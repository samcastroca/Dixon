"""M1, Elo mapped to 1X2 by an ordered logistic regression (spec section 7).

The ratings themselves are not recomputed here. Phase 3 already replays them through the
guarded history view, with the goal-difference multiplier, the between-season regression to the
mean and the promoted-team rule of spec 7.1, and stores `elo_diff` in the feature store. This
model reads that column and learns the one thing Elo does not provide: how a rating difference
turns into three probabilities.

That mapping is an ordered logistic regression, because the three outcomes are ordered
(home > draw > away) and a multinomial fit would throw that away:

    z    = slope * elo_diff / rating_scale
    P(A) = sigmoid(lower - z)
    P(D) = sigmoid(upper - z) - sigmoid(lower - z)
    P(H) = 1 - sigmoid(upper - z)

`upper` is parameterised as `lower + exp(gap)`, so `lower < upper` holds by construction rather
than by hoping the optimiser respects a constraint. Without that ordering the draw probability
could come out negative.

M1 has no scoreline distribution: a rating difference says who is likely to win, not how many
goals anybody scores. `predict_scores` says so rather than inventing one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize

from predictor.config import OptimiserSettings, Settings, get_settings
from predictor.models.base import ScoreMatrices, probability_frame

Floats = npt.NDArray[np.float64]
Counts = npt.NDArray[np.int64]

_UNFITTED = "this model has not been fitted yet"

_NO_SCORELINE = (
    "M1 has no scoreline distribution: a rating difference orders the three outcomes without "
    "saying how many goals either side scores. The goal models M2 to M4 have one."
)

#: Keeps the log of a probability finite when a fixture is nearly settled.
_FLOOR = 1e-12


def _sigmoid(values: Floats) -> Floats:
    """Numerically stable logistic, so a large rating gap cannot overflow the exponential."""
    positive = values >= 0.0
    exponential = np.exp(-np.abs(values))
    return np.asarray(
        np.where(positive, 1.0 / (1.0 + exponential), exponential / (1.0 + exponential)),
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True)
class OrderedLogit:
    """The fitted mapping from one scaled rating difference to three ordered outcomes."""

    slope: float
    lower: float
    upper: float

    def probabilities(self, scaled: Floats) -> Floats:
        """One row of home, draw and away probabilities per rating difference."""
        latent = self.slope * np.asarray(scaled, dtype=np.float64)
        away = _sigmoid(self.lower - latent)
        not_home = _sigmoid(self.upper - latent)
        # A home win is `sigmoid(latent - upper)` rather than `1 - not_home`. The two are the
        # same number, but only the first takes the same branch of the stable sigmoid as the
        # away probability, so a level fixture under mirrored cutpoints comes out exactly even
        # instead of one bit apart. That is acceptance test 3 for M1.
        home = _sigmoid(latent - self.upper)
        rows = np.column_stack([home, not_home - away, away])
        # The subtraction can leave a tiny negative where the two sigmoids coincide.
        np.clip(rows, _FLOOR, None, out=rows)
        return np.asarray(rows / rows.sum(axis=1, keepdims=True), dtype=np.float64)

    @classmethod
    def fit(cls, scaled: Floats, outcomes: Counts, *, optimiser: OptimiserSettings) -> OrderedLogit:
        """Maximum likelihood over the slope and the two cutpoints, with an analytic gradient."""
        values = np.asarray(scaled, dtype=np.float64)
        codes = np.asarray(outcomes, dtype=np.int64)
        if values.shape != codes.shape:
            msg = "the ratings and the outcomes must line up one to one"
            raise ValueError(msg)
        if values.size == 0:
            msg = "an ordered logistic regression cannot be fitted on no matches"
            raise ValueError(msg)
        result = minimize(
            _objective,
            # A positive slope and cutpoints either side of zero: the shape every league has.
            np.array([1.0, -0.5, np.log(1.0)], dtype=np.float64),
            args=(values, codes, optimiser.ridge),
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": optimiser.max_iter,
                "ftol": optimiser.tolerance,
                "gtol": optimiser.tolerance,
            },
        )
        slope, lower, gap = (float(value) for value in result.x)
        return cls(slope=slope, lower=lower, upper=lower + float(np.exp(gap)))


def _objective(
    parameters: Floats, scaled: Floats, outcomes: Counts, ridge: float
) -> tuple[float, Floats]:
    """Negative log likelihood of the ordered logit and its gradient, in one pass."""
    slope, lower, gap = float(parameters[0]), float(parameters[1]), float(parameters[2])
    spread = float(np.exp(gap))
    upper = lower + spread

    latent = slope * scaled
    away = _sigmoid(lower - latent)
    not_home = _sigmoid(upper - latent)
    # Derivative of each sigmoid with respect to its own argument.
    away_slope = away * (1.0 - away)
    not_home_slope = not_home * (1.0 - not_home)

    chosen = np.empty_like(latent)
    # d(chosen probability) / d(lower), d/d(upper) and d/d(latent), per match.
    d_lower = np.zeros_like(latent)
    d_upper = np.zeros_like(latent)
    d_latent = np.zeros_like(latent)

    is_home = outcomes == 0
    is_draw = outcomes == 1
    is_away = outcomes == 2

    # Same value as `1 - not_home`, and the same reason as in `probabilities`.
    chosen[is_home] = _sigmoid(latent - upper)[is_home]
    d_upper[is_home] = -not_home_slope[is_home]
    d_latent[is_home] = not_home_slope[is_home]

    chosen[is_draw] = not_home[is_draw] - away[is_draw]
    d_upper[is_draw] = not_home_slope[is_draw]
    d_lower[is_draw] = -away_slope[is_draw]
    d_latent[is_draw] = away_slope[is_draw] - not_home_slope[is_draw]

    chosen[is_away] = away[is_away]
    d_lower[is_away] = away_slope[is_away]
    d_latent[is_away] = -away_slope[is_away]

    chosen = np.clip(chosen, _FLOOR, None)
    total = -float(np.sum(np.log(chosen)))
    inverse = -1.0 / chosen

    gradient_slope = float(np.sum(inverse * d_latent * scaled))
    # `upper = lower + exp(gap)`, so a move in `lower` shifts both cutpoints.
    gradient_lower = float(np.sum(inverse * (d_lower + d_upper)))
    gradient_gap = float(np.sum(inverse * d_upper) * spread)

    if ridge:
        total += ridge * (slope**2 + lower**2 + upper**2)
        gradient_slope += 2.0 * ridge * slope
        gradient_lower += 2.0 * ridge * (lower + upper)
        gradient_gap += 2.0 * ridge * upper * spread

    return total, np.array([gradient_slope, gradient_lower, gradient_gap], dtype=np.float64)


class EloModel:
    """M1: the phase 3 Elo ratings, mapped to 1X2 by an ordered logistic regression."""

    name = "elo"
    deployable = True
    needs_market = False

    __slots__ = ("_mapping", "_matches", "_settings")

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._mapping: OrderedLogit | None = None
        self._matches = 0

    @property
    def mapping(self) -> OrderedLogit:
        if self._mapping is None:
            raise RuntimeError(_UNFITTED)
        return self._mapping

    def fit(self, matches: pd.DataFrame, features: pd.DataFrame | None) -> EloModel:
        elo = self._settings.models.elo
        ratings, outcomes = self._training_rows(matches, features)
        self._mapping = OrderedLogit.fit(ratings, outcomes, optimiser=elo.optimiser)
        self._matches = int(ratings.size)
        return self

    def predict_proba(self, fixtures: pd.DataFrame, features: pd.DataFrame | None) -> pd.DataFrame:
        ratings = self._ratings_of(fixtures, features)
        return probability_frame(
            [int(match_id) for match_id in fixtures["match_id"]],
            self.mapping.probabilities(ratings),
        )

    def predict_scores(
        self,
        fixtures: pd.DataFrame,
        features: pd.DataFrame | None = None,
        max_goals: int | None = None,
    ) -> ScoreMatrices:
        del fixtures, features, max_goals
        raise NotImplementedError(_NO_SCORELINE)

    def fitted_parameters(self) -> Mapping[str, float]:
        """The mapping the report shows: how steeply a rating gap moves the three outcomes."""
        mapping = self.mapping
        return {
            "slope": mapping.slope,
            "lower_cutpoint": mapping.lower,
            "upper_cutpoint": mapping.upper,
            "rating_scale": self._settings.models.elo.rating_scale,
            "matches": float(self._matches),
        }

    def _training_rows(
        self, matches: pd.DataFrame, features: pd.DataFrame | None
    ) -> tuple[Floats, Counts]:
        """Rating differences and outcomes of every training match that has both."""
        if "outcome" not in matches:
            msg = "M1 needs the `outcome` column of the training frame"
            raise ValueError(msg)
        ratings = self._ratings_of(matches, features)
        outcomes = matches["outcome"].to_numpy(dtype="float64")
        known = np.isfinite(outcomes) & np.isfinite(ratings)
        if not known.any():
            msg = "M1 was given no training match with both a rating difference and a result"
            raise ValueError(msg)
        return ratings[known], outcomes[known].astype(np.int64)

    def _ratings_of(self, frame: pd.DataFrame, features: pd.DataFrame | None) -> Floats:
        """The scaled rating difference of every row, in the frame's own order.

        A missing feature store is an error rather than a shrug: predicting without the signal
        the model is named after would silently turn M1 into a worse frequency baseline.
        """
        column = self._settings.models.elo.feature
        if features is None:
            msg = (
                f"M1 reads `{column}` from the feature store and was handed none; build it with "
                "`make features` and keep `features.groups.elo` on"
            )
            raise ValueError(msg)
        if column not in features:
            msg = (
                f"the feature store has no `{column}` column; turn `features.groups.elo` on in "
                "config/settings.yaml and rebuild the store with `make features`"
            )
            raise ValueError(msg)
        aligned = (
            features.set_index("match_id")[column]
            .reindex(frame["match_id"].to_numpy())
            .to_numpy(dtype="float64")
        )
        return np.asarray(aligned / self._settings.models.elo.rating_scale, dtype=np.float64)
