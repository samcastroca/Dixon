"""M3, Dixon-Coles (spec section 7).

Two things sit on top of M2, and both of them are about football rather than about statistics:

* the **low-score correction** `tau(x, y; rho)`, because 0-0, 1-0, 0-1 and 1-1 do not happen as
  often as two independent Poissons say they do. A negative rho, which is what leagues
  actually show, moves mass onto 0-0 and 1-1 and so onto the draw;
* **time-decay weights** `w = exp(-xi * days / 365)`, because a result from four years ago says
  less about a squad than one from April. Age is measured back from the newest training match,
  so the weights depend on the fold's cut-off and never on the calendar.

`xi` is tuned per league by an inner walk-forward over the *training* seasons only, which is
the one part of the model that could leak if it were written carelessly: the search never looks
at a season the outer fold has not already handed over. Because the model is registered behind
`PerCompetition`, the tuning is per league by construction, with no league named anywhere.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from predictor.config import Settings, get_settings
from predictor.evaluation.metrics import rps
from predictor.logging import get_logger
from predictor.models.base import ScoreMatrices, probability_frame
from predictor.models.poisson import (
    Design,
    Estimate,
    TeamStrength,
    as_strength,
    fit_goal_model,
    newcomer_strength,
    outcome_rows,
    score_matrices,
)

logger = get_logger(__name__)

Floats = npt.NDArray[np.float64]

#: Days in the year the decay is expressed in, so `xi` is a rate per year (spec section 7, M3).
DAYS_PER_YEAR = 365.0

_UNFITTED = "this model has not been fitted yet"


def time_decay_weights(age_days: Floats, xi: float) -> Floats:
    """`w = exp(-xi * days / 365)`: full weight for the newest match, less for every older one."""
    if xi == 0.0:
        return np.ones_like(age_days)
    return np.asarray(np.exp(-xi * age_days / DAYS_PER_YEAR), dtype=np.float64)


class DixonColesModel:
    """M3: the Poisson model plus the low-score correction and the time decay."""

    name = "dixon_coles"
    deployable = True
    needs_market = False

    __slots__ = ("_estimate", "_settings", "_strength", "_xi")

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._strength: TeamStrength | None = None
        self._estimate: Estimate | None = None
        self._xi: float | None = None

    @property
    def strength(self) -> TeamStrength:
        if self._strength is None:
            raise RuntimeError(_UNFITTED)
        return self._strength

    @property
    def xi(self) -> float:
        """The decay the inner walk-forward chose for this league and this fold."""
        if self._xi is None:
            raise RuntimeError(_UNFITTED)
        return self._xi

    def fit(self, matches: pd.DataFrame, features: pd.DataFrame | None) -> DixonColesModel:
        del features  # M3 reads goals and kickoffs, not the feature store
        settings = self._settings.models.dixon_coles
        self._xi = self._tuned_xi(matches)
        strength, estimate = self._fitted(matches, self._xi)
        self._strength, self._estimate = strength, estimate
        logger.info(
            "dixon_coles.fitted",
            xi=self._xi,
            rho=round(strength.rho, 4),
            home_advantage=round(strength.home_advantage, 4),
            teams=len(strength.attack),
            matches=len(matches),
            converged=estimate.converged,
            rho_bounds=list(settings.rho_bounds),
        )
        return self

    def _fitted(self, matches: pd.DataFrame, xi: float) -> tuple[TeamStrength, Estimate]:
        """One weighted fit at a given decay, which is the unit the search is built from."""
        settings = self._settings.models.dixon_coles
        design = Design.from_frame(matches)
        weighted = design.reweighted(time_decay_weights(design.age_days, xi))
        estimate = fit_goal_model(
            weighted, optimiser=settings.optimiser, rho_bounds=settings.rho_bounds
        )
        strength = as_strength(
            design,
            estimate,
            *newcomer_strength(design, estimate, matches, settings.promoted),
        )
        return strength, estimate

    def _tuned_xi(self, matches: pd.DataFrame) -> float:
        """The decay with the lowest RPS on an inner walk-forward of the training seasons.

        A single candidate needs no search. Anything else is scored by fitting on everything
        before each of the newest `inner_seasons` training seasons and predicting that season,
        which is the outer protocol of spec section 8 applied one level down.
        """
        tuning = self._settings.models.dixon_coles.xi
        if len(tuning.grid) == 1:
            return float(tuning.grid[0])

        seasons = Design.from_frame(matches).seasons
        inner = seasons[-tuning.inner_seasons :] if len(seasons) > tuning.inner_seasons else ()
        if not inner:
            logger.warning(
                "dixon_coles.xi_not_tuned",
                reason="the training history is shorter than the inner walk-forward needs",
                training_seasons=len(seasons),
                inner_seasons=tuning.inner_seasons,
                xi=tuning.default,
            )
            return float(tuning.default)

        scores = {
            float(candidate): self._inner_score(matches, seasons, inner, candidate)
            for candidate in tuning.grid
        }
        # Ties go to the smaller xi: less decay is the simpler model, and the order of
        # `grid` must not decide anything.
        chosen = min(sorted(scores), key=lambda candidate: scores[candidate])
        logger.info(
            "dixon_coles.xi_tuned",
            xi=chosen,
            inner_seasons=list(inner),
            scores={str(candidate): round(score, 6) for candidate, score in scores.items()},
        )
        return chosen

    def _inner_score(
        self,
        matches: pd.DataFrame,
        seasons: Sequence[int],
        inner: Sequence[int],
        xi: float,
    ) -> float:
        """Pooled RPS of one candidate decay over the inner folds, all inside the training set."""
        order = {season: position for position, season in enumerate(seasons)}
        rows: list[Floats] = []
        outcomes: list[npt.NDArray[np.int64]] = []
        max_goals = self._settings.models.max_goals
        for season in inner:
            position = order[season]
            past = matches[matches["season_id"].map(order) < position]
            held_out = matches[matches["season_id"] == season].dropna(subset=["outcome"])
            if past.empty or held_out.empty:
                continue
            strength, _ = self._fitted(past, xi)
            rows.append(
                outcome_rows(
                    strength,
                    held_out["home_team_id"].tolist(),
                    held_out["away_team_id"].tolist(),
                    max_goals,
                )
            )
            outcomes.append(held_out["outcome"].to_numpy(dtype=np.int64))
        if not rows:
            return float("inf")
        return rps(np.concatenate(rows), np.concatenate(outcomes))

    def predict_proba(self, fixtures: pd.DataFrame, features: pd.DataFrame | None) -> pd.DataFrame:
        del features
        return probability_frame(
            [int(match_id) for match_id in fixtures["match_id"]],
            outcome_rows(
                self.strength,
                fixtures["home_team_id"].tolist(),
                fixtures["away_team_id"].tolist(),
                self._settings.models.max_goals,
            ),
        )

    def predict_scores(
        self,
        fixtures: pd.DataFrame,
        features: pd.DataFrame | None = None,
        max_goals: int | None = None,
    ) -> ScoreMatrices:
        del features
        return score_matrices(self.strength, fixtures, max_goals or self._settings.models.max_goals)

    def fitted_parameters(self) -> Mapping[str, float]:
        """What the report shows per league: the home advantage, rho and the chosen xi."""
        strength = self.strength
        return {
            "intercept": strength.intercept,
            "home_advantage": strength.home_advantage,
            "rho": strength.rho,
            "xi": self.xi,
            "teams": float(len(strength.attack)),
            "newcomer_attack": strength.newcomer_attack,
            "newcomer_defence": strength.newcomer_defence,
            "converged": float(self._estimate.converged) if self._estimate else 0.0,
        }
