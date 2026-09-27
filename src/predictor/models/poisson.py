"""M2, the independent Poisson goal model (spec section 7).

    log lambda_home = mu + home_advantage + attack[home] + defence[away]
    log lambda_away = mu + attack[away] + defence[home]

Attack and defence are only identified up to a constant, so the spec constrains each of them
to sum to zero. That constraint is *parameterised in* rather than penalised: the free vector
carries `n - 1` of each and the last one is derived as minus the sum of the rest. The optimiser
therefore cannot violate it, and it holds to machine precision instead of to a tolerance.

The fit is a weighted maximum likelihood with an analytic gradient, which is also everything
M3 needs: this module owns the likelihood and Dixon-Coles adds rho and the weights on top.
Nothing in it is random, so the same training frame always produces the same parameters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize

from predictor.config import OptimiserSettings, PromotedRule, Settings, get_settings
from predictor.models.base import ScoreMatrices, probability_frame
from predictor.models.scorelines import outcome_probabilities, score_matrix, tau_terms

Floats = npt.NDArray[np.float64]
Counts = npt.NDArray[np.int64]

#: The training columns a goal model reads (`evaluation.frames.MATCH_COLUMNS`).
REQUIRED_COLUMNS = ("home_team_id", "away_team_id", "home_goals", "away_goals")

_UNFITTED = "this model has not been fitted yet"

#: A log goal rate is bounded before it is exponentiated. `exp(10)` is 22026 goals in a match,
#: so nothing inside the bound is a rate the data could ever support, and `exp()` overflows to
#: infinity not far past it. Without the bound, a step that overshoots makes the Dixon-Coles
#: gradient `inf / inf`, and a NaN gradient stops the optimiser wherever it happens to be
#: rather than pushing it back out. The likelihood at the bound is already astronomically bad,
#: which is what sends it back.
MAX_LOG_RATE = 10.0


@dataclass(frozen=True, slots=True)
class TeamStrength:
    """Everything a fitted goal model knows, and the only thing prediction reads.

    `newcomer_attack` and `newcomer_defence` are what a team with no training history gets
    (spec section 7.1). They are part of the fit rather than a prediction-time fallback,
    because what a promoted team is worth is a statement about the league, not about a fixture.
    """

    attack: Mapping[int, float]
    defence: Mapping[int, float]
    intercept: float
    home_advantage: float
    newcomer_attack: float
    newcomer_defence: float
    rho: float = 0.0

    def rates(self, home_team_id: int, away_team_id: int) -> tuple[float, float]:
        """The two goal rates of one fixture, falling back to the newcomer strength."""
        home_attack = self.attack.get(home_team_id, self.newcomer_attack)
        home_defence = self.defence.get(home_team_id, self.newcomer_defence)
        away_attack = self.attack.get(away_team_id, self.newcomer_attack)
        away_defence = self.defence.get(away_team_id, self.newcomer_defence)
        return (
            float(np.exp(self.intercept + self.home_advantage + home_attack + away_defence)),
            float(np.exp(self.intercept + away_attack + home_defence)),
        )


@dataclass(frozen=True, slots=True)
class Design:
    """A training frame reduced to the arrays the likelihood actually needs."""

    #: Every team with a training match, sorted, so the fit does not depend on row order.
    teams: tuple[int, ...]
    home: Counts
    away: Counts
    home_goals: Counts
    away_goals: Counts
    #: How long before the newest training match each one kicked off. M3 turns this into the
    #: time-decay weights; M2 ignores it.
    age_days: Floats
    weights: Floats
    #: The training seasons in chronological order. The promoted-team rule and the inner
    #: walk-forward of M3 both need to know which season came last.
    seasons: tuple[int, ...] = field(default=())

    @property
    def matches(self) -> int:
        return int(self.home.shape[0])

    def reweighted(self, weights: Floats) -> Design:
        """The same design under different match weights, which is all M3 changes."""
        return replace(self, weights=weights)

    @classmethod
    def from_frame(cls, matches: pd.DataFrame) -> Design:
        """Build the design, dropping anything without a result and sorting the teams."""
        missing = [column for column in REQUIRED_COLUMNS if column not in matches]
        if missing:
            msg = f"a goal model needs the training columns {missing}"
            raise ValueError(msg)
        played = matches.dropna(subset=["home_goals", "away_goals"])
        if played.empty:
            msg = "a goal model cannot be fitted on a training frame with no results"
            raise ValueError(msg)

        home_ids = played["home_team_id"].to_numpy(dtype=np.int64)
        away_ids = played["away_team_id"].to_numpy(dtype=np.int64)
        teams = np.unique(np.concatenate([home_ids, away_ids]))
        return cls(
            teams=tuple(int(team) for team in teams),
            home=np.searchsorted(teams, home_ids).astype(np.int64),
            away=np.searchsorted(teams, away_ids).astype(np.int64),
            home_goals=played["home_goals"].to_numpy(dtype=np.int64),
            away_goals=played["away_goals"].to_numpy(dtype=np.int64),
            age_days=_age_in_days(played),
            weights=np.ones(len(played), dtype=np.float64),
            seasons=_seasons_in_order(played),
        )


def _age_in_days(matches: pd.DataFrame) -> Floats:
    """Days between each match and the newest one in the frame; the newest one is zero."""
    kickoffs = pd.to_datetime(matches["kickoff_utc"], utc=True)
    return np.asarray(
        (kickoffs.max() - kickoffs).dt.total_seconds().to_numpy() / 86_400.0, dtype=np.float64
    )


def _seasons_in_order(matches: pd.DataFrame) -> tuple[int, ...]:
    """The training seasons, oldest first, ordered by when each one kicked off."""
    if "season_id" not in matches or "kickoff_utc" not in matches:
        return ()
    starts = matches.groupby("season_id")["kickoff_utc"].min().sort_values()
    return tuple(int(season) for season in starts.index)


@dataclass(frozen=True, slots=True)
class Estimate:
    """The raw output of the likelihood fit, before it is named and indexed by team."""

    intercept: float
    home_advantage: float
    attack: Floats
    defence: Floats
    rho: float
    converged: bool
    iterations: int


def fit_goal_model(
    design: Design,
    *,
    optimiser: OptimiserSettings,
    rho_bounds: tuple[float, float] | None = None,
) -> Estimate:
    """Weighted maximum likelihood with an analytic gradient (spec section 7, M2 and M3).

    `rho_bounds` is what turns M2 into M3: absent, the low-score correction is not part of the
    model at all; present, rho joins the free vector and is kept inside those bounds so tau
    cannot go negative.
    """
    teams = len(design.teams)
    with_rho = rho_bounds is not None
    start = np.zeros(2 * teams + (1 if with_rho else 0), dtype=np.float64)
    # A sensible intercept costs nothing and saves the optimiser a dozen iterations.
    start[0] = float(
        np.log(
            max(
                1e-6,
                (design.home_goals.mean() + design.away_goals.mean()) / 2.0,
            )
        )
    )
    bounds = None
    if with_rho:
        bounds = [(None, None)] * (2 * teams) + [rho_bounds]

    result = minimize(
        _objective,
        start,
        args=(design, optimiser.ridge, with_rho),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": optimiser.max_iter,
            "ftol": optimiser.tolerance,
            "gtol": optimiser.tolerance,
        },
    )
    values = np.asarray(result.x, dtype=np.float64)
    return Estimate(
        intercept=float(values[0]),
        home_advantage=float(values[1]),
        attack=_expanded(values[2 : teams + 1]),
        defence=_expanded(values[teams + 1 : 2 * teams]),
        rho=float(values[2 * teams]) if with_rho else 0.0,
        converged=bool(result.success),
        iterations=int(result.nit),
    )


def _expanded(free: Floats) -> Floats:
    """The `n - 1` free values plus the one the sum-to-zero constraint determines."""
    return np.concatenate([free, [-free.sum()]])


def _objective(
    values: Floats, design: Design, ridge: float, with_rho: bool
) -> tuple[float, Floats]:
    """The weighted negative log likelihood and its gradient, in one pass."""
    teams = len(design.teams)
    intercept, home_advantage = float(values[0]), float(values[1])
    attack = _expanded(values[2 : teams + 1])
    defence = _expanded(values[teams + 1 : 2 * teams])
    rho = float(values[2 * teams]) if with_rho else 0.0

    log_home = np.clip(
        intercept + home_advantage + attack[design.home] + defence[design.away],
        -MAX_LOG_RATE,
        MAX_LOG_RATE,
    )
    log_away = np.clip(
        intercept + attack[design.away] + defence[design.home], -MAX_LOG_RATE, MAX_LOG_RATE
    )
    rate_home, rate_away = np.exp(log_home), np.exp(log_away)
    weights = design.weights

    # Constant factorial terms are dropped: they shift the objective, never the optimum.
    total = float(
        np.sum(
            weights
            * (rate_home - design.home_goals * log_home + rate_away - design.away_goals * log_away)
        )
    )
    residual_home = weights * (rate_home - design.home_goals)
    residual_away = weights * (rate_away - design.away_goals)
    gradient_rho = 0.0

    if with_rho:
        values_tau, d_home, d_away, d_rho = tau_terms(
            design.home_goals, design.away_goals, rate_home, rate_away, rho
        )
        total -= float(np.sum(weights * np.log(values_tau)))
        # The chain rule through log lambda, which is what the strength parameters move.
        residual_home -= weights * rate_home * d_home / values_tau
        residual_away -= weights * rate_away * d_away / values_tau
        gradient_rho = -float(np.sum(weights * d_rho / values_tau))

    gradient_attack = np.bincount(design.home, residual_home, teams) + np.bincount(
        design.away, residual_away, teams
    )
    gradient_defence = np.bincount(design.away, residual_home, teams) + np.bincount(
        design.home, residual_away, teams
    )

    if ridge:
        # Only the team strengths are penalised. Shrinking the intercept or the home advantage
        # would bias every rate in the league towards nothing in particular.
        total += ridge * float(attack @ attack + defence @ defence)
        gradient_attack += 2.0 * ridge * attack
        gradient_defence += 2.0 * ridge * defence

    gradient = np.concatenate(
        [
            [float(np.sum(residual_home + residual_away)), float(np.sum(residual_home))],
            gradient_attack[:-1] - gradient_attack[-1],
            gradient_defence[:-1] - gradient_defence[-1],
            [gradient_rho] if with_rho else [],
        ]
    )
    return total, gradient


def newcomer_strength(
    design: Design, estimate: Estimate, matches: pd.DataFrame, rule: PromotedRule
) -> tuple[float, float]:
    """What a team with no training history is worth (spec section 7.1).

    `relegated_mean` is the Poisson analogue of the phase 3 Elo rule: a newcomer inherits the
    mean strength of the teams that went down at the end of the previous training season. That
    set is read off the training data itself -- teams that played in the older season and not
    in the newer one -- so no league roster, relegation count or competition code is needed
    anywhere in the logic.
    """
    if rule == "league_average" or len(design.seasons) < 2:
        return 0.0, 0.0
    older, newer = design.seasons[-2], design.seasons[-1]
    stayed = _teams_of(matches, newer)
    relegated = sorted(_teams_of(matches, older) - stayed)
    if not relegated:
        return 0.0, 0.0
    index = {team: position for position, team in enumerate(design.teams)}
    positions = [index[team] for team in relegated if team in index]
    if not positions:
        return 0.0, 0.0
    return float(estimate.attack[positions].mean()), float(estimate.defence[positions].mean())


def _teams_of(matches: pd.DataFrame, season_id: int) -> set[int]:
    season = matches[matches["season_id"] == season_id]
    return {
        int(team)
        for team in np.concatenate(
            [season["home_team_id"].to_numpy(), season["away_team_id"].to_numpy()]
        )
    }


class PoissonModel:
    """M2: independent Poisson goals with attack, defence, home advantage and an intercept."""

    name = "poisson"
    deployable = True
    needs_market = False

    __slots__ = ("_estimate", "_settings", "_strength")

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._strength: TeamStrength | None = None
        self._estimate: Estimate | None = None

    @classmethod
    def from_strength(
        cls, strength: TeamStrength, settings: Settings | None = None
    ) -> PoissonModel:
        """A model built from parameters instead of from data, for tests and for reloading."""
        model = cls(settings)
        model._strength = strength
        return model

    @property
    def strength(self) -> TeamStrength:
        if self._strength is None:
            raise RuntimeError(_UNFITTED)
        return self._strength

    @property
    def settings(self) -> Settings:
        return self._settings

    def fit(self, matches: pd.DataFrame, features: pd.DataFrame | None) -> PoissonModel:
        del features  # M2 reads goals, not the feature store
        design = Design.from_frame(matches)
        estimate = fit_goal_model(design, optimiser=self._settings.models.poisson.optimiser)
        self._estimate = estimate
        self._strength = as_strength(
            design,
            estimate,
            *newcomer_strength(design, estimate, matches, self._settings.models.poisson.promoted),
        )
        return self

    def rates(self, home_team_id: int, away_team_id: int) -> tuple[float, float]:
        """The two goal rates of one fixture, which is the whole of what the model predicts."""
        return self.strength.rates(home_team_id, away_team_id)

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
        """What the report and MLflow record about this fit (spec section 11, phase 5)."""
        strength = self.strength
        return {
            "intercept": strength.intercept,
            "home_advantage": strength.home_advantage,
            "teams": float(len(strength.attack)),
            "newcomer_attack": strength.newcomer_attack,
            "newcomer_defence": strength.newcomer_defence,
            "converged": float(self._estimate.converged) if self._estimate else 0.0,
        }


def as_strength(
    design: Design, estimate: Estimate, newcomer_attack: float, newcomer_defence: float
) -> TeamStrength:
    """Index the fitted arrays by team id, which is how prediction reads them."""
    return TeamStrength(
        attack={
            team: float(estimate.attack[position]) for position, team in enumerate(design.teams)
        },
        defence={
            team: float(estimate.defence[position]) for position, team in enumerate(design.teams)
        },
        intercept=estimate.intercept,
        home_advantage=estimate.home_advantage,
        newcomer_attack=newcomer_attack,
        newcomer_defence=newcomer_defence,
        rho=estimate.rho,
    )


def outcome_rows(
    strength: TeamStrength,
    home_team_ids: Sequence[int],
    away_team_ids: Sequence[int],
    max_goals: int,
) -> Floats:
    """The 1X2 rows of a set of fixtures, straight from the team ids.

    The inner walk-forward of M3 scores thousands of candidate fits, and building a frame for
    each of them would be all of the cost and none of the point.
    """
    return np.array(
        [
            outcome_probabilities(
                score_matrix(
                    *strength.rates(int(home), int(away)), max_goals=max_goals, rho=strength.rho
                )
            )
            for home, away in zip(home_team_ids, away_team_ids, strict=True)
        ],
        dtype=np.float64,
    ).reshape(len(home_team_ids), 3)


def score_matrices(strength: TeamStrength, fixtures: pd.DataFrame, max_goals: int) -> ScoreMatrices:
    """One scoreline matrix per fixture, shared by M2 and M3."""
    return {
        int(match_id): score_matrix(
            *strength.rates(int(home), int(away)), max_goals=max_goals, rho=strength.rho
        )
        for match_id, home, away in zip(
            fixtures["match_id"],
            fixtures["home_team_id"],
            fixtures["away_team_id"],
            strict=True,
        )
    }
