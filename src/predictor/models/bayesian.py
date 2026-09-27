"""M4, a hierarchical Poisson fitted with PyMC (spec section 7).

The likelihood is M2's. What changes is what happens to a team the data barely covers:

    mu, home                   ~ Normal(0, intercept_sigma / home_sigma)
    sigma_attack, sigma_defence ~ HalfNormal(strength_sigma)
    attack_raw, defence_raw    ~ Normal(0, 1)                     one per team
    attack   = centre(attack_raw) * sigma_attack
    defence  = centre(defence_raw) * sigma_defence
    goals    ~ Poisson(exp(log lambda))                           M2's two equations

`sigma_attack` is learned rather than assumed, so the league itself decides how far apart its
teams really are, and every team's strength is pulled towards the average by an amount the data
justifies. That partial pooling is what spec 7.1 names as the answer for promoted teams: a side
with almost no history ends up near the league mean not because someone chose a prior for it
but because nothing in the data pulls it away. A side with no history at all is predicted by
drawing its strength from `Normal(0, sigma)` using the posterior spread, which carries the right
uncertainty instead of pretending the newcomer is exactly average.

The strengths are centred inside the model (non-centred parameterisation, then mean-subtracted)
so the sum-to-zero constraint of M2 still holds and `mu` stays identified.

Sampling is NUTS through nutpie, which compiles the model with numba and therefore needs no C
compiler. The seed comes from settings, so a rerun on the same data reproduces the posterior.
Runtime is in the README; the backtest of this model is a slow job, not an interactive one.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.special import factorial

from predictor.config import BayesianModelSettings, Settings, get_settings
from predictor.logging import get_logger
from predictor.models.base import ScoreMatrices, probability_frame
from predictor.models.poisson import Design, TeamStrength
from predictor.models.scorelines import outcome_probabilities

# PyTensor warns once about a missing C compiler. nutpie compiles with numba instead, so the
# warning is noise in the middle of the CLI's own output.
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

import pymc as pm

logger = get_logger(__name__)

Floats = npt.NDArray[np.float64]

_UNFITTED = "this model has not been fitted yet"

#: The variables whose posterior draws the prediction needs.
DRAWN = ("intercept", "home_advantage", "attack", "defence", "sigma_attack", "sigma_defence")


class Posterior:
    """The posterior draws a prediction reads, flattened over chains.

    Holding the draws rather than only their means is the point of M4: the scoreline matrix is a
    posterior *predictive*, averaged over the whole posterior, so the spread of the parameters
    reaches the forecast instead of being collapsed first.
    """

    __slots__ = ("_index", "attack", "defence", "home_advantage", "intercept", "sigma")

    def __init__(
        self,
        teams: tuple[int, ...],
        intercept: Floats,
        home_advantage: Floats,
        attack: Floats,
        defence: Floats,
        sigma: tuple[Floats, Floats],
    ) -> None:
        self._index = {team: position for position, team in enumerate(teams)}
        self.intercept = intercept
        self.home_advantage = home_advantage
        self.attack = attack
        self.defence = defence
        self.sigma = sigma

    @property
    def draws(self) -> int:
        return int(self.intercept.shape[0])

    def rates(
        self, home_team_id: int, away_team_id: int, rng: np.random.Generator
    ) -> tuple[Floats, Floats]:
        """One pair of goal rates per posterior draw, for one fixture.

        An unknown team is given a strength drawn from `Normal(0, sigma)` at each draw, which is
        exactly what the hierarchy says about a team it has never seen (spec section 7.1).
        """
        home_attack, home_defence = self._strength(home_team_id, rng)
        away_attack, away_defence = self._strength(away_team_id, rng)
        return (
            np.exp(self.intercept + self.home_advantage + home_attack + away_defence),
            np.exp(self.intercept + away_attack + home_defence),
        )

    def _strength(self, team_id: int, rng: np.random.Generator) -> tuple[Floats, Floats]:
        position = self._index.get(team_id)
        if position is not None:
            return self.attack[:, position], self.defence[:, position]
        sigma_attack, sigma_defence = self.sigma
        return (
            rng.normal(0.0, sigma_attack),
            rng.normal(0.0, sigma_defence),
        )


class BayesianHierarchicalModel:
    """M4: hierarchical Poisson with partial pooling of attack and defence."""

    name = "bayesian_hierarchical"
    deployable = True
    needs_market = False

    __slots__ = ("_diagnostics", "_posterior", "_settings", "_strength")

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._posterior: Posterior | None = None
        self._strength: TeamStrength | None = None
        self._diagnostics: Mapping[str, float] = {}

    @property
    def strength(self) -> TeamStrength:
        """The posterior means, in the same shape M2 and M3 report."""
        if self._strength is None:
            raise RuntimeError(_UNFITTED)
        return self._strength

    @property
    def posterior(self) -> Posterior:
        if self._posterior is None:
            raise RuntimeError(_UNFITTED)
        return self._posterior

    def fit(
        self, matches: pd.DataFrame, features: pd.DataFrame | None
    ) -> BayesianHierarchicalModel:
        del features  # M4 reads goals, not the feature store
        sampler = self._settings.models.bayesian
        design = Design.from_frame(matches)
        trace = _sample(design, sampler)

        self._posterior = Posterior(
            teams=design.teams,
            intercept=trace["intercept"],
            home_advantage=trace["home_advantage"],
            attack=trace["attack"],
            defence=trace["defence"],
            sigma=(trace["sigma_attack"], trace["sigma_defence"]),
        )
        self._strength = TeamStrength(
            attack={
                team: float(trace["attack"][:, position].mean())
                for position, team in enumerate(design.teams)
            },
            defence={
                team: float(trace["defence"][:, position].mean())
                for position, team in enumerate(design.teams)
            },
            intercept=float(trace["intercept"].mean()),
            home_advantage=float(trace["home_advantage"].mean()),
            # A newcomer is not a point estimate in M4; it is the pooling prior, whose mean is
            # the league average. `Posterior.rates` is where the spread actually enters.
            newcomer_attack=0.0,
            newcomer_defence=0.0,
        )
        self._diagnostics = {
            "sigma_attack": float(trace["sigma_attack"].mean()),
            "sigma_defence": float(trace["sigma_defence"].mean()),
            "draws": float(self._posterior.draws),
        }
        logger.info(
            "bayesian.fitted",
            teams=len(design.teams),
            matches=design.matches,
            draws=self._posterior.draws,
            sigma_attack=round(self._diagnostics["sigma_attack"], 4),
            sigma_defence=round(self._diagnostics["sigma_defence"], 4),
            home_advantage=round(self._strength.home_advantage, 4),
        )
        return self

    def predict_proba(self, fixtures: pd.DataFrame, features: pd.DataFrame | None) -> pd.DataFrame:
        del features
        matrices = self.predict_scores(fixtures, None)
        match_ids = [int(match_id) for match_id in fixtures["match_id"]]
        return probability_frame(
            match_ids, [outcome_probabilities(matrices[match_id]) for match_id in match_ids]
        )

    def predict_scores(
        self,
        fixtures: pd.DataFrame,
        features: pd.DataFrame | None = None,
        max_goals: int | None = None,
    ) -> ScoreMatrices:
        """The posterior predictive scoreline matrix, averaged over the posterior draws."""
        del features
        sampler = self._settings.models.bayesian
        grid = max_goals or self._settings.models.max_goals
        posterior = self.posterior
        # A fixed seed per fit, so an unseen team's drawn strength is reproducible too.
        rng = np.random.default_rng(sampler.seed)
        wanted = min(sampler.posterior_samples, posterior.draws)
        chosen = np.linspace(0, posterior.draws - 1, wanted).astype(np.int64)

        matrices: ScoreMatrices = {}
        for match_id, home, away in zip(
            fixtures["match_id"], fixtures["home_team_id"], fixtures["away_team_id"], strict=True
        ):
            rate_home, rate_away = posterior.rates(int(home), int(away), rng)
            matrices[int(match_id)] = _averaged_matrix(rate_home[chosen], rate_away[chosen], grid)
        return matrices

    def fitted_parameters(self) -> Mapping[str, float]:
        """What the report shows for M4, including how wide the pooling turned out to be."""
        strength = self.strength
        return {
            "intercept": strength.intercept,
            "home_advantage": strength.home_advantage,
            "teams": float(len(strength.attack)),
            **self._diagnostics,
        }


def _averaged_matrix(rate_home: Floats, rate_away: Floats, max_goals: int) -> Floats:
    """The predictive scoreline: the mean over draws of each draw's own Poisson product."""
    goals = np.arange(max_goals + 1)
    # Shape (draws, goals) for each side, then one outer product per draw, averaged.
    home = np.exp(-rate_home[:, None]) * np.power(rate_home[:, None], goals) / _factorials(goals)
    away = np.exp(-rate_away[:, None]) * np.power(rate_away[:, None], goals) / _factorials(goals)
    return np.asarray(np.einsum("di,dj->ij", home, away) / rate_home.shape[0], dtype=np.float64)


def _factorials(goals: npt.NDArray[np.int_]) -> Floats:
    """`goals!` as floats. The grid is a dozen wide, so this is exact and cheap."""
    return np.asarray(factorial(goals), dtype=np.float64)


def _sample(design: Design, sampler: BayesianModelSettings) -> Mapping[str, Floats]:
    """Draw the posterior and return every variable flattened over chains."""
    teams = len(design.teams)
    with pm.Model() as model:
        intercept = pm.Normal("intercept", 0.0, sampler.intercept_sigma)
        home_advantage = pm.Normal("home_advantage", 0.0, sampler.home_sigma)
        sigma_attack = pm.HalfNormal("sigma_attack", sampler.strength_sigma)
        sigma_defence = pm.HalfNormal("sigma_defence", sampler.strength_sigma)

        # Non-centred, then mean-subtracted: the pooling is learned and the sum-to-zero
        # constraint of M2 still holds, so `intercept` means the same thing in both models.
        attack_raw = pm.Normal("attack_raw", 0.0, 1.0, shape=teams)
        defence_raw = pm.Normal("defence_raw", 0.0, 1.0, shape=teams)
        attack = pm.Deterministic("attack", (attack_raw - attack_raw.mean()) * sigma_attack)
        defence = pm.Deterministic("defence", (defence_raw - defence_raw.mean()) * sigma_defence)

        log_home = intercept + home_advantage + attack[design.home] + defence[design.away]
        log_away = intercept + attack[design.away] + defence[design.home]
        pm.Poisson("home_goals", mu=pm.math.exp(log_home), observed=design.home_goals)
        pm.Poisson("away_goals", mu=pm.math.exp(log_away), observed=design.away_goals)

        trace = pm.sample(
            draws=sampler.draws,
            tune=sampler.tune,
            chains=sampler.chains,
            target_accept=sampler.target_accept,
            nuts_sampler=sampler.nuts_sampler,
            random_seed=sampler.seed,
            progressbar=False,
        )
    del model
    posterior = trace.posterior
    return {name: _flattened(posterior[name].to_numpy()) for name in DRAWN}


def _flattened(values: npt.NDArray[np.float64]) -> Floats:
    """Chains and draws collapsed into one axis, keeping any per-team axis behind it."""
    return np.asarray(values.reshape(-1, *values.shape[2:]), dtype=np.float64)
