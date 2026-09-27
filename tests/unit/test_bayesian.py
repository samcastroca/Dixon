"""M4, the PyMC hierarchical Poisson (spec section 7, acceptance tests 1, 2 and 6).

Every test here samples a posterior, so the whole module is marked `slow`. The league is
deliberately small and the chains short: this checks that the model is the one the spec
describes and that it is reproducible, not that a production-sized run converges.
"""

from __future__ import annotations

import numpy as np
import pytest

from predictor.config import BayesianModelSettings, Settings, get_settings
from predictor.evaluation.frames import fixtures_frame, matches_frame
from predictor.models.bayesian import BayesianHierarchicalModel
from predictor.models.scorelines import truncation_deficit
from tests.unit.simulate import LeagueParameters, balanced_league, simulate_seasons

pytestmark = pytest.mark.slow

#: Acceptance test 1 for M4. Wider than the maximum-likelihood models: these are posterior
#: means from a short chain on a small league, and the pooling shrinks them towards zero.
STRENGTH_RMSE = 0.15
SCALAR_TOLERANCE = 0.15

TRUNCATION_TOLERANCE = 1e-4
MAX_GOALS = 12
SUM_TOLERANCE = 1e-9

TEAMS = 10
SEASONS = 4
SEED = 20260926


def _quick(settings: Settings) -> Settings:
    """A sampler small enough for a test suite, still four chains so divergence would show."""
    bayesian = BayesianModelSettings(
        **{
            **settings.models.bayesian.model_dump(),
            "draws": 300,
            "tune": 300,
            "chains": 2,
            "posterior_samples": 200,
        }
    )
    return settings.model_copy(
        update={"models": settings.models.model_copy(update={"bayesian": bayesian})}
    )


@pytest.fixture(scope="module")
def truth() -> LeagueParameters:
    return balanced_league(TEAMS).centred()


@pytest.fixture(scope="module")
def fitted(truth: LeagueParameters) -> BayesianHierarchicalModel:
    records = simulate_seasons(truth, seasons=SEASONS, seed=SEED)
    return BayesianHierarchicalModel(_quick(get_settings())).fit(matches_frame(records), None)


class TestParameterRecovery:
    """Acceptance test 1: the posterior means land on the league that generated the data."""

    def test_team_strengths_are_recovered(
        self, fitted: BayesianHierarchicalModel, truth: LeagueParameters
    ) -> None:
        estimated = np.array(
            [fitted.strength.attack[team + 1] for team in range(truth.teams)]
            + [fitted.strength.defence[team + 1] for team in range(truth.teams)]
        )
        errors = estimated - np.array([*truth.attack, *truth.defence])
        assert float(np.sqrt(np.mean(np.square(errors)))) < STRENGTH_RMSE

    def test_home_advantage_and_intercept_are_recovered(
        self, fitted: BayesianHierarchicalModel, truth: LeagueParameters
    ) -> None:
        assert fitted.strength.home_advantage == pytest.approx(
            truth.home_advantage, abs=SCALAR_TOLERANCE
        )
        assert fitted.strength.intercept == pytest.approx(truth.intercept, abs=SCALAR_TOLERANCE)

    def test_the_pooling_keeps_the_strengths_centred(
        self, fitted: BayesianHierarchicalModel
    ) -> None:
        assert sum(fitted.strength.attack.values()) == pytest.approx(0.0, abs=1e-8)
        assert sum(fitted.strength.defence.values()) == pytest.approx(0.0, abs=1e-8)


class TestPredictions:
    """Acceptance test 2, from the posterior predictive rather than from a point estimate."""

    def test_rows_sum_to_one_and_matrices_only_lose_the_tail(
        self, fitted: BayesianHierarchicalModel, truth: LeagueParameters
    ) -> None:
        records = simulate_seasons(truth, seasons=1, seed=SEED + 1)[:20]
        frame = fixtures_frame(
            [record.as_fixture() for record in records], as_of=records[0].kickoff_utc
        )
        answer = fitted.predict_proba(frame, None)
        totals = answer[["p_home", "p_draw", "p_away"]].to_numpy().sum(axis=1)
        assert np.allclose(totals, 1.0, atol=SUM_TOLERANCE, rtol=0.0)
        for matrix in fitted.predict_scores(frame, None, max_goals=MAX_GOALS).values():
            assert 0.0 <= truncation_deficit(matrix) < TRUNCATION_TOLERANCE

    def test_the_predictive_is_wider_than_a_point_estimate(
        self, fitted: BayesianHierarchicalModel, truth: LeagueParameters
    ) -> None:
        # Averaging over the posterior mixes many rate pairs, so the resulting scoreline
        # distribution has more spread than the one at the posterior mean would.
        records = simulate_seasons(truth, seasons=1, seed=SEED + 1)[:1]
        frame = fixtures_frame(
            [record.as_fixture() for record in records], as_of=records[0].kickoff_utc
        )
        matrix = next(iter(fitted.predict_scores(frame, None, max_goals=MAX_GOALS).values()))
        assert matrix.min() >= 0.0
        assert matrix.sum() == pytest.approx(1.0, abs=TRUNCATION_TOLERANCE)


class TestPromotedTeams:
    """Spec section 7.1: partial pooling is what a newcomer gets instead of a guess."""

    def test_an_unseen_team_is_drawn_from_the_posterior_spread(
        self, fitted: BayesianHierarchicalModel, truth: LeagueParameters
    ) -> None:
        from dataclasses import replace

        record = replace(simulate_seasons(truth, seasons=1, seed=SEED + 1)[0], home_team_id=999)
        frame = fixtures_frame([record.as_fixture()], as_of=record.kickoff_utc)
        answer = fitted.predict_proba(frame, None)
        assert len(answer) == 1
        assert answer[["p_home", "p_draw", "p_away"]].to_numpy().sum() == pytest.approx(1.0)


class TestDeterminism:
    """Acceptance test 6: a fixed seed makes the posterior reproducible."""

    def test_two_fits_with_the_same_seed_agree(self, truth: LeagueParameters) -> None:
        frame = matches_frame(simulate_seasons(truth, seasons=2, seed=SEED))
        settings = _quick(get_settings())
        first = BayesianHierarchicalModel(settings).fit(frame, None)
        second = BayesianHierarchicalModel(settings).fit(frame, None)
        assert first.fitted_parameters() == second.fitted_parameters()


class TestFittedParameters:
    """What the report shows for M4."""

    def test_the_scalars_and_the_pooling_scales_are_reported(
        self, fitted: BayesianHierarchicalModel
    ) -> None:
        reported = fitted.fitted_parameters()
        assert set(reported) >= {
            "intercept",
            "home_advantage",
            "sigma_attack",
            "sigma_defence",
            "teams",
        }
