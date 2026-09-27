"""M2, the independent Poisson model (spec section 7, acceptance tests 1, 2, 3 and 6)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from predictor.config import Settings, get_settings
from predictor.evaluation.frames import fixtures_frame, matches_frame
from predictor.models.poisson import PoissonModel, TeamStrength
from predictor.models.scorelines import outcome_probabilities, score_matrix, truncation_deficit
from tests.unit.simulate import (
    LeagueParameters,
    Recovery,
    balanced_league,
    recovery,
    simulate_seasons,
)

#: Acceptance test 1, measured over `SEEDS` independent three-season simulations of a
#: twenty-team league. Every figure here was measured rather than hoped for: a team plays 114
#: matches at roughly 1.3 goals, which puts the standard error of its strength near 0.08, so a
#: single fit scores an RMSE of 0.073 to 0.085 and averaging eight of them leaves 0.03.
#:
#: The bias is what "recovers" means; the per-fit figures only bound the variance.
STRENGTH_BIAS_RMSE = 0.05
STRENGTH_WORST_FIT_RMSE = 0.10
MINIMUM_CORRELATION = 0.85
#: The intercept and the home advantage are estimated from every goal at once, so averaging
#: leaves them an order of magnitude better determined than any single team's strength.
SCALAR_TOLERANCE = 0.03

#: Acceptance test 2.
SUM_TOLERANCE = 1e-9
TRUNCATION_TOLERANCE = 1e-4
#: What `config/settings.yaml` configures; see `test_scorelines.py` for why it is not 10.
MAX_GOALS = 12

TEAMS = 20
SEASONS = 3
SEED = 20260926
#: Independent simulations the recovery is averaged over. Eight fits take under two seconds.
SEEDS = tuple(range(1, 9))


@pytest.fixture(scope="module")
def truth() -> LeagueParameters:
    return balanced_league(TEAMS).centred()


@pytest.fixture(scope="module")
def fitted(truth: LeagueParameters) -> PoissonModel:
    """One fit, for everything that is about the model's behaviour rather than its accuracy."""
    records = simulate_seasons(truth, seasons=SEASONS, seed=SEED)
    return PoissonModel(get_settings()).fit(matches_frame(records), None)


@pytest.fixture(scope="module")
def recovered(truth: LeagueParameters) -> Recovery:
    """One fit per seed, summarised: this is what acceptance test 1 is measured on."""
    settings = get_settings()
    return recovery(
        truth,
        [
            PoissonModel(settings)
            .fit(matches_frame(simulate_seasons(truth, seasons=SEASONS, seed=seed)), None)
            .strength
            for seed in SEEDS
        ],
    )


class TestParameterRecovery:
    """Acceptance test 1: the fit finds the league it was generated from."""

    def test_team_strengths_are_recovered(self, recovered: Recovery) -> None:
        assert recovered.bias_rmse < STRENGTH_BIAS_RMSE

    def test_no_single_fit_is_wild(self, recovered: Recovery) -> None:
        assert recovered.worst_fit_rmse < STRENGTH_WORST_FIT_RMSE

    def test_home_advantage_is_recovered(
        self, recovered: Recovery, truth: LeagueParameters
    ) -> None:
        assert recovered.home_advantage == pytest.approx(truth.home_advantage, abs=SCALAR_TOLERANCE)

    def test_the_intercept_is_recovered(self, recovered: Recovery, truth: LeagueParameters) -> None:
        assert recovered.intercept == pytest.approx(truth.intercept, abs=SCALAR_TOLERANCE)

    def test_the_ranking_is_recovered(self, recovered: Recovery) -> None:
        # Not the exact order: twenty strengths spaced 0.037 apart cannot be sorted correctly
        # from a standard error of 0.08, and demanding it would test luck. The ranking has to
        # agree strongly, which is what a league table built from the fit would need.
        assert recovered.worst_correlation > MINIMUM_CORRELATION


class TestSumToZero:
    """The constraint of spec section 7 holds exactly, because it is parameterised in."""

    def test_attack_and_defence_each_sum_to_zero(self, fitted: PoissonModel) -> None:
        assert sum(fitted.strength.attack.values()) == pytest.approx(0.0, abs=1e-10)
        assert sum(fitted.strength.defence.values()) == pytest.approx(0.0, abs=1e-10)


class TestProbabilities:
    """Acceptance test 2, through the model rather than through the matrix."""

    def test_every_row_sums_to_one(self, fitted: PoissonModel, truth: LeagueParameters) -> None:
        records = simulate_seasons(truth, seasons=1, seed=SEED + 1)
        frame = fixtures_frame(
            [record.as_fixture() for record in records[:50]], as_of=records[0].kickoff_utc
        )
        answer = fitted.predict_proba(frame, None)
        totals = answer[["p_home", "p_draw", "p_away"]].to_numpy().sum(axis=1)
        assert np.allclose(totals, 1.0, atol=SUM_TOLERANCE, rtol=0.0)
        assert len(answer) == 50

    def test_score_matrices_keep_only_their_truncation_error(
        self, fitted: PoissonModel, truth: LeagueParameters
    ) -> None:
        records = simulate_seasons(truth, seasons=1, seed=SEED + 1)
        frame = fixtures_frame(
            [record.as_fixture() for record in records[:20]], as_of=records[0].kickoff_utc
        )
        matrices = fitted.predict_scores(frame, None, max_goals=MAX_GOALS)
        assert len(matrices) == 20
        for matrix in matrices.values():
            assert 0.0 <= truncation_deficit(matrix) < TRUNCATION_TOLERANCE

    def test_the_matrix_and_the_row_agree(
        self, fitted: PoissonModel, truth: LeagueParameters
    ) -> None:
        records = simulate_seasons(truth, seasons=1, seed=SEED + 1)
        frame = fixtures_frame(
            [record.as_fixture() for record in records[:10]], as_of=records[0].kickoff_utc
        )
        answer = fitted.predict_proba(frame, None).set_index("match_id")
        for match_id, matrix in fitted.predict_scores(frame, None).items():
            expected = outcome_probabilities(matrix)
            row = answer.loc[match_id]
            assert (row["p_home"], row["p_draw"], row["p_away"]) == pytest.approx(expected)


class TestSymmetry:
    """Acceptance test 3: no home advantage and no difference means no preference."""

    def test_identical_teams_without_home_advantage_are_a_coin_flip(
        self, settings: Settings
    ) -> None:
        strength = TeamStrength(
            attack={1: 0.0, 2: 0.0},
            defence={1: 0.0, 2: 0.0},
            intercept=0.3,
            home_advantage=0.0,
            newcomer_attack=0.0,
            newcomer_defence=0.0,
        )
        model = PoissonModel.from_strength(strength, settings)
        home_rate, away_rate = model.rates(1, 2)
        assert home_rate == away_rate
        p_home, _, p_away = outcome_probabilities(
            score_matrix(home_rate, away_rate, max_goals=settings.models.max_goals)
        )
        assert p_home == p_away

    def test_home_advantage_breaks_the_tie_towards_the_home_side(self, settings: Settings) -> None:
        strength = TeamStrength(
            attack={1: 0.0, 2: 0.0},
            defence={1: 0.0, 2: 0.0},
            intercept=0.3,
            home_advantage=0.25,
            newcomer_attack=0.0,
            newcomer_defence=0.0,
        )
        model = PoissonModel.from_strength(strength, settings)
        p_home, _, p_away = outcome_probabilities(
            score_matrix(*model.rates(1, 2), max_goals=settings.models.max_goals)
        )
        assert p_home > p_away


class TestDeterminism:
    """Acceptance test 6: the same data gives the same parameters, every time."""

    def test_two_fits_on_the_same_data_are_identical(
        self, truth: LeagueParameters, settings: Settings
    ) -> None:
        frame = matches_frame(simulate_seasons(truth, seasons=2, seed=SEED))
        first = PoissonModel(settings).fit(frame, None)
        second = PoissonModel(settings).fit(frame, None)
        assert first.fitted_parameters() == second.fitted_parameters()
        assert first.strength.attack == second.strength.attack
        assert first.strength.defence == second.strength.defence

    def test_the_row_order_of_the_training_frame_does_not_matter(
        self, truth: LeagueParameters, settings: Settings
    ) -> None:
        records = simulate_seasons(truth, seasons=2, seed=SEED)
        forward = PoissonModel(settings).fit(matches_frame(records), None)
        backward = PoissonModel(settings).fit(matches_frame(records[::-1]), None)
        assert forward.strength.intercept == pytest.approx(backward.strength.intercept, abs=1e-8)


class TestPromotedTeams:
    """Spec section 7.1: a team with no history is not treated as an average team."""

    def test_a_newcomer_inherits_the_strength_of_the_relegated(self, settings: Settings) -> None:
        truth = balanced_league(TEAMS).centred()
        records = simulate_seasons(truth, seasons=4, seed=SEED, newcomers_per_season=3)
        model = PoissonModel(settings).fit(matches_frame(records), None)
        # Below the league average, because the teams that go down are the weak ones.
        assert model.strength.newcomer_attack < 0.0
        assert model.strength.newcomer_defence > 0.0

    def test_an_unseen_team_is_predicted_rather_than_refused(self, settings: Settings) -> None:
        truth = balanced_league(TEAMS).centred()
        records = simulate_seasons(truth, seasons=3, seed=SEED, newcomers_per_season=3)
        model = PoissonModel(settings).fit(matches_frame(records), None)
        unseen = max(model.strength.attack) + 100
        fixture = replace(records[0].as_fixture(), home_team_id=unseen)
        frame = fixtures_frame([fixture], as_of=fixture.kickoff_utc)
        answer = model.predict_proba(frame, None)
        assert len(answer) == 1
        assert answer[["p_home", "p_draw", "p_away"]].to_numpy().sum() == pytest.approx(1.0)


class TestFittedParameters:
    """What the report and MLflow are handed after every fit."""

    def test_the_scalars_are_reported(self, fitted: PoissonModel) -> None:
        reported = fitted.fitted_parameters()
        assert set(reported) >= {"intercept", "home_advantage", "teams"}
        assert reported["teams"] == TEAMS
