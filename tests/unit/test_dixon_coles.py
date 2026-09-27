"""M3, Dixon-Coles (spec section 7, acceptance tests 1, 2, 5 and 6)."""

from __future__ import annotations

import numpy as np
import pytest

from predictor.config import DixonColesModelSettings, Settings, get_settings
from predictor.evaluation.bootstrap import paired_bootstrap
from predictor.evaluation.frames import fixtures_frame, matches_frame, outcome_of
from predictor.evaluation.metrics import rps
from predictor.models.dixon_coles import DixonColesModel, time_decay_weights
from predictor.models.poisson import PoissonModel
from predictor.models.scorelines import truncation_deficit
from tests.unit.simulate import (
    LeagueParameters,
    Recovery,
    balanced_league,
    recovery,
    simulate_seasons,
)

#: Acceptance test 1, averaged over `SEEDS` independent simulations for the reason spelled out
#: in `simulate.Recovery`: three seasons give one team's strength a standard error near 0.08, so
#: a single fit measures the draw's luck. Wider than M2's figures because rho costs a degree of
#: freedom and the correction pulls on the same low scorelines the strengths are read from.
STRENGTH_BIAS_RMSE = 0.06
STRENGTH_WORST_FIT_RMSE = 0.11
SCALAR_TOLERANCE = 0.04
#: Measured: a single fit puts rho within a standard deviation of 0.050 of the truth, so
#: averaging eight leaves a standard error of 0.018. This is three of those.
RHO_TOLERANCE = 0.05

TRUNCATION_TOLERANCE = 1e-4
MAX_GOALS = 12
SUM_TOLERANCE = 1e-9

TEAMS = 20
SEASONS = 3
SEED = 20260926
RHO = -0.13
#: Independent simulations the recovery is averaged over. Each one is a fit with rho.
SEEDS = tuple(range(1, 9))


def _with_xi(settings: Settings, **overrides: object) -> Settings:
    """The same settings with the time-decay block overridden, leaving the rest alone."""
    dixon_coles = DixonColesModelSettings(
        **{
            **settings.models.dixon_coles.model_dump(),
            "xi": {**settings.models.dixon_coles.xi.model_dump(), **overrides},
        }
    )
    return settings.model_copy(
        update={"models": settings.models.model_copy(update={"dixon_coles": dixon_coles})}
    )


def _without_decay(settings: Settings) -> Settings:
    """Xi pinned to zero, which isolates rho from the weighting in a recovery test."""
    return _with_xi(settings, grid=(0.0,))


@pytest.fixture(scope="module")
def truth() -> LeagueParameters:
    return balanced_league(TEAMS, rho=RHO).centred()


@pytest.fixture(scope="module")
def fitted(truth: LeagueParameters) -> DixonColesModel:
    """One fit, for everything that is about behaviour rather than about accuracy."""
    records = simulate_seasons(truth, seasons=SEASONS, seed=SEED)
    return DixonColesModel(_without_decay(get_settings())).fit(matches_frame(records), None)


@pytest.fixture(scope="module")
def recovered(truth: LeagueParameters) -> Recovery:
    """One fit per seed, summarised: this is what acceptance test 1 is measured on."""
    settings = _without_decay(get_settings())
    return recovery(
        truth,
        [
            DixonColesModel(settings)
            .fit(matches_frame(simulate_seasons(truth, seasons=SEASONS, seed=seed)), None)
            .strength
            for seed in SEEDS
        ],
    )


class TestParameterRecovery:
    """Acceptance test 1, now including the low-score correction."""

    def test_rho_is_recovered(self, recovered: Recovery, truth: LeagueParameters) -> None:
        assert recovered.rho == pytest.approx(truth.rho, abs=RHO_TOLERANCE)

    def test_team_strengths_are_recovered(self, recovered: Recovery) -> None:
        assert recovered.bias_rmse < STRENGTH_BIAS_RMSE

    def test_no_single_fit_is_wild(self, recovered: Recovery) -> None:
        assert recovered.worst_fit_rmse < STRENGTH_WORST_FIT_RMSE

    def test_home_advantage_and_intercept_are_recovered(
        self, recovered: Recovery, truth: LeagueParameters
    ) -> None:
        assert recovered.home_advantage == pytest.approx(truth.home_advantage, abs=SCALAR_TOLERANCE)
        assert recovered.intercept == pytest.approx(truth.intercept, abs=SCALAR_TOLERANCE)

    def test_a_league_without_dependence_gives_rho_near_zero(self) -> None:
        settings = _without_decay(get_settings())
        independent = balanced_league(TEAMS).centred()
        summary = recovery(
            independent,
            [
                DixonColesModel(settings)
                .fit(matches_frame(simulate_seasons(independent, seasons=SEASONS, seed=seed)), None)
                .strength
                for seed in SEEDS
            ],
        )
        assert summary.rho == pytest.approx(0.0, abs=RHO_TOLERANCE)


class TestTimeDecay:
    """`w = exp(-xi * days / 365)`, measured back from the newest training match."""

    def test_the_newest_match_carries_full_weight(self) -> None:
        days = np.array([0.0, 365.0, 730.0])
        weights = time_decay_weights(days, xi=1.0)
        assert weights[0] == pytest.approx(1.0)

    def test_a_year_of_age_costs_exactly_one_factor_of_exp_minus_xi(self) -> None:
        weights = time_decay_weights(np.array([0.0, 365.0, 730.0]), xi=0.7)
        assert weights[1] == pytest.approx(np.exp(-0.7))
        assert weights[2] == pytest.approx(np.exp(-1.4))

    def test_xi_of_zero_weights_every_match_alike(self) -> None:
        weights = time_decay_weights(np.array([0.0, 400.0, 3000.0]), xi=0.0)
        assert np.allclose(weights, 1.0)


class TestXiTuning:
    """The search runs on training seasons only, and it finds the decay that is there."""

    def test_a_single_valued_grid_is_used_as_given(self) -> None:
        records = simulate_seasons(balanced_league(TEAMS).centred(), seasons=3, seed=SEED)
        model = DixonColesModel(_with_xi(get_settings(), grid=(2.5,))).fit(
            matches_frame(records), None
        )
        assert model.xi == 2.5

    def test_a_league_that_never_changes_prefers_little_decay(self) -> None:
        # The simulated strengths are constant across seasons, so discarding the past can
        # only hurt: the search should land at or near the bottom of the grid.
        records = simulate_seasons(balanced_league(TEAMS).centred(), seasons=4, seed=SEED)
        model = DixonColesModel(get_settings()).fit(matches_frame(records), None)
        assert model.xi <= 1.0

    def test_a_short_history_falls_back_to_the_configured_default(self) -> None:
        settings = get_settings()
        records = simulate_seasons(balanced_league(TEAMS).centred(), seasons=1, seed=SEED)
        model = DixonColesModel(settings).fit(matches_frame(records), None)
        assert model.xi == settings.models.dixon_coles.xi.default


class TestProbabilities:
    """Acceptance test 2, with the correction in play."""

    def test_rows_sum_to_one_and_matrices_only_lose_the_tail(
        self, fitted: DixonColesModel, truth: LeagueParameters
    ) -> None:
        records = simulate_seasons(truth, seasons=1, seed=SEED + 1)
        frame = fixtures_frame(
            [record.as_fixture() for record in records[:40]], as_of=records[0].kickoff_utc
        )
        answer = fitted.predict_proba(frame, None)
        totals = answer[["p_home", "p_draw", "p_away"]].to_numpy().sum(axis=1)
        assert np.allclose(totals, 1.0, atol=SUM_TOLERANCE, rtol=0.0)
        for matrix in fitted.predict_scores(frame, None, max_goals=MAX_GOALS).values():
            assert 0.0 <= truncation_deficit(matrix) < TRUNCATION_TOLERANCE

    def test_the_correction_lifts_the_draw_on_a_dependent_league(
        self, fitted: DixonColesModel, truth: LeagueParameters
    ) -> None:
        # A negative rho piles mass onto 0-0 and 1-1, which are draws.
        records = simulate_seasons(truth, seasons=1, seed=SEED + 1)
        frame = fixtures_frame(
            [record.as_fixture() for record in records[:60]], as_of=records[0].kickoff_utc
        )
        plain = PoissonModel(get_settings()).fit(
            matches_frame(simulate_seasons(truth, seasons=SEASONS, seed=SEED)), None
        )
        assert (
            fitted.predict_proba(frame, None)["p_draw"].mean()
            > plain.predict_proba(frame, None)["p_draw"].mean()
        )


class TestAgainstPoisson:
    """Acceptance test 5: Dixon-Coles is not worse than plain Poisson."""

    def test_it_is_at_least_as_good_on_a_dependent_league(self, truth: LeagueParameters) -> None:
        settings = _without_decay(get_settings())
        train = simulate_seasons(truth, seasons=4, seed=SEED)
        test = simulate_seasons(truth, seasons=2, seed=SEED + 99, first_season=90)
        training = matches_frame(train)
        frame = fixtures_frame([record.as_fixture() for record in test], as_of=test[0].kickoff_utc)
        outcomes = np.array([outcome_of(record) for record in test], dtype=np.int64)

        corrected = DixonColesModel(settings).fit(training, None).predict_proba(frame, None)
        plain = PoissonModel(settings).fit(training, None).predict_proba(frame, None)
        columns = ["p_home", "p_draw", "p_away"]
        assert rps(corrected[columns].to_numpy(), outcomes) <= rps(
            plain[columns].to_numpy(), outcomes
        )

    def test_the_improvement_is_not_noise(self, truth: LeagueParameters) -> None:
        settings = _without_decay(get_settings())
        train = simulate_seasons(truth, seasons=6, seed=SEED)
        test = simulate_seasons(truth, seasons=4, seed=SEED + 99, first_season=90)
        training = matches_frame(train)
        frame = fixtures_frame([record.as_fixture() for record in test], as_of=test[0].kickoff_utc)
        outcomes = np.array([outcome_of(record) for record in test], dtype=np.int64)
        columns = ["p_home", "p_draw", "p_away"]

        interval = paired_bootstrap(
            rps,
            DixonColesModel(settings)
            .fit(training, None)
            .predict_proba(frame, None)[columns]
            .to_numpy(),
            PoissonModel(settings)
            .fit(training, None)
            .predict_proba(frame, None)[columns]
            .to_numpy(),
            outcomes,
            resamples=settings.backtest.bootstrap.resamples,
            confidence=settings.backtest.bootstrap.confidence,
            seed=settings.backtest.bootstrap.seed,
        )
        # Dixon-Coles minus Poisson: the whole interval must sit at or below zero.
        assert interval.high <= 0.0


class TestDeterminism:
    """Acceptance test 6."""

    def test_two_fits_on_the_same_data_are_identical(self, truth: LeagueParameters) -> None:
        settings = _without_decay(get_settings())
        frame = matches_frame(simulate_seasons(truth, seasons=2, seed=SEED))
        first = DixonColesModel(settings).fit(frame, None)
        second = DixonColesModel(settings).fit(frame, None)
        assert first.fitted_parameters() == second.fitted_parameters()

    def test_the_decay_search_is_reproducible_too(self, truth: LeagueParameters) -> None:
        # The fit above pins the grid to one value, so it never runs the search. Nothing in the
        # search is random -- it scores a fixed grid and breaks ties towards the smaller xi --
        # and that has to stay true, or two runs of the same backtest would disagree.
        frame = matches_frame(simulate_seasons(truth, seasons=4, seed=SEED))
        first = DixonColesModel(get_settings()).fit(frame, None)
        second = DixonColesModel(get_settings()).fit(frame, None)
        assert first.xi == second.xi
        assert first.fitted_parameters() == second.fitted_parameters()


class TestFittedParameters:
    """What the report shows for M3 (spec: home advantage, rho and xi per league)."""

    def test_rho_and_xi_are_reported(self, fitted: DixonColesModel) -> None:
        reported = fitted.fitted_parameters()
        assert set(reported) >= {"intercept", "home_advantage", "rho", "xi", "teams"}
