"""The registry after phase 5, and the floor every statistical model has to clear.

Acceptance test 4 asks whether each model beats the frequency baseline in each league with a
bootstrap interval that excludes zero. That question is answered twice: here on simulated
leagues, where it runs on every `make check`, and in `tests/integration/test_phase5_models.py`
on EPL and La Liga themselves.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from predictor.config import get_settings
from predictor.evaluation.bootstrap import paired_bootstrap
from predictor.evaluation.frames import (
    StoredFeatures,
    features_frame,
    fixtures_frame,
    matches_frame,
    outcome_of,
)
from predictor.evaluation.metrics import rps
from predictor.features.replay import MatchRecord
from predictor.models.base import MatchModel, PerCompetition
from predictor.models.baselines import FrequencyBaseline
from predictor.models.registry import MODEL_NAMES, MODELS, build, factory
from tests.unit.simulate import balanced_league, simulate_seasons

#: Every model phase 5 adds, in the order the spec numbers them.
PHASE_FIVE = ("elo", "poisson", "dixon_coles", "bayesian_hierarchical")

#: The models acceptance test 4 is run on here. M4 samples a posterior, so it is checked in
#: its own `slow` module instead of inside this comparison.
COMPARED = ("elo", "poisson", "dixon_coles")

PROBABILITY_COLUMNS = ["p_home", "p_draw", "p_away"]

TEAMS = 20
SEED = 20260926


class TestRegistry:
    def test_every_phase_five_model_is_registered(self) -> None:
        assert set(MODEL_NAMES) == {"baseline_frequency", "baseline_market", *PHASE_FIVE}

    def test_the_names_are_the_ones_the_cli_documents(self) -> None:
        assert tuple(sorted(MODELS)) == MODEL_NAMES

    @pytest.mark.parametrize("name", PHASE_FIVE)
    def test_team_strength_models_are_fitted_per_competition(self, name: str) -> None:
        # Spec 7.1: EPL and La Liga teams never meet, so one joint set of strengths could not
        # tell the leagues apart.
        assert isinstance(build(name), PerCompetition)

    @pytest.mark.parametrize("name", PHASE_FIVE)
    def test_the_factory_hands_out_a_fresh_model_every_time(self, name: str) -> None:
        make = factory(name)
        assert make() is not make()

    @pytest.mark.parametrize("name", PHASE_FIVE)
    def test_no_statistical_model_reads_the_market(self, name: str) -> None:
        model = build(name)
        assert model.needs_market is False
        assert model.deployable is True


def _elo_features(records: Sequence[MatchRecord], *, scale: float = 400.0) -> pd.DataFrame | None:
    """The one feature M1 needs, monotone in the simulated team strengths."""
    return features_frame(
        [
            StoredFeatures(
                record.match_id,
                record.kickoff_utc,
                {"elo_diff": float(record.away_team_id - record.home_team_id) * scale / 20.0},
            )
            for record in records
        ],
        ("elo_diff",),
    )


class TestAgainstTheFrequencyBaseline:
    """Acceptance test 4, on simulated leagues where the truth is known."""

    @pytest.mark.parametrize("name", COMPARED)
    def test_it_beats_knowing_nothing_but_the_frequencies(self, name: str) -> None:
        settings = get_settings()
        truth = balanced_league(TEAMS).centred()
        train = simulate_seasons(truth, seasons=4, seed=SEED)
        test = simulate_seasons(truth, seasons=2, seed=SEED + 99, first_season=90)

        training = matches_frame(train)
        frame = fixtures_frame([record.as_fixture() for record in test], as_of=test[0].kickoff_utc)
        outcomes = np.array([outcome_of(record) for record in test], dtype=np.int64)

        model: MatchModel = factory(name)()
        candidate = _predictions(model, training, frame, train, test)
        floor = _predictions(PerCompetition(FrequencyBaseline), training, frame, train, test)

        interval = paired_bootstrap(
            rps,
            candidate,
            floor,
            outcomes,
            resamples=settings.backtest.bootstrap.resamples,
            confidence=settings.backtest.bootstrap.confidence,
            seed=settings.backtest.bootstrap.seed,
        )
        # Model minus baseline on RPS, where lower is better: the whole interval below zero.
        assert interval.high < 0.0


def _predictions(
    model: MatchModel,
    training: pd.DataFrame,
    frame: pd.DataFrame,
    train: Sequence[MatchRecord],
    test: Sequence[MatchRecord],
) -> npt.NDArray[np.float64]:
    """Fit and predict. M1 needs the Elo features; the rest are handed them and ignore them."""
    fitted = model.fit(training, _elo_features(train))
    answer = fitted.predict_proba(frame, _elo_features(test))
    return np.asarray(answer[PROBABILITY_COLUMNS].to_numpy(), dtype=np.float64)
