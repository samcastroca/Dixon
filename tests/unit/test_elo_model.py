"""M1, Elo plus an ordered logistic regression (spec section 7, acceptance tests 1, 3 and 6)."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from predictor.config import get_settings
from predictor.evaluation.frames import (
    StoredFeatures,
    features_frame,
    fixtures_frame,
    matches_frame,
)
from predictor.features.replay import MatchRecord
from predictor.models.elo_model import EloModel, OrderedLogit
from tests.unit.simulate import balanced_league, simulate_seasons

Floats = npt.NDArray[np.float64]
Codes = npt.NDArray[np.int64]

#: Acceptance test 1 for the ordered mapping, on ten thousand simulated matches.
SLOPE_TOLERANCE = 0.08
CUTPOINT_TOLERANCE = 0.06

SUM_TOLERANCE = 1e-9

_OPTIMISER = get_settings().models.elo.optimiser

TRUE_SLOPE = 1.0
TRUE_LOWER = -0.55
TRUE_UPPER = 0.35
SEED = 20260926


def _simulated_ratings(n: int, seed: int) -> tuple[Floats, Codes]:
    """Rating differences and the outcomes a known ordered logit produces from them."""
    rng = np.random.default_rng(seed)
    # Scaled the way the model sees them: elo_diff divided by the rating scale.
    scaled = rng.normal(0.0, 0.6, size=n)
    latent = TRUE_SLOPE * scaled
    p_away = 1.0 / (1.0 + np.exp(-(TRUE_LOWER - latent)))
    p_not_home = 1.0 / (1.0 + np.exp(-(TRUE_UPPER - latent)))
    probabilities = np.column_stack([1.0 - p_not_home, p_not_home - p_away, p_away])
    draws = rng.random(n)
    outcomes = (draws[:, None] > probabilities.cumsum(axis=1)).sum(axis=1)
    return scaled, outcomes.astype(np.int64)


def _elo_features(records: Sequence[MatchRecord], *, scale: float = 400.0) -> pd.DataFrame | None:
    """A feature frame carrying `elo_diff`, built exactly as the backtester builds one.

    The rating difference stands in for what phase 3 stores: the home team's id minus the
    away team's, scaled up, which is monotone in the simulated strengths.
    """
    rows = [
        StoredFeatures(
            match_id=record.match_id,
            as_of_utc=record.kickoff_utc,
            values={"elo_diff": float(record.away_team_id - record.home_team_id) * scale / 20.0},
        )
        for record in records
    ]
    return features_frame(rows, ("elo_diff",))


class TestOrderedLogit:
    """The mapping from one rating difference to three ordered outcomes."""

    def test_the_cutpoints_stay_ordered_by_construction(self) -> None:
        # The fit optimises a log-gap, so `lower < upper` cannot be violated numerically.
        fitted = OrderedLogit.fit(*_simulated_ratings(2000, SEED), optimiser=_OPTIMISER)
        assert fitted.lower < fitted.upper

    def test_it_recovers_the_mapping_it_was_generated_from(self) -> None:
        scaled, outcomes = _simulated_ratings(20_000, SEED)
        fitted = OrderedLogit.fit(scaled, outcomes, optimiser=_OPTIMISER)
        assert fitted.slope == pytest.approx(TRUE_SLOPE, abs=SLOPE_TOLERANCE)
        assert fitted.lower == pytest.approx(TRUE_LOWER, abs=CUTPOINT_TOLERANCE)
        assert fitted.upper == pytest.approx(TRUE_UPPER, abs=CUTPOINT_TOLERANCE)

    def test_every_row_sums_to_one(self) -> None:
        fitted = OrderedLogit.fit(*_simulated_ratings(2000, SEED), optimiser=_OPTIMISER)
        rows = fitted.probabilities(np.linspace(-3.0, 3.0, 25))
        assert np.allclose(rows.sum(axis=1), 1.0, atol=SUM_TOLERANCE, rtol=0.0)
        assert rows.min() > 0.0

    def test_a_bigger_rating_difference_favours_the_home_side(self) -> None:
        fitted = OrderedLogit.fit(*_simulated_ratings(5000, SEED), optimiser=_OPTIMISER)
        rows = fitted.probabilities(np.linspace(-2.0, 2.0, 9))
        assert np.all(np.diff(rows[:, 0]) > 0.0)
        assert np.all(np.diff(rows[:, 2]) < 0.0)

    def test_symmetric_cutpoints_make_a_level_fixture_a_coin_flip(self) -> None:
        # Acceptance test 3 for M1: with no rating difference and cutpoints that mirror each
        # other, nothing distinguishes the two sides, so the two win probabilities are equal.
        level = OrderedLogit(slope=0.9, lower=-0.45, upper=0.45)
        row = level.probabilities(np.array([0.0]))[0]
        assert row[0] == row[2]


class TestEloModel:
    """The model the registry hands the backtester."""

    def test_it_fits_and_predicts_from_the_stored_ratings(self) -> None:
        records = simulate_seasons(balanced_league(20).centred(), seasons=2, seed=SEED)
        model = EloModel(get_settings()).fit(matches_frame(records), _elo_features(records))
        fixtures = records[:30]
        frame = fixtures_frame(
            [record.as_fixture() for record in fixtures], as_of=fixtures[0].kickoff_utc
        )
        answer = model.predict_proba(frame, _elo_features(fixtures))
        assert len(answer) == 30
        totals = answer[["p_home", "p_draw", "p_away"]].to_numpy().sum(axis=1)
        assert np.allclose(totals, 1.0, atol=SUM_TOLERANCE, rtol=0.0)

    def test_it_refuses_to_guess_without_the_elo_features(self) -> None:
        records = simulate_seasons(balanced_league(20).centred(), seasons=2, seed=SEED)
        with pytest.raises(ValueError, match="elo_diff"):
            EloModel(get_settings()).fit(matches_frame(records), None)

    def test_it_names_the_feature_group_that_is_missing(self) -> None:
        records = simulate_seasons(balanced_league(20).centred(), seasons=2, seed=SEED)
        empty = features_frame(
            [
                StoredFeatures(record.match_id, record.kickoff_utc, {"home_rest_days": 3.0})
                for record in records
            ],
            ("home_rest_days",),
        )
        with pytest.raises(ValueError, match=re.escape("features.groups.elo")):
            EloModel(get_settings()).fit(matches_frame(records), empty)

    def test_it_has_no_scoreline_distribution(self) -> None:
        records = simulate_seasons(balanced_league(20).centred(), seasons=2, seed=SEED)
        model = EloModel(get_settings()).fit(matches_frame(records), _elo_features(records))
        frame = fixtures_frame([records[0].as_fixture()], as_of=records[0].kickoff_utc)
        with pytest.raises(NotImplementedError, match="scoreline"):
            model.predict_scores(frame, None)

    def test_two_fits_on_the_same_data_are_identical(self) -> None:
        records = simulate_seasons(balanced_league(20).centred(), seasons=2, seed=SEED)
        training, features = matches_frame(records), _elo_features(records)
        first = EloModel(get_settings()).fit(training, features)
        second = EloModel(get_settings()).fit(training, features)
        assert first.fitted_parameters() == second.fitted_parameters()

    def test_the_mapping_is_reported(self) -> None:
        records = simulate_seasons(balanced_league(20).centred(), seasons=2, seed=SEED)
        model = EloModel(get_settings()).fit(matches_frame(records), _elo_features(records))
        assert set(model.fitted_parameters()) >= {"slope", "lower_cutpoint", "upper_cutpoint"}
