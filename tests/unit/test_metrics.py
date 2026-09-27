"""Probability scoring: RPS, log loss, Brier and calibration (spec section 8).

Every expected number below is calculated by hand from the formula in the comment above it,
so a change to a metric has to disagree with arithmetic rather than with a fixture.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from predictor.evaluation.metrics import (
    AWAY,
    DRAW,
    HOME,
    brier_score,
    ece,
    log_loss,
    metric_set,
    reliability,
    rps,
    rps_per_match,
)

TOLERANCE = 1e-12

#: The settings defaults, passed explicitly because no metric hides a knob.
FLOOR = 1e-15
BINS = 10

UNIFORM = (1 / 3, 1 / 3, 1 / 3)

#: Three rows, scored by hand in the docstring of the test that uses them.
EXAMPLE = ((0.6, 0.25, 0.15), (0.2, 0.3, 0.5), (0.4, 0.3, 0.3))
EXAMPLE_OUTCOMES = (HOME, AWAY, DRAW)

weight = st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False)


def distributions(count: int = 25) -> st.SearchStrategy[np.ndarray]:
    """Rows of three positive numbers, normalised so that every row is a distribution."""
    rows = st.lists(st.tuples(weight, weight, weight), min_size=1, max_size=count)
    return rows.map(lambda values: np.array(values) / np.array(values).sum(axis=1, keepdims=True))


def outcomes_for(probabilities: np.ndarray) -> np.ndarray:
    """A deterministic outcome per row: whichever one the row calls least likely."""
    return np.asarray(probabilities).argmin(axis=1)


def test_a_perfect_forecast_scores_zero_on_every_metric() -> None:
    """All the mass on what happened: RPS 0, log loss 0 and Brier 0."""
    probabilities = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    outcomes = np.array([HOME, DRAW, AWAY])
    assert rps(probabilities, outcomes) == pytest.approx(0.0, abs=TOLERANCE)
    assert log_loss(probabilities, outcomes, floor=FLOOR) == pytest.approx(0.0, abs=TOLERANCE)
    assert brier_score(probabilities, outcomes) == pytest.approx(0.0, abs=TOLERANCE)


def test_the_worst_possible_forecast_scores_one_on_rps() -> None:
    """All the mass on the away win when the home team won: (0 - 1)^2 + (0 - 1)^2, halved."""
    assert rps(np.array([[0.0, 0.0, 1.0]]), np.array([HOME])) == pytest.approx(1.0, abs=TOLERANCE)


def test_a_uniform_forecast_gives_the_hand_computed_rps() -> None:
    """RPS = 1/2 * sum_i (sum_{j<=i} (p_j - o_j))^2.

    Home or away win: (1/3 - 1)^2 + (2/3 - 1)^2 = 4/9 + 1/9 = 5/9, halved -> 5/18.
    Draw:             (1/3 - 0)^2 + (2/3 - 1)^2 = 1/9 + 1/9 = 2/9, halved -> 1/9.
    """
    probabilities = np.array([UNIFORM, UNIFORM, UNIFORM])
    outcomes = np.array([HOME, DRAW, AWAY])
    per_match = rps_per_match(probabilities, outcomes)
    assert per_match[0] == pytest.approx(5 / 18, abs=TOLERANCE)
    assert per_match[1] == pytest.approx(1 / 9, abs=TOLERANCE)
    assert per_match[2] == pytest.approx(5 / 18, abs=TOLERANCE)
    assert rps(probabilities, outcomes) == pytest.approx((5 / 18 + 1 / 9 + 5 / 18) / 3, abs=1e-12)


def test_a_uniform_forecast_gives_the_hand_computed_log_loss_and_brier() -> None:
    """Log loss is -ln(1/3) = ln 3 on every row; Brier is 4/9 + 1/9 + 1/9 = 6/9."""
    probabilities = np.array([UNIFORM, UNIFORM, UNIFORM])
    outcomes = np.array([HOME, DRAW, AWAY])
    assert log_loss(probabilities, outcomes, floor=FLOOR) == pytest.approx(math.log(3.0))
    assert brier_score(probabilities, outcomes) == pytest.approx(6 / 9, abs=TOLERANCE)


def test_the_three_row_example_matches_the_hand_computed_metrics() -> None:
    """Rows (0.6, 0.25, 0.15) H, (0.2, 0.3, 0.5) A and (0.4, 0.3, 0.3) D.

    RPS:      (0.16 + 0.0225)/2 = 0.09125, (0.04 + 0.25)/2 = 0.145, (0.16 + 0.09)/2 = 0.125.
    Brier:    0.245, 0.38 and 0.74.
    Log loss: -(ln 0.6 + ln 0.5 + ln 0.3) / 3.
    """
    probabilities = np.array(EXAMPLE)
    outcomes = np.array(EXAMPLE_OUTCOMES)
    expected_rps = (0.09125 + 0.145 + 0.125) / 3
    expected_brier = (0.245 + 0.38 + 0.74) / 3
    expected_log_loss = -(math.log(0.6) + math.log(0.5) + math.log(0.3)) / 3

    assert rps(probabilities, outcomes) == pytest.approx(expected_rps, abs=TOLERANCE)
    assert brier_score(probabilities, outcomes) == pytest.approx(expected_brier, abs=TOLERANCE)
    assert log_loss(probabilities, outcomes, floor=FLOOR) == pytest.approx(
        expected_log_loss, abs=TOLERANCE
    )


def test_a_certain_miss_is_finite_because_the_logarithm_is_floored() -> None:
    assert log_loss(np.array([[0.0, 0.0, 1.0]]), np.array([HOME]), floor=FLOOR) == pytest.approx(
        -math.log(FLOOR)
    )


def test_a_perfectly_calibrated_forecast_has_no_calibration_error() -> None:
    """Four rows claiming 50% for the home win, and the home team wins exactly half of them."""
    probabilities = np.array([[0.5, 0.25, 0.25]] * 4)
    outcomes = np.array([HOME, HOME, DRAW, AWAY])
    assert ece(probabilities, outcomes, bins=BINS) == pytest.approx(0.0, abs=TOLERANCE)


def test_confident_and_always_wrong_is_the_worst_calibration_error() -> None:
    probabilities = np.array([[1.0, 0.0, 0.0]] * 4)
    assert ece(probabilities, np.array([AWAY] * 4), bins=BINS) == pytest.approx(1.0, abs=TOLERANCE)


def test_reliability_bins_report_what_was_claimed_and_what_happened() -> None:
    """Two rows claim 0.15 for the home win (one happens) and two claim 0.95 (both do)."""
    probabilities = np.array(
        [[0.15, 0.6, 0.25], [0.15, 0.25, 0.6], [0.95, 0.03, 0.02], [0.95, 0.02, 0.03]]
    )
    outcomes = np.array([HOME, DRAW, HOME, HOME])
    diagram = reliability(probabilities, outcomes, outcome=HOME, bins=BINS)

    assert diagram.outcome == HOME
    assert len(diagram.counts) == BINS
    assert [index for index, count in enumerate(diagram.counts) if count] == [1, 9]
    assert diagram.counts[1] == 2
    assert diagram.mean_predicted[1] == pytest.approx(0.15, abs=TOLERANCE)
    assert diagram.observed[1] == pytest.approx(0.5, abs=TOLERANCE)
    assert diagram.counts[9] == 2
    assert diagram.mean_predicted[9] == pytest.approx(0.95, abs=TOLERANCE)
    assert diagram.observed[9] == pytest.approx(1.0, abs=TOLERANCE)


def test_a_metric_set_carries_the_sample_size_and_every_metric() -> None:
    probabilities = np.array(EXAMPLE)
    outcomes = np.array(EXAMPLE_OUTCOMES)
    result = metric_set(probabilities, outcomes, bins=BINS, floor=FLOOR)
    assert result.n == 3
    assert result.rps == pytest.approx(rps(probabilities, outcomes))
    assert set(result.as_mapping()) == {"n", "rps", "log_loss", "brier", "ece"}


def test_rows_that_do_not_sum_to_one_are_refused() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        rps(np.array([[0.5, 0.3, 0.1]]), np.array([HOME]))


def test_a_missing_probability_is_refused() -> None:
    with pytest.raises(ValueError, match="finite"):
        rps(np.array([[float("nan"), 0.5, 0.5]]), np.array([HOME]))


def test_an_unknown_outcome_code_is_refused() -> None:
    with pytest.raises(ValueError, match="outcome"):
        rps(np.array([list(UNIFORM)]), np.array([3]))


def test_a_shape_mismatch_is_refused() -> None:
    with pytest.raises(ValueError, match="rows"):
        rps(np.array([list(UNIFORM)]), np.array([HOME, DRAW]))


@hypothesis_settings(max_examples=200)
@given(distributions())
def test_rps_always_lies_between_zero_and_one(probabilities: np.ndarray) -> None:
    assert 0.0 <= rps(probabilities, outcomes_for(probabilities)) <= 1.0


@hypothesis_settings(max_examples=200)
@given(distributions())
def test_brier_and_log_loss_are_never_negative(probabilities: np.ndarray) -> None:
    outcomes = outcomes_for(probabilities)
    assert brier_score(probabilities, outcomes) >= 0.0
    assert log_loss(probabilities, outcomes, floor=FLOOR) >= 0.0


@hypothesis_settings(max_examples=100)
@given(distributions())
def test_the_order_of_the_matches_does_not_change_a_metric(probabilities: np.ndarray) -> None:
    outcomes = outcomes_for(probabilities)
    backwards = slice(None, None, -1)
    assert rps(probabilities[backwards], outcomes[backwards]) == pytest.approx(
        rps(probabilities, outcomes)
    )
    assert ece(probabilities[backwards], outcomes[backwards], bins=BINS) == pytest.approx(
        ece(probabilities, outcomes, bins=BINS)
    )
