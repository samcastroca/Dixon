"""Paired bootstrap intervals: a model is only better when zero is excluded (spec section 8)."""

from __future__ import annotations

import numpy as np
import pytest

from predictor.evaluation.bootstrap import paired_bootstrap
from predictor.evaluation.metrics import AWAY, DRAW, HOME, rps

RESAMPLES = 500
CONFIDENCE = 0.95
SEED = 20260926

OUTCOMES = np.array([HOME, DRAW, AWAY] * 8)
UNINFORMED = np.tile(np.array([1 / 3, 1 / 3, 1 / 3]), (len(OUTCOMES), 1))


def confident() -> np.ndarray:
    """A forecast that puts most of its mass on what actually happened."""
    rows = np.full((len(OUTCOMES), 3), 0.1)
    rows[np.arange(len(OUTCOMES)), OUTCOMES] = 0.8
    return rows


def test_two_identical_models_differ_by_nothing_and_the_interval_contains_zero() -> None:
    result = paired_bootstrap(
        rps,
        UNINFORMED,
        UNINFORMED,
        OUTCOMES,
        resamples=RESAMPLES,
        confidence=CONFIDENCE,
        seed=SEED,
    )

    assert result.difference == pytest.approx(0.0, abs=1e-12)
    assert result.low <= 0.0 <= result.high
    assert result.excludes_zero is False


def test_a_better_model_has_a_negative_interval_that_excludes_zero() -> None:
    """Lower RPS is better, so the difference of the better model minus the worse is negative."""
    result = paired_bootstrap(
        rps,
        confident(),
        UNINFORMED,
        OUTCOMES,
        resamples=RESAMPLES,
        confidence=CONFIDENCE,
        seed=SEED,
    )

    assert result.difference < 0.0
    assert result.high < 0.0
    assert result.excludes_zero is True


def test_the_interval_brackets_the_observed_difference() -> None:
    result = paired_bootstrap(
        rps,
        confident(),
        UNINFORMED,
        OUTCOMES,
        resamples=RESAMPLES,
        confidence=CONFIDENCE,
        seed=SEED,
    )

    assert result.low <= result.difference <= result.high
    assert result.resamples == RESAMPLES
    assert result.confidence == CONFIDENCE


def test_the_same_seed_gives_the_same_interval() -> None:
    first = paired_bootstrap(
        rps,
        confident(),
        UNINFORMED,
        OUTCOMES,
        resamples=RESAMPLES,
        confidence=CONFIDENCE,
        seed=SEED,
    )
    again = paired_bootstrap(
        rps,
        confident(),
        UNINFORMED,
        OUTCOMES,
        resamples=RESAMPLES,
        confidence=CONFIDENCE,
        seed=SEED,
    )

    assert (again.difference, again.low, again.high) == (first.difference, first.low, first.high)


def test_comparing_different_numbers_of_matches_is_refused() -> None:
    with pytest.raises(ValueError, match="same matches"):
        paired_bootstrap(
            rps,
            UNINFORMED[:-1],
            UNINFORMED,
            OUTCOMES,
            resamples=RESAMPLES,
            confidence=CONFIDENCE,
            seed=SEED,
        )


def test_an_empty_comparison_is_refused() -> None:
    empty = np.empty((0, 3))
    with pytest.raises(ValueError, match="no matches"):
        paired_bootstrap(
            rps,
            empty,
            empty,
            np.empty(0, dtype=int),
            resamples=RESAMPLES,
            confidence=CONFIDENCE,
            seed=SEED,
        )
