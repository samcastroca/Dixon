"""Paired bootstrap confidence intervals for a metric difference (spec section 8).

Two models are only comparable on the matches they both predicted, so the resampling is
paired: one draw of match indices scores both models, which cancels out the difficulty of the
matches that happened to be drawn. A model counts as better only when the interval excludes
zero, which is the rule the promotion decision of phase 6 will build on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from predictor.evaluation.metrics import Outcomes, Probabilities, as_outcomes, as_probabilities

#: A metric is anything that scores probabilities against outcomes, lower or higher be damned.
Metric = Callable[[Probabilities, Outcomes], float]


@dataclass(frozen=True, slots=True)
class Interval:
    """The observed difference between two models, and where resampling puts it."""

    difference: float
    low: float
    high: float
    resamples: int
    confidence: float

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval supports calling one model better than the other."""
        return self.low > 0.0 or self.high < 0.0

    def as_mapping(self) -> dict[str, float]:
        return {"difference": self.difference, "low": self.low, "high": self.high}


def paired_bootstrap(
    metric: Metric,
    left: object,
    right: object,
    outcomes: object,
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> Interval:
    """The `left - right` difference in a metric, with its percentile interval.

    `left` and `right` are two models' probabilities for the *same* matches in the same order,
    which is what the engine guarantees by intersecting their predictions before comparing.
    """
    first = as_probabilities(left)
    second = as_probabilities(right)
    if first.shape != second.shape:
        msg = "a paired bootstrap compares two models on the same matches, row for row"
        raise ValueError(msg)
    codes = as_outcomes(outcomes, first.shape[0])
    if not codes.size:
        msg = "a paired bootstrap needs matches to resample; this comparison has no matches"
        raise ValueError(msg)

    difference = metric(first, codes) - metric(second, codes)
    generator = np.random.default_rng(seed)
    drawn = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sample = generator.integers(0, codes.shape[0], size=codes.shape[0])
        drawn[index] = metric(first[sample], codes[sample]) - metric(second[sample], codes[sample])

    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(drawn, [tail, 1.0 - tail])
    return Interval(
        difference=float(difference),
        low=float(low),
        high=float(high),
        resamples=resamples,
        confidence=confidence,
    )
