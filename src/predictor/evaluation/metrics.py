"""Probability scoring for 1X2 forecasts (spec section 8).

Everything here is vectorised numpy over an `(n, 3)` matrix of probabilities and an `(n,)`
vector of outcome codes. The column order is the ordered one, home before draw before away,
because the ranked probability score only means anything if the outcomes are ordered.

No function carries a default for a tunable: the number of calibration bins and the log-loss
floor come from `settings.backtest.metrics`, so the value that produced a number is always
visible at the call site.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

#: Outcome codes, in the order RPS needs them.
HOME = 0
DRAW = 1
AWAY = 2
OUTCOMES = (HOME, DRAW, AWAY)
OUTCOME_COUNT = len(OUTCOMES)

#: How far a row of probabilities may stray from summing to one.
SUM_TOLERANCE = 1e-9

Probabilities = npt.NDArray[np.float64]
Outcomes = npt.NDArray[np.int64]


def as_probabilities(values: object) -> Probabilities:
    """An `(n, 3)` float matrix whose rows are distributions, or a clear refusal."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != OUTCOME_COUNT:
        msg = f"probabilities must be an (n, {OUTCOME_COUNT}) matrix, got shape {array.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = "every probability must be finite"
        raise ValueError(msg)
    if np.any(array < 0.0) or np.any(array > 1.0):
        msg = "every probability must lie in [0, 1]"
        raise ValueError(msg)
    if array.size and not np.allclose(array.sum(axis=1), 1.0, atol=SUM_TOLERANCE, rtol=0.0):
        msg = f"every row of probabilities must sum to 1 within {SUM_TOLERANCE}"
        raise ValueError(msg)
    return array


def as_outcomes(values: object, expected_rows: int) -> Outcomes:
    """An `(n,)` vector of outcome codes matching the probability matrix row for row."""
    array = np.asarray(values)
    if array.ndim != 1 or array.shape[0] != expected_rows:
        msg = f"outcomes must have one entry per row of probabilities ({expected_rows} rows)"
        raise ValueError(msg)
    codes = array.astype(np.int64)
    if codes.size and (codes.min() < HOME or codes.max() > AWAY):
        msg = f"every outcome must be one of {OUTCOMES}"
        raise ValueError(msg)
    return codes


def _checked(probabilities: object, outcomes: object) -> tuple[Probabilities, Outcomes]:
    matrix = as_probabilities(probabilities)
    return matrix, as_outcomes(outcomes, matrix.shape[0])


def one_hot(outcomes: Outcomes) -> Probabilities:
    """The observed distribution: all the mass on what happened."""
    observed = np.zeros((outcomes.shape[0], OUTCOME_COUNT), dtype=np.float64)
    observed[np.arange(outcomes.shape[0]), outcomes] = 1.0
    return observed


def rps_per_match(probabilities: object, outcomes: object) -> Probabilities:
    """RPS of every match: 1/(r-1) * sum over the first r-1 cumulative differences, squared."""
    matrix, codes = _checked(probabilities, outcomes)
    differences = np.cumsum(matrix - one_hot(codes), axis=1)[:, : OUTCOME_COUNT - 1]
    return np.asarray(np.square(differences).sum(axis=1) / (OUTCOME_COUNT - 1), dtype=np.float64)


def rps(probabilities: object, outcomes: object) -> float:
    """The ranked probability score, averaged over the matches. Lower is better."""
    return float(rps_per_match(probabilities, outcomes).mean())


def log_loss_per_match(probabilities: object, outcomes: object, *, floor: float) -> Probabilities:
    """The negative log of the probability given to what happened, floored so it stays finite."""
    matrix, codes = _checked(probabilities, outcomes)
    taken = matrix[np.arange(codes.shape[0]), codes]
    return np.asarray(-np.log(np.clip(taken, floor, 1.0)), dtype=np.float64)


def log_loss(probabilities: object, outcomes: object, *, floor: float) -> float:
    return float(log_loss_per_match(probabilities, outcomes, floor=floor).mean())


def brier_per_match(probabilities: object, outcomes: object) -> Probabilities:
    """The multiclass Brier score of every match: the squared distance to the observed vector."""
    matrix, codes = _checked(probabilities, outcomes)
    return np.asarray(np.square(matrix - one_hot(codes)).sum(axis=1), dtype=np.float64)


def brier_score(probabilities: object, outcomes: object) -> float:
    return float(brier_per_match(probabilities, outcomes).mean())


def _bin_of(values: Probabilities, bins: int) -> npt.NDArray[np.int64]:
    """Which equal-width bin of [0, 1] each value belongs to; 1.0 belongs to the last one."""
    if bins < 2:
        msg = "calibration needs at least two bins"
        raise ValueError(msg)
    return np.minimum((values * bins).astype(np.int64), bins - 1)


def ece(probabilities: object, outcomes: object, *, bins: int) -> float:
    """Expected calibration error over the confidence of the predicted outcome."""
    matrix, codes = _checked(probabilities, outcomes)
    if not codes.size:
        return 0.0
    confidence = matrix.max(axis=1)
    correct = (matrix.argmax(axis=1) == codes).astype(np.float64)
    index = _bin_of(confidence, bins)
    counts = np.bincount(index, minlength=bins).astype(np.float64)
    claimed = np.bincount(index, weights=confidence, minlength=bins)
    observed = np.bincount(index, weights=correct, minlength=bins)
    populated = counts > 0
    gaps = np.abs(observed[populated] - claimed[populated])
    return float(gaps.sum() / codes.shape[0])


@dataclass(frozen=True, slots=True)
class Reliability:
    """One outcome's reliability diagram: what was claimed against what happened, per bin."""

    outcome: int
    bins: int
    edges: tuple[float, ...]
    counts: tuple[int, ...]
    mean_predicted: tuple[float, ...]
    observed: tuple[float, ...]

    def as_rows(self) -> tuple[tuple[int, float, float, int, float, float], ...]:
        """The diagram as plain rows, for the CSV artifact behind the plot."""
        return tuple(
            (
                self.outcome,
                self.edges[index],
                self.edges[index + 1],
                self.counts[index],
                self.mean_predicted[index],
                self.observed[index],
            )
            for index in range(self.bins)
        )


def reliability(probabilities: object, outcomes: object, *, outcome: int, bins: int) -> Reliability:
    """Binned claim-against-frequency data for one outcome, empty bins left as NaN."""
    matrix, codes = _checked(probabilities, outcomes)
    if outcome not in OUTCOMES:
        msg = f"every outcome must be one of {OUTCOMES}"
        raise ValueError(msg)
    claimed = matrix[:, outcome]
    happened = (codes == outcome).astype(np.float64)
    index = _bin_of(claimed, bins)
    counts = np.bincount(index, minlength=bins)
    safe = np.where(counts > 0, counts, 1).astype(np.float64)
    mean_claimed = np.bincount(index, weights=claimed, minlength=bins) / safe
    frequency = np.bincount(index, weights=happened, minlength=bins) / safe
    empty = counts == 0
    return Reliability(
        outcome=outcome,
        bins=bins,
        edges=tuple(np.linspace(0.0, 1.0, bins + 1).tolist()),
        counts=tuple(int(count) for count in counts),
        mean_predicted=tuple(np.where(empty, np.nan, mean_claimed).tolist()),
        observed=tuple(np.where(empty, np.nan, frequency).tolist()),
    )


@dataclass(frozen=True, slots=True)
class MetricSet:
    """Every metric of one slice of predictions, plus how many matches it covers."""

    n: int
    rps: float
    log_loss: float
    brier: float
    ece: float

    def as_mapping(self) -> Mapping[str, float]:
        return {
            "n": float(self.n),
            "rps": self.rps,
            "log_loss": self.log_loss,
            "brier": self.brier,
            "ece": self.ece,
        }


def metric_set(probabilities: object, outcomes: object, *, bins: int, floor: float) -> MetricSet:
    """Score one slice of predictions with every metric the spec asks for."""
    matrix, codes = _checked(probabilities, outcomes)
    return MetricSet(
        n=int(codes.shape[0]),
        rps=rps(matrix, codes),
        log_loss=log_loss(matrix, codes, floor=floor),
        brier=brier_score(matrix, codes),
        ece=ece(matrix, codes, bins=bins),
    )


def reliability_diagrams(
    probabilities: object, outcomes: object, *, bins: int, wanted: Sequence[int] = OUTCOMES
) -> tuple[Reliability, ...]:
    """One diagram per outcome, which is what the spec asks the backtest to log."""
    return tuple(
        reliability(probabilities, outcomes, outcome=outcome, bins=bins) for outcome in wanted
    )
