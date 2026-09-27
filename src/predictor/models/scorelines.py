"""From two goal rates to a scoreline distribution, and from that to 1X2 (spec section 7).

Every goal model of phase 5 ends here: M2 and M3 with a point estimate of the two rates, M4
with one pair per posterior draw. Keeping the arithmetic in one place is what makes the two
halves of acceptance test 2 consistent rather than contradictory:

* a **matrix** is truncated at `max_goals` and keeps the mass that falls off the end, which is
  the "score matrices sum to 1 minus a truncation error < 1e-4" half. Nothing here renormalises
  it, because hiding the error would make the test vacuous;
* a **probability row** is normalised, which is the "every probability row sums to 1" half and
  what `validate_probabilities` demands of a model.

The low-score correction of Dixon-Coles moves mass between the four scorelines 0-0, 0-1, 1-0
and 1-1 without creating or destroying any: the four adjustments cancel exactly, so a corrected
matrix loses precisely as much to truncation as the uncorrected one.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.stats import poisson

__all__ = [
    "outcome_probabilities",
    "score_matrix",
    "tau",
    "tau_terms",
    "truncation_deficit",
]

Matrix = npt.NDArray[np.float64]
Floats = npt.NDArray[np.float64]
Counts = npt.NDArray[np.int64]

#: tau is a factor, so it must stay positive; a fit that pushes it here is pushed back.
TAU_FLOOR = 1e-10


def goal_pmf(rate: float, max_goals: int) -> Floats:
    """Poisson probabilities of 0..max_goals goals, untruncated and unnormalised."""
    return np.asarray(poisson.pmf(np.arange(max_goals + 1), rate), dtype=np.float64)


def score_matrix(
    lambda_home: float, lambda_away: float, *, max_goals: int, rho: float = 0.0
) -> Matrix:
    """The joint distribution of the scoreline, rows home goals and columns away goals.

    `rho = 0` is the independent Poisson of M2; anything else applies the Dixon-Coles
    correction of M3 and M4 to the four low scorelines.
    """
    matrix = np.outer(goal_pmf(lambda_home, max_goals), goal_pmf(lambda_away, max_goals))
    if rho and max_goals >= 1:
        matrix[0, 0] *= 1.0 - lambda_home * lambda_away * rho
        matrix[0, 1] *= 1.0 + lambda_home * rho
        matrix[1, 0] *= 1.0 + lambda_away * rho
        matrix[1, 1] *= 1.0 - rho
        np.clip(matrix, 0.0, None, out=matrix)
    return matrix


def tau(
    home_goals: int, away_goals: int, lambda_home: float, lambda_away: float, rho: float
) -> float:
    """The Dixon-Coles correction factor of one scoreline; 1 everywhere but the four low ones."""
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lambda_home * lambda_away * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lambda_home * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + lambda_away * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def tau_terms(
    home_goals: Counts,
    away_goals: Counts,
    lambda_home: Floats,
    lambda_away: Floats,
    rho: float,
) -> tuple[Floats, Floats, Floats, Floats]:
    """`tau` and its three partial derivatives, vectorised over a whole training set.

    Returns tau and the derivatives with respect to the home rate, the away rate and rho, in
    that order. The weighted likelihood of M3 needs all four, and computing them together is
    what keeps its gradient analytic (spec section 7, M3).
    """
    size = home_goals.shape[0]
    values = np.ones(size, dtype=np.float64)
    d_home = np.zeros(size, dtype=np.float64)
    d_away = np.zeros(size, dtype=np.float64)
    d_rho = np.zeros(size, dtype=np.float64)

    nil_nil = (home_goals == 0) & (away_goals == 0)
    values[nil_nil] = 1.0 - lambda_home[nil_nil] * lambda_away[nil_nil] * rho
    d_home[nil_nil] = -lambda_away[nil_nil] * rho
    d_away[nil_nil] = -lambda_home[nil_nil] * rho
    d_rho[nil_nil] = -lambda_home[nil_nil] * lambda_away[nil_nil]

    nil_one = (home_goals == 0) & (away_goals == 1)
    values[nil_one] = 1.0 + lambda_home[nil_one] * rho
    d_home[nil_one] = rho
    d_rho[nil_one] = lambda_home[nil_one]

    one_nil = (home_goals == 1) & (away_goals == 0)
    values[one_nil] = 1.0 + lambda_away[one_nil] * rho
    d_away[one_nil] = rho
    d_rho[one_nil] = lambda_away[one_nil]

    one_one = (home_goals == 1) & (away_goals == 1)
    values[one_one] = 1.0 - rho
    d_rho[one_one] = -1.0

    return np.clip(values, TAU_FLOOR, None), d_home, d_away, d_rho


def outcome_probabilities(matrix: Matrix) -> tuple[float, float, float]:
    """Home win, draw and away win, normalised so the row sums to one exactly.

    The truncated tail is spread across the three outcomes in proportion, which is the only
    assumption available: nothing in the model says a 12-0 is likelier than a 0-12.

    Away wins are summed as the lower triangle of the transpose rather than as the upper
    triangle or as a remainder. On a symmetric matrix that is the identical reduction over
    identical memory, so two evenly matched sides come out exactly equal instead of a few
    machine epsilons apart, which is what acceptance test 3 asks for.
    """
    draw = float(np.trace(matrix))
    home = float(np.tril(matrix, -1).sum())
    away = float(np.tril(matrix.T, -1).sum())
    mass = home + draw + away
    if mass <= 0.0:
        msg = "a scoreline matrix with no mass has no outcome probabilities"
        raise ValueError(msg)
    return home / mass, draw / mass, away / mass


def truncation_deficit(matrix: Matrix) -> float:
    """The probability mass that fell past `max_goals`, which is never quietly absorbed."""
    return max(0.0, 1.0 - float(matrix.sum()))
