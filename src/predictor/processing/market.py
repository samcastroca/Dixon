"""De-margined 1X2 probabilities (spec section 6.2).

Bookmaker prices carry a margin: the inverse prices sum to more than one. Two standard ways of
removing it are implemented, both pure functions over the three prices:

* proportional normalisation, which divides every inverse price by their sum;
* Shin's method, which assumes part of the margin protects the book against insider money and
  therefore shrinks the favourite more than the outsiders.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import sqrt

#: A price of 1.0 or less is not a quote: it would imply a probability of 1 or more.
MIN_PRICE = 1.0

#: Shin's z lives in [0, 1); the bisection needs enough steps to reach machine precision.
_MAX_Z = 1.0 - 1e-12
_BISECTION_STEPS = 200


@dataclass(frozen=True, slots=True)
class MarketProbabilities:
    """One de-margined 1X2 distribution, tagged with the method that produced it."""

    home: float
    draw: float
    away: float
    overround: float
    method: str

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.home, self.draw, self.away)


def _checked(prices: Sequence[float]) -> tuple[float, float, float]:
    if len(prices) != 3:
        msg = f"expected three 1X2 prices, got {len(prices)}"
        raise ValueError(msg)
    for price in prices:
        if not price > MIN_PRICE:
            msg = f"price {price!r} must be greater than {MIN_PRICE}"
            raise ValueError(msg)
    return (float(prices[0]), float(prices[1]), float(prices[2]))


def overround(prices: Sequence[float]) -> float:
    """Sum of the inverse prices: 1.0 is a fair book, more than that is the bookmaker's margin."""
    return sum(1.0 / price for price in _checked(prices))


def _normalised(values: Sequence[float]) -> tuple[float, float, float]:
    total = sum(values)
    return (values[0] / total, values[1] / total, values[2] / total)


def proportional(prices: Sequence[float]) -> MarketProbabilities:
    """Remove the margin by scaling every inverse price by the same factor."""
    inverse = [1.0 / price for price in _checked(prices)]
    home, draw, away = _normalised(inverse)
    return MarketProbabilities(home, draw, away, sum(inverse), "proportional")


def _shin_probabilities(inverse: Sequence[float], booksum: float, z: float) -> list[float]:
    return [
        (sqrt(z * z + 4.0 * (1.0 - z) * value * value / booksum) - z) / (2.0 * (1.0 - z))
        for value in inverse
    ]


def shin(prices: Sequence[float]) -> MarketProbabilities:
    """Remove the margin with Shin's model, solving for the insider fraction z by bisection."""
    inverse = [1.0 / price for price in _checked(prices)]
    booksum = sum(inverse)

    if booksum <= 1.0:  # a fair or arbitrage book leaves no insider margin to explain
        home, draw, away = _normalised(inverse)
        return MarketProbabilities(home, draw, away, booksum, "shin")

    # sum(p(z)) decreases in z and starts at sqrt(booksum) >= 1, so the root is bracketed.
    low, high = 0.0, _MAX_Z
    for _ in range(_BISECTION_STEPS):
        middle = (low + high) / 2.0
        if sum(_shin_probabilities(inverse, booksum, middle)) > 1.0:
            low = middle
        else:
            high = middle

    home, draw, away = _normalised(_shin_probabilities(inverse, booksum, (low + high) / 2.0))
    return MarketProbabilities(home, draw, away, booksum, "shin")


#: Every de-margining method the pipeline stores, by name.
METHODS: Mapping[str, Callable[[Sequence[float]], MarketProbabilities]] = {
    "proportional": proportional,
    "shin": shin,
}
