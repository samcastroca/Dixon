"""De-margined market probabilities (spec section 6.2)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from predictor.processing.market import MarketProbabilities, overround, proportional, shin

TOLERANCE = 1e-9

#: A clear favourite, used to show what Shin does to it (spec phase 2, acceptance test 4).
FAVOURITE = (1.30, 5.50, 12.00)

#: Golden example published by the reference implementation (github.com/mberk/shin).
REFERENCE_PRICES = (2.6, 2.4, 4.3)
REFERENCE_SHIN = (0.37299406033208965, 0.4047794109200184, 0.2222265287474275)

prices = st.floats(min_value=1.01, max_value=100.0, allow_nan=False, allow_infinity=False)
price_triples = st.tuples(prices, prices, prices)


def assert_is_a_distribution(probabilities: MarketProbabilities) -> None:
    values = probabilities.as_tuple()
    assert sum(values) == pytest.approx(1.0, abs=TOLERANCE)
    assert all(0.0 < value < 1.0 for value in values)


@settings(max_examples=300)
@given(price_triples)
def test_proportional_probabilities_are_a_distribution(triple: tuple[float, float, float]) -> None:
    assert_is_a_distribution(proportional(triple))


@settings(max_examples=300)
@given(price_triples)
def test_shin_probabilities_are_a_distribution(triple: tuple[float, float, float]) -> None:
    assert_is_a_distribution(shin(triple))


@settings(max_examples=200)
@given(price_triples)
def test_shin_moves_probability_from_the_longshot_to_the_favourite(
    triple: tuple[float, float, float],
) -> None:
    """Shin's correction of the favourite-longshot bias, the direction of the whole method."""
    favourite = min(range(3), key=lambda index: triple[index])
    longshot = max(range(3), key=lambda index: triple[index])
    shin_probabilities = shin(triple).as_tuple()
    proportional_probabilities = proportional(triple).as_tuple()

    assert shin_probabilities[favourite] >= proportional_probabilities[favourite] - 1e-12
    assert shin_probabilities[longshot] <= proportional_probabilities[longshot] + 1e-12


def test_shin_corrects_the_favourite_longshot_bias_on_a_known_example() -> None:
    """Proportional normalisation underestimates favourites; Shin lifts the favourite instead.

    The spec's phase 2 checklist reads "Shin gives a less extreme favourite probability than
    proportional", which is the opposite of what the published method does: basic normalisation
    is biased too low for favourites and too high for longshots, and Shin corrects exactly that.
    Verified below against the reference implementation's own example.
    """
    assert shin(FAVOURITE).home > proportional(FAVOURITE).home
    assert shin(FAVOURITE).away < proportional(FAVOURITE).away
    assert_is_a_distribution(shin(FAVOURITE))
    assert_is_a_distribution(proportional(FAVOURITE))


def test_shin_matches_the_reference_implementation() -> None:
    assert shin(REFERENCE_PRICES).as_tuple() == pytest.approx(REFERENCE_SHIN, abs=1e-12)


def test_overround_is_the_bookmaker_margin() -> None:
    assert overround(FAVOURITE) > 1.0
    fair = (3.0, 3.0, 3.0)
    assert overround(fair) == pytest.approx(1.0, abs=TOLERANCE)
    assert proportional(fair).as_tuple() == pytest.approx((1 / 3, 1 / 3, 1 / 3), abs=TOLERANCE)


def test_a_fair_book_leaves_shin_equal_to_proportional() -> None:
    fair = (3.0, 3.0, 3.0)
    assert shin(fair).as_tuple() == pytest.approx(proportional(fair).as_tuple(), abs=TOLERANCE)


@pytest.mark.parametrize("triple", [(1.0, 5.0, 5.0), (0.0, 3.0, 3.0), (-2.0, 3.0, 3.0)])
def test_prices_at_or_below_evens_are_rejected(triple: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError, match="price"):
        proportional(triple)
    with pytest.raises(ValueError, match="price"):
        shin(triple)


def test_the_method_name_travels_with_the_probabilities() -> None:
    assert proportional(FAVOURITE).method == "proportional"
    assert shin(FAVOURITE).method == "shin"
