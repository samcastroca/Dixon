"""The scoreline matrix and what is derived from it (spec section 7, acceptance test 2 and 3)."""

from __future__ import annotations

import numpy as np
import pytest

from predictor.models.scorelines import (
    outcome_probabilities,
    score_matrix,
    tau,
    truncation_deficit,
)

#: Acceptance test 2: the truncated matrix may lose this much mass and no more.
TRUNCATION_TOLERANCE = 1e-4

#: Acceptance test 2: a probability row sums to one this tightly.
SUM_TOLERANCE = 1e-9

#: Where `config/settings.yaml` truncates. The pydantic default is the spec's 10; the
#: configured value is wider, for the reason `test_ten_is_not_enough_for_the_highest_rates`
#: measures.
MAX_GOALS = 12

#: Goal rates spanning what a fitted model actually produces, the last pair being roughly
#: the strongest home side against the weakest away side in a real league.
RATES = ((0.4, 0.3), (1.6, 1.2), (2.4, 0.9), (3.0, 2.6))


class TestTau:
    """The low-score correction of Dixon-Coles, one clause at a time."""

    def test_only_the_four_low_scorelines_are_corrected(self) -> None:
        for home, away in ((0, 2), (2, 0), (1, 2), (2, 1), (3, 3)):
            assert tau(home, away, 1.5, 1.2, -0.13) == 1.0

    def test_each_corrected_cell_matches_the_published_formula(self) -> None:
        rho, home_rate, away_rate = -0.13, 1.5, 1.2
        assert tau(0, 0, home_rate, away_rate, rho) == pytest.approx(
            1.0 - home_rate * away_rate * rho
        )
        assert tau(0, 1, home_rate, away_rate, rho) == pytest.approx(1.0 + home_rate * rho)
        assert tau(1, 0, home_rate, away_rate, rho) == pytest.approx(1.0 + away_rate * rho)
        assert tau(1, 1, home_rate, away_rate, rho) == pytest.approx(1.0 - rho)

    def test_rho_of_zero_is_the_independent_model(self) -> None:
        for home in range(3):
            for away in range(3):
                assert tau(home, away, 1.5, 1.2, 0.0) == 1.0


class TestScoreMatrix:
    """Acceptance test 2: the matrix keeps its truncation error, and only that."""

    @pytest.mark.parametrize(("home_rate", "away_rate"), RATES)
    def test_truncation_error_stays_under_the_tolerance(
        self, home_rate: float, away_rate: float
    ) -> None:
        matrix = score_matrix(home_rate, away_rate, max_goals=MAX_GOALS)
        assert 0.0 <= truncation_deficit(matrix) < TRUNCATION_TOLERANCE

    def test_ten_is_not_enough_for_the_highest_rates(self) -> None:
        # The spec names 10 as the default, and it holds for an ordinary fixture. It does not
        # hold for the highest-scoring one, which is why settings.yaml configures a wider grid
        # rather than leaving the acceptance criterion to luck.
        assert truncation_deficit(score_matrix(1.6, 1.2, max_goals=10)) < TRUNCATION_TOLERANCE
        assert truncation_deficit(score_matrix(3.0, 2.6, max_goals=10)) > TRUNCATION_TOLERANCE

    @pytest.mark.parametrize(("home_rate", "away_rate"), RATES)
    def test_the_correction_moves_mass_without_creating_any(
        self, home_rate: float, away_rate: float
    ) -> None:
        # tau is built to preserve the marginal totals, so the only loss is the truncation.
        plain = score_matrix(home_rate, away_rate, max_goals=18)
        corrected = score_matrix(home_rate, away_rate, max_goals=18, rho=-0.13)
        assert corrected.sum() == pytest.approx(plain.sum(), abs=1e-12)

    def test_the_correction_actually_changes_the_low_scores(self) -> None:
        plain = score_matrix(1.5, 1.2, max_goals=10)
        corrected = score_matrix(1.5, 1.2, max_goals=10, rho=-0.13)
        # A negative rho is what makes Dixon-Coles produce more draws than plain Poisson:
        # it lifts 0-0 and 1-1 and takes the mass from 0-1 and 1-0.
        assert corrected[0, 0] > plain[0, 0]
        assert corrected[1, 1] > plain[1, 1]
        assert corrected[0, 1] < plain[0, 1]
        assert corrected[1, 0] < plain[1, 0]
        assert corrected[3, 2] == pytest.approx(plain[3, 2])

    def test_a_wider_grid_loses_less_mass(self) -> None:
        deficits = [truncation_deficit(score_matrix(2.4, 2.1, max_goals=n)) for n in (5, 8, 12)]
        assert deficits[0] > deficits[1] > deficits[2]

    def test_the_matrix_is_square_and_non_negative(self) -> None:
        matrix = score_matrix(1.5, 1.2, max_goals=7, rho=-0.13)
        assert matrix.shape == (8, 8)
        assert matrix.min() >= 0.0


class TestOutcomeProbabilities:
    """Acceptance tests 2 and 3, at the level where symmetry is exact."""

    @pytest.mark.parametrize(("home_rate", "away_rate"), RATES)
    def test_the_row_sums_to_one(self, home_rate: float, away_rate: float) -> None:
        probabilities = outcome_probabilities(
            score_matrix(home_rate, away_rate, max_goals=MAX_GOALS)
        )
        assert sum(probabilities) == pytest.approx(1.0, abs=SUM_TOLERANCE)

    @pytest.mark.parametrize("rho", [0.0, -0.13, 0.11])
    def test_equal_rates_give_equal_home_and_away(self, rho: float) -> None:
        # Two identical teams with no home advantage share a rate, and a symmetric matrix
        # cannot prefer one side. This is acceptance test 3 at its exact form.
        p_home, _, p_away = outcome_probabilities(score_matrix(1.4, 1.4, max_goals=10, rho=rho))
        assert p_home == p_away

    def test_the_stronger_side_is_the_likelier_winner(self) -> None:
        p_home, _, p_away = outcome_probabilities(score_matrix(2.2, 0.9, max_goals=10))
        assert p_home > p_away

    def test_the_three_outcomes_partition_the_matrix(self) -> None:
        matrix = score_matrix(1.6, 1.2, max_goals=10, rho=-0.13)
        p_home, p_draw, p_away = outcome_probabilities(matrix)
        total = matrix.sum()
        assert p_draw == pytest.approx(float(np.trace(matrix)) / total)
        assert p_home == pytest.approx(float(np.tril(matrix, -1).sum()) / total)
        assert p_away == pytest.approx(float(np.triu(matrix, 1).sum()) / total)
