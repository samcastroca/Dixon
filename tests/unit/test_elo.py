"""Pre-match Elo. The golden numbers below were worked out by hand, step by step."""

from __future__ import annotations

import pytest

from predictor.config import EloParameters
from predictor.features.elo import EloRatings
from predictor.features.replay import MatchRecord, ReplayEngine
from tests.support import kickoff, record

A, B, C, D, E = 1, 2, 3, 4, 5

PARAMETERS = EloParameters(
    k=20.0,
    home_advantage=100.0,
    initial_rating=1500.0,
    two_goal_multiplier=1.5,
    large_margin_base=11.0,
    large_margin_divisor=8.0,
    season_regression=0.0,
    season_regression_target=1500.0,
)

# Five matches, four teams, one season. Ratings start at 1500 and the home side gets +100.
#   E_home = 1 / (1 + 10 ** (-(Rh + 100 - Ra) / 400)),  R' = R ± K·G·(S - E)
#   G = 1 for a 0 or 1 goal margin, 1.5 for two, (11 + margin) / 8 from three upwards.
#
#   m1  A 2-0 B   d=0     E=0.6400649998028851  G=1.5   Δ=+10.798050005913446
#   m2  C 1-1 D   d=0     E=0.6400649998028851  G=1.0   Δ=-2.8012999960577023
#   m3  B 0-3 C   d=-7.997  E=0.6293929111132002  G=1.75  Δ=-22.028751888962006
#   m4  D 1-0 A   d=-7.997  E=0.6293929111132002  G=1.0   Δ=+7.412141777735997
#   m5  A 2-1 C   d=-15.841 E=0.618799308993343   G=1.0   Δ=+7.62401382013314
TOY = [
    record(1, A, B, kickoff(1), 2, 0),
    record(2, C, D, kickoff(2), 1, 1),
    record(3, B, C, kickoff(3), 0, 3),
    record(4, D, A, kickoff(4), 1, 0),
    record(5, A, C, kickoff(5), 2, 1),
]

EXPECTED_PRE = {
    1: (1500.0, 1500.0, 0.6400649998028851),
    2: (1500.0, 1500.0, 0.6400649998028851),
    3: (1489.2019499940866, 1497.1987000039423, 0.6293929111132002),
    4: (1502.8012999960577, 1510.7980500059134, 0.6293929111132002),
    5: (1503.3859082281774, 1519.2274518929044, 0.618799308993343),
}

EXPECTED_FINAL = {
    A: 1511.0099220483105,
    B: 1467.1731981051246,
    C: 1511.6034380727713,
    D: 1510.2134417737936,
}


def replay(records: list[MatchRecord], ratings: EloRatings) -> dict[int, dict[str, float | None]]:
    engine = ReplayEngine(records, as_of_offset_seconds=1)
    return {step.match.match_id: step.values for step in engine.run([ratings])}


def test_the_toy_dataset_reproduces_the_hand_calculated_ratings() -> None:
    ratings = EloRatings(PARAMETERS, relegation_slots=2)
    values = replay(TOY, ratings)

    for match_id, (home, away, expected_home) in EXPECTED_PRE.items():
        assert values[match_id]["home_elo_pre"] == pytest.approx(home, abs=1e-9), match_id
        assert values[match_id]["away_elo_pre"] == pytest.approx(away, abs=1e-9), match_id
        assert values[match_id]["elo_expected_home"] == pytest.approx(expected_home, abs=1e-12)

    for team, expected in EXPECTED_FINAL.items():
        assert ratings.rating(team) == pytest.approx(expected, abs=1e-9), team


def test_elo_diff_is_the_plain_rating_difference() -> None:
    values = replay(TOY, EloRatings(PARAMETERS, relegation_slots=2))
    assert values[5]["elo_diff"] == pytest.approx(1503.3859082281774 - 1519.2274518929044, abs=1e-9)


def test_ratings_are_zero_sum_within_a_match() -> None:
    ratings = EloRatings(PARAMETERS, relegation_slots=2)
    replay(TOY, ratings)
    assert sum(ratings.ratings().values()) == pytest.approx(4 * 1500.0, abs=1e-9)


def test_an_unplayed_match_moves_no_rating() -> None:
    ratings = EloRatings(PARAMETERS, relegation_slots=2)
    replay([record(1, A, B, kickoff(1), status="postponed")], ratings)
    assert ratings.ratings() == {}


def test_the_season_start_regresses_every_rating_towards_the_target() -> None:
    """After the toy season, a quarter of each gap to 1500 is given back."""
    parameters = PARAMETERS.model_copy(update={"season_regression": 0.25})
    ratings = EloRatings(parameters, relegation_slots=2)
    next_season = [*TOY, record(6, A, B, kickoff(40), 1, 0, season_id=2, season_label="2021-22")]
    values = replay(next_season, ratings)

    assert values[6]["home_elo_pre"] == pytest.approx(1508.2574415362328, abs=1e-9)
    assert values[6]["away_elo_pre"] == pytest.approx(1475.3798985788435, abs=1e-9)


def test_a_promoted_team_starts_from_the_lowest_rated_of_the_previous_season() -> None:
    """Two relegation slots, so the mean of B (1467.173...) and D (1510.213...)."""
    ratings = EloRatings(PARAMETERS, relegation_slots=2)
    with_newcomer = [*TOY, record(6, E, A, kickoff(40), 0, 0, season_id=2, season_label="2021-22")]
    values = replay(with_newcomer, ratings)

    assert values[6]["home_elo_pre"] == pytest.approx(1488.6933199394591, abs=1e-9)


def test_the_first_team_ever_seen_starts_from_the_configured_initial_rating() -> None:
    ratings = EloRatings(PARAMETERS, relegation_slots=2)
    values = replay([record(1, A, B, kickoff(1), 1, 0)], ratings)
    assert values[1]["home_elo_pre"] == 1500.0
    assert values[1]["away_elo_pre"] == 1500.0


@pytest.mark.parametrize(
    ("margin", "expected"),
    [(0, 1.0), (1, 1.0), (-1, 1.0), (2, 1.5), (-2, 1.5), (3, 1.75), (5, 2.0)],
)
def test_the_goal_difference_multiplier_follows_the_configured_curve(
    margin: int, expected: float
) -> None:
    assert EloRatings(PARAMETERS, relegation_slots=2).goal_difference_multiplier(margin) == expected


def test_the_feature_names_are_stable_and_complete() -> None:
    assert EloRatings(PARAMETERS, relegation_slots=2).names == (
        "home_elo_pre",
        "away_elo_pre",
        "elo_diff",
        "elo_expected_home",
    )
