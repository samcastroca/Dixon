"""Rest, congestion, season stage and the promoted-team flag."""

from __future__ import annotations

import pytest

from predictor.features.replay import MatchRecord, ReplayEngine
from predictor.features.schedule import ScheduleFeatures
from tests.support import kickoff, record

TEAM = 10
MATCHES_PER_TEAM = 38


def schedule() -> ScheduleFeatures:
    return ScheduleFeatures(congestion_window_days=14, matches_per_team=MATCHES_PER_TEAM)


def replay(records: list[MatchRecord]) -> dict[int, dict[str, float | None]]:
    engine = ReplayEngine(records, as_of_offset_seconds=1)
    return {step.match.match_id: step.values for step in engine.run([schedule()])}


def test_a_debut_has_no_rest_and_nothing_played() -> None:
    values = replay([record(1, TEAM, 20, kickoff(1), 1, 0)])[1]
    assert values["home_rest_days"] is None
    assert values["home_matches_last_14d"] == 0.0
    assert values["home_matches_played"] == 0.0
    assert values["home_season_stage"] == 0.0


def test_rest_days_count_from_the_previous_appearance() -> None:
    records = [record(1, TEAM, 20, kickoff(1), 1, 0), record(2, 30, TEAM, kickoff(8), 0, 0)]
    assert replay(records)[2]["away_rest_days"] == pytest.approx(7.0, abs=1e-12)


def test_rest_days_keep_the_fraction_of_a_day() -> None:
    records = [
        record(1, TEAM, 20, kickoff(1, hour=12), 1, 0),
        record(2, TEAM, 30, kickoff(4, hour=18), 0, 0),
    ]
    assert replay(records)[2]["home_rest_days"] == pytest.approx(3.25, abs=1e-12)


def test_the_congestion_window_includes_its_own_edge() -> None:
    """A match exactly 14 days before kickoff counts; one a day older does not."""
    on_the_edge = [record(1, TEAM, 20, kickoff(1), 1, 0), record(2, TEAM, 30, kickoff(15), 0, 0)]
    just_outside = [record(1, TEAM, 20, kickoff(1), 1, 0), record(2, TEAM, 30, kickoff(16), 0, 0)]
    assert replay(on_the_edge)[2]["home_matches_last_14d"] == 1.0
    assert replay(just_outside)[2]["home_matches_last_14d"] == 0.0


def test_congestion_counts_every_appearance_in_the_window() -> None:
    records = [
        record(1, TEAM, 20, kickoff(1), 1, 0),
        record(2, 30, TEAM, kickoff(5), 0, 0),
        record(3, TEAM, 40, kickoff(9), 0, 0),
        record(4, TEAM, 50, kickoff(12), 0, 0),
    ]
    assert replay(records)[4]["home_matches_last_14d"] == 3.0


def test_season_stage_is_the_share_of_the_season_already_played() -> None:
    records = [
        record(1, TEAM, 20, kickoff(1), 1, 0),
        record(2, TEAM, 30, kickoff(2), 1, 0),
        record(3, TEAM, 40, kickoff(3), 1, 0),
    ]
    values = replay(records)[3]
    assert values["home_matches_played"] == 2.0
    assert values["home_season_stage"] == pytest.approx(2.0 / MATCHES_PER_TEAM, abs=1e-12)


def test_matches_played_restarts_with_the_season() -> None:
    records = [
        record(1, TEAM, 20, kickoff(1), 1, 0),
        record(2, TEAM, 30, kickoff(2), 1, 0),
        record(3, TEAM, 20, kickoff(20), 1, 0, season_id=2, season_label="2021-22"),
    ]
    assert replay(records)[3]["home_matches_played"] == 0.0


def test_the_promoted_flag_is_unknown_in_the_first_season_on_record() -> None:
    values = replay([record(1, TEAM, 20, kickoff(1), 1, 0)])[1]
    assert values["home_is_promoted"] is None
    assert values["away_is_promoted"] is None


def test_a_team_absent_from_the_previous_season_is_promoted() -> None:
    records = [
        record(1, TEAM, 20, kickoff(1), 1, 0, season_id=1, season_label="2019-20"),
        record(2, TEAM, 30, kickoff(20), 1, 0, season_id=2, season_label="2020-21"),
    ]
    values = replay(records)[2]
    assert values["home_is_promoted"] == 0.0
    assert values["away_is_promoted"] == 1.0


def test_the_promoted_flag_never_reads_the_current_season_squad() -> None:
    """Team 30 only ever appears later in season 2; that cannot change match 2's flags."""
    short = [
        record(1, TEAM, 20, kickoff(1), 1, 0, season_id=1, season_label="2019-20"),
        record(2, TEAM, 20, kickoff(20), 1, 0, season_id=2, season_label="2020-21"),
    ]
    longer = [*short, record(3, TEAM, 30, kickoff(25), 1, 0, season_id=2, season_label="2020-21")]
    assert replay(short)[2] == replay(longer)[2]


def test_the_feature_names_cover_both_sides() -> None:
    assert schedule().names == (
        "home_rest_days",
        "away_rest_days",
        "home_matches_last_14d",
        "away_matches_last_14d",
        "home_matches_played",
        "away_matches_played",
        "home_season_stage",
        "away_season_stage",
        "home_is_promoted",
        "away_is_promoted",
    )


def test_the_congestion_window_shows_up_in_the_feature_name() -> None:
    assert (
        "home_matches_last_7d"
        in ScheduleFeatures(congestion_window_days=7, matches_per_team=38).names
    )
