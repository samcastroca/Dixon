"""Staging rows to clean matches: kickoff in UTC, status, stats and odds."""

from __future__ import annotations

from datetime import UTC, date, datetime, time

import pytest
from tests.support import recorded_bytes

from predictor.config import Competition, Settings, get_settings
from predictor.ingestion.staging import parse_rows
from predictor.processing.clean import MatchStatus, RowCleaner, parse_kickoff

ENCODINGS = ("utf-8-sig", "cp1252")


@pytest.fixture
def settings() -> Settings:
    return get_settings()


@pytest.fixture
def epl(settings: Settings) -> Competition:
    return settings.competition("EPL")


@pytest.fixture
def laliga(settings: Settings) -> Competition:
    return settings.competition("LALIGA")


def first_row(competition: str, season: str) -> dict[str, str]:
    rows = parse_rows(recorded_bytes(competition, season), ENCODINGS)
    return next(iter(rows))[1]


@pytest.mark.parametrize(
    ("day", "clock", "expected"),
    [
        ("11/01/2025", "21:00", datetime(2025, 1, 11, 20, 0, tzinfo=UTC)),  # CET
        ("15/09/2024", "21:00", datetime(2024, 9, 15, 19, 0, tzinfo=UTC)),  # CEST
    ],
)
def test_madrid_kickoffs_convert_across_daylight_saving(
    laliga: Competition, day: str, clock: str, expected: datetime
) -> None:
    assert parse_kickoff(day, clock, laliga, time(15, 0)).utc == expected


@pytest.mark.parametrize(
    ("day", "clock", "expected"),
    [
        ("11/01/2025", "15:00", datetime(2025, 1, 11, 15, 0, tzinfo=UTC)),  # GMT
        ("15/06/2025", "15:00", datetime(2025, 6, 15, 14, 0, tzinfo=UTC)),  # BST
    ],
)
def test_london_kickoffs_convert_across_daylight_saving(
    epl: Competition, day: str, clock: str, expected: datetime
) -> None:
    assert parse_kickoff(day, clock, epl, time(15, 0)).utc == expected


def test_a_missing_time_falls_back_to_the_configured_local_time(laliga: Competition) -> None:
    kickoff = parse_kickoff("15/09/2024", "", laliga, time(21, 0))

    assert kickoff.time_estimated is True
    assert kickoff.utc == datetime(2024, 9, 15, 19, 0, tzinfo=UTC)
    assert kickoff.local_date == date(2024, 9, 15)


def test_a_present_time_is_never_flagged_as_estimated(laliga: Competition) -> None:
    assert parse_kickoff("15/09/2024", "21:00", laliga, time(15, 0)).time_estimated is False


@pytest.mark.parametrize("day", ["23/08/14", "23/08/2014"])
def test_both_date_layouts_of_the_source_are_understood(laliga: Competition, day: str) -> None:
    assert parse_kickoff(day, "21:00", laliga, time(15, 0)).local_date == date(2014, 8, 23)


def test_an_unparseable_date_fails_loudly(laliga: Competition) -> None:
    with pytest.raises(ValueError, match="date"):
        parse_kickoff("2014-08-23", "21:00", laliga, time(15, 0))


def test_a_current_layout_row_is_cleaned_end_to_end(laliga: Competition) -> None:
    cleaner = RowCleaner.from_settings(laliga)

    match = cleaner.clean("2024-25", first_row("LALIGA", "2024-25"))

    assert (match.home, match.away) == ("Athletic Club", "Getafe CF")
    assert (match.home_goals, match.away_goals) == (1, 1)
    assert (match.ht_home_goals, match.ht_away_goals) == (1, 0)
    assert match.result == "D"
    assert match.status is MatchStatus.PLAYED
    assert match.kickoff.utc == datetime(2024, 8, 15, 16, 0, tzinfo=UTC)
    assert match.kickoff.time_estimated is False

    home, away = match.stats
    assert (home.side, home.shots, home.shots_on_target, home.corners) == ("home", 7, 4, 5)
    assert (away.side, away.shots, away.shots_on_target, away.corners) == ("away", 9, 2, 6)
    assert home.xg is None

    by_key = {(odds.bookmaker, odds.is_closing): odds for odds in match.odds}
    assert by_key[("B365", False)].price_home == pytest.approx(1.48)
    assert ("PS", True) in by_key
    assert all(odds.market == "1X2" for odds in match.odds)
    assert by_key[("B365", False)].p_home_proportional is not None


def test_an_old_layout_row_has_an_estimated_kickoff_and_fewer_bookmakers(
    laliga: Competition,
) -> None:
    cleaner = RowCleaner.from_settings(laliga)

    match = cleaner.clean("2014-15", first_row("LALIGA", "2014-15"))

    assert (match.home, match.away) == ("UD Almería", "RCD Espanyol")
    assert match.kickoff.time_estimated is True
    assert match.kickoff.local_date == date(2014, 8, 23)
    bookmakers = {(odds.bookmaker, odds.is_closing) for odds in match.odds}
    assert ("PS", True) in bookmakers  # Pinnacle closing is in every recorded season
    assert ("B365", True) not in bookmakers  # the closing columns arrive in 2019-20
    assert all(stat.shots is not None for stat in match.stats)


def test_a_row_without_a_result_is_not_marked_as_played(epl: Competition) -> None:
    cleaner = RowCleaner.from_settings(epl)
    row = dict(first_row("EPL", "2024-25")) | {"FTHG": "", "FTAG": "", "FTR": ""}

    match = cleaner.clean("2024-25", row)

    assert match.status is MatchStatus.POSTPONED
    assert (match.home_goals, match.away_goals, match.result) == (None, None, None)


def test_an_unknown_team_stops_the_row(epl: Competition) -> None:
    from predictor.processing.entities import UnknownTeamError

    cleaner = RowCleaner.from_settings(epl)
    row = dict(first_row("EPL", "2024-25")) | {"HomeTeam": "Real Madrid"}

    with pytest.raises(UnknownTeamError):
        cleaner.clean("2024-25", row)
