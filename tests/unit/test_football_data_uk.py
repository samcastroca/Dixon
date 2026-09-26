"""Parsing the recorded season CSVs of both leagues, with the HTTP layer mocked."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime

import pytest
from tests.support import FakeClock, fixture_transport, recorded_bytes

from predictor.config import get_settings
from predictor.ingestion.base import ResourceNotFound
from predictor.ingestion.football_data_uk import FootballDataUkSource
from predictor.ingestion.http import HttpClient
from predictor.ingestion.staging import parse_rows

MATCHES_PER_SEASON = 380
CORE_COLUMNS = ("Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR")


@pytest.fixture
def source() -> FootballDataUkSource:
    clock = FakeClock()
    config = get_settings().source("football_data_uk")
    client = HttpClient(
        config, transport=fixture_transport(), sleep=clock.sleep, clock=clock.monotonic
    )
    return FootballDataUkSource(client=client)


@pytest.mark.parametrize("competition_code", ["EPL", "LALIGA"])
@pytest.mark.parametrize("season", ["2014-15", "2024-25"])
def test_recorded_season_has_380_rows_and_the_core_columns(
    source: FootballDataUkSource, competition_code: str, season: str
) -> None:
    competition = get_settings().competition(competition_code)
    resource = source.list_resources(competition, [season])[0]

    document = source.fetch(resource)
    rows = list(parse_rows(document.payload, source.encodings))

    assert len(rows) == MATCHES_PER_SEASON
    indexes = [index for index, _ in rows]
    assert indexes == list(range(MATCHES_PER_SEASON))
    for _, row in rows:
        for column in CORE_COLUMNS:
            assert column in row
    assert {row["Div"] for _, row in rows} == {competition.source_code("football_data_uk")}


def test_column_layout_differs_between_seasons(source: FootballDataUkSource) -> None:
    competition = get_settings().competition("EPL")
    old, new = (
        source.fetch(source.list_resources(competition, [season])[0])
        for season in ("2014-15", "2024-25")
    )

    old_columns = next(iter(parse_rows(old.payload, source.encodings)))[1]
    new_columns = next(iter(parse_rows(new.payload, source.encodings)))[1]

    assert "Time" not in old_columns
    assert "Time" in new_columns
    # The UTF-8 BOM of the recent files must not leak into the first column name.
    assert "Div" in new_columns


def test_resources_are_built_from_configuration(source: FootballDataUkSource) -> None:
    settings = get_settings()
    resources = source.list_resources(settings.competition("LALIGA"), ["2014-15", "2024-25"])

    assert [resource.resource for resource in resources] == [
        "1415/SP1.csv",
        "2425/SP1.csv",
    ]
    assert resources[0].url.startswith(settings.source("football_data_uk").base_url)
    assert resources[0].competition_code == "LALIGA"
    assert resources[0].season == "2014-15"
    assert resources[0].source == source.name


def test_fetch_returns_raw_bytes_and_checksum(source: FootballDataUkSource) -> None:
    competition = get_settings().competition("EPL")
    resource = source.list_resources(competition, ["2024-25"])[0]

    before = datetime.now(UTC)
    document = source.fetch(resource)

    expected = recorded_bytes("EPL", "2024-25")
    assert document.payload == expected
    assert document.checksum == hashlib.sha256(expected).hexdigest()
    assert document.fetched_at >= before
    assert document.fetched_at.tzinfo is UTC


def test_missing_season_names_the_resource(source: FootballDataUkSource) -> None:
    competition = get_settings().competition("EPL")
    resource = source.list_resources(competition, ["1990-91"])[0]

    with pytest.raises(ResourceNotFound, match=re.escape("9091/E0.csv")):
        source.fetch(resource)


def test_parse_rows_drops_the_trailing_empty_row() -> None:
    payload = recorded_bytes("EPL", "2014-15")

    # The recorded file really does end with a row of commas.
    assert payload.rstrip().endswith(b",")
    assert len(list(parse_rows(payload, ("utf-8-sig", "cp1252")))) == MATCHES_PER_SEASON


def test_parse_rows_keeps_every_value_as_text() -> None:
    payload = recorded_bytes("LALIGA", "2024-25")
    header, first_line = payload.decode("utf-8-sig").splitlines()[:2]
    expected = dict(zip(header.split(","), first_line.split(","), strict=False))

    _, row = next(iter(parse_rows(payload, ("utf-8-sig",))))

    # Byte-for-byte the same text as the file: no trimming, no number parsing, no dates.
    for column in ("Date", "HomeTeam", "FTHG", "B365H", "AHh"):
        assert row[column] == expected[column]
    assert isinstance(row["FTHG"], str)
