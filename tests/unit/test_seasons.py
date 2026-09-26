"""Season labels and the range syntax used by the CLI."""

from __future__ import annotations

import pytest

from predictor.ingestion.seasons import parse_seasons, season_code, season_label


def test_range_covers_start_year_to_end_year() -> None:
    seasons = parse_seasons("2014-2025")

    assert len(seasons) == 11
    assert seasons[0] == "2014-15"
    assert seasons[-1] == "2024-25"


def test_single_season_range() -> None:
    assert parse_seasons("2024-2025") == ("2024-25",)


def test_explicit_comma_separated_list() -> None:
    assert parse_seasons("2014-15, 2020-21") == ("2014-15", "2020-21")


def test_season_code_is_the_four_digit_source_code() -> None:
    assert season_code("2014-15") == "1415"
    assert season_code("1999-00") == "9900"
    assert season_code("2024-25") == "2425"


def test_season_label_from_start_year() -> None:
    assert season_label(2014) == "2014-15"
    assert season_label(1999) == "1999-00"


@pytest.mark.parametrize(
    "value",
    ["2025-2014", "2014", "20xx-2025", "2014-2014", "", "2014-15-16"],
)
def test_rejects_malformed_ranges(value: str) -> None:
    with pytest.raises(ValueError, match="season"):
        parse_seasons(value)
