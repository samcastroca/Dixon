"""pandera schemas: a corrupted source file never reaches the clean tables."""

from __future__ import annotations

import pytest

from predictor.config import get_settings
from predictor.ingestion.staging import parse_rows
from predictor.processing.clean import RowCleaner
from predictor.processing.validate import ProcessingValidationError, validate
from tests.support import corrupt_bytes, recorded_bytes

ENCODINGS = ("utf-8-sig", "cp1252")


def clean_matches(payload: bytes, competition_code: str, season: str) -> list[object]:
    competition = get_settings().competition(competition_code)
    cleaner = RowCleaner.from_settings(competition)
    return [cleaner.clean(season, row) for _, row in parse_rows(payload, ENCODINGS)]


def test_a_recorded_season_passes_validation() -> None:
    matches = clean_matches(recorded_bytes("EPL", "2024-25"), "EPL", "2024-25")

    frames = validate(matches)  # type: ignore[arg-type]

    assert len(frames.matches) == 380
    assert len(frames.match_stats) == 760
    assert not frames.odds.empty


def test_the_corrupted_fixture_is_rejected_with_a_readable_error() -> None:
    matches = clean_matches(corrupt_bytes(), "EPL", "2024-25")

    with pytest.raises(ProcessingValidationError) as error:
        validate(matches)  # type: ignore[arg-type]

    message = str(error.value)
    assert "matches" in message and "odds" in message
    assert "home_goals" in message  # the negative score
    assert "-1" in message
    assert "duplicate" in message.lower()  # the repeated fixture
    assert "price_home" in message  # the 1.00 price
