"""The committed settings file loads both competitions; broken config is rejected."""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from predictor.config import Settings, get_settings

VALID_COMPETITION = """
competitions:
  - code: EPL
    sport: football
    name: English Premier League
    country: England
    timezone: Europe/London
    sources: {football_data_uk: E0}
    rules: {teams: 20, relegation_slots: 3, champions_league_slots: 4, tiebreaker: goal_difference}
"""


def test_loads_both_competitions() -> None:
    settings = get_settings()

    assert settings.competition_codes == ("EPL", "LALIGA")

    epl = settings.competition("EPL")
    assert epl.sport == "football"
    assert epl.timezone == "Europe/London"
    assert epl.source_code("football_data_uk") == "E0"
    assert epl.tzinfo() == ZoneInfo("Europe/London")

    laliga = settings.competition("laliga")  # lookup is case-insensitive
    assert laliga.timezone == "Europe/Madrid"
    assert laliga.source_code("football_data_uk") == "SP1"
    assert laliga.rules.tiebreaker == "head_to_head"


def test_unknown_competition_fails_loudly() -> None:
    with pytest.raises(KeyError, match="SERIEA"):
        get_settings().competition("SERIEA")


def test_unknown_source_fails_loudly() -> None:
    with pytest.raises(KeyError, match="api_football"):
        get_settings().competition("EPL").source_code("api_football")


def test_environment_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PREDICTOR_DATABASE__HOST", "localhost")
    monkeypatch.setenv("PREDICTOR_DATABASE__PASSWORD", "s3cret")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.database.host == "localhost"
    assert settings.database.password.get_secret_value() == "s3cret"
    assert "s3cret" not in repr(settings.database)


def test_rejects_unknown_timezone(settings_file: Path) -> None:
    settings_file.write_text(
        VALID_COMPETITION.replace("Europe/London", "Europe/Nowhere"), encoding="utf-8"
    )

    with pytest.raises(ValidationError, match="unknown timezone"):
        Settings()


def test_rejects_duplicate_competition_codes(settings_file: Path) -> None:
    settings_file.write_text(
        VALID_COMPETITION + VALID_COMPETITION.split("competitions:")[1], encoding="utf-8"
    )

    with pytest.raises(ValidationError, match="duplicate competition codes: EPL"):
        Settings()


def test_rejects_empty_competition_list(settings_file: Path) -> None:
    settings_file.write_text("competitions: []\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        Settings()


def test_rejects_competition_without_sources(settings_file: Path) -> None:
    settings_file.write_text(
        VALID_COMPETITION.replace("sources: {football_data_uk: E0}", "sources: {}"),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        Settings()


def test_rejects_unknown_tiebreaker(settings_file: Path) -> None:
    settings_file.write_text(
        VALID_COMPETITION.replace("goal_difference", "coin_flip"), encoding="utf-8"
    )

    with pytest.raises(ValidationError):
        Settings()
