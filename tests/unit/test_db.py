"""URL building: settings by default, DATABASE_URL when it is set."""

from __future__ import annotations

import pytest

from predictor.config import get_settings
from predictor.db import DRIVER, build_url, check_connection


def test_url_comes_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("PREDICTOR_DATABASE__PASSWORD", "s3cret")
    get_settings.cache_clear()

    url = build_url()

    assert url.drivername == DRIVER
    assert url.host == get_settings().database.host
    assert url.database == "predictor"
    assert url.password == "s3cret"
    assert "s3cret" not in str(url)  # SQLAlchemy masks it


def test_database_url_wins_and_keeps_the_psycopg_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@example:6543/other")

    url = build_url()

    assert url.drivername == DRIVER
    assert (url.host, url.port, url.database) == ("example", 6543, "other")


def test_check_connection_is_false_without_a_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@127.0.0.1:1/none")

    assert check_connection() is False
