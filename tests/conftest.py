"""Shared fixtures. No test in this suite touches the network."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from predictor import db
from predictor.config import SETTINGS_FILE_ENV, Settings, get_settings

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS_FILE = REPO_ROOT / "config" / "settings.yaml"


@pytest.fixture(autouse=True)
def _isolated_caches() -> Iterator[None]:
    """Never let a cached settings object or engine leak between tests."""
    get_settings.cache_clear()
    db.get_engine.cache_clear()
    db.get_sessionmaker.cache_clear()
    yield
    get_settings.cache_clear()
    db.get_engine.cache_clear()
    db.get_sessionmaker.cache_clear()


@pytest.fixture
def settings() -> Settings:
    return get_settings()


@pytest.fixture
def settings_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Point the loader at a temporary settings file the test writes itself."""
    path = tmp_path / "settings.yaml"
    monkeypatch.setenv(SETTINGS_FILE_ENV, str(path))
    yield path


@pytest.fixture(scope="session")
def database_available() -> bool:
    """Probe the database once per session: an unreachable host is slow to fail."""
    return db.check_connection()
