"""GET /health against a live PostgreSQL (docker compose up -d, or the CI service)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from predictor.api.main import app

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _requires_database(database_available: bool) -> None:
    if not database_available:
        pytest.skip("no database reachable; run `make up` or set DATABASE_URL")


def test_health_reports_ok() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "db": "ok"}


def test_health_reports_degraded_when_the_database_is_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("predictor.api.main.check_connection", lambda: False)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "db": "error"}
