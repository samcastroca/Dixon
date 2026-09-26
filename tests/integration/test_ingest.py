"""Ingestion against the compose database, with the HTTP layer mocked (no network)."""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
from sqlalchemy import delete, func, select
from tests.support import FakeClock, fixture_transport, recorded_bytes

from predictor.config import get_settings
from predictor.db import session_scope
from predictor.ingestion.football_data_uk import FootballDataUkSource
from predictor.ingestion.http import HttpClient
from predictor.ingestion.service import ingest
from predictor.tables import RawPayload, StgMatch

pytestmark = pytest.mark.integration

TEST_SOURCE = "football_data_uk_test"
SEASONS = ("2014-15", "2024-25")
COMPETITIONS = ("EPL", "LALIGA")
MATCHES_PER_SEASON = 380


def build_source(transport: httpx.MockTransport) -> FootballDataUkSource:
    clock = FakeClock()
    config = get_settings().source("football_data_uk")
    client = HttpClient(config, transport=transport, sleep=clock.sleep, clock=clock.monotonic)
    return FootballDataUkSource(client=client, name=TEST_SOURCE)


def counts() -> tuple[int, int]:
    with session_scope() as session:
        raw = session.scalar(
            select(func.count()).select_from(RawPayload).where(RawPayload.source == TEST_SOURCE)
        )
        staged = session.scalar(
            select(func.count())
            .select_from(StgMatch)
            .join(RawPayload, RawPayload.id == StgMatch.raw_payload_id)
            .where(RawPayload.source == TEST_SOURCE)
        )
    return int(raw or 0), int(staged or 0)


@pytest.fixture(autouse=True)
def _clean_test_rows(database_available: bool) -> Iterator[None]:
    if not database_available:
        pytest.skip("no database reachable; run `make up` or set DATABASE_URL")

    def purge() -> None:
        with session_scope() as session:
            session.execute(delete(RawPayload).where(RawPayload.source == TEST_SOURCE))

    purge()
    yield
    purge()


@pytest.fixture
def competitions() -> list[object]:
    settings = get_settings()
    return [settings.competition(code) for code in COMPETITIONS]


def test_ingest_is_idempotent(competitions: list[object]) -> None:
    source = build_source(fixture_transport())

    first = ingest(source, competitions, SEASONS)  # type: ignore[arg-type]
    after_first = counts()

    second = ingest(source, competitions, SEASONS)  # type: ignore[arg-type]
    after_second = counts()

    # Pinning the first run's numbers: equal counts must not mean "nothing was ever stored".
    assert after_first == (len(COMPETITIONS) * len(SEASONS), 4 * MATCHES_PER_SEASON)
    assert after_first == after_second
    assert all(entry.raw_inserted for entry in first.entries)
    assert not any(entry.raw_inserted for entry in second.entries)
    assert all(entry.staged_rows == 0 for entry in second.entries)


def test_every_complete_season_has_380_staging_rows(competitions: list[object]) -> None:
    source = build_source(fixture_transport())

    ingest(source, competitions, SEASONS)  # type: ignore[arg-type]

    with session_scope() as session:
        rows = session.execute(
            select(StgMatch.competition, StgMatch.season, func.count())
            .join(RawPayload, RawPayload.id == StgMatch.raw_payload_id)
            .where(RawPayload.source == TEST_SOURCE)
            .group_by(StgMatch.competition, StgMatch.season)
        ).all()

    per_season = {(competition, season): count for competition, season, count in rows}
    assert per_season == {
        (competition, season): MATCHES_PER_SEASON
        for competition in COMPETITIONS
        for season in SEASONS
    }


def test_changed_payload_is_stored_as_a_new_immutable_row(competitions: list[object]) -> None:
    epl = get_settings().competition("EPL")
    original = recorded_bytes("EPL", "2024-25")
    modified = original.replace(b"E0,", b"E0,", 1) + b"\r\nE0,26/05/2025,16:00,Extra,Row,1,1,D\r\n"

    def transport_for(payload: bytes) -> httpx.MockTransport:
        return httpx.MockTransport(lambda request: httpx.Response(200, content=payload))

    ingest(build_source(transport_for(original)), [epl], ["2024-25"])
    ingest(build_source(transport_for(modified)), [epl], ["2024-25"])

    with session_scope() as session:
        payloads = session.scalars(
            select(RawPayload)
            .where(RawPayload.source == TEST_SOURCE)
            .order_by(RawPayload.fetched_at)
        ).all()

    assert len(payloads) == 2
    assert [bytes(payload.payload) for payload in payloads] == [original, modified]
    assert payloads[0].checksum != payloads[1].checksum
    assert payloads[0].resource == payloads[1].resource
