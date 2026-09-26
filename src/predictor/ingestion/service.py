"""Orchestration: fetch every configured resource, store it raw, then stage its rows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from predictor.config import Competition
from predictor.db import session_scope
from predictor.ingestion.base import Resource, Source
from predictor.ingestion.repository import store_raw_payload, store_staging_rows
from predictor.ingestion.staging import parse_rows
from predictor.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class IngestEntry:
    competition: str
    season: str
    resource: str
    checksum: str
    payload_bytes: int
    raw_inserted: bool
    staged_rows: int


@dataclass(frozen=True, slots=True)
class IngestReport:
    entries: tuple[IngestEntry, ...]

    @property
    def raw_inserted(self) -> int:
        return sum(1 for entry in self.entries if entry.raw_inserted)

    @property
    def staged_rows(self) -> int:
        return sum(entry.staged_rows for entry in self.entries)


def ingest(
    source: Source,
    competitions: Sequence[Competition],
    seasons: Sequence[str],
) -> IngestReport:
    """Fetch and store every (competition, season) pair. Safe to run again at any time."""
    resources: list[Resource] = []
    for competition in competitions:
        resources.extend(source.list_resources(competition, seasons))

    entries: list[IngestEntry] = []
    for resource in resources:
        document = source.fetch(resource)

        with session_scope() as session:
            payload_id, inserted = store_raw_payload(session, document)
            staged = 0
            if inserted:
                staged = store_staging_rows(
                    session,
                    payload_id,
                    competition=resource.competition_code,
                    season=resource.season,
                    rows=parse_rows(document.payload, source.encodings),
                )

        logger.info(
            "ingest.stored",
            source=resource.source,
            competition=resource.competition_code,
            season=resource.season,
            resource=resource.resource,
            raw_inserted=inserted,
            staged_rows=staged,
        )
        entries.append(
            IngestEntry(
                competition=resource.competition_code,
                season=resource.season,
                resource=resource.resource,
                checksum=document.checksum,
                payload_bytes=len(document.payload),
                raw_inserted=inserted,
                staged_rows=staged,
            )
        )

    return IngestReport(entries=tuple(entries))
