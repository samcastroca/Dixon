"""Idempotent writes: the same fetch twice must not create a second row."""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from predictor.ingestion.base import RawDocument
from predictor.tables import RawPayload, StgMatch


def store_raw_payload(session: Session, document: RawDocument) -> tuple[int, bool]:
    """Insert the payload unless (source, resource, checksum) is already stored.

    Returns the row id and whether this call inserted it.
    """
    statement = (
        insert(RawPayload)
        .values(
            source=document.resource.source,
            competition=document.resource.competition_code,
            resource=document.resource.resource,
            fetched_at=document.fetched_at,
            checksum=document.checksum,
            payload=document.payload,
            content_type=document.content_type,
        )
        .on_conflict_do_nothing(constraint="uq_raw_payloads_identity")
        .returning(RawPayload.id)
    )
    inserted_id = session.execute(statement).scalar_one_or_none()
    if inserted_id is not None:
        return inserted_id, True

    existing_id = session.execute(
        select(RawPayload.id).where(
            RawPayload.source == document.resource.source,
            RawPayload.resource == document.resource.resource,
            RawPayload.checksum == document.checksum,
        )
    ).scalar_one()
    return existing_id, False


def store_staging_rows(
    session: Session,
    raw_payload_id: int,
    competition: str,
    season: str,
    rows: Iterable[tuple[int, dict[str, str]]],
) -> int:
    """Insert the CSV rows of one payload. Re-running inserts nothing new."""
    values = [
        {
            "raw_payload_id": raw_payload_id,
            "competition": competition,
            "season": season,
            "row_index": index,
            "row": row,
        }
        for index, row in rows
    ]
    if not values:
        return 0

    statement = (
        insert(StgMatch)
        .values(values)
        .on_conflict_do_nothing(constraint="uq_stg_matches_row")
        .returning(StgMatch.id)
    )
    return len(session.execute(statement).scalars().all())
