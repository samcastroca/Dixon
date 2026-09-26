"""Database tables. Every change here needs an Alembic migration.

Kept apart from `predictor.models`, which the spec reserves for the MatchModel implementations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from predictor.db import Base


class RawPayload(Base):
    """Immutable copy of exactly what a source returned (spec section 5)."""

    __tablename__ = "raw_payloads"
    __table_args__ = (
        UniqueConstraint("source", "resource", "checksum", name="uq_raw_payloads_identity"),
        Index("ix_raw_payloads_lookup", "source", "competition", "resource"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(64))
    competition: Mapped[str] = mapped_column(String(32))
    resource: Mapped[str] = mapped_column(Text)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    checksum: Mapped[str] = mapped_column(String(64))
    payload: Mapped[bytes] = mapped_column(LargeBinary)
    content_type: Mapped[str | None] = mapped_column(String(128), nullable=True)


class StgMatch(Base):
    """One CSV row, exactly as it was, with every value as text inside the JSON document."""

    __tablename__ = "stg_matches"
    __table_args__ = (
        UniqueConstraint("raw_payload_id", "row_index", name="uq_stg_matches_row"),
        Index("ix_stg_matches_season", "competition", "season"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    raw_payload_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("raw_payloads.id", ondelete="CASCADE")
    )
    competition: Mapped[str] = mapped_column(String(32))
    season: Mapped[str] = mapped_column(String(16))
    row_index: Mapped[int] = mapped_column(Integer)
    row: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
