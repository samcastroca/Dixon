"""ingestion: immutable raw payloads and the CSV staging table

Revision ID: 0002_ingestion
Revises: 0001_baseline
Create Date: 2026-09-26

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_ingestion"
down_revision: str | None = "0001_baseline"
branch_labels: None = None
depends_on: None = None


def upgrade() -> None:
    op.create_table(
        "raw_payloads",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("competition", sa.String(length=32), nullable=False),
        sa.Column("resource", sa.Text(), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("checksum", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.LargeBinary(), nullable=False),
        sa.Column("content_type", sa.String(length=128), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source", "resource", "checksum", name="uq_raw_payloads_identity"),
    )
    op.create_index("ix_raw_payloads_lookup", "raw_payloads", ["source", "competition", "resource"])

    op.create_table(
        "stg_matches",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("raw_payload_id", sa.BigInteger(), nullable=False),
        sa.Column("competition", sa.String(length=32), nullable=False),
        sa.Column("season", sa.String(length=16), nullable=False),
        sa.Column("row_index", sa.Integer(), nullable=False),
        sa.Column("row", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["raw_payload_id"], ["raw_payloads.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("raw_payload_id", "row_index", name="uq_stg_matches_row"),
    )
    op.create_index("ix_stg_matches_season", "stg_matches", ["competition", "season"])


def downgrade() -> None:
    op.drop_index("ix_stg_matches_season", table_name="stg_matches")
    op.drop_table("stg_matches")
    op.drop_index("ix_raw_payloads_lookup", table_name="raw_payloads")
    op.drop_table("raw_payloads")
