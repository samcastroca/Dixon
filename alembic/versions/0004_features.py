"""features: the point-in-time feature store

Revision ID: 0004_features
Revises: 0003_processing
Create Date: 2026-09-26

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_features"
down_revision: str | None = "0003_processing"
branch_labels: None = None
depends_on: None = None


def upgrade() -> None:
    op.create_table(
        "features",
        sa.Column("match_id", sa.BigInteger(), nullable=False),
        sa.Column("feature_version", sa.String(length=16), nullable=False),
        sa.Column("as_of_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("definition_checksum", sa.String(length=64), nullable=False),
        sa.Column("feature_values", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(["match_id"], ["matches.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("match_id", "feature_version"),
    )
    op.create_index("ix_features_version", "features", ["feature_version", "as_of_utc"])


def downgrade() -> None:
    op.drop_index("ix_features_version", table_name="features")
    op.drop_table("features")
