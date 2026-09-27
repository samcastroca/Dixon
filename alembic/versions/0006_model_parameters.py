"""model_parameters: what each walk-forward fit learned, per league and season

Revision ID: 0006_model_parameters
Revises: 0005_backtest
Create Date: 2026-09-27

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006_model_parameters"
down_revision: str | None = "0005_backtest"
branch_labels: None = None
depends_on: None = None


def upgrade() -> None:
    op.create_table(
        "model_parameters",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("model_name", sa.String(length=64), nullable=False),
        sa.Column("competition_id", sa.Integer(), nullable=False),
        sa.Column("season_id", sa.Integer(), nullable=True),
        sa.Column("fit_index", sa.Integer(), nullable=False),
        sa.Column("parameter_values", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("fit_index >= 0", name="ck_model_parameters_fit_index"),
        sa.ForeignKeyConstraint(["competition_id"], ["competitions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["season_id"], ["seasons.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "run_id",
            "model_name",
            "competition_id",
            "season_id",
            "fit_index",
            name="uq_model_parameters_fit",
            postgresql_nulls_not_distinct=True,
        ),
    )
    op.create_index("ix_model_parameters_model", "model_parameters", ["model_name", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_model_parameters_model", table_name="model_parameters")
    op.drop_table("model_parameters")
