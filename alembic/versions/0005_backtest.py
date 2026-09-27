"""backtest: walk-forward metrics mirrored from MLflow

Revision ID: 0005_backtest
Revises: 0004_features
Create Date: 2026-09-26

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0005_backtest"
down_revision: str | None = "0004_features"
branch_labels: None = None
depends_on: None = None


def upgrade() -> None:
    op.create_table(
        "backtest_results",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("model_name", sa.String(length=64), nullable=False),
        sa.Column("scope", sa.String(length=16), nullable=False),
        sa.Column("competition_id", sa.Integer(), nullable=True),
        sa.Column("season_id", sa.Integer(), nullable=True),
        sa.Column("feature_version", sa.String(length=16), nullable=False),
        sa.Column("git_sha", sa.String(length=40), nullable=True),
        sa.Column("n", sa.Integer(), nullable=False),
        sa.Column("rps", sa.Float(), nullable=False),
        sa.Column("log_loss", sa.Float(), nullable=False),
        sa.Column("brier", sa.Float(), nullable=False),
        sa.Column("ece", sa.Float(), nullable=False),
        sa.Column("reference_model", sa.String(length=64), nullable=True),
        sa.Column("rps_gap", sa.Float(), nullable=True),
        sa.Column("rps_gap_ci_low", sa.Float(), nullable=True),
        sa.Column("rps_gap_ci_high", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "scope IN ('pooled', 'competition', 'season')", name="ck_backtest_results_scope"
        ),
        sa.CheckConstraint("n >= 0", name="ck_backtest_results_sample"),
        sa.ForeignKeyConstraint(["competition_id"], ["competitions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["season_id"], ["seasons.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "run_id",
            "model_name",
            "scope",
            "competition_id",
            "season_id",
            name="uq_backtest_results_slice",
            postgresql_nulls_not_distinct=True,
        ),
    )
    op.create_index("ix_backtest_results_model", "backtest_results", ["model_name", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_backtest_results_model", table_name="backtest_results")
    op.drop_table("backtest_results")
