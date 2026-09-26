"""processing: competitions, seasons, teams, aliases, matches, stats and odds

Revision ID: 0003_processing
Revises: 0002_ingestion
Create Date: 2026-09-26

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0003_processing"
down_revision: str | None = "0002_ingestion"
branch_labels: None = None
depends_on: None = None


def upgrade() -> None:
    op.create_table(
        "competitions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("code", sa.String(length=32), nullable=False),
        sa.Column("sport", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("country", sa.String(length=64), nullable=False),
        sa.Column("timezone", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("code"),
    )

    op.create_table(
        "seasons",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("competition_id", sa.Integer(), nullable=False),
        sa.Column("label", sa.String(length=16), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=True),
        sa.Column("end_date", sa.Date(), nullable=True),
        sa.ForeignKeyConstraint(["competition_id"], ["competitions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("competition_id", "label", name="uq_seasons_label"),
    )

    op.create_table(
        "teams",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("competition_id", sa.Integer(), nullable=False),
        sa.Column("canonical_name", sa.String(length=128), nullable=False),
        sa.Column("normalised_key", sa.String(length=128), nullable=False),
        sa.ForeignKeyConstraint(["competition_id"], ["competitions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("competition_id", "canonical_name", name="uq_teams_name"),
        sa.UniqueConstraint("competition_id", "normalised_key", name="uq_teams_key"),
    )

    op.create_table(
        "team_aliases",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("competition_id", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("alias", sa.String(length=128), nullable=False),
        sa.Column("normalised_key", sa.String(length=128), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=True),
        sa.ForeignKeyConstraint(["competition_id"], ["competitions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["team_id"], ["teams.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("competition_id", "normalised_key", name="uq_alias_key"),
    )

    op.create_table(
        "matches",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("season_id", sa.Integer(), nullable=False),
        sa.Column("competition_id", sa.Integer(), nullable=False),
        sa.Column("kickoff_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("kickoff_local_date", sa.Date(), nullable=False),
        sa.Column("kickoff_time_estimated", sa.Boolean(), nullable=False),
        sa.Column("home_team_id", sa.Integer(), nullable=False),
        sa.Column("away_team_id", sa.Integer(), nullable=False),
        sa.Column("home_goals", sa.Integer(), nullable=True),
        sa.Column("away_goals", sa.Integer(), nullable=True),
        sa.Column("ht_home_goals", sa.Integer(), nullable=True),
        sa.Column("ht_away_goals", sa.Integer(), nullable=True),
        sa.Column("result", sa.String(length=1), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("stg_match_id", sa.BigInteger(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('played', 'postponed', 'abandoned', 'awarded')",
            name="ck_matches_status",
        ),
        sa.CheckConstraint(
            "(home_goals IS NULL OR home_goals >= 0) AND (away_goals IS NULL OR away_goals >= 0)",
            name="ck_matches_goals_non_negative",
        ),
        sa.CheckConstraint("home_team_id <> away_team_id", name="ck_matches_distinct_teams"),
        sa.ForeignKeyConstraint(["away_team_id"], ["teams.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["competition_id"], ["competitions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["home_team_id"], ["teams.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["season_id"], ["seasons.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["stg_match_id"], ["stg_matches.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("season_id", "home_team_id", "away_team_id", name="uq_matches_fixture"),
    )
    op.create_index("ix_matches_schedule", "matches", ["competition_id", "kickoff_utc"])

    op.create_table(
        "match_stats",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("match_id", sa.BigInteger(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("side", sa.String(length=4), nullable=False),
        sa.Column("shots", sa.Integer(), nullable=True),
        sa.Column("shots_on_target", sa.Integer(), nullable=True),
        sa.Column("corners", sa.Integer(), nullable=True),
        sa.Column("fouls", sa.Integer(), nullable=True),
        sa.Column("yellows", sa.Integer(), nullable=True),
        sa.Column("reds", sa.Integer(), nullable=True),
        sa.Column("xg", sa.Float(), nullable=True),
        sa.CheckConstraint("side IN ('home', 'away')", name="ck_match_stats_side"),
        sa.ForeignKeyConstraint(["match_id"], ["matches.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["team_id"], ["teams.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("match_id", "team_id", name="uq_match_stats_team"),
    )

    op.create_table(
        "odds",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("match_id", sa.BigInteger(), nullable=False),
        sa.Column("bookmaker", sa.String(length=16), nullable=False),
        sa.Column("market", sa.String(length=16), nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_closing", sa.Boolean(), nullable=False),
        sa.Column("price_home", sa.Float(), nullable=False),
        sa.Column("price_draw", sa.Float(), nullable=False),
        sa.Column("price_away", sa.Float(), nullable=False),
        sa.Column("overround", sa.Float(), nullable=True),
        sa.Column("p_home_proportional", sa.Float(), nullable=True),
        sa.Column("p_draw_proportional", sa.Float(), nullable=True),
        sa.Column("p_away_proportional", sa.Float(), nullable=True),
        sa.Column("p_home_shin", sa.Float(), nullable=True),
        sa.Column("p_draw_shin", sa.Float(), nullable=True),
        sa.Column("p_away_shin", sa.Float(), nullable=True),
        sa.CheckConstraint(
            "price_home > 1.0 AND price_draw > 1.0 AND price_away > 1.0",
            name="ck_odds_prices_above_evens",
        ),
        sa.ForeignKeyConstraint(["match_id"], ["matches.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("match_id", "bookmaker", "market", "is_closing", name="uq_odds_quote"),
    )
    op.create_index("ix_odds_match", "odds", ["match_id"])


def downgrade() -> None:
    op.drop_index("ix_odds_match", table_name="odds")
    op.drop_table("odds")
    op.drop_table("match_stats")
    op.drop_index("ix_matches_schedule", table_name="matches")
    op.drop_table("matches")
    op.drop_table("team_aliases")
    op.drop_table("teams")
    op.drop_table("seasons")
    op.drop_table("competitions")
