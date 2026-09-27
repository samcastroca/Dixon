"""Database tables. Every change here needs an Alembic migration.

Kept apart from `predictor.models`, which the spec reserves for the MatchModel implementations.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
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
from predictor.processing.clean import SIDES, MatchStatus


def _in_clause(column: str, values: tuple[str, ...]) -> str:
    quoted = ", ".join(f"'{value}'" for value in values)
    return f"{column} IN ({quoted})"


#: Allowed values, taken from the domain enums so the database cannot drift from the code.
STATUS_VALUES = tuple(status.value for status in MatchStatus)
STATUS_CHECK = _in_clause("status", STATUS_VALUES)
SIDE_CHECK = _in_clause("side", SIDES)


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


class Competition(Base):
    """One league, mirrored from `settings.competitions` so the data model can reference it."""

    __tablename__ = "competitions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(32), unique=True)
    sport: Mapped[str] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(String(128))
    country: Mapped[str] = mapped_column(String(64))
    timezone: Mapped[str] = mapped_column(String(64))


class Season(Base):
    """One season of one competition (spec section 5: a season belongs to one competition)."""

    __tablename__ = "seasons"
    __table_args__ = (UniqueConstraint("competition_id", "label", name="uq_seasons_label"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    competition_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("competitions.id", ondelete="CASCADE")
    )
    label: Mapped[str] = mapped_column(String(16))
    start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)


class Team(Base):
    """A club, under its canonical name. A team belongs to the competition it plays in."""

    __tablename__ = "teams"
    __table_args__ = (
        UniqueConstraint("competition_id", "normalised_key", name="uq_teams_key"),
        UniqueConstraint("competition_id", "canonical_name", name="uq_teams_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    competition_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("competitions.id", ondelete="CASCADE")
    )
    canonical_name: Mapped[str] = mapped_column(String(128))
    normalised_key: Mapped[str] = mapped_column(String(128))


class TeamAlias(Base):
    """Every spelling a source may use, resolved to one team within its competition."""

    __tablename__ = "team_aliases"
    __table_args__ = (UniqueConstraint("competition_id", "normalised_key", name="uq_alias_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    competition_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("competitions.id", ondelete="CASCADE")
    )
    team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id", ondelete="CASCADE"))
    alias: Mapped[str] = mapped_column(String(128))
    normalised_key: Mapped[str] = mapped_column(String(128))
    #: Null means the alias holds for every source.
    source: Mapped[str | None] = mapped_column(String(64), nullable=True)


class Match(Base):
    """One fixture. A league pair meets once per season at home, which is the natural key."""

    __tablename__ = "matches"
    __table_args__ = (
        UniqueConstraint("season_id", "home_team_id", "away_team_id", name="uq_matches_fixture"),
        CheckConstraint("home_team_id <> away_team_id", name="ck_matches_distinct_teams"),
        CheckConstraint(
            "(home_goals IS NULL OR home_goals >= 0) AND (away_goals IS NULL OR away_goals >= 0)",
            name="ck_matches_goals_non_negative",
        ),
        CheckConstraint(STATUS_CHECK, name="ck_matches_status"),
        Index("ix_matches_schedule", "competition_id", "kickoff_utc"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    season_id: Mapped[int] = mapped_column(Integer, ForeignKey("seasons.id", ondelete="CASCADE"))
    competition_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("competitions.id", ondelete="CASCADE")
    )
    kickoff_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    kickoff_local_date: Mapped[date] = mapped_column(Date)
    #: True when the source had no kickoff time and the configured default was used.
    kickoff_time_estimated: Mapped[bool] = mapped_column(Boolean)
    home_team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id", ondelete="RESTRICT"))
    away_team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id", ondelete="RESTRICT"))
    home_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ht_home_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ht_away_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    result: Mapped[str | None] = mapped_column(String(1), nullable=True)
    status: Mapped[str] = mapped_column(String(16))
    source: Mapped[str] = mapped_column(String(64))
    #: Provenance: the staging row this match was cleaned from.
    stg_match_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("stg_matches.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class MatchStat(Base):
    """One team's statistics in one match. Only ever used as a lagged input (spec section 5)."""

    __tablename__ = "match_stats"
    __table_args__ = (
        UniqueConstraint("match_id", "team_id", name="uq_match_stats_team"),
        CheckConstraint(SIDE_CHECK, name="ck_match_stats_side"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("matches.id", ondelete="CASCADE"))
    team_id: Mapped[int] = mapped_column(Integer, ForeignKey("teams.id", ondelete="RESTRICT"))
    side: Mapped[str] = mapped_column(String(4))
    shots: Mapped[int | None] = mapped_column(Integer, nullable=True)
    shots_on_target: Mapped[int | None] = mapped_column(Integer, nullable=True)
    corners: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fouls: Mapped[int | None] = mapped_column(Integer, nullable=True)
    yellows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: This source publishes no expected goals; later sources may.
    xg: Mapped[float | None] = mapped_column(Float, nullable=True)


class Odds(Base):
    """One bookmaker's quote for one match, with the margin already removed two ways."""

    __tablename__ = "odds"
    __table_args__ = (
        UniqueConstraint("match_id", "bookmaker", "market", "is_closing", name="uq_odds_quote"),
        CheckConstraint(
            "price_home > 1.0 AND price_draw > 1.0 AND price_away > 1.0",
            name="ck_odds_prices_above_evens",
        ),
        Index("ix_odds_match", "match_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("matches.id", ondelete="CASCADE"))
    bookmaker: Mapped[str] = mapped_column(String(16))
    market: Mapped[str] = mapped_column(String(16))
    #: Set when a source timestamps its quotes; this one only separates opening from closing.
    captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_closing: Mapped[bool] = mapped_column(Boolean)
    price_home: Mapped[float] = mapped_column(Float)
    price_draw: Mapped[float] = mapped_column(Float)
    price_away: Mapped[float] = mapped_column(Float)
    overround: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_home_proportional: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_draw_proportional: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_away_proportional: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_home_shin: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_draw_shin: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_away_shin: Mapped[float | None] = mapped_column(Float, nullable=True)


class Feature(Base):
    """One point-in-time feature row per match and feature version (spec section 5).

    The values live in a single JSONB document so that adding features in a later phase is a
    new `feature_version`, not a migration. `as_of_utc < kickoff_utc` spans two tables, so it
    cannot be a CHECK here: the builder guarantees it, the pandera schema checks it on the way
    in and an integration test asserts it against the stored rows.

    The column is `feature_values` and not `values` because VALUES is a reserved keyword in
    PostgreSQL and would need quoting in every hand-written query.
    """

    __tablename__ = "features"
    __table_args__ = (Index("ix_features_version", "feature_version", "as_of_utc"),)

    match_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("matches.id", ondelete="CASCADE"), primary_key=True
    )
    feature_version: Mapped[str] = mapped_column(String(16), primary_key=True)
    as_of_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    definition_checksum: Mapped[str] = mapped_column(String(64))
    feature_values: Mapped[dict[str, Any]] = mapped_column(JSONB)


#: The slices a backtest is scored at, mirrored from the evaluation module's scopes.
BACKTEST_SCOPES = ("pooled", "competition", "season")
BACKTEST_SCOPE_CHECK = _in_clause("scope", BACKTEST_SCOPES)


class BacktestResult(Base):
    """One scored slice of one walk-forward run, mirroring MLflow (spec section 5).

    A slice is pooled, one competition, or one competition-season, so `competition_id` and
    `season_id` are null for the coarser scopes. Postgres would treat those nulls as distinct
    and happily store the same pooled row twice, which is why the unique constraint is declared
    NULLS NOT DISTINCT: re-mirroring a run has to be idempotent.
    """

    __tablename__ = "backtest_results"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "model_name",
            "scope",
            "competition_id",
            "season_id",
            name="uq_backtest_results_slice",
            postgresql_nulls_not_distinct=True,
        ),
        CheckConstraint(BACKTEST_SCOPE_CHECK, name="ck_backtest_results_scope"),
        CheckConstraint("n >= 0", name="ck_backtest_results_sample"),
        Index("ix_backtest_results_model", "model_name", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    #: The MLflow run the numbers came from, so a row can always be traced back.
    run_id: Mapped[str] = mapped_column(String(64))
    model_name: Mapped[str] = mapped_column(String(64))
    scope: Mapped[str] = mapped_column(String(16))
    competition_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("competitions.id", ondelete="CASCADE"), nullable=True
    )
    season_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("seasons.id", ondelete="CASCADE"), nullable=True
    )
    feature_version: Mapped[str] = mapped_column(String(16))
    git_sha: Mapped[str | None] = mapped_column(String(40), nullable=True)
    n: Mapped[int] = mapped_column(Integer)
    rps: Mapped[float] = mapped_column(Float)
    log_loss: Mapped[float] = mapped_column(Float)
    brier: Mapped[float] = mapped_column(Float)
    ece: Mapped[float] = mapped_column(Float)
    #: The model this slice's RPS gap is measured against, and the paired bootstrap interval.
    reference_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    rps_gap: Mapped[float | None] = mapped_column(Float, nullable=True)
    rps_gap_ci_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    rps_gap_ci_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
