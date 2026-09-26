"""Idempotent writes for the clean tables: re-processing updates rows, it never duplicates them.

Every statement is an upsert on the table's natural key, so `predictor process` can be run as
often as wanted and leaves the database in the same state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.dialects.postgresql import Insert, insert
from sqlalchemy.orm import Session

from predictor import tables
from predictor.config import Competition
from predictor.processing.entities import TeamResolver, normalise_key

#: Columns an existing match row takes from a re-processed payload.
_MATCH_UPDATES = (
    "kickoff_utc",
    "kickoff_local_date",
    "kickoff_time_estimated",
    "home_goals",
    "away_goals",
    "ht_home_goals",
    "ht_away_goals",
    "result",
    "status",
    "source",
    "stg_match_id",
)

_STAT_UPDATES = ("side", "shots", "shots_on_target", "corners", "fouls", "yellows", "reds", "xg")

_ODDS_UPDATES = (
    "captured_at",
    "price_home",
    "price_draw",
    "price_away",
    "overround",
    "p_home_proportional",
    "p_draw_proportional",
    "p_away_proportional",
    "p_home_shin",
    "p_draw_shin",
    "p_away_shin",
)


def _excluded(statement: Insert, columns: Sequence[str]) -> dict[str, Any]:
    return {column: statement.excluded[column] for column in columns}


def sync_competitions(session: Session, competitions: Sequence[Competition]) -> dict[str, int]:
    """Mirror the configured competitions into the database. Returns code -> id."""
    values = [
        {
            "code": competition.code,
            "sport": competition.sport,
            "name": competition.name,
            "country": competition.country,
            "timezone": competition.timezone,
        }
        for competition in competitions
    ]
    if values:
        statement = insert(tables.Competition).values(values)
        session.execute(
            statement.on_conflict_do_update(
                index_elements=["code"],
                set_=_excluded(statement, ("sport", "name", "country", "timezone")),
            )
        )
    codes = [competition.code for competition in competitions]
    rows = session.execute(
        select(tables.Competition.code, tables.Competition.id).where(
            tables.Competition.code.in_(codes)
        )
    ).all()
    return dict(rows)


def sync_teams(
    session: Session, competition_id: int, competition_code: str, resolver: TeamResolver
) -> dict[str, int]:
    """Write the canonical teams and their aliases. Returns canonical name -> team id."""
    names = resolver.canonical_names(competition_code)
    team_values = [
        {
            "competition_id": competition_id,
            "canonical_name": name,
            "normalised_key": normalise_key(name),
        }
        for name in names
    ]
    if team_values:
        statement = insert(tables.Team).values(team_values)
        session.execute(
            statement.on_conflict_do_update(
                constraint="uq_teams_key", set_=_excluded(statement, ("canonical_name",))
            )
        )

    rows = session.execute(
        select(tables.Team.canonical_name, tables.Team.id).where(
            tables.Team.competition_id == competition_id
        )
    ).all()
    team_ids = dict(rows)

    alias_values = [
        {
            "competition_id": competition_id,
            "team_id": team_ids[canonical],
            "alias": alias,
            "normalised_key": key,
            "source": None,
        }
        for alias, key, canonical in resolver.alias_rows(competition_code)
        if canonical in team_ids
    ]
    if alias_values:
        statement = insert(tables.TeamAlias).values(alias_values)
        session.execute(
            statement.on_conflict_do_update(
                constraint="uq_alias_key", set_=_excluded(statement, ("team_id", "alias", "source"))
            )
        )
    return team_ids


def upsert_season(
    session: Session,
    competition_id: int,
    label: str,
    start_date: date | None,
    end_date: date | None,
) -> int:
    """Create or refresh one season and return its id."""
    statement = insert(tables.Season).values(
        competition_id=competition_id, label=label, start_date=start_date, end_date=end_date
    )
    session.execute(
        statement.on_conflict_do_update(
            constraint="uq_seasons_label", set_=_excluded(statement, ("start_date", "end_date"))
        )
    )
    return session.execute(
        select(tables.Season.id).where(
            tables.Season.competition_id == competition_id, tables.Season.label == label
        )
    ).scalar_one()


def upsert_matches(session: Session, values: Sequence[Mapping[str, Any]]) -> int:
    """Insert or refresh the fixtures of one season."""
    if not values:
        return 0
    statement = insert(tables.Match).values(list(values))
    # Only rewrite a fixture that actually changed, so `updated_at` stays a real modification
    # time and re-processing the same payload leaves every row byte for byte as it was.
    changed = or_(
        *[
            getattr(tables.Match, column).is_distinct_from(statement.excluded[column])
            for column in _MATCH_UPDATES
        ]
    )
    session.execute(
        statement.on_conflict_do_update(
            constraint="uq_matches_fixture",
            set_={**_excluded(statement, _MATCH_UPDATES), "updated_at": func.now()},
            where=changed,
        )
    )
    return len(values)


def match_ids(session: Session, season_id: int) -> dict[tuple[int, int], int]:
    """(home team id, away team id) -> match id, for one season."""
    rows = session.execute(
        select(tables.Match.home_team_id, tables.Match.away_team_id, tables.Match.id).where(
            tables.Match.season_id == season_id
        )
    ).all()
    return {(home, away): identifier for home, away, identifier in rows}


def upsert_match_stats(session: Session, values: Sequence[Mapping[str, Any]]) -> int:
    if not values:
        return 0
    statement = insert(tables.MatchStat).values(list(values))
    session.execute(
        statement.on_conflict_do_update(
            constraint="uq_match_stats_team", set_=_excluded(statement, _STAT_UPDATES)
        )
    )
    return len(values)


def upsert_odds(session: Session, values: Sequence[Mapping[str, Any]]) -> int:
    if not values:
        return 0
    statement = insert(tables.Odds).values(list(values))
    session.execute(
        statement.on_conflict_do_update(
            constraint="uq_odds_quote", set_=_excluded(statement, _ODDS_UPDATES)
        )
    )
    return len(values)


def latest_staging_rows(
    session: Session, source: str, competition_code: str, season: str
) -> list[tuple[int, dict[str, str]]]:
    """Staging rows of the most recent payload for one season: raw payloads are append-only."""
    payload_id = session.execute(
        select(tables.StgMatch.raw_payload_id)
        .join(tables.RawPayload, tables.RawPayload.id == tables.StgMatch.raw_payload_id)
        .where(
            tables.RawPayload.source == source,
            tables.StgMatch.competition == competition_code,
            tables.StgMatch.season == season,
        )
        .order_by(tables.RawPayload.fetched_at.desc(), tables.StgMatch.raw_payload_id.desc())
        .limit(1)
    ).scalar()
    if payload_id is None:
        return []

    rows = session.execute(
        select(tables.StgMatch.id, tables.StgMatch.row)
        .where(tables.StgMatch.raw_payload_id == payload_id)
        .order_by(tables.StgMatch.row_index)
    ).all()
    return [(identifier, row) for identifier, row in rows]


def staged_seasons(session: Session, source: str, competition_code: str) -> tuple[str, ...]:
    """Every season of one competition that ingestion has staged, oldest first."""
    rows = session.execute(
        select(tables.StgMatch.season)
        .join(tables.RawPayload, tables.RawPayload.id == tables.StgMatch.raw_payload_id)
        .where(
            tables.RawPayload.source == source,
            tables.StgMatch.competition == competition_code,
        )
        .distinct()
        .order_by(tables.StgMatch.season)
    ).scalars()
    return tuple(rows)
