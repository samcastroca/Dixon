"""Processing against the compose database, over the seasons ingestion has already staged."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

import pytest
from sqlalchemy import func, select
from tests.support import RECORDED_SEASONS

from predictor import tables
from predictor.config import Competition, get_settings
from predictor.db import Base, session_scope
from predictor.processing.service import process

pytestmark = pytest.mark.integration

COMPETITIONS = ("EPL", "LALIGA")
MATCHES_PER_SEASON = 380
MATCHES_PER_TEAM = 38

#: The tables a second run must leave exactly as the first one did.
SNAPSHOT_TABLES = ("competitions", "seasons", "teams", "team_aliases", "matches", "match_stats")


@pytest.fixture(autouse=True)
def _needs_database(database_available: bool) -> Iterator[None]:
    if not database_available:
        pytest.skip("no database reachable; run `make up` or set DATABASE_URL")
    yield


@pytest.fixture
def competitions() -> list[Competition]:
    settings = get_settings()
    return [settings.competition(code) for code in COMPETITIONS]


def snapshot(table: str) -> str:
    """Hash of a whole table, read in a deterministic order."""
    model = Base.metadata.tables[table]
    with session_scope() as session:
        rows = session.execute(select(model).order_by(*model.primary_key.columns)).all()
    digest = hashlib.sha256()
    for row in rows:
        digest.update(repr(tuple(row)).encode("utf-8"))
    return digest.hexdigest()


@pytest.fixture(scope="module", autouse=True)
def processed(database_available: bool) -> None:
    """Process the recorded seasons once for the whole module."""
    if not database_available:
        pytest.skip("no database reachable; run `make up` or set DATABASE_URL")
    settings = get_settings()
    process([settings.competition(code) for code in COMPETITIONS], RECORDED_SEASONS)


def test_every_processed_season_holds_the_league_invariants() -> None:
    with session_scope() as session:
        rows = session.execute(
            select(
                tables.Competition.code,
                tables.Season.label,
                func.count(tables.Match.id),
                func.count(func.distinct(tables.Match.home_team_id)),
                func.count(func.distinct(tables.Match.away_team_id)),
            )
            .select_from(tables.Match)
            .join(tables.Season, tables.Season.id == tables.Match.season_id)
            .join(tables.Competition, tables.Competition.id == tables.Match.competition_id)
            .group_by(tables.Competition.code, tables.Season.label)
        ).all()

    assert rows, "no season has been processed"
    for code, season, matches, home_teams, away_teams in rows:
        teams = get_settings().competition(code).rules.teams
        assert matches == MATCHES_PER_SEASON, f"{code} {season}"
        assert home_teams == teams, f"{code} {season}"
        assert away_teams == teams, f"{code} {season}"


def test_every_team_plays_thirty_eight_matches_in_a_complete_season() -> None:
    with session_scope() as session:
        appearances = session.execute(
            select(
                tables.Match.season_id,
                tables.Match.home_team_id.label("team_id"),
            ).union_all(select(tables.Match.season_id, tables.Match.away_team_id.label("team_id")))
        ).all()

    played: dict[tuple[int, int], int] = {}
    for season_id, team_id in appearances:
        played[(season_id, team_id)] = played.get((season_id, team_id), 0) + 1

    assert played
    assert set(played.values()) == {MATCHES_PER_TEAM}


def test_no_match_links_teams_from_different_competitions() -> None:
    home = tables.Team.__table__.alias("home")
    away = tables.Team.__table__.alias("away")
    with session_scope() as session:
        mixed = session.scalar(
            select(func.count())
            .select_from(tables.Match)
            .join(home, home.c.id == tables.Match.home_team_id)
            .join(away, away.c.id == tables.Match.away_team_id)
            .where(
                (home.c.competition_id != tables.Match.competition_id)
                | (away.c.competition_id != tables.Match.competition_id)
            )
        )
    assert mixed == 0


def test_scores_and_teams_are_sane() -> None:
    with session_scope() as session:
        offending = session.scalar(
            select(func.count())
            .select_from(tables.Match)
            .where(
                (tables.Match.home_team_id == tables.Match.away_team_id)
                | (tables.Match.home_goals < 0)
                | (tables.Match.away_goals < 0)
            )
        )
        priced = session.scalar(
            select(func.count()).select_from(tables.Odds).where(tables.Odds.price_home <= 1.0)
        )
    assert offending == 0
    assert priced == 0


def test_every_odds_row_carries_both_de_margined_distributions() -> None:
    with session_scope() as session:
        total = session.scalar(select(func.count()).select_from(tables.Odds))
        complete = session.scalar(
            select(func.count())
            .select_from(tables.Odds)
            .where(
                func.abs(
                    tables.Odds.p_home_proportional
                    + tables.Odds.p_draw_proportional
                    + tables.Odds.p_away_proportional
                    - 1.0
                )
                < 1e-9,
                func.abs(
                    tables.Odds.p_home_shin
                    + tables.Odds.p_draw_shin
                    + tables.Odds.p_away_shin
                    - 1.0
                )
                < 1e-9,
            )
        )
    assert total and complete == total


def test_closing_odds_are_stored_next_to_the_opening_ones() -> None:
    with session_scope() as session:
        closing = session.scalar(
            select(func.count()).select_from(tables.Odds).where(tables.Odds.is_closing.is_(True))
        )
        opening = session.scalar(
            select(func.count()).select_from(tables.Odds).where(tables.Odds.is_closing.is_(False))
        )
    assert closing and opening


def test_processing_twice_changes_nothing(competitions: list[Competition]) -> None:
    before = {table: snapshot(table) for table in SNAPSHOT_TABLES}
    odds_before = snapshot("odds")

    process(competitions, RECORDED_SEASONS)

    assert {table: snapshot(table) for table in SNAPSHOT_TABLES} == before
    assert snapshot("odds") == odds_before
