"""Database access for the feature store: records in, feature rows out.

Reading happens once per competition and never again during a replay, so the leakage guard
has a closed set of records to reason about instead of a live cursor.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from predictor import tables
from predictor.config import Competition, Settings, get_settings
from predictor.features.builder import FeatureRow
from predictor.features.replay import MarketQuote, MatchRecord, TeamStats

#: How many rows go into one INSERT, keeping well under the driver's parameter ceiling.
CHUNK_SIZE = 1000

#: The columns a conflicting row may overwrite.
_FEATURE_UPDATES = ("as_of_utc", "computed_at", "definition_checksum", "feature_values")


def competition_id(session: Session, competition: Competition) -> int:
    """The stored id of a configured competition, failing loudly when it was never processed."""
    found = session.scalar(
        select(tables.Competition.id).where(tables.Competition.code == competition.code)
    )
    if found is None:
        msg = f"competition {competition.code} is not in the database; run `make process` first"
        raise LookupError(msg)
    return found


def load_records(
    session: Session, competition: Competition, settings: Settings | None = None
) -> list[MatchRecord]:
    """Every match of one competition, in kickoff order, with its lagged statistics."""
    resolved = settings or get_settings()
    identifier = competition_id(session, competition)

    stats = _team_statistics(session, identifier)
    market = (
        _market_quotes(session, identifier, resolved) if resolved.features.groups.market else {}
    )

    rows = session.execute(
        select(
            tables.Match.id,
            tables.Match.season_id,
            tables.Season.label,
            tables.Match.kickoff_utc,
            tables.Match.home_team_id,
            tables.Match.away_team_id,
            tables.Match.home_goals,
            tables.Match.away_goals,
            tables.Match.status,
        )
        .join(tables.Season, tables.Season.id == tables.Match.season_id)
        .where(tables.Match.competition_id == identifier)
        .order_by(tables.Match.kickoff_utc, tables.Match.id)
    ).all()

    missing = TeamStats(shots=None, shots_on_target=None)
    return [
        MatchRecord(
            match_id=match_id,
            competition_id=identifier,
            season_id=season_id,
            season_label=label,
            kickoff_utc=kickoff_utc,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_goals=home_goals,
            away_goals=away_goals,
            status=status,
            home_stats=stats.get((match_id, home_team_id), missing),
            away_stats=stats.get((match_id, away_team_id), missing),
            market=market.get(match_id),
        )
        for (
            match_id,
            season_id,
            label,
            kickoff_utc,
            home_team_id,
            away_team_id,
            home_goals,
            away_goals,
            status,
        ) in rows
    ]


def team_names(session: Session, competition: Competition) -> dict[int, str]:
    """Canonical names of a competition's teams, for readable demo output."""
    identifier = competition_id(session, competition)
    rows = session.execute(
        select(tables.Team.id, tables.Team.canonical_name).where(
            tables.Team.competition_id == identifier
        )
    ).all()
    return dict(rows)


def _team_statistics(session: Session, identifier: int) -> dict[tuple[int, int], TeamStats]:
    rows = session.execute(
        select(
            tables.MatchStat.match_id,
            tables.MatchStat.team_id,
            tables.MatchStat.shots,
            tables.MatchStat.shots_on_target,
        )
        .join(tables.Match, tables.Match.id == tables.MatchStat.match_id)
        .where(tables.Match.competition_id == identifier)
    ).all()
    return {
        (match_id, team_id): TeamStats(shots=shots, shots_on_target=shots_on_target)
        for match_id, team_id, shots, shots_on_target in rows
    }


def _market_quotes(session: Session, identifier: int, settings: Settings) -> dict[int, MarketQuote]:
    """The configured bookmaker's de-margined 1X2 distribution, one quote per match."""
    market = settings.features.market
    columns = (
        (tables.Odds.p_home_shin, tables.Odds.p_draw_shin, tables.Odds.p_away_shin)
        if market.method == "shin"
        else (
            tables.Odds.p_home_proportional,
            tables.Odds.p_draw_proportional,
            tables.Odds.p_away_proportional,
        )
    )
    statement = (
        select(tables.Odds.match_id, tables.Odds.captured_at, *columns)
        .join(tables.Match, tables.Match.id == tables.Odds.match_id)
        .where(
            tables.Match.competition_id == identifier,
            tables.Odds.bookmaker == market.bookmaker,
            tables.Odds.market == settings.processing.market,
            tables.Odds.is_closing.is_(market.prefer_closing),
        )
    )
    quotes: dict[int, MarketQuote] = {}
    for match_id, captured_at, p_home, p_draw, p_away in session.execute(statement).all():
        if p_home is None or p_draw is None or p_away is None:
            continue
        quotes[match_id] = MarketQuote(
            p_home=p_home, p_draw=p_draw, p_away=p_away, captured_at=captured_at
        )
    return quotes


def upsert_features(session: Session, rows: Iterable[FeatureRow]) -> int:
    """Store feature rows idempotently: an unchanged row is left exactly as it was."""
    values = [row.as_record() for row in rows]
    written = 0
    for chunk in _chunks(values, CHUNK_SIZE):
        statement = insert(tables.Feature).values(chunk)
        excluded = statement.excluded
        changed = or_(
            *(
                getattr(tables.Feature, column).is_distinct_from(excluded[column])
                for column in _FEATURE_UPDATES
                if column != "computed_at"
            )
        )
        session.execute(
            statement.on_conflict_do_update(
                index_elements=["match_id", "feature_version"],
                set_={column: excluded[column] for column in _FEATURE_UPDATES},
                # Without this, a rebuild that changes nothing would still bump computed_at.
                where=changed,
            )
        )
        written += len(chunk)
    return written


def _chunks(values: Sequence[dict[str, object]], size: int) -> list[Sequence[dict[str, object]]]:
    return [values[start : start + size] for start in range(0, len(values), size)]
