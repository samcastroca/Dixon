"""Orchestration of phase 2: staging rows in, clean and validated tables out.

One season is one unit of work: its rows are cleaned, the three frames are validated together,
and only then is anything written. Every write is an upsert, so running the command twice leaves
the database exactly as it was.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from predictor.config import Competition, Settings, get_settings
from predictor.db import session_scope
from predictor.logging import get_logger
from predictor.processing import repository
from predictor.processing.clean import CleanMatch, RowCleaner
from predictor.processing.entities import TeamResolver
from predictor.processing.validate import validate

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ProcessEntry:
    """What one season of one competition contributed."""

    competition: str
    season: str
    matches: int
    match_stats: int
    odds: int


@dataclass(frozen=True, slots=True)
class ProcessReport:
    entries: tuple[ProcessEntry, ...]
    #: Competition code -> canonical team names, for the demo output.
    teams: Mapping[str, tuple[str, ...]]

    @property
    def matches(self) -> int:
        return sum(entry.matches for entry in self.entries)

    @property
    def match_stats(self) -> int:
        return sum(entry.match_stats for entry in self.entries)

    @property
    def odds(self) -> int:
        return sum(entry.odds for entry in self.entries)


def _match_values(
    match: CleanMatch, season_id: int, competition_id: int, team_ids: Mapping[str, int]
) -> dict[str, Any]:
    return {
        "season_id": season_id,
        "competition_id": competition_id,
        "kickoff_utc": match.kickoff.utc,
        "kickoff_local_date": match.kickoff.local_date,
        "kickoff_time_estimated": match.kickoff.time_estimated,
        "home_team_id": team_ids[match.home],
        "away_team_id": team_ids[match.away],
        "home_goals": match.home_goals,
        "away_goals": match.away_goals,
        "ht_home_goals": match.ht_home_goals,
        "ht_away_goals": match.ht_away_goals,
        "result": match.result,
        "status": match.status.value,
        "source": match.source,
        "stg_match_id": match.stg_match_id,
    }


def _stats_values(
    matches: Sequence[CleanMatch],
    team_ids: Mapping[str, int],
    ids: Mapping[tuple[int, int], int],
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for match in matches:
        match_id = ids[(team_ids[match.home], team_ids[match.away])]
        for stats in match.stats:
            team = match.home if stats.side == "home" else match.away
            values.append(
                {
                    "match_id": match_id,
                    "team_id": team_ids[team],
                    "side": stats.side,
                    "shots": stats.shots,
                    "shots_on_target": stats.shots_on_target,
                    "corners": stats.corners,
                    "fouls": stats.fouls,
                    "yellows": stats.yellows,
                    "reds": stats.reds,
                    "xg": stats.xg,
                }
            )
    return values


def _odds_values(
    matches: Sequence[CleanMatch],
    team_ids: Mapping[str, int],
    ids: Mapping[tuple[int, int], int],
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for match in matches:
        match_id = ids[(team_ids[match.home], team_ids[match.away])]
        for odds in match.odds:
            values.append(
                {
                    "match_id": match_id,
                    "bookmaker": odds.bookmaker,
                    "market": odds.market,
                    "captured_at": None,
                    "is_closing": odds.is_closing,
                    "price_home": odds.price_home,
                    "price_draw": odds.price_draw,
                    "price_away": odds.price_away,
                    "overround": odds.overround,
                    "p_home_proportional": odds.p_home_proportional,
                    "p_draw_proportional": odds.p_draw_proportional,
                    "p_away_proportional": odds.p_away_proportional,
                    "p_home_shin": odds.p_home_shin,
                    "p_draw_shin": odds.p_draw_shin,
                    "p_away_shin": odds.p_away_shin,
                }
            )
    return values


def process(
    competitions: Sequence[Competition],
    seasons: Sequence[str] | None = None,
    settings: Settings | None = None,
    resolver: TeamResolver | None = None,
) -> ProcessReport:
    """Clean, validate and store every staged season of the given competitions."""
    resolved = settings or get_settings()
    source = resolved.processing.source
    teams = resolver or TeamResolver.from_file()

    with session_scope() as session:
        competition_ids = repository.sync_competitions(session, competitions)
        team_ids = {
            competition.code: repository.sync_teams(
                session, competition_ids[competition.code], competition.code, teams
            )
            for competition in competitions
        }

    entries: list[ProcessEntry] = []
    for competition in competitions:
        cleaner = RowCleaner.from_settings(competition, resolved, teams)
        with session_scope() as session:
            available = repository.staged_seasons(session, source, competition.code)
        wanted = tuple(seasons) if seasons else available
        for season in wanted:
            if season not in available:
                logger.warning(
                    "process.season_not_staged", competition=competition.code, season=season
                )
                continue
            entries.append(
                _process_season(
                    competition=competition,
                    competition_id=competition_ids[competition.code],
                    season=season,
                    cleaner=cleaner,
                    team_ids=team_ids[competition.code],
                    source=source,
                )
            )

    return ProcessReport(
        entries=tuple(entries),
        teams={
            competition.code: teams.canonical_names(competition.code)
            for competition in competitions
        },
    )


def _process_season(
    competition: Competition,
    competition_id: int,
    season: str,
    cleaner: RowCleaner,
    team_ids: Mapping[str, int],
    source: str,
) -> ProcessEntry:
    with session_scope() as session:
        staged = repository.latest_staging_rows(session, source, competition.code, season)
        matches = [cleaner.clean(season, row, stg_match_id) for stg_match_id, row in staged]

        validate(matches)

        dates = [match.kickoff.local_date for match in matches]
        season_id = repository.upsert_season(
            session,
            competition_id,
            season,
            min(dates) if dates else None,
            max(dates) if dates else None,
        )
        repository.upsert_matches(
            session,
            [_match_values(match, season_id, competition_id, team_ids) for match in matches],
        )
        ids = repository.match_ids(session, season_id)
        stats = repository.upsert_match_stats(session, _stats_values(matches, team_ids, ids))
        odds = repository.upsert_odds(session, _odds_values(matches, team_ids, ids))

    logger.info(
        "process.season",
        competition=competition.code,
        season=season,
        matches=len(matches),
        match_stats=stats,
        odds=odds,
    )
    return ProcessEntry(
        competition=competition.code,
        season=season,
        matches=len(matches),
        match_stats=stats,
        odds=odds,
    )
