"""Staging rows to clean matches (spec section 6.2).

One staging row is one fixture: the teams are resolved to canonical names, the local kickoff is
converted to UTC through the competition's timezone (daylight saving included), the result
becomes an explicit status, and every configured bookmaker contributes its 1X2 prices with the
de-margined probabilities already attached. Nothing here touches the database.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from enum import StrEnum

from predictor.config import Competition, Settings, SourceColumns, get_settings
from predictor.processing.entities import TeamResolver
from predictor.processing.market import MarketProbabilities, proportional, shin

#: The layouts football-data.co.uk has used for its date column, in order.
DATE_FORMATS = ("%d/%m/%Y", "%d/%m/%y")
TIME_FORMATS = ("%H:%M", "%H:%M:%S")

#: Full-time results the source publishes.
RESULTS = frozenset({"H", "D", "A"})

SIDES = ("home", "away")


class MatchStatus(StrEnum):
    """What happened to a fixture (spec section 6.2). Training uses `played` rows only."""

    PLAYED = "played"
    POSTPONED = "postponed"
    ABANDONED = "abandoned"
    AWARDED = "awarded"


class RowError(ValueError):
    """A staging row cannot be cleaned: the message names the column and the value."""


@dataclass(frozen=True, slots=True)
class Kickoff:
    """When a match started, in UTC and in the competition's own local date."""

    utc: datetime
    local_date: date
    #: True when the source had no time and the configured default was used instead.
    time_estimated: bool


@dataclass(frozen=True, slots=True)
class CleanOdds:
    """One bookmaker's 1X2 quote for one match, with the margin removed two ways."""

    bookmaker: str
    market: str
    is_closing: bool
    price_home: float
    price_draw: float
    price_away: float
    overround: float | None = None
    p_home_proportional: float | None = None
    p_draw_proportional: float | None = None
    p_away_proportional: float | None = None
    p_home_shin: float | None = None
    p_draw_shin: float | None = None
    p_away_shin: float | None = None

    @classmethod
    def from_prices(
        cls,
        bookmaker: str,
        market: str,
        is_closing: bool,
        prices: tuple[float, float, float],
    ) -> CleanOdds:
        """Build the row, leaving the probabilities empty when the prices are not quotable.

        An impossible price is not dropped here: validation rejects the row and says why.
        """
        try:
            straight: MarketProbabilities | None = proportional(prices)
            insider: MarketProbabilities | None = shin(prices)
        except ValueError:
            straight = insider = None
        return cls(
            bookmaker=bookmaker,
            market=market,
            is_closing=is_closing,
            price_home=prices[0],
            price_draw=prices[1],
            price_away=prices[2],
            overround=straight.overround if straight else None,
            p_home_proportional=straight.home if straight else None,
            p_draw_proportional=straight.draw if straight else None,
            p_away_proportional=straight.away if straight else None,
            p_home_shin=insider.home if insider else None,
            p_draw_shin=insider.draw if insider else None,
            p_away_shin=insider.away if insider else None,
        )


@dataclass(frozen=True, slots=True)
class CleanTeamStats:
    """One team's match statistics. `xg` stays empty: this source does not publish it."""

    side: str
    shots: int | None
    shots_on_target: int | None
    corners: int | None
    fouls: int | None
    yellows: int | None
    reds: int | None
    xg: float | None = None


@dataclass(frozen=True, slots=True)
class CleanMatch:
    """One fixture, ready for validation and for the clean tables."""

    competition: str
    season: str
    kickoff: Kickoff
    home: str
    away: str
    home_goals: int | None
    away_goals: int | None
    ht_home_goals: int | None
    ht_away_goals: int | None
    result: str | None
    status: MatchStatus
    source: str
    stats: tuple[CleanTeamStats, ...] = ()
    odds: tuple[CleanOdds, ...] = ()
    #: Provenance: the staging row this match was built from.
    stg_match_id: int | None = field(default=None)


def _text(row: Mapping[str, str], column: str) -> str:
    return (row.get(column) or "").strip()


def _integer(row: Mapping[str, str], column: str) -> int | None:
    value = _text(row, column)
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError as exc:
        msg = f"column {column!r} is not a number: {value!r}"
        raise RowError(msg) from exc


def _decimal(row: Mapping[str, str], column: str) -> float | None:
    value = _text(row, column)
    if not value:
        return None
    try:
        return float(value)
    except ValueError as exc:
        msg = f"column {column!r} is not a number: {value!r}"
        raise RowError(msg) from exc


def _parse_date(text: str) -> date:
    for layout in DATE_FORMATS:
        try:
            return datetime.strptime(text, layout).date()
        except ValueError:
            continue
    tried = ", ".join(DATE_FORMATS)
    msg = f"unparseable date {text!r}; expected one of {tried}"
    raise RowError(msg)


def _parse_time(text: str) -> time:
    for layout in TIME_FORMATS:
        try:
            return datetime.strptime(text, layout).time()
        except ValueError:
            continue
    tried = ", ".join(TIME_FORMATS)
    msg = f"unparseable time {text!r}; expected one of {tried}"
    raise RowError(msg)


def parse_kickoff(
    date_text: str, time_text: str, competition: Competition, default_time: time
) -> Kickoff:
    """Combine the local date and time of a competition into an instant in UTC."""
    local_date = _parse_date(date_text.strip())
    estimated = not time_text.strip()
    local_time = default_time if estimated else _parse_time(time_text.strip())
    local = datetime.combine(local_date, local_time, tzinfo=competition.tzinfo())
    return Kickoff(utc=local.astimezone(UTC), local_date=local_date, time_estimated=estimated)


class RowCleaner:
    """Turns the staging rows of one competition into `CleanMatch` objects."""

    def __init__(
        self,
        competition: Competition,
        columns: SourceColumns,
        resolver: TeamResolver,
        default_kickoff_local_time: time,
        status_when_result_missing: MatchStatus,
        market: str,
        source: str,
    ) -> None:
        self.competition = competition
        self.columns = columns
        self.resolver = resolver
        self.default_kickoff_local_time = default_kickoff_local_time
        self.status_when_result_missing = status_when_result_missing
        self.market = market
        self.source = source

    @classmethod
    def from_settings(
        cls,
        competition: Competition,
        settings: Settings | None = None,
        resolver: TeamResolver | None = None,
    ) -> RowCleaner:
        """Build the cleaner from the configuration: no league or column name lives in code."""
        resolved = settings or get_settings()
        processing = resolved.processing
        return cls(
            competition=competition,
            columns=processing.source_columns(processing.source),
            resolver=resolver or TeamResolver.from_file(),
            default_kickoff_local_time=(
                competition.default_kickoff_local_time or processing.default_kickoff_local_time
            ),
            status_when_result_missing=MatchStatus(processing.status_when_result_missing),
            market=processing.market,
            source=processing.source,
        )

    def clean(
        self, season: str, row: Mapping[str, str], stg_match_id: int | None = None
    ) -> CleanMatch:
        """Clean one staging row. Raises `RowError` or `UnknownTeamError` on bad input."""
        columns = self.columns
        goals = columns.goals

        kickoff = parse_kickoff(
            _text(row, columns.date),
            _text(row, columns.time),
            self.competition,
            self.default_kickoff_local_time,
        )
        home = self.resolver.resolve(self.competition.code, _text(row, columns.home))
        away = self.resolver.resolve(self.competition.code, _text(row, columns.away))

        home_goals = _integer(row, goals.home)
        away_goals = _integer(row, goals.away)
        result = _text(row, goals.result).upper() or None
        if result is not None and result not in RESULTS:
            msg = f"column {goals.result!r} has an unknown result {result!r}"
            raise RowError(msg)

        played = home_goals is not None and away_goals is not None and result is not None
        status = MatchStatus.PLAYED if played else self.status_when_result_missing

        return CleanMatch(
            competition=self.competition.code,
            season=season,
            kickoff=kickoff,
            home=home,
            away=away,
            home_goals=home_goals,
            away_goals=away_goals,
            ht_home_goals=_integer(row, goals.ht_home),
            ht_away_goals=_integer(row, goals.ht_away),
            result=result,
            status=status,
            source=self.source,
            stats=self._stats(row),
            odds=self._odds(row),
            stg_match_id=stg_match_id,
        )

    def _stats(self, row: Mapping[str, str]) -> tuple[CleanTeamStats, ...]:
        stats = self.columns.stats
        return tuple(
            CleanTeamStats(
                side=side,
                shots=_integer(row, stats.shots[index]),
                shots_on_target=_integer(row, stats.shots_on_target[index]),
                corners=_integer(row, stats.corners[index]),
                fouls=_integer(row, stats.fouls[index]),
                yellows=_integer(row, stats.yellows[index]),
                reds=_integer(row, stats.reds[index]),
            )
            for index, side in enumerate(SIDES)
        )

    def _odds(self, row: Mapping[str, str]) -> tuple[CleanOdds, ...]:
        quotes: list[CleanOdds] = []
        for bookmaker in self.columns.bookmakers:
            for prefix, is_closing in ((bookmaker.open, False), (bookmaker.closing, True)):
                if prefix is None:
                    continue
                prices = tuple(
                    _decimal(row, f"{prefix}{suffix}") for suffix in self.columns.outcome_suffixes
                )
                if any(price is None for price in prices):
                    continue  # the season file does not carry this bookmaker
                quotes.append(
                    CleanOdds.from_prices(
                        bookmaker.code,
                        self.market,
                        is_closing,
                        (prices[0], prices[1], prices[2]),  # type: ignore[arg-type]
                    )
                )
        return tuple(quotes)
