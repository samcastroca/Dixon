"""Chronological replay and the guarded history view (spec section 6.3).

Every feature is computed by walking matches in kickoff order. The only way a feature can
reach historical data is a `HistoryView` bound to a cut-off: it refuses anything timestamped
at or after that instant, so a leak is an exception rather than a silent mistake.

Matches sharing a kickoff are processed as one batch. Features for the whole batch are
computed first and the ratings only move afterwards, so simultaneous matches cannot see each
other's results. That is the common case, not an edge case: the older season files carry no
kickoff time, so a whole Saturday lands on the same timestamp.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Protocol

from predictor.processing.clean import MatchStatus

#: What one accumulator contributes for one match. `None` means "not knowable yet".
FeatureValues = dict[str, float | None]

#: History scopes a team can be asked about: every match, or only those at that venue.
OVERALL = "overall"
HOME = "home"
AWAY = "away"
SCOPES = (OVERALL, HOME, AWAY)


class LeakageError(RuntimeError):
    """Raised when something asks the history for data at or after the cut-off."""


@dataclass(frozen=True, slots=True)
class TeamStats:
    """The per-team match statistics the feature store uses, always as a lagged input."""

    shots: int | None
    shots_on_target: int | None


@dataclass(frozen=True, slots=True)
class MarketQuote:
    """De-margined pre-match market probabilities, already computed in phase 2."""

    p_home: float
    p_draw: float
    p_away: float
    #: When the price was captured. `None` means the source never told us.
    captured_at: datetime | None


@dataclass(frozen=True, slots=True)
class MatchRecord:
    """One match as the replay engine sees it: the only data that enters a feature."""

    match_id: int
    competition_id: int
    season_id: int
    season_label: str
    kickoff_utc: datetime
    home_team_id: int
    away_team_id: int
    home_goals: int | None
    away_goals: int | None
    status: str
    home_stats: TeamStats
    away_stats: TeamStats
    market: MarketQuote | None = None

    @property
    def played(self) -> bool:
        """Whether a result is known, which is the only thing that may move a rating."""
        return (
            self.status == MatchStatus.PLAYED.value
            and self.home_goals is not None
            and self.away_goals is not None
        )

    def as_fixture(self) -> MatchRecord:
        """The same match with every result stripped, as the database held it before kickoff."""
        return replace(
            self,
            home_goals=None,
            away_goals=None,
            home_stats=TeamStats(shots=None, shots_on_target=None),
            away_stats=TeamStats(shots=None, shots_on_target=None),
        )

    def appearances(self) -> tuple[TeamAppearance, TeamAppearance]:
        """This match seen from each team's side. Only valid once the result is known."""
        if self.home_goals is None or self.away_goals is None:
            msg = f"match {self.match_id} has no result to orient"
            raise ValueError(msg)
        home = TeamAppearance(
            match_id=self.match_id,
            team_id=self.home_team_id,
            opponent_id=self.away_team_id,
            season_id=self.season_id,
            kickoff_utc=self.kickoff_utc,
            at_home=True,
            goals_for=self.home_goals,
            goals_against=self.away_goals,
            shots_for=self.home_stats.shots,
            shots_against=self.away_stats.shots,
            sot_for=self.home_stats.shots_on_target,
            sot_against=self.away_stats.shots_on_target,
        )
        away = TeamAppearance(
            match_id=self.match_id,
            team_id=self.away_team_id,
            opponent_id=self.home_team_id,
            season_id=self.season_id,
            kickoff_utc=self.kickoff_utc,
            at_home=False,
            goals_for=self.away_goals,
            goals_against=self.home_goals,
            shots_for=self.away_stats.shots,
            shots_against=self.home_stats.shots,
            sot_for=self.away_stats.shots_on_target,
            sot_against=self.home_stats.shots_on_target,
        )
        return home, away


@dataclass(frozen=True, slots=True)
class TeamAppearance:
    """A past match seen from one team's side, so "for" and "against" need no re-orienting."""

    match_id: int
    team_id: int
    opponent_id: int
    season_id: int
    kickoff_utc: datetime
    at_home: bool
    goals_for: int
    goals_against: int
    shots_for: int | None
    shots_against: int | None
    sot_for: int | None
    sot_against: int | None


class HistoryView:
    """A read-only window on the past. Every accessor refuses data at or after the cut-off."""

    __slots__ = ("_as_of", "_index")

    def __init__(self, index: HistoryIndex, as_of: datetime) -> None:
        self._index = index
        self._as_of = as_of

    @property
    def as_of(self) -> datetime:
        return self._as_of

    def record(self, match_id: int) -> MatchRecord:
        """The match itself, refused unless it kicked off before the cut-off."""
        found = self._index.record(match_id)
        if found.kickoff_utc >= self._as_of:
            msg = (
                f"match {match_id} kicks off at {found.kickoff_utc.isoformat()}, "
                f"which is not before the cut-off {self._as_of.isoformat()}"
            )
            raise LeakageError(msg)
        return found

    def result_of(self, match_id: int) -> tuple[int, int]:
        """The final score of a past match."""
        found = self.record(match_id)
        if found.home_goals is None or found.away_goals is None:
            msg = f"match {match_id} has no result"
            raise ValueError(msg)
        return found.home_goals, found.away_goals

    def played(self) -> tuple[MatchRecord, ...]:
        """Every match with a result that kicked off before the cut-off, in kickoff order.

        This is the training set of a backtest fold: a model is fitted on the whole past
        rather than on one team's, and it still cannot reach past the cut-off.
        """
        records, kickoffs = self._index.played()
        return tuple(records[: bisect_left(kickoffs, self._as_of)])

    def recent(
        self, team_id: int, scope: str = OVERALL, limit: int | None = None
    ) -> tuple[TeamAppearance, ...]:
        """A team's appearances before the cut-off, most recent first."""
        entries, kickoffs = self._index.scoped(team_id, scope)
        end = bisect_left(kickoffs, self._as_of)
        if end == 0:
            return ()
        start = 0 if limit is None else max(0, end - limit)
        return tuple(reversed(entries[start:end]))

    def last_appearance(self, team_id: int) -> TeamAppearance | None:
        """The team's most recent match before the cut-off, if it has ever played."""
        recent = self.recent(team_id, OVERALL, limit=1)
        return recent[0] if recent else None

    def appearances_since(self, team_id: int, cutoff: datetime) -> int:
        """How many matches the team played in [cutoff, as_of)."""
        _, kickoffs = self._index.scoped(team_id, OVERALL)
        end = bisect_left(kickoffs, self._as_of)
        return end - bisect_left(kickoffs, cutoff)

    def matches_played(self, team_id: int, season_id: int) -> int:
        """How many matches the team has completed in that season before the cut-off."""
        kickoffs = self._index.season_kickoffs(team_id, season_id)
        return bisect_left(kickoffs, self._as_of)

    def participants(self, season_id: int) -> frozenset[int]:
        """Teams known to have played in that season before the cut-off."""
        entries, kickoffs = self._index.season_participants(season_id)
        end = bisect_left(kickoffs, self._as_of)
        return frozenset(team_id for _, team_id in entries[:end])

    def previous_season(self, season_id: int) -> int | None:
        """The season before this one in kickoff order, or None if none is on record."""
        return self._index.previous_season(season_id)


class HistoryIndex:
    """Every record, indexed so the guarded view can answer in logarithmic time.

    The index holds the future too, deliberately: a view can only refuse to reveal a match it
    knows about, and phase 4 needs exactly that to catch a model reading its own result.
    """

    __slots__ = (
        "_batches",
        "_by_id",
        "_participants",
        "_played",
        "_played_kickoffs",
        "_previous_season",
        "_scoped",
        "_season_kickoffs",
    )

    def __init__(self, records: Iterable[MatchRecord]) -> None:
        ordered = sorted(records, key=lambda record: (record.kickoff_utc, record.match_id))
        self._by_id: dict[int, MatchRecord] = {record.match_id: record for record in ordered}
        self._scoped: dict[tuple[int, str], tuple[list[TeamAppearance], list[datetime]]] = {}
        self._season_kickoffs: dict[tuple[int, int], list[datetime]] = {}
        self._participants: dict[int, tuple[list[tuple[datetime, int]], list[datetime]]] = {}
        self._previous_season: dict[int, int | None] = {}
        self._batches: list[tuple[datetime, tuple[MatchRecord, ...]]] = []
        self._played: list[MatchRecord] = []
        self._played_kickoffs: list[datetime] = []

        season_order: list[int] = []
        batch: list[MatchRecord] = []
        for record in ordered:
            if record.season_id not in self._previous_season:
                self._previous_season[record.season_id] = season_order[-1] if season_order else None
                season_order.append(record.season_id)
            if batch and record.kickoff_utc != batch[0].kickoff_utc:
                self._batches.append((batch[0].kickoff_utc, tuple(batch)))
                batch = []
            batch.append(record)
            if record.played:
                self._played.append(record)
                self._played_kickoffs.append(record.kickoff_utc)
                for appearance in record.appearances():
                    self._add(appearance)
        if batch:
            self._batches.append((batch[0].kickoff_utc, tuple(batch)))

    @classmethod
    def from_records(cls, records: Iterable[MatchRecord]) -> HistoryIndex:
        return cls(records)

    def _add(self, appearance: TeamAppearance) -> None:
        venue = HOME if appearance.at_home else AWAY
        for scope in (OVERALL, venue):
            entries, kickoffs = self._scoped.setdefault((appearance.team_id, scope), ([], []))
            entries.append(appearance)
            kickoffs.append(appearance.kickoff_utc)
        self._season_kickoffs.setdefault((appearance.team_id, appearance.season_id), []).append(
            appearance.kickoff_utc
        )
        entries_by_season, kickoffs_by_season = self._participants.setdefault(
            appearance.season_id, ([], [])
        )
        entries_by_season.append((appearance.kickoff_utc, appearance.team_id))
        kickoffs_by_season.append(appearance.kickoff_utc)

    def record(self, match_id: int) -> MatchRecord:
        try:
            return self._by_id[match_id]
        except KeyError as exc:
            msg = f"no match {match_id} in this history"
            raise KeyError(msg) from exc

    def scoped(self, team_id: int, scope: str) -> tuple[list[TeamAppearance], list[datetime]]:
        if scope not in SCOPES:
            msg = f"unknown history scope {scope!r}; known: {', '.join(SCOPES)}"
            raise KeyError(msg)
        return self._scoped.get((team_id, scope), ([], []))

    def season_kickoffs(self, team_id: int, season_id: int) -> list[datetime]:
        return self._season_kickoffs.get((team_id, season_id), [])

    def season_participants(
        self, season_id: int
    ) -> tuple[list[tuple[datetime, int]], list[datetime]]:
        return self._participants.get(season_id, ([], []))

    def previous_season(self, season_id: int) -> int | None:
        return self._previous_season.get(season_id)

    def played(self) -> tuple[list[MatchRecord], list[datetime]]:
        """Every match with a result, in kickoff order, with its kickoffs alongside."""
        return self._played, self._played_kickoffs

    def view(self, as_of: datetime) -> HistoryView:
        return HistoryView(self, as_of)

    def batches(self) -> Iterator[tuple[datetime, tuple[MatchRecord, ...]]]:
        """Matches grouped by identical kickoff, in chronological order."""
        yield from self._batches


class Accumulator(Protocol):
    """One feature group. Its state may only ever be fed with matches that are already past."""

    @property
    def names(self) -> tuple[str, ...]:
        """The features this group contributes, in a stable order."""

    def begin(self, batch: Sequence[MatchRecord], view: HistoryView) -> None:
        """Called once per batch, before any feature of that batch is computed."""

    def features(self, match: MatchRecord, view: HistoryView) -> FeatureValues:
        """The group's values for one match, read only through `view`.

        `match` arrives with its result stripped, so a group cannot read the outcome it
        is about to describe even by accident.
        """

    def update(self, batch: Sequence[MatchRecord]) -> None:
        """Called once per batch, after every feature of that batch has been computed."""


@dataclass(frozen=True, slots=True)
class ReplayStep:
    """One match, the cut-off it was computed at, and the features that came out."""

    match: MatchRecord
    as_of_utc: datetime
    view: HistoryView
    values: FeatureValues


class ReplayEngine:
    """Walks a competition's matches in kickoff order, one simultaneous batch at a time."""

    __slots__ = ("_index", "_offset")

    def __init__(self, records: Iterable[MatchRecord], as_of_offset_seconds: int) -> None:
        if as_of_offset_seconds < 1:
            msg = "the cut-off must sit at least one second before kickoff"
            raise ValueError(msg)
        self._index = HistoryIndex.from_records(records)
        self._offset = timedelta(seconds=as_of_offset_seconds)

    @property
    def index(self) -> HistoryIndex:
        return self._index

    def run(self, accumulators: Sequence[Accumulator]) -> Iterator[ReplayStep]:
        """Yield one step per match. Ratings move only once a whole batch has been served."""
        for kickoff, batch in self._index.batches():
            view = self._index.view(kickoff - self._offset)
            for accumulator in accumulators:
                accumulator.begin(batch, view)
            for match in batch:
                # A group only ever sees the fixture, never the result it is describing.
                fixture = match.as_fixture()
                values: FeatureValues = {}
                for accumulator in accumulators:
                    values.update(accumulator.features(fixture, view))
                yield ReplayStep(match=match, as_of_utc=view.as_of, view=view, values=values)
            for accumulator in accumulators:
                accumulator.update(batch)


def merge_names(accumulators: Sequence[Accumulator]) -> tuple[str, ...]:
    """Every group's feature names, in group order, failing loudly on a collision."""
    names: list[str] = []
    seen: set[str] = set()
    for accumulator in accumulators:
        for name in accumulator.names:
            if name in seen:
                msg = f"two feature groups both define {name!r}"
                raise ValueError(msg)
            seen.add(name)
            names.append(name)
    return tuple(names)


def as_of_for(kickoff: datetime, offset_seconds: int) -> datetime:
    """The cut-off of a match, which is what makes as_of_utc < kickoff_utc structural."""
    return kickoff - timedelta(seconds=offset_seconds)


#: Re-exported so callers do not have to know the Mapping type of a feature row.
FeatureMapping = Mapping[str, float | None]
