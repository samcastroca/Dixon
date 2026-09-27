"""Rest, congestion, season stage and the promoted-team flag (spec section 6.3).

The promoted flag is deliberately derived from the *previous* season's participants only.
Deriving it from the current season's squad would read a fixture list that stretches past
kickoff, and a feature would then change depending on how much of the season already exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

from predictor.config import Competition, Settings, get_settings
from predictor.features.replay import FeatureValues, HistoryView, MatchRecord

SECONDS_PER_DAY = 86400.0


class ScheduleFeatures:
    """The schedule accumulator. Everything it reports is read through the guarded view."""

    __slots__ = ("_matches_per_team", "_names", "_promoted", "_window")

    def __init__(self, congestion_window_days: int, matches_per_team: int) -> None:
        if congestion_window_days < 1:
            msg = "the congestion window must cover at least one day"
            raise ValueError(msg)
        if matches_per_team < 1:
            msg = "a season must hold at least one match per team"
            raise ValueError(msg)
        self._window = congestion_window_days
        self._matches_per_team = matches_per_team
        self._names = (
            "home_rest_days",
            "away_rest_days",
            f"home_matches_last_{congestion_window_days}d",
            f"away_matches_last_{congestion_window_days}d",
            "home_matches_played",
            "away_matches_played",
            "home_season_stage",
            "away_season_stage",
            "home_is_promoted",
            "away_is_promoted",
        )
        #: (team, season) -> the flag, which cannot change once the season has started.
        self._promoted: dict[tuple[int, int], float | None] = {}

    @classmethod
    def for_competition(
        cls, competition: Competition, settings: Settings | None = None
    ) -> ScheduleFeatures:
        resolved = settings or get_settings()
        return cls(
            congestion_window_days=resolved.features.schedule.congestion_window_days,
            matches_per_team=matches_per_team(competition),
        )

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    def begin(self, batch: Sequence[MatchRecord], view: HistoryView) -> None:
        del batch, view  # every value is read per match

    def features(self, match: MatchRecord, view: HistoryView) -> FeatureValues:
        cutoff = match.kickoff_utc - timedelta(days=self._window)
        rest: dict[str, float | None] = {}
        congestion: dict[str, float | None] = {}
        played: dict[str, float | None] = {}
        stage: dict[str, float | None] = {}
        promoted: dict[str, float | None] = {}
        for side, team_id in (("home", match.home_team_id), ("away", match.away_team_id)):
            last = view.last_appearance(team_id)
            rest[side] = (
                None
                if last is None
                else (match.kickoff_utc - last.kickoff_utc).total_seconds() / SECONDS_PER_DAY
            )
            congestion[side] = float(view.appearances_since(team_id, cutoff))
            count = view.matches_played(team_id, match.season_id)
            played[side] = float(count)
            stage[side] = count / self._matches_per_team
            promoted[side] = self._is_promoted(team_id, match.season_id, view)
        return {
            "home_rest_days": rest["home"],
            "away_rest_days": rest["away"],
            f"home_matches_last_{self._window}d": congestion["home"],
            f"away_matches_last_{self._window}d": congestion["away"],
            "home_matches_played": played["home"],
            "away_matches_played": played["away"],
            "home_season_stage": stage["home"],
            "away_season_stage": stage["away"],
            "home_is_promoted": promoted["home"],
            "away_is_promoted": promoted["away"],
        }

    def update(self, batch: Sequence[MatchRecord]) -> None:
        del batch  # the schedule keeps no derived state of its own

    def _is_promoted(self, team_id: int, season_id: int, view: HistoryView) -> float | None:
        """Whether the team is new to this league, judged only on the previous season."""
        key = (team_id, season_id)
        if key in self._promoted:
            return self._promoted[key]
        previous = view.previous_season(season_id)
        participants = view.participants(previous) if previous is not None else frozenset()
        # With no previous season on record the question is unanswerable, not answerable "no".
        flag = None if not participants else float(team_id not in participants)
        self._promoted[key] = flag
        return flag


def matches_per_team(competition: Competition) -> int:
    """How many league matches a team plays in a season: everyone else, home and away."""
    return 2 * (competition.rules.teams - 1)
