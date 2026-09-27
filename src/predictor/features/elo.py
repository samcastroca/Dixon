"""Pre-match Elo ratings, fitted per competition (spec sections 6.3 and 7.1).

    E_home = 1 / (1 + 10 ** (-(R_home + hfa - R_away) / 400))
    R' = R +- K * G * (S - E)

`G` scales the update with the goal margin, and every constant in it comes from
settings.yaml. A rating only ever moves once a match has been played, and every match in a
batch of simultaneous kickoffs is rated against the ratings as they stood before the batch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from predictor.config import Competition, EloParameters, Settings, get_settings
from predictor.features.replay import FeatureValues, HistoryView, MatchRecord

#: Elo's traditional scale: 400 rating points are a factor of ten in expected score.
RATING_SCALE = 400.0

NAMES = ("home_elo_pre", "away_elo_pre", "elo_diff", "elo_expected_home")


class EloRatings:
    """The Elo accumulator: pre-match features out, post-match updates in."""

    __slots__ = (
        "_current_season",
        "_parameters",
        "_pre",
        "_ratings",
        "_relegation_slots",
        "_season_end",
    )

    def __init__(self, parameters: EloParameters, relegation_slots: int) -> None:
        if relegation_slots < 1:
            msg = "a promoted team's rating needs at least one relegated team to average"
            raise ValueError(msg)
        self._parameters = parameters
        self._relegation_slots = relegation_slots
        self._ratings: dict[int, float] = {}
        #: Ratings as they stood when each season finished, for the promoted-team rule.
        self._season_end: dict[int, dict[int, float]] = {}
        self._current_season: int | None = None
        #: Ratings of the teams in the batch being served, frozen before any of them updates.
        self._pre: dict[int, float] = {}

    @classmethod
    def for_competition(
        cls, competition: Competition, settings: Settings | None = None
    ) -> EloRatings:
        resolved = settings or get_settings()
        return cls(
            resolved.features.elo.for_competition(competition.code),
            competition.rules.relegation_slots,
        )

    @property
    def names(self) -> tuple[str, ...]:
        return NAMES

    @property
    def parameters(self) -> EloParameters:
        return self._parameters

    def goal_difference_multiplier(self, margin: int) -> float:
        """G: flat for a tight match, then steeper the wider the margin (spec section 7, M1)."""
        size = abs(margin)
        if size <= 1:
            return 1.0
        if size == 2:
            return self._parameters.two_goal_multiplier
        return (self._parameters.large_margin_base + size) / self._parameters.large_margin_divisor

    def expected_home(self, home_rating: float, away_rating: float) -> float:
        """The home side's expected score, home advantage included."""
        edge = home_rating + self._parameters.home_advantage - away_rating
        return 1.0 / (1.0 + float(10.0 ** (-edge / RATING_SCALE)))

    def rating(self, team_id: int) -> float | None:
        """The team's current rating, or None if it has never played."""
        return self._ratings.get(team_id)

    def ratings(self) -> Mapping[int, float]:
        return dict(self._ratings)

    def begin(self, batch: Sequence[MatchRecord], view: HistoryView) -> None:
        self._cross_season_boundary(batch)
        self._pre = {}
        for match in batch:
            for team_id in (match.home_team_id, match.away_team_id):
                self._pre[team_id] = self._rating_of(team_id, match.season_id, view)

    def features(self, match: MatchRecord, view: HistoryView) -> FeatureValues:
        del view  # the ratings are derived state; the view guards the data they came from
        home = self._pre[match.home_team_id]
        away = self._pre[match.away_team_id]
        return {
            "home_elo_pre": home,
            "away_elo_pre": away,
            "elo_diff": home - away,
            "elo_expected_home": self.expected_home(home, away),
        }

    def update(self, batch: Sequence[MatchRecord]) -> None:
        for match in batch:
            if not match.played or match.home_goals is None or match.away_goals is None:
                continue
            home = self._pre[match.home_team_id]
            away = self._pre[match.away_team_id]
            expected = self.expected_home(home, away)
            scored = _outcome(match.home_goals, match.away_goals)
            multiplier = self.goal_difference_multiplier(match.home_goals - match.away_goals)
            delta = self._parameters.k * multiplier * (scored - expected)
            self._ratings[match.home_team_id] = home + delta
            self._ratings[match.away_team_id] = away - delta

    def _cross_season_boundary(self, batch: Sequence[MatchRecord]) -> None:
        """Snapshot the season that just ended, then regress every rating to the mean."""
        season = batch[0].season_id
        if season == self._current_season:
            return
        if self._current_season is not None:
            self._season_end[self._current_season] = dict(self._ratings)
            share = self._parameters.season_regression
            target = self._parameters.season_regression_target
            if share:
                self._ratings = {
                    team_id: target + (1.0 - share) * (rating - target)
                    for team_id, rating in self._ratings.items()
                }
        self._current_season = season

    def _rating_of(self, team_id: int, season_id: int, view: HistoryView) -> float:
        known = self._ratings.get(team_id)
        if known is not None:
            return known
        return self._promoted_rating(season_id, view)

    def _promoted_rating(self, season_id: int, view: HistoryView) -> float:
        """A newcomer enters on the mean of the lowest-rated teams of the previous season.

        Those are the teams that went down, and using only the previous season keeps the
        rating a function of the past alone: nothing here looks at who is in the league now.
        A newcomer has no history to revert, so this estimate is not season-regressed.
        """
        previous = view.previous_season(season_id)
        snapshot = self._season_end.get(previous) if previous is not None else None
        if not snapshot:
            return self._parameters.initial_rating
        lowest = sorted(snapshot.values())[: self._relegation_slots]
        return sum(lowest) / len(lowest)


def _outcome(home_goals: int, away_goals: int) -> float:
    """The home side's score: a win, a draw or a loss."""
    if home_goals > away_goals:
        return 1.0
    if home_goals == away_goals:
        return 0.5
    return 0.0
