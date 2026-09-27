"""Rolling and exponentially weighted form (spec section 6.3).

Every metric comes in two flavours: an exponentially weighted average over the whole past,
and the plain mean of the last N appearances. Both are measured overall and at the team's own
venue, so the home side is described by its home record and the away side by its away one.

The rolling means are read through the guarded history view; the exponentially weighted ones
are derived state that only ever receives matches the replay has already passed.
"""

from __future__ import annotations

from collections.abc import Sequence

from predictor.config import Settings, get_settings
from predictor.features.replay import (
    AWAY,
    HOME,
    OVERALL,
    FeatureValues,
    HistoryView,
    MatchRecord,
    TeamAppearance,
)

#: The quantities the form group tracks, as named on `TeamAppearance`.
METRICS = (
    "goals_for",
    "goals_against",
    "shots_for",
    "shots_against",
    "sot_for",
    "sot_against",
)

#: (feature prefix, the venue scope that side is measured at).
SIDES = (("home", HOME), ("away", AWAY))


class FormFeatures:
    """The form accumulator: rolling windows from the view, EWMA from its own state."""

    __slots__ = ("_alpha", "_ewma", "_halflife", "_max_window", "_names", "_windows")

    def __init__(self, windows: Sequence[int], ewma_halflife: float) -> None:
        if not windows:
            msg = "the form group needs at least one rolling window"
            raise ValueError(msg)
        if ewma_halflife <= 0:
            msg = "the exponential half-life must be positive"
            raise ValueError(msg)
        self._windows = tuple(windows)
        self._halflife = ewma_halflife
        self._alpha = float(1.0 - 2.0 ** (-1.0 / ewma_halflife))
        self._max_window = max(self._windows)
        self._names = tuple(self._generate_names())
        #: (team, venue scope, metric) -> its exponentially weighted average so far.
        self._ewma: dict[tuple[int, str, str], float] = {}

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> FormFeatures:
        resolved = (settings or get_settings()).features.form
        return cls(windows=resolved.windows, ewma_halflife=resolved.ewma_halflife)

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def alpha(self) -> float:
        """The weight the newest match carries, derived from the configured half-life."""
        return self._alpha

    def _generate_names(self) -> list[str]:
        names: list[str] = []
        for side, _ in SIDES:
            for scope in ("overall", "venue"):
                names.extend(f"{side}_{scope}_ewma_{metric}" for metric in METRICS)
                for window in self._windows:
                    names.extend(f"{side}_{scope}_last{window}_{metric}" for metric in METRICS)
        return names

    def begin(self, batch: Sequence[MatchRecord], view: HistoryView) -> None:
        del batch, view  # nothing to prepare: every window is read per match

    def features(self, match: MatchRecord, view: HistoryView) -> FeatureValues:
        values: FeatureValues = {}
        for side, team_id, venue in (
            ("home", match.home_team_id, HOME),
            ("away", match.away_team_id, AWAY),
        ):
            for label, scope in (("overall", OVERALL), ("venue", venue)):
                recent = view.recent(team_id, scope, limit=self._max_window)
                for metric in METRICS:
                    values[f"{side}_{label}_ewma_{metric}"] = self._ewma.get(
                        (team_id, scope, metric)
                    )
                for window in self._windows:
                    for metric in METRICS:
                        values[f"{side}_{label}_last{window}_{metric}"] = _window_mean(
                            recent, window, metric
                        )
        return values

    def update(self, batch: Sequence[MatchRecord]) -> None:
        for match in batch:
            if not match.played:
                continue
            for appearance in match.appearances():
                venue = HOME if appearance.at_home else AWAY
                for scope in (OVERALL, venue):
                    self._absorb(appearance, scope)

    def _absorb(self, appearance: TeamAppearance, scope: str) -> None:
        for metric in METRICS:
            observed = getattr(appearance, metric)
            if observed is None:
                # A statistic the source never published cannot pull the average down.
                continue
            key = (appearance.team_id, scope, metric)
            previous = self._ewma.get(key)
            value = float(observed)
            self._ewma[key] = (
                value if previous is None else self._alpha * value + (1.0 - self._alpha) * previous
            )


def _window_mean(recent: Sequence[TeamAppearance], window: int, metric: str) -> float | None:
    """Mean of a metric over the most recent `window` appearances, oldest summed first."""
    total = 0.0
    seen = 0
    for appearance in reversed(recent[:window]):
        observed = getattr(appearance, metric)
        if observed is None:
            continue
        total += observed
        seen += 1
    return total / seen if seen else None
