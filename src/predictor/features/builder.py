"""Assembles one wide, versioned feature row per match (spec sections 5 and 6.3).

A row is identified by `(match_id, feature_version)` and stamped with `as_of_utc`, which the
builder derives from kickoff so the point-in-time invariant cannot be forgotten. Alongside the
version label it also stores a checksum of the actual definitions, so a rebuild under the same
label but different parameters is visible instead of silently overwriting the old numbers.

Every value is rounded to `features.value_decimals` on the way out. Without it a rebuild is
only reproducible on the machine that ran it: `pow` differs by one bit between C libraries,
so the container and the host would take turns rewriting the same rows forever.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from predictor.config import Competition, Settings, get_settings
from predictor.features.elo import EloRatings
from predictor.features.form import FormFeatures
from predictor.features.market import MarketFeatures
from predictor.features.replay import (
    Accumulator,
    FeatureValues,
    MatchRecord,
    ReplayEngine,
    as_of_for,
    merge_names,
)
from predictor.features.schedule import ScheduleFeatures
from predictor.features.validate import KEY_COLUMNS


@dataclass(frozen=True, slots=True)
class FeatureRow:
    """One match's features, ready to be stored."""

    match_id: int
    feature_version: str
    as_of_utc: datetime
    kickoff_utc: datetime
    computed_at: datetime
    definition_checksum: str
    values: FeatureValues

    def as_record(self) -> dict[str, object]:
        """The row as the `features` table holds it; kickoff lives in `matches`."""
        return {
            "match_id": self.match_id,
            "feature_version": self.feature_version,
            "as_of_utc": self.as_of_utc,
            "computed_at": self.computed_at,
            "definition_checksum": self.definition_checksum,
            "feature_values": dict(self.values),
        }


class FeatureBuilder:
    """Turns a competition's match history into feature rows, one replay at a time."""

    __slots__ = (
        "_checksum",
        "_competition",
        "_decimals",
        "_elo",
        "_names",
        "_offset",
        "_settings",
        "_version",
    )

    def __init__(
        self, competition: Competition, settings: Settings, version: str | None = None
    ) -> None:
        self._competition = competition
        self._settings = settings
        self._version = version or settings.features.version
        self._offset = settings.features.as_of_offset_seconds
        self._decimals = settings.features.value_decimals
        self._names = merge_names(self._accumulators()[0])
        self._checksum = _definition_checksum(settings, self._names)
        self._elo: EloRatings | None = None

    @classmethod
    def for_competition(
        cls, competition: Competition, settings: Settings | None = None, version: str | None = None
    ) -> FeatureBuilder:
        return cls(competition, settings or get_settings(), version)

    @property
    def competition(self) -> Competition:
        return self._competition

    @property
    def version(self) -> str:
        return self._version

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def definition_checksum(self) -> str:
        return self._checksum

    def _accumulators(self) -> tuple[list[Accumulator], EloRatings | None]:
        """A fresh set of accumulators, so two builds never share state."""
        groups = self._settings.features.groups
        accumulators: list[Accumulator] = []
        elo: EloRatings | None = None
        if groups.elo:
            elo = EloRatings.for_competition(self._competition, self._settings)
            accumulators.append(elo)
        if groups.form:
            accumulators.append(FormFeatures.from_settings(self._settings))
        if groups.schedule:
            accumulators.append(ScheduleFeatures.for_competition(self._competition, self._settings))
        if groups.market:
            accumulators.append(MarketFeatures.from_settings(self._settings))
        if not accumulators:
            msg = "no feature group is enabled; there would be nothing to build"
            raise ValueError(msg)
        return accumulators, elo

    def rows(self, records: Iterable[MatchRecord]) -> Iterator[FeatureRow]:
        """Replay the history and yield a row per match, lazily and in kickoff order."""
        accumulators, elo = self._accumulators()
        self._elo = elo
        computed_at = datetime.now(UTC)
        engine = ReplayEngine(records, as_of_offset_seconds=self._offset)
        for step in engine.run(accumulators):
            yield FeatureRow(
                match_id=step.match.match_id,
                feature_version=self._version,
                as_of_utc=as_of_for(step.match.kickoff_utc, self._offset),
                kickoff_utc=step.match.kickoff_utc,
                computed_at=computed_at,
                definition_checksum=self._checksum,
                values=self._rounded(step.values),
            )

    def _rounded(self, values: FeatureValues) -> FeatureValues:
        return {
            name: None if value is None else round(value, self._decimals)
            for name, value in values.items()
        }

    def frame(self, rows: Iterable[FeatureRow]) -> pd.DataFrame:
        """The wide frame the schema validates and later phases train on."""
        data = [
            {
                "match_id": row.match_id,
                "kickoff_utc": row.kickoff_utc,
                "as_of_utc": row.as_of_utc,
                **row.values,
            }
            for row in rows
        ]
        return pd.DataFrame(data, columns=[*KEY_COLUMNS, *self._names])

    def final_ratings(self) -> Mapping[int, float]:
        """Elo ratings left standing after the last replay, for the demo output."""
        return self._elo.ratings() if self._elo is not None else {}


def _definition_checksum(settings: Settings, names: Sequence[str]) -> str:
    """Hash of what a feature *is*, independent of the label the build was given."""
    features = settings.features.model_dump(mode="json")
    features.pop("version", None)
    payload = json.dumps(
        {"definitions": features, "features": list(names)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
