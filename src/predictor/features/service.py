"""Orchestration of phase 3: match history in, a validated feature store out.

One competition is one unit of work. Its whole history is replayed from the beginning even
when only a few seasons are being written, because a point-in-time feature of a 2024 match is
a function of everything before it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from predictor.config import Competition, Settings, get_settings
from predictor.db import session_scope
from predictor.features import repository
from predictor.features.builder import FeatureBuilder, FeatureRow
from predictor.features.validate import validate_features
from predictor.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class FeatureEntry:
    """What one competition contributed."""

    competition: str
    seasons: int
    matches: int
    rows: int


@dataclass(frozen=True, slots=True)
class FeatureReport:
    """Everything the CLI needs to show, including the demo output."""

    version: str
    definition_checksum: str
    feature_count: int
    entries: tuple[FeatureEntry, ...]
    #: Competition code -> (team name, rating), strongest first.
    ratings: Mapping[str, tuple[tuple[str, float], ...]]
    #: Competition code -> the most recent row written, as a sample.
    samples: Mapping[str, FeatureRow]

    @property
    def rows(self) -> int:
        return sum(entry.rows for entry in self.entries)

    @property
    def matches(self) -> int:
        return sum(entry.matches for entry in self.entries)


def build_features(
    competitions: Sequence[Competition],
    version: str | None = None,
    seasons: Sequence[str] | None = None,
    settings: Settings | None = None,
) -> FeatureReport:
    """Build and store the feature store for every given competition."""
    resolved = settings or get_settings()
    wanted = frozenset(seasons) if seasons else None

    entries: list[FeatureEntry] = []
    ratings: dict[str, tuple[tuple[str, float], ...]] = {}
    samples: dict[str, FeatureRow] = {}
    builder: FeatureBuilder | None = None

    for competition in competitions:
        builder = FeatureBuilder.for_competition(competition, resolved, version)
        with session_scope() as session:
            records = repository.load_records(session, competition, resolved)
            rows = list(builder.rows(records))
            validate_features(builder.frame(rows), builder.names)

            seasons_of = {record.match_id: record.season_label for record in records}
            selected = (
                rows
                if wanted is None
                else [row for row in rows if seasons_of[row.match_id] in wanted]
            )
            written = repository.upsert_features(session, selected)
            names = repository.team_names(session, competition)

        ratings[competition.code] = tuple(
            (names.get(team_id, str(team_id)), rating)
            for team_id, rating in sorted(
                builder.final_ratings().items(), key=lambda item: (-item[1], item[0])
            )
        )
        if selected:
            samples[competition.code] = selected[-1]
        entries.append(
            FeatureEntry(
                competition=competition.code,
                seasons=len({seasons_of[row.match_id] for row in selected}),
                matches=len(records),
                rows=written,
            )
        )
        logger.info(
            "features.competition",
            competition=competition.code,
            version=builder.version,
            matches=len(records),
            rows=written,
            features=len(builder.names),
        )

    if builder is None:
        msg = "no competition was selected, so there is nothing to build"
        raise ValueError(msg)

    return FeatureReport(
        version=builder.version,
        definition_checksum=builder.definition_checksum,
        feature_count=len(builder.names),
        entries=tuple(entries),
        ratings=ratings,
        samples=samples,
    )
