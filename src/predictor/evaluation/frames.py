"""The frames the backtester hands a model, built from guarded records only.

Every frame here is assembled from `MatchRecord`s that the walk-forward engine read through a
`HistoryView`, so a column that exists at all is a column the model was allowed to see. The
fixtures frame is built from `MatchRecord.as_fixture()`, which is why no result can reach a
prediction even by accident.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from predictor.config import MarketFeatureSettings
from predictor.evaluation.metrics import AWAY, DRAW, HOME
from predictor.features.market import guarded_quote
from predictor.features.replay import MatchRecord

#: The training frame of spec section 7, in a stable order.
MATCH_COLUMNS = (
    "match_id",
    "competition_id",
    "season_id",
    "kickoff_utc",
    "home_team_id",
    "away_team_id",
    "home_goals",
    "away_goals",
    "outcome",
)

#: The prediction frame: the same identity columns, and nothing that reveals the result.
FIXTURE_COLUMNS = (
    "match_id",
    "competition_id",
    "season_id",
    "kickoff_utc",
    "home_team_id",
    "away_team_id",
)

#: Added only for a model that declares `needs_market` (the undeployable benchmark).
MARKET_COLUMNS = ("market_p_home", "market_p_draw", "market_p_away")

#: The identity columns of a feature frame; the rest are the features themselves.
FEATURE_KEY_COLUMNS = ("match_id", "as_of_utc")

_INTEGER_COLUMNS = (
    "match_id",
    "competition_id",
    "season_id",
    "home_team_id",
    "away_team_id",
)


@dataclass(frozen=True, slots=True)
class StoredFeatures:
    """One row of the feature store, as the backtest reads it back."""

    match_id: int
    as_of_utc: datetime
    values: Mapping[str, float | None]


def outcome_of(record: MatchRecord) -> int | None:
    """The outcome code of a played match, or None when there is no result to score."""
    if not record.played or record.home_goals is None or record.away_goals is None:
        return None
    if record.home_goals > record.away_goals:
        return HOME
    if record.home_goals == record.away_goals:
        return DRAW
    return AWAY


def matches_frame(records: Iterable[MatchRecord]) -> pd.DataFrame:
    """The training frame: one row per match, with its result and outcome code."""
    rows = [
        {
            "match_id": record.match_id,
            "competition_id": record.competition_id,
            "season_id": record.season_id,
            "kickoff_utc": record.kickoff_utc,
            "home_team_id": record.home_team_id,
            "away_team_id": record.away_team_id,
            "home_goals": record.home_goals,
            "away_goals": record.away_goals,
            "outcome": outcome_of(record),
        }
        for record in records
    ]
    frame = pd.DataFrame(rows, columns=list(MATCH_COLUMNS))
    return _with_nullable_integers(frame, ("home_goals", "away_goals", "outcome"))


def fixtures_frame(
    records: Iterable[MatchRecord],
    *,
    as_of: datetime,
    market: MarketFeatureSettings | None = None,
) -> pd.DataFrame:
    """The prediction frame, with the market columns only when a model may read them."""
    prepared = list(records)
    rows = [
        {
            "match_id": record.match_id,
            "competition_id": record.competition_id,
            "season_id": record.season_id,
            "kickoff_utc": record.kickoff_utc,
            "home_team_id": record.home_team_id,
            "away_team_id": record.away_team_id,
        }
        for record in prepared
    ]
    frame = pd.DataFrame(rows, columns=list(FIXTURE_COLUMNS))
    if market is None:
        return frame
    quotes = [guarded_quote(record.match_id, record.market, as_of, market) for record in prepared]
    for column, side in zip(MARKET_COLUMNS, ("p_home", "p_draw", "p_away"), strict=True):
        frame[column] = [None if quote is None else getattr(quote, side) for quote in quotes]
    return frame


def features_frame(rows: Sequence[StoredFeatures], names: Sequence[str]) -> pd.DataFrame | None:
    """The feature frame, or None when the store has nothing for these matches."""
    if not rows:
        return None
    data = [
        {"match_id": row.match_id, "as_of_utc": row.as_of_utc, **dict.fromkeys(names), **row.values}
        for row in rows
    ]
    return pd.DataFrame(data, columns=[*FEATURE_KEY_COLUMNS, *names])


def _with_nullable_integers(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Goals and outcomes are integers that can be missing, which only Int64 can express."""
    for column in _INTEGER_COLUMNS:
        if column in frame:
            frame[column] = frame[column].astype("int64")
    for column in columns:
        frame[column] = frame[column].astype("Int64")
    return frame
