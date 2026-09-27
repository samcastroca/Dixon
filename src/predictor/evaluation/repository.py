"""Database access for the backtest: history and features in, scored slices out.

Reading happens once per competition, before any model is fitted, so the guard has a closed
set of records to reason about — the same discipline phase 3 uses for the feature store.

The market quotes are loaded here under `backtest.market_baseline`, not under the feature
store's market settings. That separation is the whole point: the undeployable benchmark may
use closing prices this source publishes without a capture time, and the feature store may not.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace

from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from predictor import tables
from predictor.config import Competition, Settings, get_settings
from predictor.evaluation.backtest import CompetitionHistory
from predictor.evaluation.frames import StoredFeatures
from predictor.evaluation.report import ReportRow
from predictor.features import repository as features_repository
from predictor.logging import get_logger

logger = get_logger(__name__)

#: How many rows go into one INSERT, as in the feature store.
CHUNK_SIZE = 1000

#: The columns a conflicting result row may overwrite; the key itself never changes.
_RESULT_UPDATES = (
    "feature_version",
    "git_sha",
    "n",
    "rps",
    "log_loss",
    "brier",
    "ece",
    "reference_model",
    "rps_gap",
    "rps_gap_ci_low",
    "rps_gap_ci_high",
)


def load_history(
    session: Session,
    competition: Competition,
    version: str,
    settings: Settings | None = None,
) -> CompetitionHistory:
    """One competition's matches, market quotes and stored features, ready for the engine."""
    resolved = settings or get_settings()
    identifier = features_repository.competition_id(session, competition)
    records = features_repository.load_records(session, competition, resolved)
    quotes = features_repository.market_quotes(
        session, identifier, resolved.backtest.market_baseline, resolved.processing.market
    )
    features, names, checksum = load_features(session, identifier, version)
    logger.info(
        "backtest.history_loaded",
        competition=competition.code,
        matches=len(records),
        quotes=len(quotes),
        feature_rows=len(features),
        feature_version=version,
    )
    return CompetitionHistory(
        competition=competition,
        records=tuple(replace(record, market=quotes.get(record.match_id)) for record in records),
        features=features,
        feature_names=names,
        definition_checksum=checksum,
    )


def load_features(
    session: Session, identifier: int, version: str
) -> tuple[dict[int, StoredFeatures], tuple[str, ...], str | None]:
    """The stored feature rows of one competition and version, keyed by match."""
    rows = session.execute(
        select(
            tables.Feature.match_id,
            tables.Feature.as_of_utc,
            tables.Feature.feature_values,
            tables.Feature.definition_checksum,
        )
        .join(tables.Match, tables.Match.id == tables.Feature.match_id)
        .where(tables.Match.competition_id == identifier, tables.Feature.feature_version == version)
        .order_by(tables.Feature.as_of_utc, tables.Feature.match_id)
    ).all()
    stored: dict[int, StoredFeatures] = {}
    names: list[str] = []
    checksum: str | None = None
    for match_id, as_of_utc, values, definition_checksum in rows:
        stored[match_id] = StoredFeatures(match_id=match_id, as_of_utc=as_of_utc, values=values)
        if not names:
            names = sorted(values)
        checksum = definition_checksum
    return stored, tuple(names), checksum


def competition_ids(session: Session) -> dict[str, int]:
    """Competition code to stored id, for the rows that mirror MLflow."""
    rows = session.execute(select(tables.Competition.code, tables.Competition.id)).all()
    return dict(rows)


def season_ids(session: Session) -> dict[tuple[int, str], int]:
    """(competition id, season label) to stored season id."""
    rows = session.execute(
        select(tables.Season.competition_id, tables.Season.label, tables.Season.id)
    ).all()
    return {(competition_id, label): identifier for competition_id, label, identifier in rows}


def upsert_backtest_results(session: Session, records: Sequence[Mapping[str, object]]) -> int:
    """Store scored slices idempotently: mirroring the same run twice changes nothing."""
    written = 0
    for chunk in _chunks(records, CHUNK_SIZE):
        statement = insert(tables.BacktestResult).values(list(chunk))
        excluded = statement.excluded
        changed = or_(
            *(
                getattr(tables.BacktestResult, column).is_distinct_from(excluded[column])
                for column in _RESULT_UPDATES
            )
        )
        session.execute(
            statement.on_conflict_do_update(
                constraint="uq_backtest_results_slice",
                set_={column: excluded[column] for column in _RESULT_UPDATES},
                where=changed,
            )
        )
        written += len(chunk)
    return written


def _chunks(
    values: Sequence[Mapping[str, object]], size: int
) -> Iterable[Sequence[Mapping[str, object]]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def latest_results(session: Session) -> tuple[ReportRow, ...]:
    """Every slice of each model's most recent run: what the comparison report shows.

    The table keeps every run, so "the report" has to mean the newest run per model rather
    than everything ever stored.
    """
    rows = session.execute(
        select(
            tables.BacktestResult,
            tables.Competition.code,
            tables.Season.label,
        )
        .outerjoin(
            tables.Competition, tables.Competition.id == tables.BacktestResult.competition_id
        )
        .outerjoin(tables.Season, tables.Season.id == tables.BacktestResult.season_id)
        .order_by(tables.BacktestResult.created_at, tables.BacktestResult.id)
    ).all()
    newest: dict[str, str] = {row.model_name: row.run_id for row, _, _ in rows}
    return tuple(
        ReportRow(
            model_name=result.model_name,
            scope=result.scope,
            competition=code,
            season_label=label,
            n=result.n,
            rps=result.rps,
            log_loss=result.log_loss,
            brier=result.brier,
            ece=result.ece,
            rps_gap=result.rps_gap,
            rps_gap_ci_low=result.rps_gap_ci_low,
            rps_gap_ci_high=result.rps_gap_ci_high,
        )
        for result, code, label in rows
        if newest[result.model_name] == result.run_id
    )
