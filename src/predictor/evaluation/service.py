"""Orchestration of phase 4: history in, an honest measuring stick out.

One invocation loads every competition once, runs every requested model over exactly the same
folds, scores each of them per season, per league and pooled, compares them against the
reference model with a paired bootstrap, records all of it in MLflow and mirrors the numbers
into `backtest_results` so the dashboard and `make report` never have to recompute anything.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from predictor.config import BootstrapSettings, Competition, Settings, get_settings
from predictor.db import session_scope
from predictor.evaluation import repository, tracking
from predictor.evaluation.backtest import (
    POOLED,
    SEASON,
    CompetitionHistory,
    FittedParameters,
    ModelRun,
    Prediction,
    ScopedMetrics,
    WalkForward,
    as_arrays,
    score,
)
from predictor.evaluation.bootstrap import Interval, paired_bootstrap
from predictor.evaluation.metrics import rps
from predictor.evaluation.report import ParameterRow, ReportRow, write_report
from predictor.logging import get_logger
from predictor.models.registry import MODEL_NAMES, factory

logger = get_logger(__name__)

#: Columns of the predictions artifact.
_PREDICTION_HEADER = (
    "model",
    "match_id",
    "competition",
    "season",
    "kickoff_utc",
    "as_of_utc",
    "p_home",
    "p_draw",
    "p_away",
    "outcome",
)

#: Columns of the fitted-parameters artifact. The parameter itself is one row per name, because
#: every model reports a different set and a fixed column list would not fit all of them.
_PARAMETER_HEADER = ("model", "competition", "season", "fit", "parameter", "value")


@dataclass(frozen=True, slots=True)
class Comparison:
    """One model against the reference, on the matches both of them predicted."""

    model_name: str
    reference_model: str
    scope: str
    competition: str | None
    season_label: str | None
    n: int
    difference: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval supports calling one of the two models better."""
        return self.low > 0.0 or self.high < 0.0


@dataclass(frozen=True, slots=True)
class BacktestOutcome:
    """Everything one backtest produced, for the CLI, the report and the tests."""

    run_id: str
    tracking_uri: str
    feature_version: str
    definition_checksum: str | None
    reference_model: str
    seasons: tuple[str, ...]
    runs: tuple[ModelRun, ...]
    scored: Mapping[str, tuple[ScopedMetrics, ...]]
    comparisons: tuple[Comparison, ...]
    rows: tuple[ReportRow, ...]
    parameters: tuple[ParameterRow, ...]
    records: tuple[Mapping[str, object], ...]
    stored: int
    #: How many `model_parameters` rows this run wrote.
    stored_parameters: int

    def run_of(self, model_name: str) -> ModelRun:
        return next(run for run in self.runs if run.model_name == model_name)


def run_backtest(
    models: Sequence[str],
    competitions: Sequence[Competition],
    seasons: Sequence[str] | None = None,
    version: str | None = None,
    settings: Settings | None = None,
) -> BacktestOutcome:
    """Run the walk-forward backtest of every requested model and record the result."""
    resolved = settings or get_settings()
    names = tuple(models) if models else MODEL_NAMES
    unknown = [name for name in names if name not in MODEL_NAMES]
    if unknown:
        msg = f"unknown model(s) {unknown}; registered models: {', '.join(MODEL_NAMES)}"
        raise KeyError(msg)
    feature_version = version or resolved.backtest.feature_version or resolved.features.version

    with session_scope() as session:
        histories = [
            repository.load_history(session, competition, feature_version, resolved)
            for competition in competitions
        ]
        engine = WalkForward(histories, resolved, seasons)
        runs = tuple(engine.run(factory(name)) for name in names)
        scored = {run.model_name: score(run, resolved) for run in runs}
        comparisons = compare(runs, scored, resolved)
        rows = report_rows(scored, comparisons)

        with tracking.tracked_run(resolved, run_name=_run_name(names)) as tracker:
            _log_run(tracker, resolved, names, competitions, seasons, histories, feature_version)
            _log_fitted_parameters(tracker, runs)
            _log_metrics(tracker, scored, comparisons)
            _log_artifacts(tracker, resolved, runs, scored)
            run_id = tracker.run_id

        competition_ids = repository.competition_ids(session)
        season_ids = repository.season_ids(session)
        records = _records(
            rows,
            run_id=run_id,
            feature_version=feature_version,
            reference_model=resolved.backtest.reference_model,
            competitions=competition_ids,
            seasons=season_ids,
        )
        stored = repository.upsert_backtest_results(session, records)
        stored_parameters = repository.upsert_model_parameters(
            session,
            _parameter_records(
                runs, run_id=run_id, competitions=competition_ids, seasons=season_ids
            ),
        )

    logger.info(
        "backtest.finished",
        run_id=run_id,
        models=list(names),
        folds=sum(len(run.folds) for run in runs),
        predictions=sum(len(run.predictions) for run in runs),
        stored=stored,
        stored_parameters=stored_parameters,
    )
    return BacktestOutcome(
        run_id=run_id,
        tracking_uri=resolved.mlflow.tracking_uri,
        feature_version=feature_version,
        definition_checksum=_checksum(histories),
        reference_model=resolved.backtest.reference_model,
        seasons=tuple(seasons or ()),
        runs=runs,
        scored=scored,
        comparisons=comparisons,
        rows=rows,
        parameters=parameter_rows(runs),
        records=records,
        stored=stored,
        stored_parameters=stored_parameters,
    )


def compare(
    runs: Sequence[ModelRun],
    scored: Mapping[str, Sequence[ScopedMetrics]],
    settings: Settings,
) -> tuple[Comparison, ...]:
    """Paired bootstrap of the RPS gap against the reference model, at every shared scope."""
    reference_name = settings.backtest.reference_model
    if reference_name not in scored:
        logger.warning(
            "backtest.no_reference",
            reference=reference_name,
            models=[run.model_name for run in runs],
        )
        return ()
    bootstrap = settings.backtest.bootstrap
    reference = {_key(entry): entry for entry in scored[reference_name]}
    comparisons: list[Comparison] = []
    for run in runs:
        if run.model_name == reference_name:
            continue
        for entry in scored[run.model_name]:
            against = reference.get(_key(entry))
            if against is None:
                continue
            paired = _interval_of(entry, against, bootstrap)
            if paired is None:
                continue
            interval, shared = paired
            comparisons.append(
                Comparison(
                    model_name=run.model_name,
                    reference_model=reference_name,
                    scope=entry.scope,
                    competition=entry.competition,
                    season_label=entry.season_label,
                    n=shared,
                    difference=interval.difference,
                    low=interval.low,
                    high=interval.high,
                )
            )
    return tuple(comparisons)


def report_rows(
    scored: Mapping[str, Sequence[ScopedMetrics]], comparisons: Sequence[Comparison]
) -> tuple[ReportRow, ...]:
    """One report row per model and scope, carrying the gap where a comparison exists."""
    gaps = {
        (
            comparison.model_name,
            comparison.scope,
            comparison.competition,
            comparison.season_label,
        ): comparison
        for comparison in comparisons
    }
    rows: list[ReportRow] = []
    for model_name, slices in scored.items():
        for entry in slices:
            gap = gaps.get((model_name, entry.scope, entry.competition, entry.season_label))
            rows.append(
                ReportRow(
                    model_name=model_name,
                    scope=entry.scope,
                    competition=entry.competition,
                    season_label=entry.season_label,
                    n=entry.metrics.n,
                    rps=entry.metrics.rps,
                    log_loss=entry.metrics.log_loss,
                    brier=entry.metrics.brier,
                    ece=entry.metrics.ece,
                    rps_gap=None if gap is None else gap.difference,
                    rps_gap_ci_low=None if gap is None else gap.low,
                    rps_gap_ci_high=None if gap is None else gap.high,
                )
            )
    return tuple(rows)


def parameter_rows(runs: Sequence[ModelRun]) -> tuple[ParameterRow, ...]:
    """Every fit of a run as the report holds it, in a stable order."""
    return tuple(
        ParameterRow(
            model_name=fit.model_name,
            competition=fit.competition,
            season_label=fit.season_label,
            fit_index=fit.fit_index,
            values=dict(fit.values),
        )
        for run in runs
        for fit in run.parameters
    )


def stored_rows() -> tuple[tuple[ReportRow, ...], tuple[ParameterRow, ...]]:
    """The latest stored run of every model: the metrics and the parameters behind them."""
    with session_scope() as session:
        return repository.latest_results(session), repository.latest_parameters(session)


def write_stored_report(settings: Settings | None = None) -> tuple[Path, ...]:
    """Render the comparison table and the fitted parameters from what the database holds."""
    resolved = settings or get_settings()
    rows, parameters = stored_rows()
    return write_report(
        rows,
        reference_model=resolved.backtest.reference_model,
        parameters=parameters,
        settings=resolved,
    )


def _key(entry: ScopedMetrics) -> tuple[str, str | None, str | None]:
    return (entry.scope, entry.competition, entry.season_label)


def _interval_of(
    left: ScopedMetrics, right: ScopedMetrics, bootstrap: BootstrapSettings
) -> tuple[Interval, int] | None:
    """The paired interval over the matches both slices predicted, or None when there are none."""
    shared = sorted(
        {prediction.match_id for prediction in left.predictions}
        & {prediction.match_id for prediction in right.predictions}
    )
    if not shared:
        return None
    order = {match_id: index for index, match_id in enumerate(shared)}
    first, outcomes = as_arrays(_aligned(left.predictions, order))
    second, _ = as_arrays(_aligned(right.predictions, order))
    interval = paired_bootstrap(
        rps,
        first,
        second,
        outcomes,
        resamples=bootstrap.resamples,
        confidence=bootstrap.confidence,
        seed=bootstrap.seed,
    )
    return interval, len(shared)


def _aligned(predictions: Sequence[Prediction], order: Mapping[int, int]) -> tuple[Prediction, ...]:
    chosen = [prediction for prediction in predictions if prediction.match_id in order]
    return tuple(sorted(chosen, key=lambda prediction: order[prediction.match_id]))


def _run_name(names: Sequence[str]) -> str:
    return "backtest-" + "-".join(names)


def _checksum(histories: Sequence[CompetitionHistory]) -> str | None:
    checksums = {
        history.definition_checksum for history in histories if history.definition_checksum
    }
    return next(iter(checksums)) if len(checksums) == 1 else None


def _log_run(
    tracker: tracking.Tracker,
    settings: Settings,
    names: Sequence[str],
    competitions: Sequence[Competition],
    seasons: Sequence[str] | None,
    histories: Sequence[CompetitionHistory],
    feature_version: str,
) -> None:
    tracker.log_params(
        {
            "models": ",".join(names),
            "competitions": ",".join(competition.code for competition in competitions),
            "seasons": ",".join(seasons) if seasons else "every season on record",
            "feature_version": feature_version,
            "definition_checksum": _checksum(histories) or "none",
            "refit_every_matchweeks": settings.backtest.refit_every_matchweeks,
            "min_train_matches": settings.backtest.min_train_matches,
            "reference_model": settings.backtest.reference_model,
            "bootstrap_resamples": settings.backtest.bootstrap.resamples,
            "bootstrap_seed": settings.backtest.bootstrap.seed,
            "ece_bins": settings.backtest.metrics.ece_bins,
            "git_sha": tracking.git_sha(),
        }
    )


def _parameter_records(
    runs: Sequence[ModelRun],
    *,
    run_id: str,
    competitions: Mapping[str, int],
    seasons: Mapping[tuple[int, str], int],
) -> tuple[Mapping[str, object], ...]:
    """The rows of `model_parameters`, with the league and season resolved to their ids."""
    records: list[Mapping[str, object]] = []
    for run in runs:
        for fit in run.parameters:
            competition_id = competitions[fit.competition]
            records.append(
                {
                    "run_id": run_id,
                    "model_name": fit.model_name,
                    "competition_id": competition_id,
                    "season_id": seasons.get((competition_id, fit.season_label)),
                    "fit_index": fit.fit_index,
                    "parameter_values": dict(fit.values),
                }
            )
    return tuple(records)


def _parameters_csv(directory: Path, runs: Sequence[ModelRun]) -> Path:
    """Every fit of every model, long format, so one file fits every model's parameter set."""
    path = directory / "fitted_parameters.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_PARAMETER_HEADER)
        for run in runs:
            for fit in run.parameters:
                for name, value in sorted(fit.values.items()):
                    writer.writerow(
                        [
                            fit.model_name,
                            fit.competition,
                            fit.season_label,
                            fit.fit_index,
                            name,
                            value,
                        ]
                    )
    return path


def _log_fitted_parameters(tracker: tracking.Tracker, runs: Sequence[ModelRun]) -> None:
    """The parameters of the first fit of each league as MLflow params, the rest as an artifact.

    MLflow params are scalars and a run refitting every matchweek would produce hundreds, so
    only the fit that opens each league's earliest test season goes there; the artifact carries
    every fit.
    """
    opening: dict[tuple[str, str], FittedParameters] = {}
    for run in runs:
        for fit in run.parameters:
            key = (fit.model_name, fit.competition)
            current = opening.get(key)
            if current is None or (fit.season_label, fit.fit_index) < (
                current.season_label,
                current.fit_index,
            ):
                opening[key] = fit
    if not opening:
        return
    tracker.log_params(
        {
            f"{model_name}.{competition}.{name}": value
            for (model_name, competition), fit in sorted(opening.items())
            for name, value in sorted(fit.values.items())
        }
    )


def _log_metrics(
    tracker: tracking.Tracker,
    scored: Mapping[str, Sequence[ScopedMetrics]],
    comparisons: Sequence[Comparison],
) -> None:
    metrics: dict[str, float] = {}
    for model_name, slices in scored.items():
        for entry in slices:
            for metric, value in entry.metrics.as_mapping().items():
                key = tracking.metric_key(model_name, metric, entry.competition, entry.season_label)
                metrics[key] = value
    for comparison in comparisons:
        scope = (comparison.competition, comparison.season_label)
        for metric, value in (
            ("rps_gap", comparison.difference),
            ("rps_gap_low", comparison.low),
            ("rps_gap_high", comparison.high),
        ):
            metrics[tracking.metric_key(comparison.model_name, metric, *scope)] = value
    tracker.log_metrics(metrics)


def _log_artifacts(
    tracker: tracking.Tracker,
    settings: Settings,
    runs: Sequence[ModelRun],
    scored: Mapping[str, Sequence[ScopedMetrics]],
) -> None:
    """Reliability diagrams per model and league, plus every prediction behind them."""
    bins = settings.backtest.metrics.ece_bins
    with TemporaryDirectory() as workspace:
        directory = Path(workspace)
        for model_name, slices in scored.items():
            for entry in slices:
                # One season is too thin a slice for a reliability curve to say anything.
                if entry.scope == SEASON:
                    continue
                probabilities, outcomes = entry.arrays()
                for path in tracking.reliability_artifacts(
                    directory,
                    model_name=model_name,
                    scope=entry.competition or POOLED,
                    probabilities=probabilities,
                    outcomes=outcomes,
                    bins=bins,
                ):
                    tracker.log_artifact(path, artifact_path="reliability")
        tracker.log_artifact(_predictions_csv(directory, runs), artifact_path="predictions")
        if any(run.parameters for run in runs):
            tracker.log_artifact(_parameters_csv(directory, runs), artifact_path="parameters")


def _predictions_csv(directory: Path, runs: Sequence[ModelRun]) -> Path:
    path = directory / "predictions.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_PREDICTION_HEADER)
        for run in runs:
            for prediction in run.predictions:
                writer.writerow(
                    [
                        run.model_name,
                        prediction.match_id,
                        prediction.competition,
                        prediction.season_label,
                        prediction.kickoff_utc.isoformat(),
                        prediction.as_of_utc.isoformat(),
                        prediction.p_home,
                        prediction.p_draw,
                        prediction.p_away,
                        prediction.outcome,
                    ]
                )
    return path


def _records(
    rows: Sequence[ReportRow],
    *,
    run_id: str,
    feature_version: str,
    reference_model: str,
    competitions: Mapping[str, int],
    seasons: Mapping[tuple[int, str], int],
) -> tuple[Mapping[str, object], ...]:
    """The rows of `backtest_results`, with competitions and seasons resolved to their ids."""
    sha = tracking.git_sha()
    records: list[Mapping[str, object]] = []
    for row in rows:
        competition_id = None if row.competition is None else competitions[row.competition]
        season_id = (
            seasons[(competition_id, row.season_label)]
            if row.season_label is not None and competition_id is not None
            else None
        )
        records.append(
            {
                "run_id": run_id,
                "model_name": row.model_name,
                "scope": row.scope,
                "competition_id": competition_id,
                "season_id": season_id,
                "feature_version": feature_version,
                "git_sha": sha,
                "n": row.n,
                "rps": row.rps,
                "log_loss": row.log_loss,
                "brier": row.brier,
                "ece": row.ece,
                "reference_model": None if row.rps_gap is None else reference_model,
                "rps_gap": row.rps_gap,
                "rps_gap_ci_low": row.rps_gap_ci_low,
                "rps_gap_ci_high": row.rps_gap_ci_high,
            }
        )
    return tuple(records)
