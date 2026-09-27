"""The walk-forward backtest against the compose database (spec section 11, phase 4).

The two baselines are run once for the whole module over the seasons ingestion has staged.
MLflow is pointed at a SQLite file inside the test's own directory, so nothing here touches
the network or the real tracking server.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import mlflow
import pytest
from sqlalchemy import func, select

from predictor import tables
from predictor.config import Competition, MlflowSettings, Settings, get_settings
from predictor.db import session_scope
from predictor.evaluation import repository
from predictor.evaluation.service import BacktestOutcome, run_backtest
from predictor.ingestion.seasons import parse_seasons

pytestmark = pytest.mark.integration

COMPETITIONS = ("EPL", "LALIGA")
MODELS = ("baseline_market", "baseline_frequency")
SEASONS = "2019-2025"

#: The spec asks for five or more test seasons per league.
MIN_TEST_SEASONS = 5


@pytest.fixture(autouse=True)
def _needs_database(database_available: bool) -> Iterator[None]:
    if not database_available:
        pytest.skip("no database reachable; run `make up` or set DATABASE_URL")
    yield


@pytest.fixture(scope="module")
def competitions() -> list[Competition]:
    settings = get_settings()
    return [settings.competition(code) for code in COMPETITIONS]


def tracked_settings(workspace: Path) -> Settings:
    """Settings whose MLflow lives in a SQLite file: no server, no network."""
    settings = get_settings()
    uri = f"sqlite:///{(workspace / 'mlflow.db').as_posix()}"
    return settings.model_copy(update={"mlflow": MlflowSettings(tracking_uri=uri)})


@pytest.fixture(scope="module")
def outcome(
    database_available: bool, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[BacktestOutcome]:
    """Run both baselines once; MLflow artifacts land in the temporary working directory."""
    if not database_available:
        pytest.skip("no database reachable; run `make up` or set DATABASE_URL")
    workspace = tmp_path_factory.mktemp("backtest")
    settings = tracked_settings(workspace)
    previous = Path.cwd()
    os.chdir(workspace)
    try:
        yield run_backtest(
            MODELS,
            [settings.competition(code) for code in COMPETITIONS],
            seasons=parse_seasons(SEASONS),
            settings=settings,
        )
    finally:
        os.chdir(previous)
        mlflow.set_tracking_uri(get_settings().mlflow.tracking_uri)


@pytest.mark.slow
def test_both_baselines_run_for_at_least_five_test_seasons_in_each_league(
    outcome: BacktestOutcome,
) -> None:
    """Acceptance test 4: the walk-forward covers 5+ test seasons per league, per model."""
    assert {run.model_name for run in outcome.runs} == set(MODELS)

    for run in outcome.runs:
        assert run.predictions
        for code in COMPETITIONS:
            seasons = {
                prediction.season_label
                for prediction in run.predictions
                if prediction.competition == code
            }
            assert len(seasons) >= MIN_TEST_SEASONS, f"{run.model_name} in {code}: {seasons}"


@pytest.mark.slow
def test_the_run_is_logged_to_mlflow_with_its_parameters_metrics_and_plots(
    outcome: BacktestOutcome, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Acceptance test 4: params, per-league and pooled metrics, and the reliability plots."""
    assert outcome.run_id is not None
    client = mlflow.MlflowClient(tracking_uri=outcome.tracking_uri)
    run = client.get_run(outcome.run_id)

    assert run.data.params["models"] == ",".join(MODELS)
    assert run.data.params["feature_version"]
    assert "git_sha" in run.data.params
    assert "baseline_market.rps_pooled" in run.data.metrics
    assert "baseline_frequency.rps_EPL" in run.data.metrics
    assert "baseline_frequency.rps_EPL_2023-24" in run.data.metrics

    artifacts = {entry.path for entry in client.list_artifacts(run.info.run_id, "reliability")}
    assert any(path.endswith(".png") for path in artifacts)
    assert any(path.endswith(".csv") for path in artifacts)


@pytest.mark.slow
def test_the_market_baseline_beats_the_frequency_baseline_in_every_league(
    outcome: BacktestOutcome,
) -> None:
    """Acceptance test 5: if the market does not win here, the plumbing is wrong."""
    for code in COMPETITIONS:
        comparison = next(
            entry
            for entry in outcome.comparisons
            if entry.model_name == "baseline_frequency"
            and entry.scope == "competition"
            and entry.competition == code
        )
        assert comparison.reference_model == "baseline_market"
        assert comparison.difference > 0.0, f"{code}: {comparison}"
        assert comparison.excludes_zero, f"{code}: {comparison}"
        assert comparison.low > 0.0


@pytest.mark.slow
def test_the_fold_boundary_holds_on_the_real_calendar(outcome: BacktestOutcome) -> None:
    """Acceptance test 3, over both leagues and every fold the engine actually ran."""
    for run in outcome.runs:
        assert run.folds
        for fold in run.folds:
            assert fold.latest_train_kickoff is not None
            assert fold.latest_train_kickoff < fold.earliest_test_kickoff


@pytest.mark.slow
def test_every_prediction_is_stamped_before_kickoff(outcome: BacktestOutcome) -> None:
    for run in outcome.runs:
        for prediction in run.predictions:
            assert prediction.as_of_utc < prediction.kickoff_utc


@pytest.mark.slow
def test_the_metrics_are_mirrored_in_the_backtest_results_table(outcome: BacktestOutcome) -> None:
    with session_scope() as session:
        rows = session.execute(
            select(tables.BacktestResult).where(tables.BacktestResult.run_id == outcome.run_id)
        ).scalars()
        stored = {
            (row.model_name, row.scope, row.competition_id, row.season_id): row for row in rows
        }

    assert len(stored) == outcome.stored
    pooled = [row for key, row in stored.items() if key[1] == "pooled"]
    assert len(pooled) == len(MODELS)
    for row in pooled:
        assert row.n > 0
        assert 0.0 < row.rps < 1.0
        assert row.feature_version


@pytest.mark.slow
def test_storing_the_same_results_twice_changes_nothing(outcome: BacktestOutcome) -> None:
    """The table mirrors MLflow, so re-mirroring one run must not duplicate its rows."""
    with session_scope() as session:
        before = session.scalar(
            select(func.count())
            .select_from(tables.BacktestResult)
            .where(tables.BacktestResult.run_id == outcome.run_id)
        )
        repository.upsert_backtest_results(session, outcome.records)
        after = session.scalar(
            select(func.count())
            .select_from(tables.BacktestResult)
            .where(tables.BacktestResult.run_id == outcome.run_id)
        )

    assert before == after == outcome.stored
