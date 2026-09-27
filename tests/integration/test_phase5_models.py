"""The statistical models against the real leagues (spec section 11, phase 5).

The unit suite answers acceptance tests 4 and 5 on simulated leagues, where the truth is known
and the run takes seconds. This module answers the same two questions the way the spec words
them -- on EPL and La Liga themselves -- which needs the compose database and several minutes,
so every test is marked `slow` and the module skips itself when no database is reachable.

M4 samples a posterior per fold and is therefore left out of this comparison: its own module
checks that it recovers what it was given, and running it here would multiply the runtime of
`make check` by an order of magnitude for a question already answered.

MLflow is pointed at a SQLite file inside the test's own directory, so nothing here touches the
network or the real tracking server.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import mlflow
import pytest
from sqlalchemy import select

from predictor import tables
from predictor.config import MlflowSettings, Settings, get_settings
from predictor.db import session_scope
from predictor.evaluation.service import BacktestOutcome, Comparison, compare, run_backtest
from predictor.ingestion.seasons import parse_seasons

pytestmark = [pytest.mark.integration, pytest.mark.slow]

COMPETITIONS = ("EPL", "LALIGA")

#: The floor of acceptance test 4, plus the three models measured against it.
BASELINE = "baseline_frequency"
MODELS = ("elo", "poisson", "dixon_coles")

SEASONS = "2019-2025"

#: The parameters the spec asks to see per league (section 7.1).
PER_LEAGUE_PARAMETERS = {"elo": "slope", "poisson": "home_advantage", "dixon_coles": "xi"}


@pytest.fixture(autouse=True)
def _needs_database(database_available: bool) -> Iterator[None]:
    if not database_available:
        pytest.skip("no database reachable; run `make up` or set DATABASE_URL")
    yield


def tracked_settings(workspace: Path) -> Settings:
    """Settings whose MLflow lives in a SQLite file: no server, no network."""
    settings = get_settings()
    uri = f"sqlite:///{(workspace / 'mlflow.db').as_posix()}"
    return settings.model_copy(update={"mlflow": MlflowSettings(tracking_uri=uri)})


@pytest.fixture(scope="module")
def outcome(
    database_available: bool, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[BacktestOutcome]:
    """One walk-forward over the baseline and the three models, shared by every test here."""
    if not database_available:
        pytest.skip("no database reachable; run `make up` or set DATABASE_URL")
    workspace = tmp_path_factory.mktemp("phase5")
    settings = tracked_settings(workspace)
    previous = Path.cwd()
    os.chdir(workspace)
    try:
        yield run_backtest(
            (BASELINE, *MODELS),
            [settings.competition(code) for code in COMPETITIONS],
            seasons=parse_seasons(SEASONS),
            settings=settings,
        )
    finally:
        os.chdir(previous)
        mlflow.set_tracking_uri(get_settings().mlflow.tracking_uri)


def _gap(outcome: BacktestOutcome, model: str, competition: str | None) -> Comparison:
    """One model's RPS gap against the frequency baseline at one scope.

    The run's own comparisons are measured against `backtest.reference_model`, which is the
    market. Acceptance test 4 asks about the frequency floor instead, so this recomputes the
    paired interval between the two runs on exactly the matches both predicted.
    """
    settings = get_settings()
    against = settings.model_copy(
        update={"backtest": settings.backtest.model_copy(update={"reference_model": BASELINE})}
    )
    scope = "pooled" if competition is None else "competition"
    comparisons = compare(outcome.runs, outcome.scored, against)
    matching = [
        comparison
        for comparison in comparisons
        if comparison.model_name == model
        and comparison.scope == scope
        and comparison.competition == competition
        and comparison.season_label is None
    ]
    assert matching, f"no {scope} comparison of {model} against {BASELINE}"
    return matching[0]


class TestAgainstTheFrequencyBaseline:
    """Acceptance test 4, on the real leagues."""

    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("competition", COMPETITIONS)
    def test_it_beats_the_floor_with_an_interval_that_excludes_zero(
        self, outcome: BacktestOutcome, model: str, competition: str
    ) -> None:
        gap = _gap(outcome, model, competition)
        # RPS: lower is better, so a model that beats the floor has a negative gap, and the
        # whole interval has to sit below zero for the difference to count.
        assert gap.difference < 0.0, f"{model} in {competition}: gap {gap.difference:+.4f}"
        assert gap.excludes_zero, f"{model} in {competition}: [{gap.low:+.4f}, {gap.high:+.4f}]"


class TestDixonColesAgainstPoisson:
    """Acceptance test 5: Dixon-Coles is not worse than plain Poisson on pooled RPS."""

    def test_the_correction_does_not_cost_anything(self, outcome: BacktestOutcome) -> None:
        corrected = _pooled_rps(outcome, "dixon_coles")
        plain = _pooled_rps(outcome, "poisson")
        assert corrected <= plain, f"dixon_coles {corrected:.4f} vs poisson {plain:.4f}"


def _pooled_rps(outcome: BacktestOutcome, model: str) -> float:
    pooled = [entry for entry in outcome.scored[model] if entry.scope == "pooled"]
    assert pooled, f"{model} produced no pooled slice"
    return pooled[0].metrics.rps


class TestFittedParameters:
    """The parameters the spec asks for reach the database, one row per league and season."""

    def test_every_model_reports_a_fit_for_every_league(self, outcome: BacktestOutcome) -> None:
        for model, parameter in PER_LEAGUE_PARAMETERS.items():
            for competition in COMPETITIONS:
                fits = [
                    row
                    for row in outcome.parameters
                    if row.model_name == model and row.competition == competition
                ]
                assert fits, f"{model} reported nothing for {competition}"
                assert all(parameter in row.values for row in fits)

    def test_the_fits_are_persisted(self, outcome: BacktestOutcome) -> None:
        with session_scope() as session:
            stored = session.execute(
                select(tables.ModelParameters).where(
                    tables.ModelParameters.run_id == outcome.run_id
                )
            ).scalars()
            rows = list(stored)
        assert len(rows) == len(outcome.parameters)
        assert {row.model_name for row in rows} == set(MODELS)

    def test_the_baseline_reports_nothing(self, outcome: BacktestOutcome) -> None:
        # A frequency has no parameters worth a table row, and the engine must not invent any.
        assert not [row for row in outcome.parameters if row.model_name == BASELINE]
