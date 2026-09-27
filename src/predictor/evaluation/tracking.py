"""MLflow tracking for the walk-forward backtest (spec sections 8 and 11, phase 4).

One MLflow run holds a whole invocation: its params say what was run against which feature
version and which commit, its metrics carry every model's numbers per season, per league and
pooled, and its artifacts carry the reliability diagrams and the predictions behind them.

Nothing here is optional or best-effort. A backtest whose numbers were not recorded is a
backtest nobody can check, so an unreachable tracking server is an error with a clear message
rather than a warning in a log nobody reads.
"""

from __future__ import annotations

import csv
import os
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# A backtest runs in a container and in CI, where there is no display to draw on.
matplotlib.use("Agg")

# The client prints an unsolicited hint about a tracing skill on import, which would land in
# the middle of this CLI's own output; the library documents this variable to silence it.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
from matplotlib import pyplot as plt

from predictor.config import Settings
from predictor.evaluation.metrics import OUTCOMES, Reliability, reliability_diagrams
from predictor.logging import get_logger

logger = get_logger(__name__)

#: Environment variable that carries the commit when git itself is not in the image.
GIT_SHA_ENV = "GIT_SHA"

#: What is recorded when neither the environment nor git can name the commit.
UNKNOWN_SHA = "unknown"

#: Human names of the outcome codes, for legends and CSV rows.
OUTCOME_NAMES = ("home", "draw", "away")

_CSV_HEADER = ("outcome", "bin_low", "bin_high", "n", "mean_predicted", "observed")

#: Marker area in points: the floor keeps an empty-ish bin visible, the range shows the weight.
MIN_MARKER = 6.0
MAX_MARKER = 70.0


def git_sha() -> str:
    """The commit the backtest ran from: the environment first, then git, then `unknown`."""
    from_environment = os.environ.get(GIT_SHA_ENV)
    if from_environment:
        return from_environment.strip()
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - resolved from PATH on purpose
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN_SHA
    return completed.stdout.strip() if completed.returncode == 0 else UNKNOWN_SHA


@dataclass(frozen=True, slots=True)
class Tracker:
    """The open MLflow run, with the few operations the backtest needs."""

    run_id: str
    tracking_uri: str

    def log_params(self, params: Mapping[str, object]) -> None:
        mlflow.log_params({key: str(value) for key, value in params.items()})

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        mlflow.log_metrics(dict(metrics))

    def log_artifact(self, path: Path, artifact_path: str) -> None:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


@contextmanager
def tracked_run(settings: Settings, *, run_name: str) -> Iterator[Tracker]:
    """Open one MLflow run against the configured tracking server."""
    uri = settings.mlflow.tracking_uri
    mlflow.set_tracking_uri(uri)
    try:
        mlflow.set_experiment(settings.backtest.experiment)
        with mlflow.start_run(run_name=run_name) as run:
            logger.info("backtest.tracking", tracking_uri=uri, run_id=run.info.run_id)
            yield Tracker(run_id=run.info.run_id, tracking_uri=uri)
    except mlflow.exceptions.MlflowException as exc:
        msg = (
            f"MLflow at {uri} could not record this backtest ({exc}); start the stack with "
            "`make up` or point PREDICTOR_MLFLOW__TRACKING_URI somewhere reachable"
        )
        raise RuntimeError(msg) from exc


def metric_key(model_name: str, metric: str, competition: str | None, season: str | None) -> str:
    """The MLflow metric name of one number: `model.metric_LEAGUE_SEASON`, pooled when absent."""
    scope = "_".join(part for part in (competition, season) if part) or "pooled"
    return f"{model_name}.{metric}_{scope}"


def reliability_artifacts(
    directory: Path,
    *,
    model_name: str,
    scope: str,
    probabilities: object,
    outcomes: object,
    bins: int,
) -> tuple[Path, ...]:
    """Write one reliability diagram per outcome as a PNG, plus the bins behind it as a CSV."""
    diagrams = reliability_diagrams(probabilities, outcomes, bins=bins)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name}_{scope}"
    return (
        _write_csv(directory / f"{stem}.csv", diagrams),
        _write_plot(directory / f"{stem}.png", diagrams, title=f"{model_name} — {scope}"),
    )


def _write_csv(path: Path, diagrams: Sequence[Reliability]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_CSV_HEADER)
        for diagram in diagrams:
            for outcome, low, high, count, predicted, observed in diagram.as_rows():
                writer.writerow(
                    [
                        OUTCOME_NAMES[outcome],
                        f"{low:.4f}",
                        f"{high:.4f}",
                        count,
                        predicted,
                        observed,
                    ]
                )
    return path


def _write_plot(path: Path, diagrams: Sequence[Reliability], *, title: str) -> Path:
    figure, axes = plt.subplots(figsize=(5.0, 5.0))
    axes.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, color="grey")
    for diagram in diagrams:
        line = axes.plot(
            diagram.mean_predicted,
            diagram.observed,
            linewidth=1.0,
            label=OUTCOME_NAMES[diagram.outcome],
        )[0]
        # Marker area follows the bin's share of the sample: a bin holding three matches
        # should not look as trustworthy as one holding eight hundred.
        busiest = max(diagram.counts) or 1
        axes.scatter(
            diagram.mean_predicted,
            diagram.observed,
            s=[MIN_MARKER + MAX_MARKER * count / busiest for count in diagram.counts],
            color=line.get_color(),
            zorder=3,
        )
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.set_xlabel("predicted probability")
    axes.set_ylabel("observed frequency")
    axes.set_title(title)
    axes.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def outcome_name(outcome: int) -> str:
    """The readable name of an outcome code, for legends and reports."""
    return OUTCOME_NAMES[OUTCOMES.index(outcome)]
