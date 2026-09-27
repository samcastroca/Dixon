"""Typer CLI. Every pipeline command is a stub until its phase implements it."""

from __future__ import annotations

from typing import Annotated

import typer

from predictor.config import Competition, Settings, get_settings
from predictor.evaluation.report import (
    COUNT_PARAMETERS,
)
from predictor.evaluation.report import (
    by_model as _report_parameters,
)
from predictor.evaluation.report import (
    ordered as _report_order,
)
from predictor.evaluation.report import (
    parameter_columns as _parameter_columns,
)
from predictor.evaluation.service import run_backtest, write_stored_report
from predictor.features.service import build_features as run_features
from predictor.ingestion.football_data_uk import FootballDataUkSource
from predictor.ingestion.seasons import parse_seasons
from predictor.ingestion.service import ingest as run_ingest
from predictor.logging import bind_run_id, configure_logging, get_logger
from predictor.models.registry import MODEL_NAMES, SLOW_MODELS
from predictor.processing.service import process as run_process

app = typer.Typer(help="Sports results prediction platform", no_args_is_help=True)
config_app = typer.Typer(help="Inspect the loaded configuration", no_args_is_help=True)
app.add_typer(config_app, name="config")

logger = get_logger(__name__)

#: command name -> the phase (spec section 11) that implements it
PLANNED_COMMANDS: dict[str, int] = {
    "train": 6,
    "predict": 7,
}


def _selected_competitions(settings: Settings, competition: str | None) -> list[Competition]:
    """Resolve a comma-separated list of codes, defaulting to every configured competition."""
    codes = (
        [code.strip().upper() for code in competition.split(",") if code.strip()]
        if competition
        else list(settings.competition_codes)
    )
    return [settings.competition(code) for code in codes]


def _not_yet(command: str) -> None:
    phase = PLANNED_COMMANDS[command]
    typer.echo(f"`{command}` is not implemented yet: it is delivered in phase {phase}.", err=True)
    raise typer.Exit(code=1)


@app.callback()
def main() -> None:
    """Configure logging once for every command."""
    configure_logging()
    bind_run_id()


@app.command()
def ingest(
    source: Annotated[
        str | None, typer.Option(help="Source name as configured in settings.yaml")
    ] = None,
    competition: Annotated[
        str | None, typer.Option(help="Comma-separated competition codes; default: all")
    ] = None,
    seasons: Annotated[
        str | None, typer.Option(help="Range like 2014-2025, or a list like 2014-15,2020-21")
    ] = None,
) -> None:
    """Fetch season files from a source and store them raw plus staged."""
    settings = get_settings()
    source_name = source or settings.ingestion.default_source
    if source_name != FootballDataUkSource.SOURCE_KEY:
        typer.echo(f"unknown source {source_name!r}; only football_data_uk exists", err=True)
        raise typer.Exit(code=2)

    competitions = _selected_competitions(settings, competition)
    season_labels = parse_seasons(seasons or settings.ingestion.default_seasons)

    fetcher = FootballDataUkSource(settings)
    try:
        report = run_ingest(fetcher, competitions, season_labels)
    finally:
        fetcher.close()

    typer.echo(f"{'COMPETITION':<12}{'SEASON':<10}{'ROWS':>6}  {'NEW':>4}  RESOURCE")
    for entry in report.entries:
        typer.echo(
            f"{entry.competition:<12}{entry.season:<10}{entry.staged_rows:>6}  "
            f"{'yes' if entry.raw_inserted else 'no':>4}  {entry.resource}"
        )
    typer.echo(
        f"{len(report.entries)} resources, {report.raw_inserted} new payloads, "
        f"{report.staged_rows} staged rows"
    )


@app.command()
def process(
    competition: Annotated[
        str | None, typer.Option(help="Comma-separated competition codes; default: all")
    ] = None,
    seasons: Annotated[
        str | None,
        typer.Option(help="Range like 2014-2025, or a list; default: every staged season"),
    ] = None,
) -> None:
    """Clean and validate the staged rows into the matches, stats and odds tables."""
    settings = get_settings()
    competitions = _selected_competitions(settings, competition)
    season_labels = parse_seasons(seasons) if seasons else None

    report = run_process(competitions, season_labels)

    typer.echo(f"{'COMPETITION':<12}{'SEASON':<10}{'MATCHES':>8}{'STATS':>8}{'ODDS':>8}")
    for entry in report.entries:
        typer.echo(
            f"{entry.competition:<12}{entry.season:<10}{entry.matches:>8}"
            f"{entry.match_stats:>8}{entry.odds:>8}"
        )
    typer.echo(
        f"{len(report.entries)} seasons, {report.matches} matches, "
        f"{report.match_stats} stat rows, {report.odds} odds rows"
    )
    for code, names in report.teams.items():
        typer.echo(f"\n{code} ({len(names)} canonical teams): {', '.join(names)}")


#: How many of the strongest and weakest ratings the demo output prints per competition.
RATINGS_SHOWN = 5


def _fitted(name: str, value: float | None) -> str:
    """One fitted parameter for the backtest table, formatted as the report formats it."""
    if value is None:
        return "-"
    return str(int(value)) if name in COUNT_PARAMETERS else format(value, ".4f")


@app.command()
def features(
    version: Annotated[
        str | None, typer.Option(help="Feature version label; default: features.version")
    ] = None,
    competition: Annotated[
        str | None, typer.Option(help="Comma-separated competition codes; default: all")
    ] = None,
    seasons: Annotated[
        str | None,
        typer.Option(help="Seasons to write; the replay always starts at the first one on record"),
    ] = None,
) -> None:
    """Build the point-in-time feature store from the clean match tables."""
    settings = get_settings()
    competitions = _selected_competitions(settings, competition)
    season_labels = parse_seasons(seasons) if seasons else None

    report = run_features(competitions, version=version, seasons=season_labels)

    typer.echo(f"version {report.version}  {report.feature_count} features per row")
    typer.echo(f"definitions {report.definition_checksum}")
    typer.echo("")
    typer.echo(f"{'COMPETITION':<12}{'SEASONS':>8}{'MATCHES':>9}{'ROWS':>7}")
    for entry in report.entries:
        typer.echo(f"{entry.competition:<12}{entry.seasons:>8}{entry.matches:>9}{entry.rows:>7}")
    typer.echo(f"{len(report.entries)} competitions, {report.rows} feature rows")

    for code, ranking in report.ratings.items():
        typer.echo("")
        typer.echo(f"{code} Elo today: top and bottom {RATINGS_SHOWN} of {len(ranking)}")
        for name, rating in ranking[:RATINGS_SHOWN]:
            typer.echo(f"  {rating:8.1f}  {name}")
        typer.echo("  ...")
        for name, rating in ranking[-RATINGS_SHOWN:]:
            typer.echo(f"  {rating:8.1f}  {name}")

    for code, sample in report.samples.items():
        typer.echo("")
        typer.echo(
            f"{code} sample: match {sample.match_id}, "
            f"kickoff {sample.kickoff_utc.isoformat()}, "
            f"as_of {sample.as_of_utc.isoformat()}"
        )
        for name, value in sample.values.items():
            typer.echo(f"  {name:<40}{'-' if value is None else format(value, '.6g')}")


def _selected_models(models: str | None) -> list[str]:
    """Resolve a comma-separated list of model names, defaulting to every registered one."""
    if not models:
        return list(MODEL_NAMES)
    names = [name.strip() for name in models.split(",") if name.strip()]
    unknown = [name for name in names if name not in MODEL_NAMES]
    if unknown:
        typer.echo(
            f"unknown model(s) {', '.join(unknown)}; registered: {', '.join(MODEL_NAMES)}", err=True
        )
        raise typer.Exit(code=2)
    return names


@app.command()
def backtest(
    model: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated model names; default: every registered model, which is what "
            "the RPS gap against the market benchmark needs. That default now includes "
            f"{' and '.join(SLOW_MODELS)}, which sample and search rather than solve: name the "
            "models explicitly to keep a run short"
        ),
    ] = None,
    seasons: Annotated[
        str | None,
        typer.Option(help="Test seasons, like 2019-2025; default: backtest.default_seasons"),
    ] = None,
    competition: Annotated[
        str | None, typer.Option(help="Comma-separated competition codes; default: all")
    ] = None,
    version: Annotated[
        str | None, typer.Option(help="Feature version to read; default: backtest.feature_version")
    ] = None,
) -> None:
    """Run the walk-forward backtest and record it in MLflow and backtest_results."""
    settings = get_settings()
    names = _selected_models(model)
    competitions = _selected_competitions(settings, competition)
    season_labels = parse_seasons(seasons or settings.backtest.default_seasons)

    outcome = run_backtest(names, competitions, seasons=season_labels, version=version)

    typer.echo(
        f"run {outcome.run_id}  features {outcome.feature_version}  "
        f"seasons {season_labels[0]}..{season_labels[-1]}"
    )
    typer.echo("")
    typer.echo(
        f"{'MODEL':<22}{'SCOPE':<12}{'LEAGUE':<8}{'SEASON':<9}{'N':>6}"
        f"{'RPS':>9}{'LOGLOSS':>9}{'BRIER':>9}{'ECE':>8}{'GAP':>9}  95% CI"
    )
    for row in _report_order(outcome.rows):
        interval = (
            ""
            if row.rps_gap_ci_low is None or row.rps_gap_ci_high is None
            else f"  [{row.rps_gap_ci_low:+.4f}, {row.rps_gap_ci_high:+.4f}]"
        )
        typer.echo(
            f"{row.model_name:<22}{row.scope:<12}{row.competition or '-':<8}"
            f"{row.season_label or '-':<9}{row.n:>6}{row.rps:>9.4f}{row.log_loss:>9.4f}"
            f"{row.brier:>9.4f}{row.ece:>8.4f}"
            f"{'' if row.rps_gap is None else format(row.rps_gap, '>+9.4f')}{interval}"
        )
    typer.echo("")
    for comparison in outcome.comparisons:
        if comparison.scope != "competition":
            continue
        verdict = "significant" if comparison.excludes_zero else "not significant"
        typer.echo(
            f"{comparison.model_name} vs {comparison.reference_model} in {comparison.competition}: "
            f"RPS {comparison.difference:+.4f} "
            f"[{comparison.low:+.4f}, {comparison.high:+.4f}] on {comparison.n} matches, {verdict}"
        )
    for name, fits in _report_parameters(outcome.parameters):
        columns = _parameter_columns(fits)
        typer.echo("")
        typer.echo(f"{name} fitted parameters")
        typer.echo(
            f"{'LEAGUE':<8}{'SEASON':<9}{'FIT':>4}  "
            + "".join(f"{column:>18}" for column in columns)
        )
        for fit in fits:
            typer.echo(
                f"{fit.competition:<8}{fit.season_label or '-':<9}{fit.fit_index:>4}  "
                + "".join(f"{_fitted(name, fit.values.get(name)):>18}" for name in columns)
            )

    typer.echo("")
    typer.echo(
        f"{outcome.stored} rows mirrored into backtest_results, "
        f"{outcome.stored_parameters} into model_parameters"
    )


@app.command()
def report() -> None:
    """Write the model comparison table from the stored backtest results."""
    written = write_stored_report()
    for path in written:
        typer.echo(f"wrote {path}")


@app.command()
def train() -> None:
    """Fit models and log them to MLflow (phase 6)."""
    _not_yet("train")


@app.command()
def predict() -> None:
    """Write predictions for upcoming fixtures (phase 7)."""
    _not_yet("predict")


@config_app.command("show")
def config_show() -> None:
    """Print the configured competitions."""
    settings = get_settings()
    for competition in settings.competitions:
        sources = ", ".join(f"{name}={code}" for name, code in sorted(competition.sources.items()))
        typer.echo(
            f"{competition.code}  {competition.name}  sport={competition.sport}  "
            f"tz={competition.timezone}  [{sources}]"
        )
