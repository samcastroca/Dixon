"""Typer CLI. Every pipeline command is a stub until its phase implements it."""

from __future__ import annotations

import typer

from predictor.config import get_settings
from predictor.logging import bind_run_id, configure_logging, get_logger

app = typer.Typer(help="Sports results prediction platform", no_args_is_help=True)
config_app = typer.Typer(help="Inspect the loaded configuration", no_args_is_help=True)
app.add_typer(config_app, name="config")

logger = get_logger(__name__)

#: command name -> the phase (spec section 11) that implements it
PLANNED_COMMANDS: dict[str, int] = {
    "ingest": 1,
    "process": 2,
    "features": 3,
    "backtest": 4,
    "train": 6,
    "predict": 7,
}


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
def ingest() -> None:
    """Fetch raw source data (phase 1)."""
    _not_yet("ingest")


@app.command()
def process() -> None:
    """Clean and validate raw data (phase 2)."""
    _not_yet("process")


@app.command()
def features() -> None:
    """Build the point-in-time feature store (phase 3)."""
    _not_yet("features")


@app.command()
def backtest() -> None:
    """Run the walk-forward backtest (phase 4)."""
    _not_yet("backtest")


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
