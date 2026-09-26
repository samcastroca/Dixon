"""The CLI exposes the pipeline commands as stubs that name their phase."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from predictor.cli import PLANNED_COMMANDS, app

runner = CliRunner()


@pytest.mark.parametrize(("command", "phase"), sorted(PLANNED_COMMANDS.items()))
def test_pipeline_commands_are_stubs(command: str, phase: int) -> None:
    result = runner.invoke(app, [command])

    assert result.exit_code == 1
    assert f"phase {phase}" in result.output


def test_ingest_is_no_longer_a_stub() -> None:
    assert "ingest" not in PLANNED_COMMANDS

    result = runner.invoke(app, ["ingest", "--help"])

    assert result.exit_code == 0
    assert "--seasons" in result.output


def test_process_is_no_longer_a_stub() -> None:
    assert "process" not in PLANNED_COMMANDS

    result = runner.invoke(app, ["process", "--help"])

    assert result.exit_code == 0
    assert "--competition" in result.output
    assert "--seasons" in result.output


def test_ingest_rejects_an_unknown_source() -> None:
    result = runner.invoke(app, ["ingest", "--source", "api_football"])

    assert result.exit_code == 2
    assert "unknown source" in result.output


def test_config_show_lists_both_competitions() -> None:
    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0
    assert "EPL" in result.output
    assert "football_data_uk=E0" in result.output
    assert "LALIGA" in result.output
    assert "football_data_uk=SP1" in result.output


def test_bare_invocation_shows_help() -> None:
    result = runner.invoke(app, [])

    assert "ingest" in result.output
