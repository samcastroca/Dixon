"""Logs are one JSON object per line and carry the run_id."""

from __future__ import annotations

import json

import pytest
import structlog

from predictor.logging import bind_run_id, configure_logging, get_logger


def test_emits_json_with_run_id(capsys: pytest.CaptureFixture[str]) -> None:
    structlog.contextvars.clear_contextvars()
    configure_logging("INFO")
    run_id = bind_run_id()

    get_logger("test").info("pipeline.started", competition="EPL")

    record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert record["event"] == "pipeline.started"
    assert record["competition"] == "EPL"
    assert record["run_id"] == run_id
    assert record["level"] == "info"
    assert record["timestamp"].endswith("Z")


def test_bind_run_id_accepts_an_explicit_id() -> None:
    structlog.contextvars.clear_contextvars()
    assert bind_run_id("run-42") == "run-42"
