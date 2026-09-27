"""The model comparison table that `make report` writes."""

from __future__ import annotations

from pathlib import Path

import pytest

from predictor.config import BacktestSettings, ReportSettings, Settings, get_settings
from predictor.evaluation.report import ReportRow, render_html, render_markdown, write_report

ROWS = (
    ReportRow(
        model_name="baseline_market",
        scope="pooled",
        competition=None,
        season_label=None,
        n=4560,
        rps=0.1932,
        log_loss=0.9725,
        brier=0.5731,
        ece=0.0181,
        rps_gap=0.0,
        rps_gap_ci_low=0.0,
        rps_gap_ci_high=0.0,
    ),
    ReportRow(
        model_name="baseline_frequency",
        scope="competition",
        competition="EPL",
        season_label=None,
        n=2280,
        rps=0.2214,
        log_loss=1.0421,
        brier=0.6302,
        ece=0.0294,
        rps_gap=0.0282,
        rps_gap_ci_low=0.0221,
        rps_gap_ci_high=0.0344,
    ),
    ReportRow(
        model_name="baseline_frequency",
        scope="season",
        competition="LALIGA",
        season_label="2023-24",
        n=380,
        rps=0.2189,
        log_loss=1.0333,
        brier=0.6240,
        ece=0.0330,
        rps_gap=None,
        rps_gap_ci_low=None,
        rps_gap_ci_high=None,
    ),
)


def report_settings(directory: Path, formats: tuple[str, ...]) -> Settings:
    settings = get_settings()
    report = ReportSettings(directory=str(directory), formats=formats)
    backtest = BacktestSettings(**{**settings.backtest.model_dump(), "report": report})
    return settings.model_copy(update={"backtest": backtest})


def test_the_markdown_table_has_a_row_per_scope_and_a_column_per_metric() -> None:
    text = render_markdown(ROWS, reference_model="baseline_market")

    header, separator, *body = [line for line in text.splitlines() if line.startswith("|")]
    assert "RPS" in header
    assert "log loss" in header
    assert "Brier" in header
    assert "ECE" in header
    assert set(separator) <= set("|- :")
    assert len(body) == len(ROWS)
    assert "0.1932" in body[0]
    assert "2023-24" in body[2]


def test_a_missing_interval_is_shown_as_absent_rather_than_as_zero() -> None:
    text = render_markdown(ROWS, reference_model="baseline_market")

    assert text.splitlines()[-1].endswith("| - |")


def test_the_reference_model_is_named_so_the_gap_column_can_be_read() -> None:
    text = render_markdown(ROWS, reference_model="baseline_market")

    assert "baseline_market" in text
    assert "gap" in text.lower()


def test_pooled_rows_come_before_per_league_and_per_season_rows() -> None:
    shuffled = (ROWS[2], ROWS[1], ROWS[0])
    text = render_markdown(shuffled, reference_model="baseline_market")
    body = [line for line in text.splitlines() if line.startswith("|")][2:]

    assert "pooled" in body[0]
    assert "2023-24" in body[-1]


def test_the_html_table_is_a_document_with_the_same_numbers() -> None:
    markup = render_html(ROWS, reference_model="baseline_market")

    assert markup.lstrip().startswith("<!DOCTYPE html>")
    assert "<table" in markup
    assert markup.count("<tr") == len(ROWS) + 1
    assert "0.2214" in markup


def test_writing_the_report_creates_exactly_the_configured_formats(tmp_path: Path) -> None:
    settings = report_settings(tmp_path / "reports", ("markdown",))

    written = write_report(ROWS, reference_model="baseline_market", settings=settings)

    assert [path.name for path in written] == ["backtest.md"]
    assert (tmp_path / "reports" / "backtest.md").read_text(encoding="utf-8").startswith("#")
    assert not (tmp_path / "reports" / "backtest.html").exists()


def test_writing_both_formats_puts_them_side_by_side(tmp_path: Path) -> None:
    settings = report_settings(tmp_path / "reports", ("markdown", "html"))

    written = write_report(ROWS, reference_model="baseline_market", settings=settings)

    assert sorted(path.name for path in written) == ["backtest.html", "backtest.md"]


def test_a_report_without_rows_says_so_instead_of_writing_an_empty_table() -> None:
    with pytest.raises(ValueError, match="no backtest"):
        render_markdown((), reference_model="baseline_market")
