"""The model comparison table (spec section 11, phase 4: `make report`).

The rows come from `backtest_results`, so the report is a view of what was stored rather than a
second computation: the numbers in the table are the numbers in MLflow, by construction.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from predictor.config import Settings, get_settings
from predictor.evaluation.backtest import COMPETITION, POOLED, SEASON

#: Filenames written into `backtest.report.directory`.
FILENAMES = {"markdown": "backtest.md", "html": "backtest.html"}

#: The order scopes are listed in: coarsest first.
SCOPE_ORDER = {POOLED: 0, COMPETITION: 1, SEASON: 2}

HEADERS = (
    "model",
    "scope",
    "league",
    "season",
    "n",
    "RPS",
    "log loss",
    "Brier",
    "ECE",
    "RPS gap",
    "95% CI",
)

#: Metrics are reported to four decimals; nothing in this domain is meaningful beyond that.
DECIMALS = 4

ABSENT = "-"


@dataclass(frozen=True, slots=True)
class ReportRow:
    """One scored slice, exactly as `backtest_results` holds it."""

    model_name: str
    scope: str
    competition: str | None
    season_label: str | None
    n: int
    rps: float
    log_loss: float
    brier: float
    ece: float
    rps_gap: float | None = None
    rps_gap_ci_low: float | None = None
    rps_gap_ci_high: float | None = None

    def cells(self) -> tuple[str, ...]:
        return (
            self.model_name,
            self.scope,
            self.competition or ABSENT,
            self.season_label or ABSENT,
            str(self.n),
            _number(self.rps),
            _number(self.log_loss),
            _number(self.brier),
            _number(self.ece),
            _signed(self.rps_gap),
            _interval(self.rps_gap_ci_low, self.rps_gap_ci_high),
        )


def ordered(rows: Sequence[ReportRow]) -> tuple[ReportRow, ...]:
    """Pooled rows first, then per league, then per season, with a stable order inside each."""
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                SCOPE_ORDER.get(row.scope, len(SCOPE_ORDER)),
                row.competition or "",
                row.season_label or "",
                row.model_name,
            ),
        )
    )


def render_markdown(rows: Sequence[ReportRow], *, reference_model: str) -> str:
    """The comparison table as markdown, with the gap column explained underneath the title."""
    prepared = _checked(rows)
    lines = [
        "# Walk-forward backtest",
        "",
        f"RPS gap is a model's RPS minus that of `{reference_model}` on the matches both "
        "predicted; the interval is a paired bootstrap, and a model is only better when it "
        "excludes zero.",
        "",
        "| " + " | ".join(HEADERS) + " |",
        "| " + " | ".join("---" for _ in HEADERS) + " |",
    ]
    lines.extend("| " + " | ".join(row.cells()) + " |" for row in prepared)
    return "\n".join(lines) + "\n"


def render_html(rows: Sequence[ReportRow], *, reference_model: str) -> str:
    """The same table as a standalone HTML page, for opening straight from the file system."""
    prepared = _checked(rows)
    head = "".join(f"<th>{html.escape(header)}</th>" for header in HEADERS)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row.cells()) + "</tr>"
        for row in prepared
    )
    note = (
        f"RPS gap is a model's RPS minus that of {html.escape(reference_model)} on the matches "
        "both predicted; a model is only better when the bootstrap interval excludes zero."
    )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        "<title>Walk-forward backtest</title>\n"
        "<style>\n"
        "body { font-family: system-ui, sans-serif; margin: 2rem; }\n"
        "table { border-collapse: collapse; }\n"
        "th, td { border: 1px solid #ccc; padding: 0.3rem 0.6rem; text-align: right; }\n"
        "th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }\n"
        "</style>\n</head>\n<body>\n"
        "<h1>Walk-forward backtest</h1>\n"
        f"<p>{note}</p>\n"
        f"<table>\n<thead><tr>{head}</tr></thead>\n<tbody>{body}</tbody>\n</table>\n"
        "</body>\n</html>\n"
    )


def write_report(
    rows: Sequence[ReportRow], *, reference_model: str, settings: Settings | None = None
) -> tuple[Path, ...]:
    """Write exactly the configured formats into the configured directory."""
    resolved = settings or get_settings()
    report = resolved.backtest.report
    directory = Path(report.directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in report.formats:
        path = directory / FILENAMES[fmt]
        text = (
            render_markdown(rows, reference_model=reference_model)
            if fmt == "markdown"
            else render_html(rows, reference_model=reference_model)
        )
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return tuple(written)


def _checked(rows: Sequence[ReportRow]) -> tuple[ReportRow, ...]:
    if not rows:
        msg = "there are no backtest results to report; run `make backtest` first"
        raise ValueError(msg)
    return ordered(rows)


def _number(value: float) -> str:
    return f"{value:.{DECIMALS}f}"


def _signed(value: float | None) -> str:
    return ABSENT if value is None else f"{value:+.{DECIMALS}f}"


def _interval(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return ABSENT
    return f"[{low:+.{DECIMALS}f}, {high:+.{DECIMALS}f}]"
