"""The model comparison table and the fitted parameters (`make report`).

The rows come from `backtest_results` and `model_parameters`, so the report is a view of what
was stored rather than a second computation: the numbers in the table are the numbers in
MLflow, by construction.

Phase 5 added the second table. The spec fits the home advantage, rho and the time decay per
league (section 7.1) and asks for them alongside the metrics; a model that reports nothing --
a baseline -- simply does not appear there.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
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

#: The identity columns of a fitted-parameter table; the rest are the parameters themselves.
PARAMETER_HEADERS = ("league", "season", "fit")

#: Parameters the spec asks for by name come first, in the order it names them. Anything else
#: a model chooses to report follows alphabetically, so the column order is always stable.
PARAMETER_ORDER = (
    "home_advantage",
    "rho",
    "xi",
    "intercept",
    "slope",
    "lower_cutpoint",
    "upper_cutpoint",
    "sigma_attack",
    "sigma_defence",
)

#: Parameters that count something rather than measure it, so they print without decimals.
COUNT_PARAMETERS = frozenset({"teams", "matches", "draws", "converged"})


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


@dataclass(frozen=True, slots=True)
class ParameterRow:
    """What one fit learned, exactly as `model_parameters` holds it."""

    model_name: str
    competition: str
    season_label: str | None
    fit_index: int
    values: Mapping[str, float]


def parameter_columns(rows: Sequence[ParameterRow]) -> tuple[str, ...]:
    """The parameter names one model reports, the ones the spec names first."""
    seen = {name for row in rows for name in row.values}
    preferred = [name for name in PARAMETER_ORDER if name in seen]
    return (*preferred, *sorted(seen - set(preferred)))


def ordered_parameters(rows: Sequence[ParameterRow]) -> tuple[ParameterRow, ...]:
    """League, then season, then the refits inside it, which is how a reader scans them."""
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.model_name,
                row.competition,
                row.season_label or "",
                row.fit_index,
            ),
        )
    )


def by_model(rows: Sequence[ParameterRow]) -> tuple[tuple[str, tuple[ParameterRow, ...]], ...]:
    """The parameter rows grouped into one table per model, in model-name order."""
    grouped: dict[str, list[ParameterRow]] = {}
    for row in ordered_parameters(rows):
        grouped.setdefault(row.model_name, []).append(row)
    return tuple((name, tuple(grouped[name])) for name in sorted(grouped))


def _parameter_cells(row: ParameterRow, columns: Sequence[str]) -> tuple[str, ...]:
    """One row of a fitted-parameter table: the fold it belongs to, then its numbers."""
    return (
        row.competition,
        row.season_label or ABSENT,
        str(row.fit_index),
        *(_parameter(column, row.values.get(column)) for column in columns),
    )


def _parameter(name: str, value: float | None) -> str:
    """A fitted value, formatted by what it is rather than by what it happens to equal.

    A count prints as a count. Everything else gets the usual four decimals even when it lands
    on a round number, so a column of decays reads `1.0000` and `0.5000` rather than `1` and
    `0.5000`.
    """
    if value is None:
        return ABSENT
    if name in COUNT_PARAMETERS:
        return str(int(value))
    return f"{value:.{DECIMALS}f}"


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


def render_markdown(
    rows: Sequence[ReportRow],
    *,
    reference_model: str,
    parameters: Sequence[ParameterRow] = (),
) -> str:
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
    for name, fits in by_model(parameters):
        columns = parameter_columns(fits)
        headers = (*PARAMETER_HEADERS, *columns)
        lines.extend(
            [
                "",
                f"## Fitted parameters: `{name}`",
                "",
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join("---" for _ in headers) + " |",
            ]
        )
        lines.extend("| " + " | ".join(_parameter_cells(row, columns)) + " |" for row in fits)
    return "\n".join(lines) + "\n"


def render_html(
    rows: Sequence[ReportRow],
    *,
    reference_model: str,
    parameters: Sequence[ParameterRow] = (),
) -> str:
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
    sections = "".join(_html_parameters(name, fits) for name, fits in by_model(parameters))
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
        f"{sections}"
        "</body>\n</html>\n"
    )


def _html_parameters(name: str, fits: Sequence[ParameterRow]) -> str:
    """One fitted-parameter table as HTML, matching the markdown one column for column."""
    columns = parameter_columns(fits)
    headers = (*PARAMETER_HEADERS, *columns)
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(cell)}</td>" for cell in _parameter_cells(row, columns))
        + "</tr>"
        for row in fits
    )
    return (
        f"<h2>Fitted parameters: {html.escape(name)}</h2>\n"
        f"<table>\n<thead><tr>{head}</tr></thead>\n<tbody>{body}</tbody>\n</table>\n"
    )


def write_report(
    rows: Sequence[ReportRow],
    *,
    reference_model: str,
    parameters: Sequence[ParameterRow] = (),
    settings: Settings | None = None,
) -> tuple[Path, ...]:
    """Write exactly the configured formats into the configured directory."""
    resolved = settings or get_settings()
    report = resolved.backtest.report
    directory = Path(report.directory)
    directory.mkdir(parents=True, exist_ok=True)
    renderers = {"markdown": render_markdown, "html": render_html}
    written: list[Path] = []
    for fmt in report.formats:
        path = directory / FILENAMES[fmt]
        text = renderers[fmt](rows, reference_model=reference_model, parameters=parameters)
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
