"""pandera schema for the feature frame, the gate before anything is written.

`as_of_utc < kickoff_utc` crosses two tables, so PostgreSQL cannot hold it as a CHECK. It is
guaranteed by construction in the builder, checked here on the way out, and asserted against
the stored table in the integration tests.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

#: Columns the schema expects besides the features themselves.
KEY_COLUMNS = ("match_id", "kickoff_utc", "as_of_utc")

MAX_REPORTED_FAILURES = 25

INFINITIES = [float("inf"), float("-inf")]


class FeatureValidationError(ValueError):
    """Raised when a feature frame breaks its schema."""


def _cut_off_precedes_kickoff(frame: pd.DataFrame) -> pd.Series:
    return frame["as_of_utc"] < frame["kickoff_utc"]


def _finite(series: pd.Series) -> pd.Series:
    return ~series.isin(INFINITIES)


def features_schema(names: Sequence[str]) -> pa.DataFrameSchema:
    """The schema of a wide feature frame with exactly these feature columns."""
    columns: dict[str, pa.Column] = {
        "match_id": pa.Column("Int64", nullable=False, unique=True),
        "kickoff_utc": pa.Column("datetime64[ns, UTC]", nullable=False),
        "as_of_utc": pa.Column("datetime64[ns, UTC]", nullable=False),
    }
    for name in names:
        columns[name] = pa.Column("Float64", nullable=True, checks=pa.Check(_finite, name="finite"))
    return pa.DataFrameSchema(
        columns=columns,
        checks=pa.Check(_cut_off_precedes_kickoff, name="as_of_utc_before_kickoff_utc"),
        strict=True,
        coerce=True,
        name="features",
    )


def validate_features(frame: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    """Validate a feature frame, raising once with every violation found."""
    try:
        return features_schema(names).validate(frame, lazy=True)
    except SchemaErrors as error:
        raise FeatureValidationError("\n".join(_report(error))) from error


def _report(error: SchemaErrors) -> list[str]:
    """One line per distinct violation, so a single bad row never hides the others."""
    failures = error.failure_cases.fillna({"column": "-"}).copy()
    table_wide = failures["schema_context"] != "Column"
    failures.loc[table_wide, "column"] = "-"
    failures = failures.drop_duplicates(subset=["check", "column", "index"])
    groups = list(failures.groupby(["check", "column"], sort=True))
    lines = [
        f"features: {check} | column={column} | {len(rows)} row(s), "
        f"first at row {rows.iloc[0]['index']} with value {rows.iloc[0]['failure_case']!r}"
        for (check, column), rows in groups[:MAX_REPORTED_FAILURES]
    ]
    if len(groups) > MAX_REPORTED_FAILURES:
        lines.append(f"features: and {len(groups) - MAX_REPORTED_FAILURES} more kinds of violation")
    return lines
