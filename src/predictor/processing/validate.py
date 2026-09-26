"""pandera schemas for the clean tables (spec section 6.2).

Cleaning is permissive on purpose: it parses whatever the source published. This module is the
gate that decides what is allowed into the database, and it reports every violation at once,
naming the table, the column, the row and the offending value.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from predictor.processing.clean import RESULTS, CleanMatch, MatchStatus

#: Columns identifying a fixture before it has a database id.
FIXTURE_KEY = ["competition", "season", "home", "away"]

#: At most this many violations are quoted in one error message.
MAX_REPORTED_FAILURES = 25

PROBABILITY_TOLERANCE = 1e-9

_PROBABILITY_COLUMNS = (
    ("p_home_proportional", "p_draw_proportional", "p_away_proportional"),
    ("p_home_shin", "p_draw_shin", "p_away_shin"),
)


class ProcessingValidationError(ValueError):
    """The cleaned data breaks an invariant, so nothing is written."""


def _no_duplicate_fixtures(frame: pd.DataFrame) -> pd.Series:
    return ~frame.duplicated(subset=FIXTURE_KEY, keep=False)


def _played_matches_have_a_score(frame: pd.DataFrame) -> pd.Series:
    played = frame["status"] == MatchStatus.PLAYED.value
    scored = frame["home_goals"].notna() & frame["away_goals"].notna()
    return ~played | scored


def _probabilities_sum_to_one(frame: pd.DataFrame) -> pd.Series:
    ok = pd.Series(True, index=frame.index)
    for columns in _PROBABILITY_COLUMNS:
        present = frame[list(columns)].notna().all(axis=1)
        total = frame[list(columns)].sum(axis=1)
        ok &= ~present | ((total - 1.0).abs() <= PROBABILITY_TOLERANCE)
    return ok


def _goal_column() -> pa.Column:
    return pa.Column("Int64", nullable=True, checks=pa.Check.ge(0), required=True)


def _count_column() -> pa.Column:
    return pa.Column("Int64", nullable=True, checks=pa.Check.ge(0), required=True)


def _probability_column() -> pa.Column:
    return pa.Column("Float64", nullable=True, checks=pa.Check.in_range(0.0, 1.0))


MATCHES_SCHEMA = pa.DataFrameSchema(
    name="matches",
    columns={
        "competition": pa.Column("string"),
        "season": pa.Column("string"),
        "kickoff_utc": pa.Column("datetime64[ns, UTC]"),
        "kickoff_local_date": pa.Column("datetime64[ns]"),
        "kickoff_time_estimated": pa.Column("boolean"),
        "home": pa.Column("string"),
        "away": pa.Column("string"),
        "home_goals": _goal_column(),
        "away_goals": _goal_column(),
        "ht_home_goals": _goal_column(),
        "ht_away_goals": _goal_column(),
        "result": pa.Column("string", nullable=True, checks=pa.Check.isin(sorted(RESULTS))),
        "status": pa.Column(
            "string", checks=pa.Check.isin([status.value for status in MatchStatus])
        ),
        "source": pa.Column("string"),
    },
    checks=[
        pa.Check(
            lambda frame: frame["home"] != frame["away"],
            error="a match cannot link a team to itself",
        ),
        pa.Check(
            _no_duplicate_fixtures, error="duplicate fixture for (competition, season, teams)"
        ),
        pa.Check(_played_matches_have_a_score, error="a played match needs both full-time scores"),
    ],
    coerce=True,
    strict=True,
)

MATCH_STATS_SCHEMA = pa.DataFrameSchema(
    name="match_stats",
    columns={
        "competition": pa.Column("string"),
        "season": pa.Column("string"),
        "home": pa.Column("string"),
        "away": pa.Column("string"),
        "side": pa.Column("string", checks=pa.Check.isin(["home", "away"])),
        "team": pa.Column("string"),
        "shots": _count_column(),
        "shots_on_target": _count_column(),
        "corners": _count_column(),
        "fouls": _count_column(),
        "yellows": _count_column(),
        "reds": _count_column(),
        "xg": pa.Column("Float64", nullable=True, checks=pa.Check.ge(0.0)),
    },
    unique=[*FIXTURE_KEY, "side"],
    coerce=True,
    strict=True,
)

ODDS_SCHEMA = pa.DataFrameSchema(
    name="odds",
    columns={
        "competition": pa.Column("string"),
        "season": pa.Column("string"),
        "home": pa.Column("string"),
        "away": pa.Column("string"),
        "bookmaker": pa.Column("string"),
        "market": pa.Column("string"),
        "is_closing": pa.Column("boolean"),
        "price_home": pa.Column("Float64", checks=pa.Check.gt(1.0)),
        "price_draw": pa.Column("Float64", checks=pa.Check.gt(1.0)),
        "price_away": pa.Column("Float64", checks=pa.Check.gt(1.0)),
        "overround": pa.Column("Float64", nullable=True, checks=pa.Check.gt(0.0)),
        "p_home_proportional": _probability_column(),
        "p_draw_proportional": _probability_column(),
        "p_away_proportional": _probability_column(),
        "p_home_shin": _probability_column(),
        "p_draw_shin": _probability_column(),
        "p_away_shin": _probability_column(),
    },
    checks=[
        pa.Check(_probabilities_sum_to_one, error="de-margined probabilities must sum to 1"),
    ],
    unique=[*FIXTURE_KEY, "bookmaker", "market", "is_closing"],
    coerce=True,
    strict=True,
)


@dataclass(frozen=True, slots=True)
class Frames:
    """The three validated frames the writer turns into rows."""

    matches: pd.DataFrame
    match_stats: pd.DataFrame
    odds: pd.DataFrame


def matches_frame(matches: Sequence[CleanMatch]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "competition": match.competition,
                "season": match.season,
                "kickoff_utc": match.kickoff.utc,
                "kickoff_local_date": match.kickoff.local_date,
                "kickoff_time_estimated": match.kickoff.time_estimated,
                "home": match.home,
                "away": match.away,
                "home_goals": match.home_goals,
                "away_goals": match.away_goals,
                "ht_home_goals": match.ht_home_goals,
                "ht_away_goals": match.ht_away_goals,
                "result": match.result,
                "status": match.status.value,
                "source": match.source,
            }
            for match in matches
        ],
        columns=list(MATCHES_SCHEMA.columns),
    )


def match_stats_frame(matches: Sequence[CleanMatch]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "competition": match.competition,
                "season": match.season,
                "home": match.home,
                "away": match.away,
                "side": stats.side,
                "team": match.home if stats.side == "home" else match.away,
                "shots": stats.shots,
                "shots_on_target": stats.shots_on_target,
                "corners": stats.corners,
                "fouls": stats.fouls,
                "yellows": stats.yellows,
                "reds": stats.reds,
                "xg": stats.xg,
            }
            for match in matches
            for stats in match.stats
        ],
        columns=list(MATCH_STATS_SCHEMA.columns),
    )


def odds_frame(matches: Sequence[CleanMatch]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "competition": match.competition,
                "season": match.season,
                "home": match.home,
                "away": match.away,
                "bookmaker": odds.bookmaker,
                "market": odds.market,
                "is_closing": odds.is_closing,
                "price_home": odds.price_home,
                "price_draw": odds.price_draw,
                "price_away": odds.price_away,
                "overround": odds.overround,
                "p_home_proportional": odds.p_home_proportional,
                "p_draw_proportional": odds.p_draw_proportional,
                "p_away_proportional": odds.p_away_proportional,
                "p_home_shin": odds.p_home_shin,
                "p_draw_shin": odds.p_draw_shin,
                "p_away_shin": odds.p_away_shin,
            }
            for match in matches
            for odds in match.odds
        ],
        columns=list(ODDS_SCHEMA.columns),
    )


def _report(table: str, error: SchemaErrors) -> list[str]:
    """One line per distinct violation, so a single bad row never hides the others."""
    failures = error.failure_cases.fillna({"column": "-"}).copy()
    # A table-wide check fails once per column of the offending row; report the row, not every cell.
    table_wide = failures["schema_context"] != "Column"
    failures.loc[table_wide, "column"] = "-"
    failures = failures.drop_duplicates(subset=["check", "column", "index"])
    groups = list(failures.groupby(["check", "column"], sort=True))
    lines = [
        f"{table}: {check} | column={column} | {len(rows)} row(s), "
        f"first at row {rows.iloc[0]['index']} with value {rows.iloc[0]['failure_case']!r}"
        for (check, column), rows in groups[:MAX_REPORTED_FAILURES]
    ]
    if len(groups) > MAX_REPORTED_FAILURES:
        lines.append(f"{table}: and {len(groups) - MAX_REPORTED_FAILURES} more kinds of violation")
    return lines


def _validated(
    table: str, schema: pa.DataFrameSchema, frame: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    try:
        return schema.validate(frame, lazy=True), []
    except SchemaErrors as error:
        return frame, _report(table, error)


def validate(matches: Sequence[CleanMatch]) -> Frames:
    """Validate the three frames together, raising once with every violation found."""
    frames: dict[str, pd.DataFrame] = {}
    problems: list[str] = []
    for table, schema, builder in (
        ("matches", MATCHES_SCHEMA, matches_frame),
        ("match_stats", MATCH_STATS_SCHEMA, match_stats_frame),
        ("odds", ODDS_SCHEMA, odds_frame),
    ):
        frame, found = _validated(table, schema, builder(matches))
        frames[table] = frame
        problems.extend(found)

    if problems:
        raise ProcessingValidationError("\n".join(problems))
    return Frames(frames["matches"], frames["match_stats"], frames["odds"])
