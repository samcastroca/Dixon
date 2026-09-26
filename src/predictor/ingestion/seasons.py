"""Season labels (`2014-15`) and the range syntax the CLI accepts (`2014-2025`)."""

from __future__ import annotations

import re

SEASON_LABEL = re.compile(r"^(\d{4})-(\d{2})$")
SEASON_RANGE = re.compile(r"^(\d{4})-(\d{4})$")

MAX_SEASONS = 60


def season_label(start_year: int) -> str:
    """2014 -> '2014-15' (the season that starts in 2014)."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def season_code(label: str) -> str:
    """'2014-15' -> '1415', the four-digit code football-data.co.uk uses in its paths."""
    match = SEASON_LABEL.match(label)
    if match is None:
        msg = f"malformed season {label!r}; expected YYYY-YY, e.g. 2014-15"
        raise ValueError(msg)
    start, end = match.groups()
    return f"{start[2:]}{end}"


def parse_seasons(value: str) -> tuple[str, ...]:
    """Parse `2014-2025` (start year to end year) or an explicit `2014-15,2020-21` list."""
    text = value.strip()
    if not text:
        msg = "empty season selection; expected a range like 2014-2025"
        raise ValueError(msg)

    range_match = SEASON_RANGE.match(text)
    if range_match is not None:
        first, last = (int(group) for group in range_match.groups())
        if last <= first:
            msg = f"malformed season range {value!r}: the end year must be after the start year"
            raise ValueError(msg)
        if last - first > MAX_SEASONS:
            msg = f"season range {value!r} covers more than {MAX_SEASONS} seasons"
            raise ValueError(msg)
        return tuple(season_label(year) for year in range(first, last))

    labels = tuple(part.strip() for part in text.split(",") if part.strip())
    if not labels:
        msg = f"malformed season selection {value!r}"
        raise ValueError(msg)
    for label in labels:
        season_code(label)  # validates the shape, raises with a clear message
    return labels
