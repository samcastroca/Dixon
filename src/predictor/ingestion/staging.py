"""Turn a raw CSV payload into staging rows: text in, text out, nothing converted."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator, Sequence


def decode(payload: bytes, encodings: Sequence[str]) -> str:
    """Decode with the configured encodings in order; the source mixes them across seasons."""
    for encoding in encodings:
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    tried = ", ".join(encodings)
    msg = f"could not decode payload with any configured encoding ({tried})"
    raise UnicodeDecodeError(encodings[0], payload, 0, 1, msg)


def parse_rows(payload: bytes, encodings: Sequence[str]) -> Iterator[tuple[int, dict[str, str]]]:
    """Yield (row_index, row) with every value kept as text. Empty rows are dropped.

    Cleaning, typing and timezone conversion belong to phase 2; nothing happens here.
    """
    reader = csv.DictReader(io.StringIO(decode(payload, encodings), newline=""))
    index = 0
    for raw_row in reader:
        row = {
            key: (value or "")
            for key, value in raw_row.items()
            if key is not None and key.strip() != ""
        }
        if not any(value.strip() for value in row.values()):
            continue  # the files end with rows of bare commas
        yield index, row
        index += 1
