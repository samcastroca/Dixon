"""Test helpers: serve the recorded CSVs over a mocked HTTP transport, never the network."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import httpx

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "football_data_uk"

#: (competition code, season label) -> recorded file
RECORDED: dict[tuple[str, str], str] = {
    ("EPL", "2014-15"): "E0_1415.csv",
    ("EPL", "2024-25"): "E0_2425.csv",
    ("LALIGA", "2014-15"): "SP1_1415.csv",
    ("LALIGA", "2024-25"): "SP1_2425.csv",
}

RECORDED_SEASONS = ("2014-15", "2024-25")

#: Deliberately broken copy of an EPL file: negative goals, a duplicated fixture, a 1.00 price.
CORRUPT = "E0_corrupt.csv"


def recorded_bytes(competition: str, season: str) -> bytes:
    return (FIXTURES / RECORDED[(competition, season)]).read_bytes()


def corrupt_bytes() -> bytes:
    return (FIXTURES / CORRUPT).read_bytes()


def fixture_transport() -> httpx.MockTransport:
    """Serve `.../<season_code>/<source_code>.csv` from the recorded files, 404 otherwise."""
    by_filename = {name: (FIXTURES / name).read_bytes() for name in RECORDED.values()}

    def handler(request: httpx.Request) -> httpx.Response:
        parts = request.url.path.strip("/").split("/")
        season_code, filename = parts[-2], parts[-1]
        name = f"{filename.removesuffix('.csv')}_{season_code}.csv"
        if name not in by_filename:
            return httpx.Response(404, text="not found")
        return httpx.Response(200, content=by_filename[name], headers={"content-type": "text/csv"})

    return httpx.MockTransport(handler)


def scripted_transport(outcomes: Sequence[httpx.Response | Exception]) -> httpx.MockTransport:
    """Return the given responses in order; an Exception entry is raised instead."""
    remaining = list(outcomes)

    def handler(request: httpx.Request) -> httpx.Response:
        outcome = remaining.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return httpx.MockTransport(handler)


class FakeClock:
    """Monotonic clock plus sleep, so retry and rate-limit tests take no real time."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds
