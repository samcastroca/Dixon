"""Retry policy and rate limiting. No network, no real sleeping."""

from __future__ import annotations

import httpx
import pytest

from predictor.config import get_settings
from predictor.ingestion.base import ResourceNotFound, TransientSourceError
from predictor.ingestion.http import HttpClient, RateLimiter
from tests.support import FakeClock, scripted_transport

URL = "https://example.test/mmz4281/2425/E0.csv"
RESOURCE = "mmz4281/2425/E0.csv"


def build_client(outcomes: list[httpx.Response | Exception], clock: FakeClock) -> HttpClient:
    config = get_settings().source("football_data_uk")
    return HttpClient(
        config,
        transport=scripted_transport(outcomes),
        sleep=clock.sleep,
        clock=clock.monotonic,
    )


def test_retries_server_errors_then_succeeds() -> None:
    clock = FakeClock()
    client = build_client(
        [httpx.Response(500), httpx.Response(500), httpx.Response(200, content=b"ok")], clock
    )

    assert client.get(URL, resource=RESOURCE).content == b"ok"
    assert len(clock.slept) == 2  # two waits, three attempts
    assert clock.slept[1] > clock.slept[0]  # exponential


def test_retries_timeouts() -> None:
    clock = FakeClock()
    client = build_client(
        [
            httpx.ConnectTimeout("too slow"),
            httpx.ReadTimeout("too slow"),
            httpx.Response(200, content=b"ok"),
        ],
        clock,
    )

    assert client.get(URL, resource=RESOURCE).content == b"ok"
    assert len(clock.slept) == 2


def test_gives_up_after_max_attempts() -> None:
    clock = FakeClock()
    attempts = get_settings().source("football_data_uk").max_attempts
    client = build_client([httpx.Response(503) for _ in range(attempts)], clock)

    with pytest.raises(TransientSourceError, match=RESOURCE):
        client.get(URL, resource=RESOURCE)

    assert len(clock.slept) == attempts - 1


def test_does_not_retry_client_errors() -> None:
    clock = FakeClock()
    client = build_client([httpx.Response(404)], clock)

    with pytest.raises(ResourceNotFound) as excinfo:
        client.get(URL, resource=RESOURCE)

    assert RESOURCE in str(excinfo.value)
    assert URL in str(excinfo.value)
    assert clock.slept == []  # failed immediately


def test_rate_limiter_spaces_requests() -> None:
    clock = FakeClock()
    limiter = RateLimiter(min_interval=1.5, clock=clock.monotonic, sleep=clock.sleep)

    limiter.wait()  # first call is free
    assert clock.slept == []

    limiter.wait()
    assert clock.slept == [pytest.approx(1.5)]

    clock.advance(5.0)
    limiter.wait()  # enough time has passed already
    assert len(clock.slept) == 1
