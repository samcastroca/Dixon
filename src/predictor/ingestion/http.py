"""HTTP layer: timeouts, retry with exponential backoff and jitter, per-source rate limit."""

from __future__ import annotations

import time
from collections.abc import Callable
from http import HTTPStatus
from types import TracebackType

import httpx
from tenacity import RetryError, Retrying, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_exponential_jitter

from predictor.config import SourceSettings
from predictor.ingestion.base import ResourceNotFound, TransientSourceError
from predictor.logging import get_logger

logger = get_logger(__name__)


class _Retryable(Exception):
    """Internal marker: this attempt failed in a way that is worth retrying."""


class RateLimiter:
    """Keep at least `min_interval` seconds between calls. The clock and sleep are injectable."""

    def __init__(
        self,
        min_interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last_call: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last_call is not None:
            remaining = self._min_interval - (now - self._last_call)
            if remaining > 0:
                self._sleep(remaining)
                now = self._clock()
        self._last_call = now


class HttpClient:
    """A polite HTTP client for one source: 4xx fails immediately, 5xx and timeouts retry."""

    def __init__(
        self,
        config: SourceSettings,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._sleep = sleep
        self._limiter = RateLimiter(config.min_interval_seconds, clock=clock, sleep=sleep)
        self._client = httpx.Client(
            timeout=config.timeout_seconds,
            transport=transport,
            follow_redirects=True,
            headers={"user-agent": config.user_agent},
        )

    def get(self, url: str, resource: str) -> httpx.Response:
        """Fetch `url`, retrying transient failures; `resource` is what error messages name."""
        retrying = Retrying(
            stop=stop_after_attempt(self._config.max_attempts),
            wait=wait_exponential_jitter(
                initial=self._config.backoff_initial_seconds,
                max=self._config.backoff_max_seconds,
                jitter=self._config.backoff_jitter_seconds,
            ),
            retry=retry_if_exception_type(_Retryable),
            sleep=self._sleep,
        )
        try:
            return retrying(self._attempt, url, resource)
        except RetryError as error:
            reason = str(error.last_attempt.exception())
            raise TransientSourceError(
                resource, url, self._config.max_attempts, reason
            ) from error.last_attempt.exception()

    def _attempt(self, url: str, resource: str) -> httpx.Response:
        self._limiter.wait()
        try:
            response = self._client.get(url)
        except (httpx.TimeoutException, httpx.TransportError) as error:
            logger.warning("http.transient", resource=resource, url=url, error=str(error))
            raise _Retryable(f"{type(error).__name__}: {error}") from error

        if response.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
            logger.warning("http.server_error", resource=resource, status=response.status_code)
            raise _Retryable(f"HTTP {response.status_code}")
        if response.status_code >= HTTPStatus.BAD_REQUEST:
            raise ResourceNotFound(resource, url, response.status_code)
        return response

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
