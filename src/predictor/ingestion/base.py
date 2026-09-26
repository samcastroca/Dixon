"""The common Source interface every data source implements (spec section 4)."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from predictor.config import Competition


class SourceError(Exception):
    """Base class for every ingestion failure."""


class ResourceNotFound(SourceError):
    """The source answered 4xx: the resource does not exist. Never retried."""

    def __init__(self, resource: str, url: str, status_code: int) -> None:
        super().__init__(f"resource {resource!r} not available at {url} (HTTP {status_code})")
        self.resource = resource
        self.url = url
        self.status_code = status_code


class TransientSourceError(SourceError):
    """A retryable failure (5xx, timeout, connection error) that outlived every attempt."""

    def __init__(self, resource: str, url: str, attempts: int, reason: str) -> None:
        super().__init__(
            f"resource {resource!r} at {url} failed after {attempts} attempts: {reason}"
        )
        self.resource = resource
        self.url = url
        self.attempts = attempts


@dataclass(frozen=True, slots=True)
class Resource:
    """One fetchable unit of a source: here, one season file of one competition."""

    source: str
    competition_code: str
    season: str
    resource: str
    url: str


@dataclass(frozen=True, slots=True)
class RawDocument:
    """Exactly what the source returned, plus the metadata needed to store it."""

    resource: Resource
    payload: bytes
    fetched_at: datetime
    checksum: str
    content_type: str | None = None

    @classmethod
    def create(
        cls,
        resource: Resource,
        payload: bytes,
        content_type: str | None = None,
    ) -> RawDocument:
        return cls(
            resource=resource,
            payload=payload,
            fetched_at=datetime.now(UTC),
            checksum=hashlib.sha256(payload).hexdigest(),
            content_type=content_type,
        )


class Source(ABC):
    """A data source. Implementations are swappable and never know about a specific league."""

    name: str

    @property
    @abstractmethod
    def encodings(self) -> tuple[str, ...]:
        """Text encodings to try when decoding this source's payloads, in order."""

    @abstractmethod
    def list_resources(
        self, competition: Competition, seasons: Sequence[str]
    ) -> tuple[Resource, ...]:
        """Enumerate what would be fetched, without fetching anything."""

    @abstractmethod
    def fetch(self, resource: Resource) -> RawDocument:
        """Fetch one resource and return its exact bytes plus metadata."""
