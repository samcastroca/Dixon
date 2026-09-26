"""football-data.co.uk: one CSV per season and competition (spec section 4)."""

from __future__ import annotations

from collections.abc import Sequence

from predictor.config import Competition, Settings, get_settings
from predictor.ingestion.base import RawDocument, Resource, Source
from predictor.ingestion.http import HttpClient
from predictor.ingestion.seasons import season_code
from predictor.logging import get_logger

logger = get_logger(__name__)


class FootballDataUkSource(Source):
    """Fetches the season files for whatever code each competition has configured."""

    #: key of this source's block in settings.yaml
    SOURCE_KEY = "football_data_uk"

    def __init__(
        self,
        settings: Settings | None = None,
        client: HttpClient | None = None,
        name: str | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._config = self._settings.source(self.SOURCE_KEY)
        self._client = client or HttpClient(self._config)
        #: recorded in raw_payloads.source; overridden by tests so they never touch real rows
        self.name = name or self.SOURCE_KEY

    @property
    def encodings(self) -> tuple[str, ...]:
        return self._config.encodings

    def list_resources(
        self, competition: Competition, seasons: Sequence[str]
    ) -> tuple[Resource, ...]:
        source_code = competition.source_code(self.SOURCE_KEY)
        resources = []
        for season in seasons:
            path = self._config.resource_template.format(
                season_code=season_code(season), source_code=source_code
            )
            resources.append(
                Resource(
                    source=self.name,
                    competition_code=competition.code,
                    season=season,
                    resource=path,
                    url=f"{self._config.base_url.rstrip('/')}/{path}",
                )
            )
        return tuple(resources)

    def fetch(self, resource: Resource) -> RawDocument:
        response = self._client.get(resource.url, resource=resource.resource)
        document = RawDocument.create(
            resource,
            payload=response.content,
            content_type=response.headers.get("content-type"),
        )
        logger.info(
            "source.fetched",
            source=self.name,
            competition=resource.competition_code,
            season=resource.season,
            resource=resource.resource,
            bytes=len(document.payload),
            checksum=document.checksum,
        )
        return document

    def close(self) -> None:
        self._client.close()
