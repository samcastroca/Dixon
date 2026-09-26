"""Application settings: config/settings.yaml, overridable by environment variables.

Everything tunable lives in the YAML file; secrets live only in the environment.
Competitions (codes, source codes, timezones, rules) are configuration too, so no
league name is ever written into logic.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import time
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

#: Environment variable pointing at an alternative settings file (used by tests and Docker).
SETTINGS_FILE_ENV = "PREDICTOR_SETTINGS_FILE"

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]


def settings_file_path() -> Path:
    """Resolve the settings file: env override, then repo root, then working directory."""
    override = os.environ.get(SETTINGS_FILE_ENV)
    if override:
        return Path(override)
    for candidate in (REPO_ROOT / "config" / "settings.yaml", Path("config/settings.yaml")):
        if candidate.is_file():
            return candidate
    return REPO_ROOT / "config" / "settings.yaml"


def team_aliases_file_path() -> Path:
    """The alias seed file (spec section 6.2) always sits next to the settings file."""
    return settings_file_path().parent / "team_aliases.yaml"


class CompetitionRules(BaseModel):
    """Competition rules used later by validation (phase 2) and simulation (phase 7)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    teams: Annotated[int, Field(ge=2)]
    relegation_slots: Annotated[int, Field(ge=0)]
    champions_league_slots: Annotated[int, Field(ge=0)]
    tiebreaker: Literal["goal_difference", "head_to_head"]

    @model_validator(mode="after")
    def _slots_fit_in_the_league(self) -> CompetitionRules:
        if self.relegation_slots + self.champions_league_slots > self.teams:
            msg = "relegation_slots + champions_league_slots cannot exceed teams"
            raise ValueError(msg)
        return self


class Competition(BaseModel):
    """One league. Adding a league is a config change, never a code change."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: Annotated[str, Field(min_length=2, pattern=r"^[A-Z0-9_]+$")]
    sport: Annotated[str, Field(min_length=3)]
    name: Annotated[str, Field(min_length=3)]
    country: Annotated[str, Field(min_length=2)]
    timezone: str
    #: source name -> that source's code for this competition, e.g. {"football_data_uk": "E0"}
    sources: Annotated[Mapping[str, str], Field(min_length=1)]
    rules: CompetitionRules
    #: Local kickoff time assumed when a source has no time column; falls back to `processing`.
    default_kickoff_local_time: time | None = None

    @field_validator("timezone")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            msg = f"unknown timezone {value!r}"
            raise ValueError(msg) from exc
        return value

    def source_code(self, source: str) -> str:
        """Return this competition's code for `source`, failing loudly if unconfigured."""
        try:
            return self.sources[source]
        except KeyError as exc:
            msg = f"competition {self.code} has no code configured for source {source!r}"
            raise KeyError(msg) from exc

    def tzinfo(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


class SourceSettings(BaseModel):
    """Everything the HTTP layer of one source needs (spec section 6.1)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: str
    resource_template: str
    user_agent: str
    timeout_seconds: Annotated[float, Field(gt=0)]
    max_attempts: Annotated[int, Field(ge=1, le=10)]
    backoff_initial_seconds: Annotated[float, Field(gt=0)]
    backoff_max_seconds: Annotated[float, Field(gt=0)]
    backoff_jitter_seconds: Annotated[float, Field(ge=0)]
    min_interval_seconds: Annotated[float, Field(ge=0)]
    encodings: Annotated[tuple[str, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _backoff_is_bounded(self) -> SourceSettings:
        if self.backoff_max_seconds < self.backoff_initial_seconds:
            msg = "backoff_max_seconds cannot be smaller than backoff_initial_seconds"
            raise ValueError(msg)
        return self


class IngestionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    default_source: str = "football_data_uk"
    default_seasons: str = "2014-2025"


class GoalColumns(BaseModel):
    """Score columns of one source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    home: str
    away: str
    ht_home: str
    ht_away: str
    result: str


class StatColumns(BaseModel):
    """Per-match team statistics: each entry is the (home column, away column) pair."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    shots: tuple[str, str]
    shots_on_target: tuple[str, str]
    corners: tuple[str, str]
    fouls: tuple[str, str]
    yellows: tuple[str, str]
    reds: tuple[str, str]


class BookmakerColumns(BaseModel):
    """One bookmaker: the column prefix of its opening prices and, if any, of its closing ones."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: Annotated[str, Field(min_length=1, max_length=16)]
    name: Annotated[str, Field(min_length=2)]
    open: Annotated[str, Field(min_length=1)]
    closing: str | None = None


class SourceColumns(BaseModel):
    """How one source names the fields the cleaning step needs (spec section 6.2)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    date: str
    time: str
    home: str
    away: str
    goals: GoalColumns
    stats: StatColumns
    bookmakers: Annotated[tuple[BookmakerColumns, ...], Field(min_length=1)]

    #: Suffixes appended to a bookmaker prefix for the home, draw and away prices.
    outcome_suffixes: tuple[str, str, str] = ("H", "D", "A")


class ProcessingSettings(BaseModel):
    """Phase 2 knobs: which source is cleaned and how its columns are read."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = "football_data_uk"
    market: str = "1X2"
    default_kickoff_local_time: time = time(15, 0)
    status_when_result_missing: Literal["postponed", "abandoned", "awarded"] = "postponed"
    columns: Mapping[str, SourceColumns] = {}

    def source_columns(self, name: str) -> SourceColumns:
        """Look up a source's column map, failing loudly on an unconfigured one."""
        try:
            return self.columns[name]
        except KeyError as exc:
            configured = ", ".join(sorted(self.columns)) or "none"
            msg = f"no column map for source {name!r}; configured: {configured}"
            raise KeyError(msg) from exc


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = "predictor"
    version: str = "0.1.0"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    storage_timezone: Literal["UTC"] = "UTC"


class DatabaseSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str = "db"
    port: Annotated[int, Field(ge=1, le=65535)] = 5432
    name: str = "predictor"
    user: str = "predictor"
    password: SecretStr = SecretStr("")
    connect_timeout: Annotated[int, Field(ge=1, le=60)] = 5


class MlflowSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tracking_uri: str = "http://mlflow:5000"


class ApiSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str = "0.0.0.0"  # noqa: S104 - inside the compose network, published per environment
    port: Annotated[int, Field(ge=1, le=65535)] = 8000
    admin_token: SecretStr = SecretStr("")


class Settings(BaseSettings):
    """Root settings, validated at startup so a broken config fails fast."""

    model_config = SettingsConfigDict(
        env_prefix="PREDICTOR_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    app: AppSettings = AppSettings()
    database: DatabaseSettings = DatabaseSettings()
    mlflow: MlflowSettings = MlflowSettings()
    api: ApiSettings = ApiSettings()
    ingestion: IngestionSettings = IngestionSettings()
    processing: ProcessingSettings = ProcessingSettings()
    sources: Mapping[str, SourceSettings] = {}
    competitions: Annotated[tuple[Competition, ...], Field(min_length=1)]

    @field_validator("competitions")
    @classmethod
    def _unique_codes(cls, value: tuple[Competition, ...]) -> tuple[Competition, ...]:
        codes = [competition.code for competition in value]
        duplicates = sorted({code for code in codes if codes.count(code) > 1})
        if duplicates:
            msg = f"duplicate competition codes: {', '.join(duplicates)}"
            raise ValueError(msg)
        return value

    @property
    def competition_codes(self) -> tuple[str, ...]:
        return tuple(competition.code for competition in self.competitions)

    def competition(self, code: str) -> Competition:
        """Look up a competition by code, failing loudly on an unknown one."""
        for competition in self.competitions:
            if competition.code == code.upper():
                return competition
        msg = f"unknown competition {code!r}; configured: {', '.join(self.competition_codes)}"
        raise KeyError(msg)

    def source(self, name: str) -> SourceSettings:
        """Look up a source's configuration, failing loudly on an unconfigured one."""
        try:
            return self.sources[name]
        except KeyError as exc:
            configured = ", ".join(sorted(self.sources)) or "none"
            msg = f"unknown source {name!r}; configured: {configured}"
            raise KeyError(msg) from exc

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Environment wins over the YAML file, which wins over the model defaults.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=settings_file_path()),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache the settings. Call `get_settings.cache_clear()` in tests."""
    return Settings()
