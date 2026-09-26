"""Application settings: config/settings.yaml, overridable by environment variables.

Everything tunable lives in the YAML file; secrets live only in the environment.
Competitions (codes, source codes, timezones, rules) are configuration too, so no
league name is ever written into logic.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
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
