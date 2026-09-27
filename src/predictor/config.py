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


class EloParameters(BaseModel):
    """Everything the Elo rating needs, per competition (spec section 7.1)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    k: Annotated[float, Field(gt=0)] = 20.0
    home_advantage: float = 60.0
    initial_rating: Annotated[float, Field(gt=0)] = 1500.0
    #: G in R' = R + K*G*(S - E): 1.0 up to a one-goal margin, then this curve.
    two_goal_multiplier: Annotated[float, Field(gt=0)] = 1.5
    large_margin_base: Annotated[float, Field(gt=0)] = 11.0
    large_margin_divisor: Annotated[float, Field(gt=0)] = 8.0
    #: Share of the gap to the target given back at every season start.
    season_regression: Annotated[float, Field(ge=0, le=1)] = 0.25
    season_regression_target: Annotated[float, Field(gt=0)] = 1500.0


class EloSettings(BaseModel):
    """Elo defaults plus per-competition overrides, so a new league needs no new values."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    defaults: EloParameters = EloParameters()
    #: Competition code -> the subset of knobs that competition overrides.
    per_competition: Mapping[str, Mapping[str, float]] = {}

    @model_validator(mode="after")
    def _overrides_are_valid(self) -> EloSettings:
        # Merge every override now, so a typo in the YAML fails at startup and not mid-build.
        for code in self.per_competition:
            self.for_competition(code)
        return self

    def for_competition(self, code: str) -> EloParameters:
        """Defaults with that competition's overrides applied on top."""
        override = self.per_competition.get(code.upper(), {})
        return EloParameters(**{**self.defaults.model_dump(), **override})


class FormSettings(BaseModel):
    """Rolling and exponentially weighted form windows."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    windows: Annotated[tuple[int, ...], Field(min_length=1)] = (5, 10)
    ewma_halflife: Annotated[float, Field(gt=0)] = 5.0

    @field_validator("windows")
    @classmethod
    def _windows_are_distinct_and_positive(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(window < 1 for window in value):
            msg = "every form window must cover at least one match"
            raise ValueError(msg)
        if len(set(value)) != len(value):
            msg = f"duplicate form windows: {value}"
            raise ValueError(msg)
        return value


class ScheduleSettings(BaseModel):
    """Schedule congestion knobs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    congestion_window_days: Annotated[int, Field(ge=1)] = 14


class MarketFeatureSettings(BaseModel):
    """The optional pre-match market group. Off by default (spec section 6.3)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bookmaker: Annotated[str, Field(min_length=1, max_length=16)] = "PS"
    method: Literal["shin", "proportional"] = "shin"
    prefer_closing: bool = True
    #: A quote without a capture time cannot be proven pre-kickoff; using it is a choice.
    allow_missing_captured_at: bool = False


class FeatureGroups(BaseModel):
    """Which feature groups a build assembles."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    elo: bool = True
    form: bool = True
    schedule: bool = True
    market: bool = False


class FeaturesSettings(BaseModel):
    """Phase 3 knobs: every number a point-in-time feature depends on."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Annotated[
        str, Field(min_length=1, max_length=16, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    ] = "v1"
    #: How long before kickoff a row is valid, which is what keeps as_of_utc < kickoff_utc.
    as_of_offset_seconds: Annotated[int, Field(ge=1)] = 1
    #: Decimals every stored value is rounded to, so two C libraries agree bit for bit.
    value_decimals: Annotated[int, Field(ge=3, le=15)] = 9
    groups: FeatureGroups = FeatureGroups()
    elo: EloSettings = EloSettings()
    form: FormSettings = FormSettings()
    schedule: ScheduleSettings = ScheduleSettings()
    market: MarketFeatureSettings = MarketFeatureSettings()


class MetricsSettings(BaseModel):
    """Phase 4 knobs: how a probability is scored (spec section 8)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Bins of the expected calibration error; the spec asks for ten.
    ece_bins: Annotated[int, Field(ge=2, le=100)] = 10
    #: Probabilities are clipped to this floor, so one confident miss is not an infinite loss.
    log_loss_floor: Annotated[float, Field(gt=0, lt=0.5)] = 1e-15


class BootstrapSettings(BaseModel):
    """Paired bootstrap of a metric difference: a model is only better if zero is excluded."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    resamples: Annotated[int, Field(ge=100, le=100_000)] = 2000
    confidence: Annotated[float, Field(gt=0.5, lt=1.0)] = 0.95
    #: Fixed, so the same predictions always give the same interval.
    seed: int = 20260926


class ReportSettings(BaseModel):
    """Where the model comparison is written and in which formats."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    directory: str = "reports"
    formats: Annotated[tuple[Literal["markdown", "html"], ...], Field(min_length=1)] = (
        "markdown",
        "html",
    )


class BacktestSettings(BaseModel):
    """Phase 4 knobs: the walk-forward protocol (spec section 8).

    This lives outside `features` on purpose: the feature store's definition checksum hashes
    that whole section, so a backtesting knob in there would invalidate every stored row.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_seasons: str = "2019-2025"
    experiment: Annotated[str, Field(min_length=1)] = "predictor-backtest"
    #: Which feature store the models read; None means `features.version`.
    feature_version: Annotated[str | None, Field(max_length=16)] = None
    #: 0 fits once per test season; N refits whenever the team with the fewest matches
    #: played in that season has played N more. The source files carry no matchweek.
    refit_every_matchweeks: Annotated[int, Field(ge=0, le=38)] = 0
    #: A fold with less history than this is not worth fitting, and is skipped loudly.
    min_train_matches: Annotated[int, Field(ge=0)] = 380
    #: The model every other model's RPS gap is measured against.
    reference_model: Annotated[str, Field(min_length=1)] = "baseline_market"
    metrics: MetricsSettings = MetricsSettings()
    bootstrap: BootstrapSettings = BootstrapSettings()
    report: ReportSettings = ReportSettings()
    #: The market benchmark reads closing prices this source publishes without a capture
    #: time. That is why it has its own block: the exemption belongs to one undeployable
    #: benchmark, not to the feature store, whose market group stays off.
    market_baseline: MarketFeatureSettings = MarketFeatureSettings(allow_missing_captured_at=True)


class OptimiserSettings(BaseModel):
    """How far the maximum-likelihood fits of M1-M3 are allowed to go (spec section 7)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_iter: Annotated[int, Field(ge=10, le=10_000)] = 500
    tolerance: Annotated[float, Field(gt=0, lt=1)] = 1e-8
    #: A small L2 penalty. It never changes a well-determined fit, and it keeps a thin fold
    #: (a team with two matches, a league with few draws) from running a parameter to infinity.
    ridge: Annotated[float, Field(ge=0, lt=1)] = 1e-6


class EloModelSettings(BaseModel):
    """M1: the ordered logistic regression laid over the phase 3 Elo ratings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: The feature the rating difference is read from; it must exist in the feature store.
    feature: Annotated[str, Field(min_length=1)] = "elo_diff"
    #: Ratings run in the hundreds, so the slope is tiny; dividing keeps the fit conditioned.
    rating_scale: Annotated[float, Field(gt=0)] = 400.0
    optimiser: OptimiserSettings = OptimiserSettings()


#: How a team with no training history is given attack and defence (spec section 7.1).
PromotedRule = Literal["relegated_mean", "league_average"]


class PoissonModelSettings(BaseModel):
    """M2: the independent Poisson goal model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    optimiser: OptimiserSettings = OptimiserSettings()
    #: `relegated_mean` gives a newcomer the mean strength of the teams that went down at the
    #: end of the previous training season, mirroring the Elo rule of phase 3. `league_average`
    #: gives it zero, which the sum-to-zero constraint makes the average team.
    promoted: PromotedRule = "relegated_mean"


class TimeDecaySettings(BaseModel):
    """M3: the time-decay half-life search, tuned per league on training seasons only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Candidate xi in `w = exp(-xi * days / 365)`; 0 means no decay at all.
    grid: Annotated[tuple[float, ...], Field(min_length=1)] = (0.0, 0.5, 1.0, 2.0, 4.0)
    #: How many of the newest training seasons are held out as inner walk-forward folds.
    inner_seasons: Annotated[int, Field(ge=1, le=10)] = 2
    #: Used when the training history is too short for the inner folds.
    default: Annotated[float, Field(ge=0)] = 1.0


class DixonColesModelSettings(BaseModel):
    """M3: M2 plus the low-score correction and the time decay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    optimiser: OptimiserSettings = OptimiserSettings()
    promoted: PromotedRule = "relegated_mean"
    #: rho is bounded so tau stays positive on every one of the four corrected scorelines.
    rho_bounds: tuple[float, float] = (-0.4, 0.4)
    xi: TimeDecaySettings = TimeDecaySettings()

    @model_validator(mode="after")
    def _bounds_are_ordered(self) -> DixonColesModelSettings:
        low, high = self.rho_bounds
        if low >= high:
            msg = f"rho_bounds must be increasing, got {self.rho_bounds}"
            raise ValueError(msg)
        return self


class BayesianModelSettings(BaseModel):
    """M4: the PyMC hierarchical Poisson sampler (spec section 7, M4)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    draws: Annotated[int, Field(ge=50, le=100_000)] = 1000
    tune: Annotated[int, Field(ge=50, le=100_000)] = 1000
    chains: Annotated[int, Field(ge=1, le=16)] = 4
    target_accept: Annotated[float, Field(gt=0.5, lt=1.0)] = 0.9
    #: Fixed, so the same data always gives the same posterior.
    seed: int = 20260926
    #: NUTS implementation. nutpie compiles with numba, which needs no C compiler.
    nuts_sampler: Literal["nutpie", "pymc"] = "nutpie"
    #: How many posterior draws are averaged into the predictive scoreline matrix.
    posterior_samples: Annotated[int, Field(ge=50, le=100_000)] = 500
    #: Prior scales, so the model's assumptions are configuration and not buried constants.
    intercept_sigma: Annotated[float, Field(gt=0)] = 1.0
    home_sigma: Annotated[float, Field(gt=0)] = 1.0
    strength_sigma: Annotated[float, Field(gt=0)] = 1.0


class ModelsSettings(BaseModel):
    """Phase 5 knobs: the statistical models of spec section 7.

    Like `backtest`, this lives outside `features` on purpose: the feature store's definition
    checksum hashes that whole section, so a model knob in there would invalidate every row.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Where the scoreline matrix is truncated. The residual mass is reported, never hidden.
    max_goals: Annotated[int, Field(ge=3, le=30)] = 10
    elo: EloModelSettings = EloModelSettings()
    poisson: PoissonModelSettings = PoissonModelSettings()
    dixon_coles: DixonColesModelSettings = DixonColesModelSettings()
    bayesian: BayesianModelSettings = BayesianModelSettings()


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
    features: FeaturesSettings = FeaturesSettings()
    backtest: BacktestSettings = BacktestSettings()
    models: ModelsSettings = ModelsSettings()
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
