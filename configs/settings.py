"""Validated experiment configuration schemas for Hydra outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf, SCMode
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator
from pydantic_core import MultiHostUrl

__all__ = [
    "SecretRefConfig",
    "DatabaseConfigNode",
    "AnalyticsConfigNode",
    "DataConfigNode",
    "TrackingConfigNode",
    "ExperimentConfigNode",
    "SecretRef",
    "SecretResolutionError",
    "SecretLoader",
    "DatabaseSettings",
    "AnalyticsSettings",
    "DataSettings",
    "TrackingSettings",
    "ExperimentSettings",
    "register_structured_configs",
    "load_experiment_settings",
]


_STRUCTURED_CONFIGS_REGISTERED = False


@dataclass
class SecretRefConfig:
    """Structured config describing a secret stored in an external backend."""

    backend: str = MISSING
    path: str = MISSING
    key: str = MISSING
    version: str | None = None


@dataclass
class DatabaseConfigNode:
    """Structured config node for experiment database connectivity."""

    uri: str | None = None
    driver: str | None = None
    user: str | None = None
    password: SecretRefConfig | None = None
    host: str | None = None
    port: int | None = None
    name: str | None = None


@dataclass
class AnalyticsConfigNode:
    """Structured config node for analytics parameters."""

    window: int = 256
    bins: int = 48
    delta: float = 0.005


@dataclass
class DataConfigNode:
    """Structured config node describing experiment data sources."""

    price_csv: str = "sample.csv"
    price_column: str = "price"


@dataclass
class TrackingConfigNode:
    """Structured config node controlling artifact tracking output."""

    enabled: bool = True
    base_dir: str = "outputs"


@dataclass
class ExperimentConfigNode:
    """Structured config node for the entire experiment configuration."""

    name: str = "local"
    debug: bool = False
    log_level: str = "INFO"
    random_seed: int = 42
    database: DatabaseConfigNode = field(default_factory=DatabaseConfigNode)
    data: DataConfigNode = field(default_factory=DataConfigNode)
    analytics: AnalyticsConfigNode = field(default_factory=AnalyticsConfigNode)
    tracking: TrackingConfigNode = field(default_factory=TrackingConfigNode)


def register_structured_configs() -> None:
    """Register structured configuration schemas for Hydra composition."""

    global _STRUCTURED_CONFIGS_REGISTERED
    if _STRUCTURED_CONFIGS_REGISTERED:
        return

    cs = ConfigStore.instance()
    cs.store(group="schema", name="experiment", node=ExperimentConfigNode)
    cs.store(
        group="secrets",
        name="stage_db_password",
        node=SecretRefConfig(
            backend="vault",
            path="secret/data/tradepulse/stage/postgres",  # pragma: allowlist secret
            key="password",
        ),
    )
    cs.store(
        group="secrets",
        name="prod_db_password",
        node=SecretRefConfig(
            backend="vault",
            path="secret/data/tradepulse/prod/postgres",  # pragma: allowlist secret
            key="password",
        ),
    )
    _STRUCTURED_CONFIGS_REGISTERED = True


class SecretResolutionError(RuntimeError):
    """Raised when a secret cannot be retrieved from its backend."""


class SecretRef(BaseModel):
    """Pydantic representation of a secret reference."""

    model_config = ConfigDict(extra="forbid")

    backend: str
    path: str
    key: str
    version: str | None = None

    @field_validator("backend")
    @classmethod
    def _normalize_backend(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"vault", "aws_secrets_manager"}:
            raise ValueError(
                "backend must be either 'vault' or 'aws_secrets_manager'"
            )
        return normalized

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        candidate = value.strip()
        if not candidate:
            raise ValueError("secret path cannot be empty")
        if candidate.startswith("/"):
            candidate = candidate[1:]
        return candidate

    @field_validator("key")
    @classmethod
    def _validate_key(cls, value: str) -> str:
        candidate = value.strip()
        if not candidate:
            raise ValueError("secret key cannot be empty")
        return candidate


SecretLoader = Callable[[SecretRef], str]


class DatabaseSettings(BaseModel):
    """Validated database connection settings."""

    model_config = ConfigDict(extra="forbid")

    uri: MultiHostUrl | None = None
    driver: str | None = None
    user: str | None = None
    password: SecretRef | str | None = None
    host: str | None = None
    port: int | None = None
    name: str | None = None

    @field_validator("driver")
    @classmethod
    def _normalize_driver(cls, value: str | None) -> str | None:
        if value is None:
            return None
        candidate = value.strip().lower()
        if candidate not in {"sqlite", "postgresql", "mysql", "mariadb"}:
            raise ValueError(
                "driver must be one of 'sqlite', 'postgresql', 'mysql', or 'mariadb'"
            )
        return candidate

    @model_validator(mode="after")
    def _ensure_uri(self) -> "DatabaseSettings":
        if self.uri is not None:
            if any(
                value is not None
                for value in (self.driver, self.user, self.password, self.host, self.port, self.name)
            ):
                raise ValueError(
                    "database 'uri' cannot be combined with individual connection components"
                )
            return self

        required_fields = {
            "driver": self.driver,
            "user": self.user,
            "password": self.password,
            "host": self.host,
            "name": self.name,
        }
        missing = [name for name, value in required_fields.items() if value is None]
        if missing:
            raise ValueError(
                "database configuration is missing required fields: " + ", ".join(sorted(missing))
            )
        if self.driver == "sqlite":
            raise ValueError(
                "use 'uri' for sqlite connections instead of discrete connection parameters"
            )
        return self

    def resolve(self, secret_loader: SecretLoader | None = None) -> MultiHostUrl:
        """Return a fully resolved database URI, retrieving secrets if necessary."""

        if self.uri is not None:
            return self.uri

        assert self.driver is not None  # for type checkers
        assert self.user is not None
        assert self.host is not None
        assert self.name is not None

        password: str | None
        if isinstance(self.password, SecretRef):
            if secret_loader is None:
                raise SecretResolutionError(
                    f"secret loader missing for backend '{self.password.backend}'"
                )
            password = secret_loader(self.password)
        else:
            password = self.password

        if password is None:
            raise SecretResolutionError("database password is missing")

        credentials = f"{self.user}:{password}"
        host = self.host
        if self.port is not None:
            host = f"{host}:{self.port}"
        uri = f"{self.driver}://{credentials}@{host}/{self.name}"
        return MultiHostUrl(uri)


class AnalyticsSettings(BaseModel):
    """Validated analytics parameters for experiment runs."""

    model_config = ConfigDict(extra="forbid")

    window: PositiveInt = Field(..., description="Rolling window size for analytics")
    bins: PositiveInt = Field(..., description="Number of bins for entropy calculations")
    delta: float = Field(..., gt=0.0, description="Ricci graph delta parameter")

    @field_validator("window")
    @classmethod
    def _window_reasonable(cls, value: int) -> int:
        if value < 32:
            raise ValueError("analytics window must be at least 32 observations")
        return value

    @field_validator("bins")
    @classmethod
    def _bins_reasonable(cls, value: int) -> int:
        if value < 8:
            raise ValueError("analytics bins must be at least 8")
        return value

    @field_validator("delta")
    @classmethod
    def _delta_range(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("analytics delta must be between 0 and 1")
        return value


class DataSettings(BaseModel):
    """Data source configuration for experiments."""

    model_config = ConfigDict(extra="forbid")

    price_csv: Path
    price_column: str

    @field_validator("price_column")
    @classmethod
    def _normalize_column(cls, value: str) -> str:
        candidate = value.strip()
        if not candidate:
            raise ValueError("price column cannot be empty")
        return candidate


class TrackingSettings(BaseModel):
    """Tracking output configuration ensuring valid directories."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    base_dir: Path

    @field_validator("base_dir", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Any:
        if isinstance(value, Path):
            return value
        return Path(str(value))

    @model_validator(mode="after")
    def _validate_path(self) -> "TrackingSettings":
        if self.enabled:
            if not str(self.base_dir):
                raise ValueError("tracking base_dir cannot be empty when tracking is enabled")
            if ".." in self.base_dir.parts:
                raise ValueError("tracking base_dir cannot traverse parent directories")
        return self


class ExperimentSettings(BaseModel):
    """Top-level settings object for analytics experiments."""

    model_config = ConfigDict(extra="forbid")

    name: str
    debug: bool
    log_level: str
    random_seed: int
    database: DatabaseSettings
    data: DataSettings
    analytics: AnalyticsSettings
    tracking: TrackingSettings

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        candidate = value.strip().upper()
        if candidate not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("log_level must be a valid logging level name")
        return candidate

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        candidate = value.strip()
        if not candidate:
            raise ValueError("experiment name cannot be empty")
        return candidate

    @property
    def database_uri(self) -> MultiHostUrl:
        """Return the resolved database URI for downstream consumers."""

        return self.database.resolve()


def _convert_experiment_cfg(cfg: DictConfig) -> MutableMapping[str, Any]:
    container = OmegaConf.to_container(
        cfg, resolve=False, structured_config_mode=SCMode.DICT
    )
    if not isinstance(container, MutableMapping):
        raise TypeError("experiment configuration must be a mapping")
    return container


def load_experiment_settings(
    cfg: DictConfig,
    *,
    secret_loader: SecretLoader | None = None,
) -> ExperimentSettings:
    """Load and validate experiment settings from a Hydra DictConfig."""

    experiment_cfg = cfg.get("experiment")
    if experiment_cfg is None:
        raise ValueError("expected 'experiment' section in Hydra config")

    raw = _convert_experiment_cfg(experiment_cfg)

    context = {"secret_loader": secret_loader}
    settings = ExperimentSettings.model_validate(raw, context=context)

    # Resolve database secrets after validation to keep the model immutable.
    resolved_uri = settings.database.resolve(secret_loader)
    object.__setattr__(settings.database, "uri", resolved_uri)

    return settings
