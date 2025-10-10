"""Structured configuration for Kuramoto–Ricci composite workflows."""
from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator

import yaml
from pydantic import BaseModel, BaseSettings, Field, ValidationError, root_validator, validator
from pydantic.env_settings import SettingsSourceCallable

from core.indicators.multiscale_kuramoto import TimeFrame

DEFAULT_CONFIG_PATH = Path("configs/kuramoto_ricci_composite.yaml")


class ConfigError(ValueError):
    """Raised when a configuration value is invalid."""


class SettingsError(ValueError):
    """Raised when settings configuration cannot be resolved."""


def _parse_timeframe(value: Any) -> TimeFrame:
    if isinstance(value, TimeFrame):
        return value
    if isinstance(value, str):
        token = value.strip()
        if token.isdigit():
            value = int(token)
        else:
            try:
                return TimeFrame[token]
            except KeyError as exc:  # pragma: no cover - defensive branch
                raise ValueError(f"unknown timeframe label '{token}'") from exc
    try:
        return TimeFrame(int(value))
    except (ValueError, TypeError) as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"invalid timeframe value: {value!r}") from exc


def _parse_env_file(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        entries[key] = value
    return entries


@contextmanager
def _temporary_env_overrides(assignments: Mapping[str, str]) -> Iterator[None]:
    added_keys: list[str] = []
    for key, value in assignments.items():
        if key in os.environ:
            continue
        os.environ[key] = value
        added_keys.append(key)
    try:
        yield
    finally:
        for key in added_keys:
            os.environ.pop(key, None)


class KuramotoConfig(BaseModel):
    """Configuration payload for :class:`MultiScaleKuramoto`."""

    class Config:
        extra = "forbid"
        allow_mutation = False

    timeframes: tuple[TimeFrame, ...] = Field(
        default=(TimeFrame.M1, TimeFrame.M5, TimeFrame.M15, TimeFrame.H1),
        description="Ordered set of timeframes analysed by the indicator.",
    )
    use_adaptive_window: bool = Field(
        default=True,
        description="Whether to adapt the base window dynamically.",
    )
    base_window: int = Field(
        default=200,
        gt=0,
        description="Base lookback window used by the Kuramoto estimator.",
    )
    min_samples_per_scale: int = Field(
        default=64,
        gt=0,
        description="Minimum number of samples required per analysed scale.",
    )

    @root_validator(pre=True)
    def _merge_adaptive_window(cls, data: Any) -> Mapping[str, Any]:
        if data is None:
            return {}
        if not isinstance(data, Mapping):
            raise TypeError("kuramoto configuration must be a mapping")
        payload = dict(data)
        adaptive = payload.pop("adaptive_window", None)
        if isinstance(adaptive, Mapping):
            if "enabled" in adaptive and "use_adaptive_window" not in payload:
                payload["use_adaptive_window"] = adaptive["enabled"]
            if "base_window" in adaptive and "base_window" not in payload:
                payload["base_window"] = adaptive["base_window"]
        return payload

    @validator("timeframes", pre=True)
    def _coerce_timeframes(cls, value: Any) -> Sequence[Any] | tuple[TimeFrame, ...]:
        if value is None:
            return value
        if isinstance(value, (str, bytes)):
            raise TypeError("kuramoto.timeframes must be a sequence")
        if isinstance(value, Iterable):
            return tuple(_parse_timeframe(item) for item in value)
        raise TypeError("kuramoto.timeframes must be a sequence")

    @root_validator
    def _ensure_timeframes_non_empty(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        timeframes = values.get("timeframes")
        if not timeframes:
            raise ValueError("kuramoto.timeframes cannot be empty")
        return values

    def to_engine_kwargs(self) -> dict[str, Any]:
        return {
            "timeframes": self.timeframes,
            "use_adaptive_window": self.use_adaptive_window,
            "base_window": self.base_window,
            "min_samples_per_scale": self.min_samples_per_scale,
        }


class RicciTemporalConfig(BaseModel):
    class Config:
        extra = "forbid"
        allow_mutation = False

    window_size: int = Field(default=100, gt=0)
    n_snapshots: int = Field(default=8, gt=0)
    retain_history: bool = Field(default=True)


class RicciGraphConfig(BaseModel):
    class Config:
        extra = "forbid"
        allow_mutation = False

    n_levels: int = Field(default=20, gt=0)
    connection_threshold: float = Field(default=0.1, gt=0, lt=1)


class RicciConfig(BaseModel):
    class Config:
        extra = "forbid"
        allow_mutation = False

    temporal: RicciTemporalConfig = Field(default_factory=RicciTemporalConfig)
    graph: RicciGraphConfig = Field(default_factory=RicciGraphConfig)

    def to_engine_kwargs(self) -> dict[str, Any]:
        return {
            "window_size": self.temporal.window_size,
            "n_snapshots": self.temporal.n_snapshots,
            "n_levels": self.graph.n_levels,
            "retain_history": self.temporal.retain_history,
            "connection_threshold": self.graph.connection_threshold,
        }


class CompositeThresholds(BaseModel):
    class Config:
        extra = "forbid"
        allow_mutation = False

    R_strong_emergent: float = Field(default=0.8, ge=0, le=1)
    R_proto_emergent: float = Field(default=0.4, ge=0, le=1)
    coherence_min: float = Field(default=0.6, ge=0, le=1)
    ricci_negative: float = Field(default=-0.3)
    temporal_ricci: float = Field(default=-0.2)
    topological_transition: float = Field(default=0.7, ge=0, le=1)

    @root_validator
    def _validate_thresholds(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        strong = values.get("R_strong_emergent")
        proto = values.get("R_proto_emergent")
        if strong is not None and proto is not None and strong <= proto:
            raise ValueError("R_strong_emergent must exceed R_proto_emergent")
        return values


class CompositeSignals(BaseModel):
    class Config:
        extra = "forbid"
        allow_mutation = False

    min_confidence: float = Field(default=0.5, ge=0, le=1)


class CompositeConfig(BaseModel):
    class Config:
        extra = "forbid"
        allow_mutation = False

    thresholds: CompositeThresholds = Field(default_factory=CompositeThresholds)
    signals: CompositeSignals = Field(default_factory=CompositeSignals)

    def to_engine_kwargs(self) -> dict[str, Any]:
        return {
            "R_strong_emergent": self.thresholds.R_strong_emergent,
            "R_proto_emergent": self.thresholds.R_proto_emergent,
            "coherence_threshold": self.thresholds.coherence_min,
            "ricci_negative_threshold": self.thresholds.ricci_negative,
            "temporal_ricci_threshold": self.thresholds.temporal_ricci,
            "transition_threshold": self.thresholds.topological_transition,
            "min_confidence": self.signals.min_confidence,
        }


class KuramotoRicciIntegrationConfig(BaseModel):
    """Composite configuration for the Kuramoto–Ricci integration workflow."""

    class Config:
        extra = "forbid"
        allow_mutation = False

    kuramoto: KuramotoConfig = Field(default_factory=KuramotoConfig)
    ricci: RicciConfig = Field(default_factory=RicciConfig)
    composite: CompositeConfig = Field(default_factory=CompositeConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "KuramotoRicciIntegrationConfig":
        try:
            return cls.parse_obj(data or {})
        except ValidationError as exc:  # pragma: no cover - error propagation
            messages = "; ".join(error["msg"] for error in exc.errors())
            raise ConfigError(messages) from exc

    @classmethod
    def from_file(cls, path: str | Path | None) -> "KuramotoRicciIntegrationConfig":
        if path is None:
            return cls()
        payload_path = Path(path)
        if not payload_path.exists():
            raise FileNotFoundError(payload_path)
        with payload_path.open("r", encoding="utf8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, Mapping):
            raise ConfigError("configuration file must define a mapping")
        return cls.from_mapping(loaded)

    def to_engine_kwargs(self) -> dict[str, dict[str, Any]]:
        return {
            "kuramoto_config": self.kuramoto.to_engine_kwargs(),
            "ricci_config": self.ricci.to_engine_kwargs(),
            "composite_config": self.composite.to_engine_kwargs(),
        }


class YamlSettingsSource:
    """Lowest-priority settings source that loads values from YAML files."""

    def __init__(
        self,
        settings_cls: type["TradePulseSettings"],
        init_source: SettingsSourceCallable,
        env_source: SettingsSourceCallable,
        dotenv_source: SettingsSourceCallable | None = None,
    ) -> None:
        self._settings_cls = settings_cls
        self._init_source = init_source
        self._env_source = env_source
        self._dotenv_source = dotenv_source

    def __call__(self, settings: BaseSettings | None = None) -> dict[str, Any]:
        config_path = self._resolve_path(settings)
        if config_path is None:
            return {}
        try:
            text = config_path.read_text(encoding="utf8")
        except FileNotFoundError:
            return {}
        try:
            payload = yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:  # pragma: no cover - YAML parser errors
            raise SettingsError(f"failed to parse YAML configuration at {config_path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise SettingsError(f"configuration file {config_path} must define a mapping")
        return dict(payload)

    def _resolve_path(self, settings: BaseSettings | None) -> Path | None:
        for source in (self._init_source, self._env_source, self._dotenv_source):
            if source is None:
                continue
            data = self._invoke_source(source, settings)
            candidate = data.get("config_file") or data.get("config")
            if candidate:
                return Path(candidate).expanduser()

        default_path = self._default_config_path()
        if default_path is not None:
            return default_path
        return None

    def _invoke_source(
        self,
        source: SettingsSourceCallable,
        settings: BaseSettings | None,
    ) -> Mapping[str, Any]:
        if settings is not None:
            try:
                return source(settings)  # type: ignore[arg-type]
            except TypeError:
                pass
            except Exception:  # pragma: no cover - defensive guard
                return {}
        try:
            return source()  # type: ignore[call-arg]
        except Exception:  # pragma: no cover - defensive guard
            return {}

    def _default_config_path(self) -> Path | None:
        fields: Dict[str, Any] = {}
        if hasattr(self._settings_cls, "model_fields"):
            fields = getattr(self._settings_cls, "model_fields")
        elif hasattr(self._settings_cls, "__fields__"):
            fields = getattr(self._settings_cls, "__fields__")
        field = fields.get("config_file")
        default_value = getattr(field, "default", None)
        if default_value:
            return Path(default_value).expanduser()
        return None


class TradePulseSettings(BaseSettings):
    """Application-wide configuration powered by ``pydantic`` settings."""

    config_file: Path | None = Field(
        default=DEFAULT_CONFIG_PATH,
        description="Primary YAML configuration file.",
    )
    kuramoto: KuramotoConfig = Field(default_factory=KuramotoConfig)
    ricci: RicciConfig = Field(default_factory=RicciConfig)
    composite: CompositeConfig = Field(default_factory=CompositeConfig)

    @root_validator(pre=True)
    def _apply_config_alias(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "config" in values and "config_file" not in values:
            values["config_file"] = values.pop("config")
        return values

    class Config:
        env_prefix = "TRADEPULSE_"
        env_nested_delimiter = "__"
        extra = "ignore"

        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,
            file_secret_settings: SettingsSourceCallable,
        ) -> tuple[SettingsSourceCallable, ...]:
            yaml_source = YamlSettingsSource(TradePulseSettings, init_settings, env_settings)
            return (
                init_settings,
                env_settings,
                yaml_source,
                file_secret_settings,
            )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: SettingsSourceCallable,
        env_settings: SettingsSourceCallable,
        dotenv_settings: SettingsSourceCallable,
        file_secret_settings: SettingsSourceCallable,
    ) -> tuple[SettingsSourceCallable, ...]:
        yaml_source = YamlSettingsSource(settings_cls, init_settings, env_settings, dotenv_settings)
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            yaml_source,
            file_secret_settings,
        )

    def as_kuramoto_ricci_config(self) -> KuramotoRicciIntegrationConfig:
        return KuramotoRicciIntegrationConfig(
            kuramoto=self.kuramoto,
            ricci=self.ricci,
            composite=self.composite,
        )


def parse_cli_overrides(pairs: Sequence[str] | None) -> dict[str, Any]:
    """Convert CLI ``key=value`` pairs into nested dictionaries."""

    overrides: dict[str, Any] = {}
    if not pairs:
        return overrides

    for raw in pairs:
        if "=" not in raw:
            raise ConfigError(f"Invalid override '{raw}', expected format key=value")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigError("Override keys cannot be empty")
        target = overrides
        parts = [segment.strip() for segment in key.split(".") if segment.strip()]
        if not parts:
            raise ConfigError("Override keys cannot be empty")
        for segment in parts[:-1]:
            target = target.setdefault(segment, {})
            if not isinstance(target, dict):
                raise ConfigError(f"Override path '{key}' collides with a scalar value")
        parsed_value: Any
        try:
            parsed_value = yaml.safe_load(value)
        except yaml.YAMLError as exc:  # pragma: no cover - YAML parser errors
            raise ConfigError(f"Unable to parse override '{raw}': {exc}") from exc
        target[parts[-1]] = parsed_value

    return overrides


def load_kuramoto_ricci_config(
    path: str | Path | None,
    *,
    cli_overrides: Mapping[str, Any] | None = None,
) -> KuramotoRicciIntegrationConfig:
    """Load a Kuramoto–Ricci integration config with layered sources."""

    overrides = dict(cli_overrides or {})
    if path is not None:
        overrides.setdefault("config_file", Path(path))
    dotenv_path = Path.cwd() / ".env"
    assignments: Mapping[str, str] = {}
    if dotenv_path.exists():
        assignments = _parse_env_file(dotenv_path)
    with _temporary_env_overrides(assignments):
        settings = TradePulseSettings(**overrides)
    return settings.as_kuramoto_ricci_config()


__all__ = [
    "ConfigError",
    "CompositeConfig",
    "CompositeSignals",
    "CompositeThresholds",
    "KuramotoConfig",
    "KuramotoRicciIntegrationConfig",
    "RicciConfig",
    "RicciGraphConfig",
    "RicciTemporalConfig",
    "TradePulseSettings",
    "YamlSettingsSource",
    "load_kuramoto_ricci_config",
    "parse_cli_overrides",
]
