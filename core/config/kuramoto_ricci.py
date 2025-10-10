"""Structured configuration for Kuramoto–Ricci composite workflows."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
import pydantic
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.fields import FieldInfo
from importlib import import_module
from importlib.util import find_spec
import os

if find_spec("pydantic_settings") is not None:
    pydantic_settings = import_module("pydantic_settings")
    settings_sources = import_module("pydantic_settings.sources")

    BaseSettings = pydantic_settings.BaseSettings  # type: ignore[attr-defined]
    SettingsConfigDict = pydantic_settings.SettingsConfigDict  # type: ignore[attr-defined]
    SettingsError = pydantic_settings.SettingsError  # type: ignore[attr-defined]
    PydanticBaseSettingsSource = settings_sources.PydanticBaseSettingsSource  # type: ignore[attr-defined]
    HAS_PYDANTIC_SETTINGS = True
else:  # pragma: no cover - exercised indirectly under legacy environments
    BaseSettings = pydantic.BaseSettings  # type: ignore[attr-defined]
    SettingsConfigDict = dict  # type: ignore[assignment]

    class SettingsError(ValueError):
        """Fallback settings error used when ``pydantic-settings`` is unavailable."""

    class PydanticBaseSettingsSource:  # type: ignore[too-many-ancestors]
        """Minimal shim that mimics the :mod:`pydantic-settings` API."""

        def __init__(self, settings_cls: type[BaseSettings], *sources: Any) -> None:
            self.settings_cls = settings_cls

        def __call__(self, settings_cls: type[BaseSettings] | None = None) -> dict[str, Any]:
            if settings_cls is not None:
                self.settings_cls = settings_cls
            return {}

        def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
            return None, field_name, False

    def _expand_env_pairs(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
        prefix = getattr(TradePulseSettings.Config, "env_prefix", "") or ""
        delimiter = getattr(TradePulseSettings.Config, "env_nested_delimiter", "__") or "__"
        expanded: dict[str, Any] = {}
        for key, raw_value in pairs:
            cleaned_key = key.strip()
            if prefix and cleaned_key.startswith(prefix):
                cleaned_key = cleaned_key[len(prefix) :]
            parts = [segment.strip().lower() for segment in cleaned_key.split(delimiter) if segment.strip()]
            if not parts:
                continue
            target: dict[str, Any] = expanded
            for segment in parts[:-1]:
                current = target.get(segment)
                if not isinstance(current, dict):
                    current = {}
                    target[segment] = current
                target = current
            try:
                parsed_value: Any = yaml.safe_load(raw_value)
            except yaml.YAMLError:
                parsed_value = raw_value.strip()
            target[parts[-1]] = parsed_value
        return expanded

    class DotEnvSettingsSource(PydanticBaseSettingsSource):
        """Lightweight ``.env`` reader used when ``python-dotenv`` is unavailable."""

        def __init__(
            self,
            settings_cls: type[BaseSettings],
            env_file: str | os.PathLike[str] | None,
        ) -> None:
            super().__init__(settings_cls)
            self._env_file = Path(env_file).expanduser() if env_file else None

        def __call__(self, settings_cls: type[BaseSettings] | None = None) -> dict[str, Any]:
            super().__call__(settings_cls)
            if self._env_file is None:
                return {}
            path = self._env_file
            if not path.is_absolute():
                path = Path.cwd() / path
            try:
                text = path.read_text(encoding="utf8")
            except FileNotFoundError:
                return {}
            pairs: list[tuple[str, str]] = []
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                pairs.append((key, value))
            return _expand_env_pairs(pairs)

    HAS_PYDANTIC_SETTINGS = False

AliasChoices = getattr(pydantic, "AliasChoices", None)

if AliasChoices is None:  # pragma: no cover - exercised implicitly via configuration parsing
    class AliasChoices(tuple):
        """Lightweight stand-in for :mod:`pydantic`'s ``AliasChoices`` helper."""

        __slots__ = ()

        def __new__(cls, *choices: str) -> "AliasChoices":
            normalized = tuple(str(choice) for choice in choices if str(choice))
            if not normalized:
                msg = "AliasChoices requires at least one non-empty alias"
                raise ValueError(msg)
            return super().__new__(cls, normalized)  # type: ignore[misc]

        def __repr__(self) -> str:  # pragma: no cover - debug helper
            joined = ", ".join(self)
            return f"AliasChoices({joined})"


PYDANTIC_V2 = hasattr(BaseModel, "model_fields")

if PYDANTIC_V2:
    from pydantic import field_validator, model_validator
else:  # pragma: no cover - exercised indirectly by tests under pydantic v1
    from pydantic import root_validator, validator

from core.indicators.multiscale_kuramoto import TimeFrame

DEFAULT_CONFIG_PATH = Path("configs/kuramoto_ricci_composite.yaml")


class ConfigError(ValueError):
    """Raised when a configuration value is invalid."""


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


def _merge_adaptive_window_payload(data: Any) -> Mapping[str, Any]:
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


def _coerce_timeframes_payload(value: Any) -> Sequence[Any] | tuple[TimeFrame, ...]:
    if value is None:
        return value
    if isinstance(value, (str, bytes)):
        raise TypeError("kuramoto.timeframes must be a sequence")
    if isinstance(value, Iterable):
        return tuple(_parse_timeframe(item) for item in value)
    raise TypeError("kuramoto.timeframes must be a sequence")


def _ensure_timeframes_non_empty_payload(timeframes: Sequence[TimeFrame] | None) -> None:
    if not timeframes:
        raise ValueError("kuramoto.timeframes cannot be empty")


def _deep_merge(base: Mapping[str, Any] | None, updates: Mapping[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = dict(base or {})
    for key, value in (updates or {}).items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result.get(key), value)
        else:
            result[key] = value
    return result


class KuramotoConfig(BaseModel):
    """Configuration payload for :class:`MultiScaleKuramoto`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    if not PYDANTIC_V2:
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

    if PYDANTIC_V2:

        @model_validator(mode="before")
        @classmethod
        def _merge_adaptive_window(cls, data: Any) -> Mapping[str, Any]:
            return _merge_adaptive_window_payload(data)

        @field_validator("timeframes", mode="before")
        @classmethod
        def _coerce_timeframes(cls, value: Any) -> Sequence[Any] | tuple[TimeFrame, ...]:
            return _coerce_timeframes_payload(value)

        @model_validator(mode="after")
        def _ensure_timeframes_non_empty(self) -> "KuramotoConfig":
            _ensure_timeframes_non_empty_payload(self.timeframes)
            return self

    else:

        @root_validator(pre=True)
        def _merge_adaptive_window(cls, values: Mapping[str, Any]) -> Mapping[str, Any]:
            return _merge_adaptive_window_payload(values)

        @validator("timeframes", pre=True)
        def _coerce_timeframes(cls, value: Any) -> Sequence[Any] | tuple[TimeFrame, ...]:
            return _coerce_timeframes_payload(value)

        @root_validator
        def _ensure_timeframes_non_empty(cls, values: Mapping[str, Any]) -> Mapping[str, Any]:
            _ensure_timeframes_non_empty_payload(values.get("timeframes"))
            return values

    def to_engine_kwargs(self) -> dict[str, Any]:
        return {
            "timeframes": self.timeframes,
            "use_adaptive_window": self.use_adaptive_window,
            "base_window": self.base_window,
            "min_samples_per_scale": self.min_samples_per_scale,
        }


class RicciTemporalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    if not PYDANTIC_V2:
        class Config:
            extra = "forbid"
            allow_mutation = False

    window_size: int = Field(default=100, gt=0)
    n_snapshots: int = Field(default=8, gt=0)
    retain_history: bool = Field(default=True)


class RicciGraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    if not PYDANTIC_V2:
        class Config:
            extra = "forbid"
            allow_mutation = False

    n_levels: int = Field(default=20, gt=0)
    connection_threshold: float = Field(default=0.1, gt=0, lt=1)


class RicciConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    if not PYDANTIC_V2:
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
    model_config = ConfigDict(extra="forbid", frozen=True)

    if not PYDANTIC_V2:
        class Config:
            extra = "forbid"
            allow_mutation = False

    R_strong_emergent: float = Field(default=0.8, ge=0, le=1)
    R_proto_emergent: float = Field(default=0.4, ge=0, le=1)
    coherence_min: float = Field(default=0.6, ge=0, le=1)
    ricci_negative: float = Field(default=-0.3)
    temporal_ricci: float = Field(default=-0.2)
    topological_transition: float = Field(default=0.7, ge=0, le=1)

    if PYDANTIC_V2:

        @model_validator(mode="after")
        def _validate_thresholds(self) -> "CompositeThresholds":
            if self.R_strong_emergent <= self.R_proto_emergent:
                raise ValueError("R_strong_emergent must exceed R_proto_emergent")
            return self

    else:

        @root_validator
        def _validate_thresholds(cls, values: Mapping[str, Any]) -> Mapping[str, Any]:
            if values.get("R_strong_emergent") <= values.get("R_proto_emergent"):
                raise ValueError("R_strong_emergent must exceed R_proto_emergent")
            return values


class CompositeSignals(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    if not PYDANTIC_V2:
        class Config:
            extra = "forbid"
            allow_mutation = False

    min_confidence: float = Field(default=0.5, ge=0, le=1)


class CompositeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    if not PYDANTIC_V2:
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

    model_config = ConfigDict(extra="forbid", frozen=True)

    if not PYDANTIC_V2:
        class Config:
            extra = "forbid"
            allow_mutation = False

    kuramoto: KuramotoConfig = Field(default_factory=KuramotoConfig)
    ricci: RicciConfig = Field(default_factory=RicciConfig)
    composite: CompositeConfig = Field(default_factory=CompositeConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "KuramotoRicciIntegrationConfig":
        try:
            if PYDANTIC_V2:
                return cls.model_validate(data or {})
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


class YamlSettingsSource(PydanticBaseSettingsSource):
    """Lowest-priority settings source that loads values from YAML files."""

    def __init__(
        self,
        settings_cls: type[BaseSettings] | None,
        init_source: PydanticBaseSettingsSource | None = None,
        env_source: PydanticBaseSettingsSource | None = None,
        dotenv_source: PydanticBaseSettingsSource | None = None,
    ) -> None:
        settings_cls = settings_cls or BaseSettings
        super().__init__(settings_cls)
        self.settings_cls = settings_cls
        self._init_source = init_source
        self._env_source = env_source
        self._dotenv_source = dotenv_source

    def __call__(self, settings_cls: type[BaseSettings] | None = None) -> dict[str, Any]:
        if settings_cls is not None:
            self.settings_cls = settings_cls
        config_path = self._resolve_path()
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

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def _resolve_path(self) -> Path | None:
        for source in (self._init_source, self._env_source, self._dotenv_source):
            if source is None:
                continue
            try:
                data = source()
            except SettingsError:  # pragma: no cover - defensive guard
                data = {}
            except Exception:  # pragma: no cover - defensive guard
                data = {}
            candidate = data.get("config_file") or data.get("config")
            if candidate:
                return Path(candidate).expanduser()

        if PYDANTIC_V2:
            field = self.settings_cls.model_fields.get("config_file")
            default_value = getattr(field, "default", None) if field else None
            default_factory = getattr(field, "default_factory", None) if field else None
        else:
            field = getattr(self.settings_cls, "__fields__", {}).get("config_file")
            default_value = getattr(field, "default", None) if field else None
            default_factory = getattr(field, "default_factory", None) if field else None
        if default_value is None and callable(default_factory):
            default_value = default_factory()
        if default_value:
            return Path(default_value).expanduser()
        return None


class TradePulseSettings(BaseSettings):
    """Application-wide configuration powered by ``pydantic-settings``."""

    model_config = SettingsConfigDict(
        env_prefix="TRADEPULSE_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    if not PYDANTIC_V2:
        class Config:
            env_prefix = "TRADEPULSE_"
            env_nested_delimiter = "__"
            env_file = None
            extra = "ignore"
            dotenv_path = Path(".env")

    config_file: Path | None = Field(
        default=DEFAULT_CONFIG_PATH,
        description="Primary YAML configuration file.",
        validation_alias=AliasChoices("config_file", "config"),
    )
    kuramoto: KuramotoConfig = Field(default_factory=KuramotoConfig)
    ricci: RicciConfig = Field(default_factory=RicciConfig)
    composite: CompositeConfig = Field(default_factory=CompositeConfig)

    if HAS_PYDANTIC_SETTINGS:

        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ) -> tuple[PydanticBaseSettingsSource, ...]:
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


if not HAS_PYDANTIC_SETTINGS:

    def _customise_sources(cls, init_settings, env_settings, file_secret_settings):
        env_file = getattr(TradePulseSettings.Config, "dotenv_path", None)
        dotenv_source = DotEnvSettingsSource(TradePulseSettings, env_file)
        yaml_source = YamlSettingsSource(TradePulseSettings, init_settings, env_settings, dotenv_source)
        return (
            init_settings,
            env_settings,
            dotenv_source,
            yaml_source,
            file_secret_settings,
        )

    TradePulseSettings.Config.customise_sources = classmethod(_customise_sources)  # type: ignore[attr-defined]


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

    if not HAS_PYDANTIC_SETTINGS:
        payload: dict[str, Any] = {}
        config_path: Path | None = Path(path) if path is not None else None
        if config_path is not None and config_path.exists():
            with config_path.open("r", encoding="utf8") as handle:
                loaded = yaml.safe_load(handle) or {}
            if not isinstance(loaded, Mapping):
                raise ConfigError("configuration file must define a mapping")
            payload = _deep_merge(payload, loaded)
        dotenv_file = getattr(TradePulseSettings.Config, "dotenv_path", None)
        if dotenv_file:
            pairs: list[tuple[str, str]] = []
            dotenv_path = Path(dotenv_file)
            if not dotenv_path.is_absolute():
                dotenv_path = Path.cwd() / dotenv_path
            if dotenv_path.exists():
                for raw_line in dotenv_path.read_text(encoding="utf8").splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    pairs.append((key, value))
            payload = _deep_merge(payload, _expand_env_pairs(pairs))
        env_pairs = [
            (key, value)
            for key, value in os.environ.items()
            if key.startswith((getattr(TradePulseSettings.Config, "env_prefix", "") or ""))
        ]
        payload = _deep_merge(payload, _expand_env_pairs(env_pairs))
        payload = _deep_merge(payload, cli_overrides)
        return KuramotoRicciIntegrationConfig.from_mapping(payload)

    overrides = dict(cli_overrides or {})
    if path is not None:
        overrides.setdefault("config_file", Path(path))
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
