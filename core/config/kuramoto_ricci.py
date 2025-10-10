# SPDX-License-Identifier: MIT
"""Configuration utilities for the Kuramoto–Ricci analytics stack."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict

import yaml
from pydantic import BaseModel, Field, ValidationError

try:  # pragma: no cover - runtime feature detection
    from pydantic import ConfigDict, field_validator, model_validator
except ImportError:  # pragma: no cover - executed on pydantic < 2
    ConfigDict = None  # type: ignore[assignment]
    field_validator = None  # type: ignore[assignment]
    model_validator = None  # type: ignore[assignment]

try:  # pragma: no cover - runtime feature detection
    from pydantic import validator as v1_validator
except ImportError:  # pragma: no cover - executed on pydantic >= 2
    v1_validator = None  # type: ignore[assignment]

try:  # pragma: no cover - runtime feature detection
    from pydantic import root_validator as v1_root_validator
except ImportError:  # pragma: no cover - executed on pydantic >= 2
    v1_root_validator = None  # type: ignore[assignment]

if v1_validator is None:  # pragma: no cover - type checking helper
    def v1_validator(*_: Any, **__: Any) -> Any:  # type: ignore[no-redef]
        raise RuntimeError("Pydantic v1 validator helpers unavailable")

if v1_root_validator is None:  # pragma: no cover - type checking helper
    def v1_root_validator(*_: Any, **__: Any) -> Any:  # type: ignore[no-redef]
        raise RuntimeError("Pydantic v1 root validator helpers unavailable")

from core.indicators.multiscale_kuramoto import TimeFrame

DEFAULT_CONFIG_PATH = Path("configs/kuramoto_ricci_composite.yaml")

SettingsSource = Callable[[], Mapping[str, Any]]


class ConfigError(ValueError):
    """Raised when a configuration value is invalid."""


class SettingsError(ValueError):
    """Raised when settings configuration cannot be resolved."""


class ImmutableModel(BaseModel):
    """Base model that is strict and immutable across Pydantic releases."""

    if ConfigDict is not None:  # pragma: no branch - runtime flag
        model_config = ConfigDict(extra="forbid", frozen=True)
    else:
        class Config:  # type: ignore[too-many-ancestors]
            extra = "forbid"
            allow_mutation = False


def _parse_scalar(value: str) -> Any:
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:  # pragma: no cover - defensive guard
        return value
    return parsed


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


def _normalise_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        normalized_key = key
        if isinstance(key, str) and key.isupper():
            normalized_key = key.lower()
        if isinstance(value, Mapping):
            result[str(normalized_key)] = _normalise_mapping(value)
        else:
            result[str(normalized_key)] = value
    return result


def _deep_update(target: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), MutableMapping):
            _deep_update(target[key], value)  # type: ignore[index]
        else:
            target[key] = value  # type: ignore[index]
    return target


def _parse_env_variables(prefix: str, delimiter: str = "__") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        remainder = key[len(prefix) :]
        if remainder.startswith("_"):
            remainder = remainder[1:]
        parts = [segment.strip().lower() for segment in remainder.split(delimiter) if segment.strip()]
        if not parts:
            continue
        cursor: MutableMapping[str, Any] = result
        for segment in parts[:-1]:
            next_value = cursor.get(segment)
            if not isinstance(next_value, MutableMapping):
                next_value = {}
                cursor[segment] = next_value
            cursor = next_value  # type: ignore[assignment]
        cursor[parts[-1]] = _parse_scalar(value)
    return result


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf8")
    except FileNotFoundError:
        return {}
    try:
        payload = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - YAML parser errors
        raise SettingsError(f"failed to parse YAML configuration at {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SettingsError(f"configuration file {path} must define a mapping")
    return _normalise_mapping(payload)


def _resolve_config_path(*sources: Mapping[str, Any], default: Path | None = None) -> Path | None:
    for source in sources:
        if not source:
            continue
        candidate = source.get("config_file") or source.get("config")
        if candidate is None:
            continue
        return Path(candidate).expanduser()
    return default


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


class KuramotoConfig(ImmutableModel):
    """Configuration payload for :class:`MultiScaleKuramoto`."""

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

    if field_validator is not None:
        @field_validator("timeframes", mode="before")
        def _coerce_timeframes(cls, value: Any) -> Sequence[Any] | tuple[TimeFrame, ...]:
            if value is None:
                return value
            if isinstance(value, (str, bytes)):
                raise TypeError("kuramoto.timeframes must be a sequence")
            if isinstance(value, Iterable):
                sequence = tuple(_parse_timeframe(item) for item in value)
                if not sequence:
                    raise ValueError("kuramoto.timeframes cannot be empty")
                return sequence
            raise TypeError("kuramoto.timeframes must be a sequence")

        @model_validator(mode="before")
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

    else:
        @v1_validator("timeframes", pre=True)  # type: ignore[misc]
        def _coerce_timeframes(cls, value: Any) -> Sequence[Any] | tuple[TimeFrame, ...]:
            if value is None:
                return value
            if isinstance(value, (str, bytes)):
                raise TypeError("kuramoto.timeframes must be a sequence")
            if isinstance(value, Iterable):
                sequence = tuple(_parse_timeframe(item) for item in value)
                if not sequence:
                    raise ValueError("kuramoto.timeframes cannot be empty")
                return sequence
            raise TypeError("kuramoto.timeframes must be a sequence")

        @v1_root_validator(pre=True)  # type: ignore[misc]
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


    def to_engine_kwargs(self) -> dict[str, Any]:
        return {
            "timeframes": self.timeframes,
            "use_adaptive_window": self.use_adaptive_window,
            "base_window": self.base_window,
            "min_samples_per_scale": self.min_samples_per_scale,
        }


class RicciTemporalConfig(ImmutableModel):
    window_size: int = Field(default=100, gt=0)
    n_snapshots: int = Field(default=8, gt=0)
    retain_history: bool = Field(default=True)


class RicciGraphConfig(ImmutableModel):
    n_levels: int = Field(default=20, gt=0)
    connection_threshold: float = Field(default=0.1, gt=0, lt=1)


class RicciConfig(ImmutableModel):
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


class CompositeThresholds(ImmutableModel):
    R_strong_emergent: float = Field(default=0.8, ge=0, le=1)
    R_proto_emergent: float = Field(default=0.4, ge=0, le=1)
    coherence_min: float = Field(default=0.6, ge=0, le=1)
    ricci_negative: float = Field(default=-0.3)
    temporal_ricci: float = Field(default=-0.2)
    topological_transition: float = Field(default=0.7, ge=0, le=1)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.R_strong_emergent <= self.R_proto_emergent:
            raise ValueError("R_strong_emergent must exceed R_proto_emergent")


class CompositeSignals(ImmutableModel):
    min_confidence: float = Field(default=0.5, ge=0, le=1)


class CompositeConfig(ImmutableModel):
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


class KuramotoRicciIntegrationConfig(ImmutableModel):
    """Composite configuration for the Kuramoto–Ricci integration workflow."""

    kuramoto: KuramotoConfig = Field(default_factory=KuramotoConfig)
    ricci: RicciConfig = Field(default_factory=RicciConfig)
    composite: CompositeConfig = Field(default_factory=CompositeConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "KuramotoRicciIntegrationConfig":
        normalised = _normalise_mapping(data or {})
        try:
            if ConfigDict is not None:
                return cls.model_validate(normalised)
            return cls.parse_obj(normalised)
        except ValidationError as exc:  # pragma: no cover - error propagation
            messages = "; ".join(error["msg"] for error in exc.errors())
            raise ConfigError(messages) from exc

    @classmethod
    def from_file(cls, path: str | Path | None) -> "KuramotoRicciIntegrationConfig":
        if path is None:
            return cls()
        payload_path = Path(path)
        payload = _load_yaml_mapping(payload_path)
        return cls.from_mapping(payload)

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
        init_source: SettingsSource,
        env_source: SettingsSource,
        dotenv_source: SettingsSource | None = None,
    ) -> None:
        self._settings_cls = settings_cls
        self._init_source = init_source
        self._env_source = env_source
        self._dotenv_source = dotenv_source

    def __call__(self) -> dict[str, Any]:
        config_path = self._resolve_path()
        if config_path is None:
            return {}
        return _load_yaml_mapping(config_path)

    def _resolve_path(self) -> Path | None:
        init_payload = _normalise_mapping(dict(self._init_source()))
        env_payload = _normalise_mapping(dict(self._env_source()))
        dotenv_payload: Mapping[str, Any] = {}
        if self._dotenv_source is not None:
            dotenv_payload = _normalise_mapping(dict(self._dotenv_source()))
        default = self._settings_cls.default_config_path()
        return _resolve_config_path(init_payload, env_payload, dotenv_payload, default=default)


class TradePulseSettings(ImmutableModel):
    """Application-wide configuration powered by layered sources."""

    config_file: Path | None = Field(
        default=DEFAULT_CONFIG_PATH,
        description="Primary YAML configuration file.",
    )
    kuramoto: KuramotoConfig = Field(default_factory=KuramotoConfig)
    ricci: RicciConfig = Field(default_factory=RicciConfig)
    composite: CompositeConfig = Field(default_factory=CompositeConfig)

    if field_validator is not None:
        @field_validator("config_file", mode="before")
        def _coerce_config_path(cls, value: Any) -> Path | None:
            if value is None or isinstance(value, Path):
                return value
            return Path(str(value)).expanduser()

        @model_validator(mode="before")
        def _apply_config_alias(cls, data: Any) -> Mapping[str, Any]:
            if data is None:
                return {}
            if not isinstance(data, Mapping):
                raise TypeError("settings payload must be a mapping")
            payload = dict(data)
            if "config" in payload and "config_file" not in payload:
                payload["config_file"] = payload.pop("config")
            return payload
    else:
        @v1_validator("config_file", pre=True)  # type: ignore[misc]
        def _coerce_config_path(cls, value: Any) -> Path | None:
            if value is None or isinstance(value, Path):
                return value
            return Path(str(value)).expanduser()

        @v1_root_validator(pre=True)  # type: ignore[misc]
        def _apply_config_alias(cls, data: Any) -> Mapping[str, Any]:
            if data is None:
                return {}
            if not isinstance(data, Mapping):
                raise TypeError("settings payload must be a mapping")
            payload = dict(data)
            if "config" in payload and "config_file" not in payload:
                payload["config_file"] = payload.pop("config")
            return payload

    @classmethod
    def default_config_path(cls) -> Path | None:
        return DEFAULT_CONFIG_PATH

    @classmethod
    def load(
        cls,
        *,
        cli_overrides: Mapping[str, Any] | None = None,
        env_prefix: str = "TRADEPULSE_",
        env_delimiter: str = "__",
    ) -> "TradePulseSettings":
        init_payload = _normalise_mapping(dict(cli_overrides or {}))
        env_payload = _parse_env_variables(env_prefix, env_delimiter)
        config_path = _resolve_config_path(init_payload, env_payload, default=cls.default_config_path())
        yaml_payload: Mapping[str, Any] = {}
        if config_path is not None:
            yaml_payload = _load_yaml_mapping(config_path)
        merged: dict[str, Any] = {}
        _deep_update(merged, yaml_payload)
        _deep_update(merged, env_payload)
        _deep_update(merged, init_payload)
        if config_path is not None:
            merged.setdefault("config_file", config_path)
        if ConfigDict is not None:
            return cls.model_validate(merged)
        return cls.parse_obj(merged)

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
            next_value = target.setdefault(segment, {})
            if not isinstance(next_value, dict):
                raise ConfigError(f"Override path '{key}' collides with a scalar value")
            target = next_value
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
        settings = TradePulseSettings.load(cli_overrides=overrides)
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
