"""Structured configuration for Kuramoto–Ricci composite workflows."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from core.indicators.multiscale_kuramoto import TimeFrame


class ConfigError(ValueError):
    """Raised when a configuration value is invalid."""


def _as_positive_int(value: Any, *, name: str) -> int:
    try:
        ivalue = int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive branch
        raise ConfigError(f"{name} must be an integer") from exc
    if ivalue <= 0:
        raise ConfigError(f"{name} must be positive")
    return ivalue


def _as_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    raise ConfigError(f"{name} must be a boolean")


def _as_timeframe(value: Any) -> TimeFrame:
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
                raise ConfigError(f"unknown timeframe label '{token}'") from exc
    try:
        return TimeFrame(int(value))
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"invalid timeframe value: {value!r}") from exc


def _as_float(value: Any, *, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive branch
        raise ConfigError(f"{name} must be numeric") from exc


@dataclass(frozen=True)
class KuramotoConfig:
    """Configuration payload for :class:`MultiScaleKuramoto`."""

    timeframes: tuple[TimeFrame, ...] = field(
        default_factory=lambda: (
            TimeFrame.M1,
            TimeFrame.M5,
            TimeFrame.M15,
            TimeFrame.H1,
        )
    )
    use_adaptive_window: bool = True
    base_window: int = 200
    min_samples_per_scale: int = 64

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "KuramotoConfig":
        data = data or {}
        raw_timeframes = data.get("timeframes")
        if raw_timeframes is None:
            timeframes: tuple[TimeFrame, ...] = cls().timeframes
        else:
            if not isinstance(raw_timeframes, Iterable) or isinstance(raw_timeframes, (str, bytes)):
                raise ConfigError("kuramoto.timeframes must be a sequence")
            parsed = tuple(_as_timeframe(v) for v in raw_timeframes)
            if not parsed:
                raise ConfigError("kuramoto.timeframes cannot be empty")
            timeframes = parsed

        base_window = _as_positive_int(data.get("base_window", cls().base_window), name="kuramoto.base_window")
        min_samples = _as_positive_int(
            data.get("min_samples_per_scale", cls().min_samples_per_scale), name="kuramoto.min_samples_per_scale"
        )
        adaptive_cfg_raw = data.get("adaptive_window")
        adaptive_cfg = adaptive_cfg_raw if isinstance(adaptive_cfg_raw, Mapping) else {}

        use_adaptive_value = data.get("use_adaptive_window")
        if use_adaptive_value is not None:
            use_adaptive = _as_bool(use_adaptive_value, name="kuramoto.use_adaptive_window")
        elif "enabled" in adaptive_cfg:
            use_adaptive = _as_bool(adaptive_cfg["enabled"], name="kuramoto.adaptive_window.enabled")
        else:
            use_adaptive = cls().use_adaptive_window

        if "base_window" in adaptive_cfg:
            base_window = _as_positive_int(adaptive_cfg["base_window"], name="kuramoto.adaptive_window.base_window")

        return cls(timeframes=timeframes, use_adaptive_window=use_adaptive, base_window=base_window, min_samples_per_scale=min_samples)

    def to_engine_kwargs(self) -> dict[str, Any]:
        return {
            "timeframes": self.timeframes,
            "use_adaptive_window": self.use_adaptive_window,
            "base_window": self.base_window,
            "min_samples_per_scale": self.min_samples_per_scale,
        }


@dataclass(frozen=True)
class RicciTemporalConfig:
    window_size: int = 100
    n_snapshots: int = 8
    retain_history: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "RicciTemporalConfig":
        data = data or {}
        window = _as_positive_int(data.get("window_size", cls().window_size), name="ricci.temporal.window_size")
        n_snaps = _as_positive_int(data.get("n_snapshots", cls().n_snapshots), name="ricci.temporal.n_snapshots")
        retain = _as_bool(data.get("retain_history", cls().retain_history), name="ricci.temporal.retain_history")
        return cls(window_size=window, n_snapshots=n_snaps, retain_history=retain)


@dataclass(frozen=True)
class RicciGraphConfig:
    n_levels: int = 20
    connection_threshold: float = 0.1

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "RicciGraphConfig":
        data = data or {}
        n_levels = _as_positive_int(data.get("n_levels", cls().n_levels), name="ricci.graph.n_levels")
        conn = _as_float(data.get("connection_threshold", cls().connection_threshold), name="ricci.graph.connection_threshold")
        if conn <= 0 or conn >= 1:
            raise ConfigError("ricci.graph.connection_threshold must be between 0 and 1")
        return cls(n_levels=n_levels, connection_threshold=conn)


@dataclass(frozen=True)
class RicciConfig:
    temporal: RicciTemporalConfig = field(default_factory=RicciTemporalConfig)
    graph: RicciGraphConfig = field(default_factory=RicciGraphConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "RicciConfig":
        data = data or {}
        temporal = RicciTemporalConfig.from_mapping(data.get("temporal"))
        graph = RicciGraphConfig.from_mapping(data.get("graph"))
        return cls(temporal=temporal, graph=graph)

    def to_engine_kwargs(self) -> dict[str, Any]:
        return {
            "window_size": self.temporal.window_size,
            "n_snapshots": self.temporal.n_snapshots,
            "n_levels": self.graph.n_levels,
            "retain_history": self.temporal.retain_history,
            "connection_threshold": self.graph.connection_threshold,
        }


@dataclass(frozen=True)
class CompositeThresholds:
    R_strong_emergent: float = 0.8
    R_proto_emergent: float = 0.4
    coherence_min: float = 0.6
    ricci_negative: float = -0.3
    temporal_ricci: float = -0.2
    topological_transition: float = 0.7

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "CompositeThresholds":
        data = data or {}
        strong = _as_float(data.get("R_strong_emergent", cls().R_strong_emergent), name="composite.thresholds.R_strong_emergent")
        proto = _as_float(data.get("R_proto_emergent", cls().R_proto_emergent), name="composite.thresholds.R_proto_emergent")
        coherence = _as_float(data.get("coherence_min", cls().coherence_min), name="composite.thresholds.coherence_min")
        ricci_neg = _as_float(data.get("ricci_negative", cls().ricci_negative), name="composite.thresholds.ricci_negative")
        temporal = _as_float(data.get("temporal_ricci", cls().temporal_ricci), name="composite.thresholds.temporal_ricci")
        transition = _as_float(data.get("topological_transition", cls().topological_transition), name="composite.thresholds.topological_transition")

        if not (0.0 <= proto <= 1.0 and 0.0 <= strong <= 1.0):
            raise ConfigError("composite.thresholds R values must be between 0 and 1")
        if strong <= proto:
            raise ConfigError("R_strong_emergent must exceed R_proto_emergent")
        if not (0.0 <= coherence <= 1.0):
            raise ConfigError("coherence_min must be between 0 and 1")
        if not (0.0 <= transition <= 1.0):
            raise ConfigError("topological_transition must be between 0 and 1")
        return cls(
            R_strong_emergent=strong,
            R_proto_emergent=proto,
            coherence_min=coherence,
            ricci_negative=ricci_neg,
            temporal_ricci=temporal,
            topological_transition=transition,
        )


@dataclass(frozen=True)
class CompositeSignals:
    min_confidence: float = 0.5

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "CompositeSignals":
        data = data or {}
        min_conf = _as_float(data.get("min_confidence", cls().min_confidence), name="composite.signals.min_confidence")
        if not (0.0 <= min_conf <= 1.0):
            raise ConfigError("min_confidence must be between 0 and 1")
        return cls(min_confidence=min_conf)


@dataclass(frozen=True)
class CompositeConfig:
    thresholds: CompositeThresholds = field(default_factory=CompositeThresholds)
    signals: CompositeSignals = field(default_factory=CompositeSignals)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "CompositeConfig":
        data = data or {}
        thresholds = CompositeThresholds.from_mapping(data.get("thresholds"))
        signals = CompositeSignals.from_mapping(data.get("signals"))
        return cls(thresholds=thresholds, signals=signals)

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


@dataclass(frozen=True)
class KuramotoRicciIntegrationConfig:
    kuramoto: KuramotoConfig = field(default_factory=KuramotoConfig)
    ricci: RicciConfig = field(default_factory=RicciConfig)
    composite: CompositeConfig = field(default_factory=CompositeConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "KuramotoRicciIntegrationConfig":
        data = data or {}
        kuramoto = KuramotoConfig.from_mapping(data.get("kuramoto"))
        ricci = RicciConfig.from_mapping(data.get("ricci"))
        composite = CompositeConfig.from_mapping(data.get("composite"))
        return cls(kuramoto=kuramoto, ricci=ricci, composite=composite)

    @classmethod
    def from_file(cls, path: str | Path | None) -> "KuramotoRicciIntegrationConfig":
        if path is None:
            return cls()
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf8") as fh:
            payload = yaml.safe_load(fh) or {}
        if not isinstance(payload, Mapping):
            raise ConfigError("configuration file must define a mapping")
        return cls.from_mapping(payload)

    def to_engine_kwargs(self) -> dict[str, dict[str, Any]]:
        return {
            "kuramoto_config": self.kuramoto.to_engine_kwargs(),
            "ricci_config": self.ricci.to_engine_kwargs(),
            "composite_config": self.composite.to_engine_kwargs(),
        }


def load_kuramoto_ricci_config(path: str | Path | None) -> KuramotoRicciIntegrationConfig:
    """Load a Kuramoto–Ricci integration config, falling back to defaults."""

    if path is None:
        return KuramotoRicciIntegrationConfig()
    try:
        return KuramotoRicciIntegrationConfig.from_file(path)
    except FileNotFoundError:
        return KuramotoRicciIntegrationConfig()
