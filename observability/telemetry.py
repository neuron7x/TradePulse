"""Unified OpenTelemetry configuration helpers for TradePulse services."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from .tracing import TracingConfig, configure_tracing

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency import guarded at runtime
    from opentelemetry import metrics as otel_metrics
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
    from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler, set_logger_provider
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource

    _OTEL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency missing
    otel_metrics = None  # type: ignore[assignment]
    OTLPMetricExporter = OTLPLogExporter = None  # type: ignore[assignment]
    LoggerProvider = LoggingHandler = set_logger_provider = None  # type: ignore[assignment]
    BatchLogRecordProcessor = MeterProvider = PeriodicExportingMetricReader = None  # type: ignore[assignment]
    Resource = None  # type: ignore[assignment]
    _OTEL_AVAILABLE = False


@dataclass(frozen=True)
class MetricsConfig:
    """Configuration options for OpenTelemetry metrics."""

    enabled: bool = True
    exporter_endpoint: str | None = None
    exporter_insecure: bool = True
    resource_attributes: Mapping[str, Any] | None = None
    collection_interval: float = 60.0


@dataclass(frozen=True)
class LoggingConfig:
    """Configuration options for OpenTelemetry logs."""

    enabled: bool = True
    exporter_endpoint: str | None = None
    exporter_insecure: bool = True
    resource_attributes: Mapping[str, Any] | None = None
    log_level: int = logging.NOTSET


@dataclass(frozen=True)
class TelemetryConfig:
    """Aggregate configuration for traces, metrics, and logs."""

    service_name: str = "tradepulse"
    environment: str | None = None
    resource_attributes: Mapping[str, Any] | None = None
    tracing: TracingConfig | None = None
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


@dataclass(frozen=True)
class TelemetryStatus:
    """Return value describing which telemetry signals were configured."""

    tracing: bool
    metrics: bool
    logging: bool


_METRIC_PROVIDER: MeterProvider | None = None
_LOGGER_PROVIDER: LoggerProvider | None = None
_LOGGING_HANDLER: logging.Handler | None = None


def _resource_from_attributes(
    service_name: str,
    environment: str | None,
    base_attributes: Mapping[str, Any] | None,
    override: Mapping[str, Any] | None,
) -> Resource | None:
    if not _OTEL_AVAILABLE or Resource is None:  # pragma: no cover - defensive guard
        return None

    attributes: dict[str, Any] = {}
    if base_attributes:
        attributes.update(dict(base_attributes))
    if override:
        attributes.update(dict(override))

    if environment:
        attributes["deployment.environment"] = environment
    if service_name:
        attributes["service.name"] = service_name
        attributes.setdefault("service.namespace", "tradepulse")

    return Resource.create(dict(attributes))


def _configure_metrics(config: MetricsConfig, resource: Resource | None) -> bool:
    global _METRIC_PROVIDER

    if not (_OTEL_AVAILABLE and otel_metrics and MeterProvider and PeriodicExportingMetricReader and OTLPMetricExporter):
        return False
    if not config.enabled:
        return False

    resolved_resource = resource
    if config.resource_attributes:
        override = _resource_from_attributes("", None, None, config.resource_attributes)
        if override is not None:
            resolved_resource = (resolved_resource or Resource.create({})).merge(override)

    interval = max(1.0, float(config.collection_interval))
    exporter = OTLPMetricExporter(endpoint=config.exporter_endpoint, insecure=config.exporter_insecure)
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=int(interval * 1000))
    provider = MeterProvider(resource=resolved_resource, metric_readers=[reader])
    otel_metrics.set_meter_provider(provider)
    _METRIC_PROVIDER = provider
    return True


def _configure_logging(config: LoggingConfig, resource: Resource | None) -> bool:
    global _LOGGER_PROVIDER, _LOGGING_HANDLER

    if not (
        _OTEL_AVAILABLE
        and LoggerProvider
        and LoggingHandler
        and BatchLogRecordProcessor
        and OTLPLogExporter
        and set_logger_provider
    ):
        return False
    if not config.enabled:
        return False

    resolved_resource = resource
    if config.resource_attributes:
        override = _resource_from_attributes("", None, None, config.resource_attributes)
        if override is not None:
            resolved_resource = (resolved_resource or Resource.create({})).merge(override)

    provider = LoggerProvider(resource=resolved_resource)
    exporter = OTLPLogExporter(endpoint=config.exporter_endpoint, insecure=config.exporter_insecure)
    processor = BatchLogRecordProcessor(exporter)
    provider.add_log_record_processor(processor)
    set_logger_provider(provider)

    handler = LoggingHandler(level=config.log_level, logger_provider=provider)
    root = logging.getLogger()
    if _LOGGING_HANDLER and _LOGGING_HANDLER in root.handlers:
        root.removeHandler(_LOGGING_HANDLER)
    root.addHandler(handler)

    _LOGGER_PROVIDER = provider
    _LOGGING_HANDLER = handler
    return True


def configure_telemetry(config: TelemetryConfig | None = None) -> TelemetryStatus:
    """Configure tracing, metrics, and logs using the supplied configuration."""

    cfg = config or TelemetryConfig()
    trace_config = cfg.tracing or TracingConfig(
        service_name=cfg.service_name,
        environment=cfg.environment,
        resource_attributes=cfg.resource_attributes,
    )

    tracing_enabled = configure_tracing(trace_config)
    resource = _resource_from_attributes(cfg.service_name, cfg.environment, cfg.resource_attributes, None)
    metrics_enabled = _configure_metrics(cfg.metrics, resource)
    logs_enabled = _configure_logging(cfg.logging, resource)

    LOGGER.info(
        "Telemetry configured",
        extra={
            "extra_fields": {
                "tracing": tracing_enabled,
                "metrics": metrics_enabled,
                "logs": logs_enabled,
            }
        },
    )
    return TelemetryStatus(tracing=bool(tracing_enabled), metrics=metrics_enabled, logging=logs_enabled)


__all__ = [
    "MetricsConfig",
    "LoggingConfig",
    "TelemetryConfig",
    "TelemetryStatus",
    "configure_telemetry",
]

