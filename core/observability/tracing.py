# SPDX-License-Identifier: MIT
"""OpenTelemetry tracing utilities for TradePulse."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    trace = None  # type: ignore
    TracerProvider = object  # type: ignore
    BatchSpanProcessor = object  # type: ignore
    ConsoleSpanExporter = object  # type: ignore
    TraceIdRatioBased = object  # type: ignore
    JaegerExporter = object  # type: ignore
    OTLPSpanExporter = object  # type: ignore
    Resource = object  # type: ignore
    OTEL_AVAILABLE = False


logger = logging.getLogger(__name__)

_tracer_provider: Optional[TracerProvider] = None


@dataclass
class TracingConfig:
    """Configuration for TradePulse tracing exporters."""

    service_name: str = "tradepulse"
    exporter: str = "otlp"
    endpoint: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    insecure: bool = False
    sample_ratio: float = 1.0
    console_debug: bool = False


def configure_tracing(config: TracingConfig) -> None:
    """Configure global tracing according to the provided configuration."""

    if not OTEL_AVAILABLE:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "opentelemetry packages are not installed. Install opentelemetry-api, "
            "opentelemetry-sdk, and the desired exporters to enable tracing."
        )

    resource = Resource.create({"service.name": config.service_name})
    provider = TracerProvider(resource=resource, sampler=TraceIdRatioBased(config.sample_ratio))

    exporter_name = config.exporter.lower()
    if exporter_name == "otlp":
        exporter = OTLPSpanExporter(endpoint=config.endpoint, headers=config.headers or None)
    elif exporter_name == "jaeger":
        exporter = JaegerExporter(
            collector_endpoint=config.endpoint,
            insecure=config.insecure,
        )
    elif exporter_name == "console":
        exporter = ConsoleSpanExporter()
    else:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported tracing exporter: {config.exporter}")

    provider.add_span_processor(BatchSpanProcessor(exporter))

    if config.console_debug and exporter_name != "console":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)

    global _tracer_provider
    _tracer_provider = provider

    logger.info(
        "Tracing configured",
        extra={
            "extra_fields": {
                "service_name": config.service_name,
                "exporter": exporter_name,
                "endpoint": config.endpoint,
                "sample_ratio": config.sample_ratio,
            }
        },
    )


def shutdown_tracing() -> None:
    """Shutdown the active tracer provider to flush spans."""

    if not OTEL_AVAILABLE or _tracer_provider is None:  # pragma: no cover - guard
        return

    _tracer_provider.shutdown()


def is_tracing_enabled() -> bool:
    """Return ``True`` when tracing is configured."""

    return _tracer_provider is not None


def get_tracer(name: str) -> "trace.Tracer":  # type: ignore[name-defined]
    """Return a tracer for the given module name."""

    if not OTEL_AVAILABLE:  # pragma: no cover - guard
        raise RuntimeError(
            "Tracing requested but opentelemetry is not available. Install the necessary "
            "packages or guard the call with `is_tracing_enabled()`."
        )

    return trace.get_tracer(name)


__all__ = [
    "TracingConfig",
    "configure_tracing",
    "shutdown_tracing",
    "is_tracing_enabled",
    "get_tracer",
]

