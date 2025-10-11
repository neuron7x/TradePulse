"""OpenTelemetry tracing utilities for TradePulse pipelines.

This module centralises the configuration of OpenTelemetry tracing and provides
helpers that make it easy to instrument the ingest → features → signals → orders
pipeline.  All helpers gracefully degrade to no-ops when the optional
``opentelemetry`` dependencies are not installed so that the rest of the code
base keeps functioning in lightweight environments.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
import os
from typing import Any, Dict, Iterator, Mapping

try:  # pragma: no cover - optional dependency import guarded at runtime
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.trace import Status, StatusCode

    _TRACE_AVAILABLE = True
except Exception:  # pragma: no cover - the dependencies are optional
    trace = None  # type: ignore[assignment]
    Resource = TracerProvider = BatchSpanProcessor = OTLPSpanExporter = None  # type: ignore[assignment]
    Status = StatusCode = None  # type: ignore[assignment]
    _TRACE_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

_DEFAULT_TRACER_NAME = "tradepulse.pipeline"


@dataclass(frozen=True)
class TracingConfig:
    """Container with tracing configuration options."""

    service_name: str = "tradepulse"
    environment: str | None = None
    exporter_endpoint: str | None = None
    exporter_insecure: bool = True
    resource_attributes: Mapping[str, Any] | None = None


def configure_tracing(config: TracingConfig | None = None) -> bool:
    """Configure OpenTelemetry tracing using ``config``.

    The function returns ``True`` when tracing is successfully initialised and
    ``False`` when the optional ``opentelemetry`` stack is not available.  When
    tracing is disabled the rest of the application keeps functioning normally.
    """

    if not _TRACE_AVAILABLE:
        LOGGER.warning("OpenTelemetry not installed; tracing is disabled")
        return False

    cfg = config or TracingConfig()

    resource_attrs: Dict[str, Any] = {
        "service.name": cfg.service_name,
        "service.namespace": "tradepulse",
    }
    if cfg.environment:
        resource_attrs["deployment.environment"] = cfg.environment
    if cfg.resource_attributes:
        resource_attrs.update(dict(cfg.resource_attributes))

    # Allow environment variables to override the endpoint for compatibility
    # with standard OTEL configuration used in managed collectors.
    endpoint = cfg.exporter_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    insecure_env = os.environ.get("OTEL_EXPORTER_OTLP_INSECURE")
    insecure = cfg.exporter_insecure if insecure_env is None else insecure_env.lower() == "true"

    provider = TracerProvider(resource=Resource.create(resource_attrs))

    exporter: OTLPSpanExporter | None
    if endpoint:
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=insecure)
    else:
        exporter = OTLPSpanExporter()

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    LOGGER.info(
        "OpenTelemetry tracing configured",
        extra={
            "extra_fields": {
                "service_name": cfg.service_name,
                "environment": cfg.environment,
                "endpoint": endpoint,
                "insecure": insecure,
            }
        },
    )
    return True


def get_tracer(name: str = _DEFAULT_TRACER_NAME):
    """Return the configured tracer or a no-op tracer when tracing is disabled."""

    if not _TRACE_AVAILABLE:
        return _NoOpTracer()
    return trace.get_tracer(name)


@contextmanager
def pipeline_span(stage: str, **attributes: Any) -> Iterator[Any]:
    """Create a span representing one pipeline stage.

    The helper ensures the stage span is recorded with the ``stage`` attribute so
    that Grafana or Jaeger views can display the ingest → features → signals →
    orders flow.  Exceptions raised inside the context automatically mark the
    span with ``StatusCode.ERROR`` and re-raise the original exception.
    """

    if not _TRACE_AVAILABLE:
        yield None
        return

    tracer = get_tracer()
    with tracer.start_as_current_span(stage) as span:  # type: ignore[assignment]
        if attributes:
            span.set_attributes(attributes)  # type: ignore[call-arg]
        try:
            yield span
        except Exception as exc:  # pragma: no cover - exercised via integration
            span.record_exception(exc)  # type: ignore[call-arg]
            span.set_status(Status(StatusCode.ERROR, str(exc)))  # type: ignore[call-arg]
            raise


class _NoOpSpan:
    """Minimal span used when OpenTelemetry is not installed."""

    def set_attributes(self, _attrs: Mapping[str, Any]) -> None:  # pragma: no cover - trivial
        return

    def record_exception(self, _exc: BaseException) -> None:  # pragma: no cover - trivial
        return

    def set_status(self, _status: Any) -> None:  # pragma: no cover - trivial
        return


class _NoOpSpanContext:
    def __enter__(self) -> _NoOpSpan:  # pragma: no cover - trivial
        return _NoOpSpan()

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        return None


class _NoOpTracer:
    """Tracer compatible stub used when tracing is disabled."""

    def start_as_current_span(self, _name: str, **_kwargs: Any) -> _NoOpSpanContext:  # pragma: no cover - trivial
        return _NoOpSpanContext()


__all__ = [
    "TracingConfig",
    "configure_tracing",
    "get_tracer",
    "pipeline_span",
]
