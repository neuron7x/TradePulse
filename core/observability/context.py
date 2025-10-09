# SPDX-License-Identifier: MIT
"""Observability context helpers that unify logs, metrics, and traces."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional

from core.utils.logging import (
    StructuredLogger,
    correlation_context,
    generate_correlation_id,
    get_logger,
)
from core.utils.metrics import MetricsCollector, get_metrics_collector

from .tracing import get_tracer, is_tracing_enabled


@dataclass
class ObservabilityScope:
    """Composite observability handles exposed inside :func:`observability_scope`."""

    correlation_id: str
    logger: StructuredLogger
    attributes: Dict[str, Any] = field(default_factory=dict)
    span: Any = None
    log_context: Optional[Dict[str, Any]] = None
    metric_context: Optional[Dict[str, Any]] = None

    def set_status(self, status: str) -> None:
        """Propagate a status indicator to logs, metrics, and traces."""

        self.attributes["status"] = status

        if isinstance(self.log_context, dict):
            self.log_context["status"] = status
        if isinstance(self.metric_context, dict):
            self.metric_context["status"] = status
        if self.span is not None:
            try:
                self.span.set_attribute("status", status)
            except AttributeError:  # pragma: no cover - defensive
                pass

    def set_attribute(self, key: str, value: Any) -> None:
        """Attach an attribute to the active span and cached attribute payload."""

        self.attributes[key] = value
        if isinstance(self.log_context, dict):
            self.log_context[key] = value
        if isinstance(self.metric_context, dict):
            self.metric_context[key] = value
        if self.span is not None:
            try:
                self.span.set_attribute(key, value)
            except AttributeError:  # pragma: no cover - defensive
                pass

    def add_event(self, name: str, **fields: Any) -> None:
        """Emit a structured event to logs and traces."""

        payload = dict(fields)
        payload.setdefault("event", name)
        self.logger.info(name, **payload)
        if self.span is not None:
            try:
                self.span.add_event(name, attributes=payload)
            except AttributeError:  # pragma: no cover - defensive
                pass

    def record_exception(self, exc: Exception) -> None:
        """Attach exception details to the active span."""

        self.set_status("error")
        if self.span is not None:
            try:
                self.span.record_exception(exc)
                self.span.set_attribute("exception.type", type(exc).__name__)
                self.span.set_attribute("exception.message", str(exc))
            except AttributeError:  # pragma: no cover - defensive
                pass


@contextmanager
def observability_scope(
    operation: str,
    *,
    component: str = "core",
    correlation_id: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    metrics: Optional[MetricsCollector] = None,
    logger_name: Optional[str] = None,
) -> Iterator[ObservabilityScope]:
    """Create a unified observability scope for an operation."""

    corr_id = correlation_id or generate_correlation_id()
    log_attrs = {"component": component}
    if attributes:
        log_attrs.update(attributes)

    structured_logger = get_logger(logger_name or component, correlation_id=corr_id)
    metrics_collector = metrics or get_metrics_collector()

    metrics_cm = metrics_collector.measure_operation(component, operation)
    tracer_cm: Any

    if is_tracing_enabled():
        tracer = get_tracer(component)
        span_attrs = {**log_attrs, "operation": operation, "correlation_id": corr_id}
        tracer_cm = tracer.start_as_current_span(operation, attributes=span_attrs)
    else:
        tracer_cm = nullcontext(None)

    with correlation_context(corr_id):
        with structured_logger.operation(operation, correlation_id=corr_id, **log_attrs) as log_ctx:
            with metrics_cm as metric_ctx:
                with tracer_cm as span_obj:
                    scope_attributes = {**log_attrs, "operation": operation, "correlation_id": corr_id}
                    scope = ObservabilityScope(
                        correlation_id=corr_id,
                        logger=structured_logger,
                        attributes=scope_attributes,
                        span=span_obj,
                        log_context=log_ctx,
                        metric_context=metric_ctx,
                    )
                    if isinstance(scope.log_context, dict):
                        scope.log_context.setdefault("correlation_id", corr_id)
                        scope.log_context.setdefault("operation", operation)
                    if isinstance(scope.metric_context, dict):
                        scope.metric_context.setdefault("component", component)
                        scope.metric_context.setdefault("operation", operation)
                        scope.metric_context.setdefault("correlation_id", corr_id)
                        for attr_key, attr_value in log_attrs.items():
                            scope.metric_context.setdefault(attr_key, attr_value)
                    yield scope


__all__ = ["ObservabilityScope", "observability_scope"]

