# SPDX-License-Identifier: MIT
"""Unit tests for the observability helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, ContextManager, Dict, Iterator

from core.observability import (
    ObservabilityConfig,
    ObservabilityScope,
    bootstrap_observability,
    observability_scope,
)
from core.utils.logging import correlation_context, get_correlation_id, get_logger


def test_correlation_context_propagates_id() -> None:
    """The correlation context should expose the ID via helper functions."""

    logger = get_logger(__name__)
    with correlation_context("unit-test-id"):
        assert get_correlation_id() == "unit-test-id"
        # StructuredLogger should pick up the correlation ID without explicit wiring
        logger.info("inside correlation context")
    assert get_correlation_id() is None


def test_observability_scope_exposes_handles() -> None:
    """The observability scope wires logging, metrics, and tracing handles."""

    # Logging is configured lazily inside bootstrap; tracing is optional here.
    bootstrap_observability(ObservabilityConfig(logging_json=False, metrics_port=None))

    with observability_scope("unit-operation", component="tests", attributes={"key": "value"}) as scope:
        assert scope.correlation_id
        assert scope.log_context is not None
        assert scope.metric_context is not None
        assert scope.attributes["component"] == "tests"
        assert scope.attributes["key"] == "value"
        assert scope.attributes["operation"] == "unit-operation"
        assert "correlation_id" in scope.attributes

        scope.set_status("degraded")
        assert scope.log_context["status"] == "degraded"
        assert scope.metric_context["status"] == "degraded"
        assert scope.attributes["status"] == "degraded"

        scope.add_event("checkpoint", step=1)


class _DummyMetricsCollector:
    """Test double that mimics the metrics collector interface."""

    def measure_operation(self, component: str, operation: str) -> ContextManager[Dict[str, Any]]:
        @contextmanager
        def _manager() -> Iterator[Dict[str, Any]]:
            ctx: Dict[str, Any] = {}
            yield ctx

        return _manager()


def test_observability_scope_attribute_propagation_updates_contexts() -> None:
    """Custom attributes should be reflected across all observability surfaces."""

    bootstrap_observability(
        ObservabilityConfig(logging_json=False, metrics_enabled=False, metrics_port=None)
    )

    metrics = _DummyMetricsCollector()

    with observability_scope(
        "propagation-operation",
        component="tests",
        attributes={"initial": "value"},
        metrics=metrics,
    ) as scope:
        scope.set_attribute("step", 1)

        assert scope.log_context["component"] == "tests"
        assert scope.metric_context["component"] == "tests"
        assert scope.metric_context["initial"] == "value"
        assert scope.log_context["step"] == 1
        assert scope.metric_context["step"] == 1
        assert scope.attributes["step"] == 1


def test_observability_scope_record_exception_sets_status_and_span() -> None:
    """Recording an exception should annotate all surfaces with error details."""

    class DummySpan:
        def __init__(self) -> None:
            self.attributes: Dict[str, Any] = {}
            self.exceptions: list[Exception] = []

        def set_attribute(self, key: str, value: Any) -> None:
            self.attributes[key] = value

        def record_exception(self, exc: Exception) -> None:
            self.exceptions.append(exc)

    span = DummySpan()
    scope = ObservabilityScope(
        correlation_id="unit-test",
        logger=get_logger(__name__),
        attributes={},
        span=span,
        log_context={},
        metric_context={},
    )

    exc = ValueError("boom")
    scope.record_exception(exc)

    assert scope.log_context["status"] == "error"
    assert scope.metric_context["status"] == "error"
    assert scope.attributes["status"] == "error"
    assert span.attributes["status"] == "error"
    assert span.attributes["exception.type"] == "ValueError"
    assert span.attributes["exception.message"] == "boom"
    assert exc in span.exceptions
