# SPDX-License-Identifier: MIT
"""Unit tests for the observability helpers."""

from __future__ import annotations

from core.observability import ObservabilityConfig, bootstrap_observability, observability_scope
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

        scope.set_status("degraded")
        assert scope.log_context["status"] == "degraded"
        assert scope.metric_context["status"] == "degraded"

        scope.add_event("checkpoint", step=1)
