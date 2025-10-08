# SPDX-License-Identifier: MIT
"""Tests for error handling functionality."""

from __future__ import annotations

import time

import pytest

from core.indicators.base import (
    BaseFeature,
    ErrorPolicy,
    ExecutionStatus,
    FeatureResult,
)
from core.indicators.errors import (
    CircuitBreaker,
    ErrorAggregator,
    ErrorContext,
    create_error_result,
    handle_transform_error,
    with_error_handling,
)


class ErrorFeature(BaseFeature):
    """Feature that always raises an error."""
    
    def transform(self, data, **kwargs):
        raise ValueError("Test error")


class SuccessFeature(BaseFeature):
    """Feature that always succeeds."""
    
    def transform(self, data, **kwargs):
        return FeatureResult(name=self.name, value=float(data) * 2)


def test_error_context_creation():
    """Test ErrorContext creation."""
    try:
        raise ValueError("Test error")
    except ValueError as e:
        import traceback
        context = ErrorContext(
            error_type="ValueError",
            error_message="Test error",
            stack_trace=traceback.format_exc(),
            feature_name="test_feature",
        )
    
    assert context.error_type == "ValueError"
    assert context.error_message == "Test error"
    assert "ValueError" in context.stack_trace
    assert context.feature_name == "test_feature"


def test_error_context_to_dict():
    """Test ErrorContext serialization."""
    context = ErrorContext(
        error_type="ValueError",
        error_message="Test",
        stack_trace="trace",
    )
    
    data = context.to_dict()
    
    assert data["error_type"] == "ValueError"
    assert data["error_message"] == "Test"
    assert data["stack_trace"] == "trace"
    assert "timestamp" in data


def test_create_error_result():
    """Test error result creation."""
    error = ValueError("Test error")
    result = create_error_result("test_feature", error)
    
    assert result.name == "test_feature"
    assert result.status == ExecutionStatus.FAILED
    assert result.error == "Test error"
    assert result.value is None


def test_handle_transform_error_raise_policy():
    """Test error handling with RAISE policy."""
    error = ValueError("Test error")
    
    with pytest.raises(ValueError, match="Test error"):
        handle_transform_error(
            "test_feature",
            error,
            ErrorPolicy.RAISE,
        )


def test_handle_transform_error_warn_policy():
    """Test error handling with WARN policy."""
    error = ValueError("Test error")
    
    result = handle_transform_error(
        "test_feature",
        error,
        ErrorPolicy.WARN,
    )
    
    assert result.status == ExecutionStatus.FAILED
    assert result.error == "Test error"


def test_handle_transform_error_skip_policy():
    """Test error handling with SKIP policy."""
    error = ValueError("Test error")
    
    result = handle_transform_error(
        "test_feature",
        error,
        ErrorPolicy.SKIP,
    )
    
    assert result.status == ExecutionStatus.SKIPPED
    assert result.value is None


def test_handle_transform_error_default_policy():
    """Test error handling with DEFAULT policy."""
    error = ValueError("Test error")
    
    result = handle_transform_error(
        "test_feature",
        error,
        ErrorPolicy.DEFAULT,
        default_value=42.0,
    )
    
    assert result.status == ExecutionStatus.PARTIAL
    assert result.value == 42.0
    assert result.metadata["fallback"] is True


def test_with_error_handling_decorator_raise():
    """Test error handling decorator with RAISE policy."""
    class TestFeature(BaseFeature):
        @with_error_handling(policy=ErrorPolicy.RAISE)
        def transform(self, data, **kwargs):
            raise ValueError("Test error")
    
    feature = TestFeature(name="test")
    
    with pytest.raises(ValueError, match="Test error"):
        feature.transform(10)


def test_with_error_handling_decorator_warn():
    """Test error handling decorator with WARN policy."""
    class TestFeature(BaseFeature):
        @with_error_handling(policy=ErrorPolicy.WARN)
        def transform(self, data, **kwargs):
            raise ValueError("Test error")
    
    feature = TestFeature(name="test")
    result = feature.transform(10)
    
    assert result.status == ExecutionStatus.FAILED


def test_with_error_handling_decorator_default():
    """Test error handling decorator with DEFAULT policy."""
    class TestFeature(BaseFeature):
        @with_error_handling(policy=ErrorPolicy.DEFAULT, default_value=99.0)
        def transform(self, data, **kwargs):
            raise ValueError("Test error")
    
    feature = TestFeature(name="test")
    result = feature.transform(10)
    
    assert result.status == ExecutionStatus.PARTIAL
    assert result.value == 99.0


def test_circuit_breaker_initial_state():
    """Test circuit breaker starts in CLOSED state."""
    breaker = CircuitBreaker(threshold=3)
    
    assert breaker.state == "CLOSED"
    assert breaker.allow_call()


def test_circuit_breaker_opens_after_threshold():
    """Test circuit breaker opens after threshold failures."""
    breaker = CircuitBreaker(threshold=3)
    
    # Record failures
    for _ in range(3):
        breaker.record_failure()
    
    assert breaker.state == "OPEN"
    assert not breaker.allow_call()


def test_circuit_breaker_half_open_after_timeout():
    """Test circuit breaker transitions to HALF_OPEN after timeout."""
    breaker = CircuitBreaker(threshold=2, timeout=0.1)
    
    # Open the circuit
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "OPEN"
    
    # Wait for timeout
    time.sleep(0.15)
    
    # Should transition to HALF_OPEN
    assert breaker.allow_call()
    assert breaker.state == "HALF_OPEN"


def test_circuit_breaker_closes_on_success():
    """Test circuit breaker closes after successful call."""
    breaker = CircuitBreaker(threshold=2)
    
    # Open the circuit
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "OPEN"
    
    # Record success
    breaker.record_success()
    
    assert breaker.state == "CLOSED"
    assert breaker.failures == 0


def test_circuit_breaker_reset():
    """Test manual circuit breaker reset."""
    breaker = CircuitBreaker(threshold=2)
    
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "OPEN"
    
    breaker.reset()
    
    assert breaker.state == "CLOSED"
    assert breaker.failures == 0


def test_with_error_handling_circuit_breaker():
    """Test error handling with circuit breaker."""
    breaker = CircuitBreaker(threshold=2)
    
    class TestFeature(BaseFeature):
        @with_error_handling(
            policy=ErrorPolicy.WARN,
            circuit_breaker=breaker
        )
        def transform(self, data, **kwargs):
            raise ValueError("Test error")
    
    feature = TestFeature(name="test")
    
    # First two calls should execute and fail
    result1 = feature.transform(10)
    assert result1.status == ExecutionStatus.FAILED
    
    result2 = feature.transform(10)
    assert result2.status == ExecutionStatus.FAILED
    
    # Circuit should be open now
    result3 = feature.transform(10)
    assert result3.status == ExecutionStatus.SKIPPED
    assert "Circuit breaker" in result3.error


def test_error_aggregator_record():
    """Test error aggregator records errors."""
    aggregator = ErrorAggregator()
    
    aggregator.record("feature1", ValueError("Error 1"))
    aggregator.record("feature2", TypeError("Error 2"))
    aggregator.record("feature1", ValueError("Error 3"))
    
    summary = aggregator.summary()
    
    assert summary["total"] == 3
    assert summary["by_feature"]["feature1"] == 2
    assert summary["by_feature"]["feature2"] == 1
    assert summary["by_type"]["ValueError"] == 2
    assert summary["by_type"]["TypeError"] == 1


def test_error_aggregator_empty():
    """Test error aggregator with no errors."""
    aggregator = ErrorAggregator()
    
    summary = aggregator.summary()
    
    assert summary["total"] == 0
    assert summary["by_feature"] == {}
    assert summary["by_type"] == {}


def test_error_aggregator_clear():
    """Test error aggregator clear."""
    aggregator = ErrorAggregator()
    
    aggregator.record("feature1", ValueError("Error"))
    assert aggregator.summary()["total"] == 1
    
    aggregator.clear()
    assert aggregator.summary()["total"] == 0
