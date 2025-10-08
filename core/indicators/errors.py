# SPDX-License-Identifier: MIT
"""Advanced error handling for indicator transformations.

This module provides comprehensive error handling with customizable policies,
error recovery strategies, and detailed audit trails. It enables production-grade
resilience and debugging capabilities.

Features:
- Customizable error policies (raise, warn, skip, default)
- Error recovery with fallback values
- Detailed error provenance and stack traces
- Circuit breaker pattern for cascading failures
- Error aggregation for batch processing
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, cast

from .base import (
    BaseFeature,
    ErrorPolicy,
    ExecutionStatus,
    FeatureInput,
    FeatureResult,
)
from .observability import get_logger

F = TypeVar("F", bound=Callable[..., FeatureResult])

logger = get_logger("tradepulse.indicators.errors")


@dataclass
class ErrorContext:
    """Detailed context about an error that occurred during transformation.

    Provides comprehensive error information for debugging and audit trails.

    Attributes:
        error_type: Type of exception that occurred
        error_message: Human-readable error message
        stack_trace: Full stack trace
        timestamp: When error occurred
        feature_name: Name of feature that failed
        input_summary: Summary of input data (for debugging)
        kwargs_summary: Summary of kwargs (for debugging)
    """

    error_type: str
    error_message: str
    stack_trace: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    feature_name: Optional[str] = None
    input_summary: Optional[str] = None
    kwargs_summary: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "timestamp": self.timestamp.isoformat(),
            "feature_name": self.feature_name,
            "input_summary": self.input_summary,
            "kwargs_summary": self.kwargs_summary,
        }


@dataclass
class ErrorRecoveryConfig:
    """Configuration for error recovery behavior.

    Defines how to handle errors and what fallback values to use.

    Attributes:
        policy: Error policy to apply
        default_value: Value to return when policy is DEFAULT
        max_retries: Number of retries for transient errors
        retry_delay: Delay between retries in seconds
        include_stack_trace: Whether to include full stack traces
    """

    policy: ErrorPolicy = ErrorPolicy.RAISE
    default_value: Any = None
    max_retries: int = 0
    retry_delay: float = 0.0
    include_stack_trace: bool = True


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures.

    Tracks error rates and automatically "opens" the circuit when
    errors exceed threshold, preventing further calls until recovery.

    States:
    - CLOSED: Normal operation, all calls go through
    - OPEN: Too many errors, all calls fail fast
    - HALF_OPEN: Testing if system recovered, limited calls allowed

    Example:
        >>> breaker = CircuitBreaker(threshold=5, timeout=60)
        >>> if breaker.allow_call():
        ...     try:
        ...         result = feature.transform(data)
        ...         breaker.record_success()
        ...     except Exception as e:
        ...         breaker.record_failure()
        ...         raise
    """

    def __init__(
        self,
        threshold: int = 5,
        timeout: float = 60.0,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            threshold: Number of failures before opening circuit
            timeout: Seconds to wait before trying half-open state
        """
        self.threshold = threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def allow_call(self) -> bool:
        """Check if call should be allowed.

        Returns:
            True if call allowed, False if circuit is open
        """
        if self.state == "CLOSED":
            return True

        if self.state == "OPEN":
            # Check if timeout elapsed
            if self.last_failure_time:
                elapsed = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
                if elapsed >= self.timeout:
                    self.state = "HALF_OPEN"
                    return True
            return False

        # HALF_OPEN: allow one call to test
        return True

    def record_success(self) -> None:
        """Record successful call."""
        self.failures = 0
        self.state = "CLOSED"

    def record_failure(self) -> None:
        """Record failed call."""
        self.failures += 1
        self.last_failure_time = datetime.now(timezone.utc)

        if self.failures >= self.threshold:
            self.state = "OPEN"
            logger.warning(
                "circuit_breaker_opened",
                failures=self.failures,
                threshold=self.threshold,
            )

    def reset(self) -> None:
        """Manually reset circuit breaker."""
        self.failures = 0
        self.last_failure_time = None
        self.state = "CLOSED"


def create_error_result(
    name: str,
    error: Exception,
    context: Optional[ErrorContext] = None,
) -> FeatureResult:
    """Create a FeatureResult representing an error.

    Args:
        name: Feature name
        error: Exception that occurred
        context: Optional error context

    Returns:
        FeatureResult with FAILED status
    """
    error_msg = str(error)

    result = FeatureResult(
        name=name,
        value=None,
        status=ExecutionStatus.FAILED,
        error=error_msg,
        metadata={"error_type": type(error).__name__},
    )

    if context:
        result.provenance["error_context"] = context.to_dict()

    return result


def handle_transform_error(
    feature_name: str,
    error: Exception,
    policy: ErrorPolicy,
    default_value: Any = None,
    input_data: Optional[FeatureInput] = None,
    **kwargs: Any,
) -> FeatureResult:
    """Handle transformation error according to policy.

    Args:
        feature_name: Name of feature that failed
        error: Exception that occurred
        policy: Error policy to apply
        default_value: Default value for DEFAULT policy
        input_data: Input data (for context)
        **kwargs: Additional kwargs (for context)

    Returns:
        FeatureResult based on policy

    Raises:
        Exception: If policy is RAISE
    """
    # Create error context
    context = ErrorContext(
        error_type=type(error).__name__,
        error_message=str(error),
        stack_trace=traceback.format_exc(),
        feature_name=feature_name,
        input_summary=_summarize_input(input_data),
        kwargs_summary=_summarize_kwargs(kwargs),
    )

    if policy == ErrorPolicy.RAISE:
        logger.error(
            "transform_error_raised",
            feature=feature_name,
            error_type=context.error_type,
            error_message=context.error_message,
        )
        raise error

    elif policy == ErrorPolicy.WARN:
        logger.warning(
            "transform_error_warn",
            feature=feature_name,
            error_type=context.error_type,
            error_message=context.error_message,
        )
        return create_error_result(feature_name, error, context)

    elif policy == ErrorPolicy.SKIP:
        logger.debug(
            "transform_error_skip",
            feature=feature_name,
            error_type=context.error_type,
        )
        return FeatureResult(
            name=feature_name,
            value=None,
            status=ExecutionStatus.SKIPPED,
            error=str(error),
        )

    elif policy == ErrorPolicy.DEFAULT:
        logger.info(
            "transform_error_default",
            feature=feature_name,
            error_type=context.error_type,
            default_value=default_value,
        )
        return FeatureResult(
            name=feature_name,
            value=default_value,
            status=ExecutionStatus.PARTIAL,
            error=str(error),
            metadata={"fallback": True},
        )

    # Fallback: raise
    raise error


def _summarize_input(data: Optional[FeatureInput]) -> Optional[str]:
    """Create safe summary of input data for logging."""
    if data is None:
        return None

    try:
        if hasattr(data, "shape"):
            return f"array(shape={data.shape}, dtype={data.dtype})"
        elif isinstance(data, (list, tuple)):
            return f"{type(data).__name__}(len={len(data)})"
        elif isinstance(data, dict):
            return f"dict(keys={list(data.keys())})"
        else:
            return f"{type(data).__name__}"
    except Exception:
        return "unknown"


def _summarize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Create safe summary of kwargs for logging."""
    summary = {}
    for key, value in kwargs.items():
        try:
            if hasattr(value, "shape"):
                summary[key] = f"array(shape={value.shape})"
            elif isinstance(value, (list, tuple)):
                summary[key] = f"{type(value).__name__}(len={len(value)})"
            else:
                summary[key] = type(value).__name__
        except Exception:
            summary[key] = "unknown"
    return summary


def with_error_handling(
    policy: ErrorPolicy = ErrorPolicy.RAISE,
    default_value: Any = None,
    circuit_breaker: Optional[CircuitBreaker] = None,
) -> Callable[[F], F]:
    """Decorator to add error handling to transform methods.

    Wraps feature transform methods with configurable error handling,
    recovery strategies, and circuit breaker protection.

    Args:
        policy: Error policy to apply
        default_value: Default value for DEFAULT policy
        circuit_breaker: Optional circuit breaker instance

    Returns:
        Decorated function with error handling

    Example:
        >>> @with_error_handling(policy=ErrorPolicy.DEFAULT, default_value=0.0)
        ... def transform(self, data, **kwargs):
        ...     return FeatureResult(name="test", value=compute(data))
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self: BaseFeature, data: FeatureInput, **kwargs: Any) -> FeatureResult:
            # Check circuit breaker
            if circuit_breaker and not circuit_breaker.allow_call():
                logger.warning(
                    "circuit_breaker_blocked",
                    feature=self.name,
                    state=circuit_breaker.state,
                )
                return FeatureResult(
                    name=self.name,
                    value=None,
                    status=ExecutionStatus.SKIPPED,
                    error="Circuit breaker open",
                )

            try:
                result = func(self, data, **kwargs)

                # Record success in circuit breaker
                if circuit_breaker:
                    circuit_breaker.record_success()

                return result

            except Exception as e:
                # Record failure in circuit breaker
                if circuit_breaker:
                    circuit_breaker.record_failure()

                # Handle error according to policy
                return handle_transform_error(
                    feature_name=self.name,
                    error=e,
                    policy=policy,
                    default_value=default_value,
                    input_data=data,
                    **kwargs,
                )

        return cast(F, wrapper)

    return decorator


class ErrorAggregator:
    """Aggregate errors from batch processing.

    Collects errors from multiple transform calls and provides
    summary statistics and detailed error reports.

    Example:
        >>> aggregator = ErrorAggregator()
        >>> for item in batch:
        ...     try:
        ...         result = feature.transform(item)
        ...     except Exception as e:
        ...         aggregator.record(feature.name, e)
        >>> print(aggregator.summary())
    """

    def __init__(self) -> None:
        """Initialize error aggregator."""
        self.errors: list[tuple[str, Exception, datetime]] = []

    def record(self, feature_name: str, error: Exception) -> None:
        """Record an error.

        Args:
            feature_name: Name of feature that failed
            error: Exception that occurred
        """
        self.errors.append((
            feature_name,
            error,
            datetime.now(timezone.utc)
        ))

    def summary(self) -> dict[str, Any]:
        """Get error summary statistics.

        Returns:
            Dictionary with error counts and types
        """
        if not self.errors:
            return {"total": 0, "by_feature": {}, "by_type": {}}

        by_feature: dict[str, int] = {}
        by_type: dict[str, int] = {}

        for feature_name, error, _ in self.errors:
            by_feature[feature_name] = by_feature.get(feature_name, 0) + 1
            error_type = type(error).__name__
            by_type[error_type] = by_type.get(error_type, 0) + 1

        return {
            "total": len(self.errors),
            "by_feature": by_feature,
            "by_type": by_type,
        }

    def clear(self) -> None:
        """Clear all recorded errors."""
        self.errors.clear()


__all__ = [
    "ErrorContext",
    "ErrorRecoveryConfig",
    "CircuitBreaker",
    "create_error_result",
    "handle_transform_error",
    "with_error_handling",
    "ErrorAggregator",
]
