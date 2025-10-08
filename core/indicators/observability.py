# SPDX-License-Identifier: MIT
"""Observability support for indicators: logging, metrics, and tracing.

This module provides structured logging, Prometheus metrics, and distributed
tracing capabilities for all feature transformations. It enables production-grade
monitoring and debugging of indicator pipelines.

Features:
- Structured JSON logging with trace_id
- Prometheus metrics for latency, throughput, and errors
- OpenTelemetry tracing support (when available)
- Contextual logging with automatic metadata extraction
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, cast

from .base import FeatureInput, FeatureResult

# Try to import prometheus_client
try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None
    Histogram = None
    Gauge = None

# Try to import OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.trace import Span, Tracer
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    Span = None
    Tracer = None


F = TypeVar("F", bound=Callable[..., Any])


class StructuredLogger:
    """Structured JSON logger for feature transformations.
    
    Provides structured logging with automatic metadata extraction,
    trace IDs, and JSON formatting for log aggregation systems.
    
    Example:
        >>> logger = StructuredLogger("indicators")
        >>> logger.info("transform_start", feature="kuramoto", data_size=1000)
    """
    
    def __init__(self, name: str, level: int = logging.INFO) -> None:
        """Initialize structured logger.
        
        Args:
            name: Logger name (usually module or component name)
            level: Logging level (default: INFO)
        """
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        
        # Configure JSON formatter if no handlers exist
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(self._create_json_formatter())
            self._logger.addHandler(handler)
    
    @staticmethod
    def _create_json_formatter() -> logging.Formatter:
        """Create JSON formatter for structured logging."""
        
        class JSONFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                log_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                }
                
                # Add exception info if present
                if record.exc_info:
                    log_data["exception"] = self.formatException(record.exc_info)
                
                # Add custom fields
                if hasattr(record, "extra_fields"):
                    log_data.update(record.extra_fields)
                
                return json.dumps(log_data)
        
        return JSONFormatter()
    
    def _log(
        self,
        level: int,
        event: str,
        trace_id: Optional[str] = None,
        **fields: Any,
    ) -> None:
        """Internal logging method with structured fields.
        
        Args:
            level: Logging level
            event: Event name/message
            trace_id: Optional trace identifier
            **fields: Additional structured fields
        """
        extra_fields = {"event": event, **fields}
        if trace_id:
            extra_fields["trace_id"] = trace_id
        
        self._logger.log(
            level,
            event,
            extra={"extra_fields": extra_fields},
        )
    
    def debug(self, event: str, trace_id: Optional[str] = None, **fields: Any) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, event, trace_id, **fields)
    
    def info(self, event: str, trace_id: Optional[str] = None, **fields: Any) -> None:
        """Log info message."""
        self._log(logging.INFO, event, trace_id, **fields)
    
    def warning(self, event: str, trace_id: Optional[str] = None, **fields: Any) -> None:
        """Log warning message."""
        self._log(logging.WARNING, event, trace_id, **fields)
    
    def error(self, event: str, trace_id: Optional[str] = None, **fields: Any) -> None:
        """Log error message."""
        self._log(logging.ERROR, event, trace_id, **fields)
    
    def critical(self, event: str, trace_id: Optional[str] = None, **fields: Any) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, event, trace_id, **fields)


class IndicatorMetrics:
    """Prometheus metrics for indicator transformations.
    
    Tracks key metrics:
    - Transform latency (histogram)
    - Transform count (counter)
    - Active transforms (gauge)
    - Error count (counter)
    
    Example:
        >>> metrics = IndicatorMetrics()
        >>> with metrics.measure_transform("kuramoto"):
        ...     result = compute_indicator(data)
    """
    
    def __init__(self, prefix: str = "tradepulse_indicator") -> None:
        """Initialize metrics collector.
        
        Args:
            prefix: Metric name prefix
        """
        self.prefix = prefix
        
        if PROMETHEUS_AVAILABLE:
            self.transform_latency = Histogram(
                f"{prefix}_transform_latency_seconds",
                "Feature transformation latency",
                ["feature_name"],
            )
            
            self.transform_count = Counter(
                f"{prefix}_transform_total",
                "Total feature transformations",
                ["feature_name", "status"],
            )
            
            self.active_transforms = Gauge(
                f"{prefix}_active_transforms",
                "Currently executing transforms",
                ["feature_name"],
            )
            
            self.error_count = Counter(
                f"{prefix}_errors_total",
                "Total transformation errors",
                ["feature_name", "error_type"],
            )
        else:
            # No-op metrics if prometheus not available
            self.transform_latency = None
            self.transform_count = None
            self.active_transforms = None
            self.error_count = None
    
    @contextmanager
    def measure_transform(self, feature_name: str):
        """Context manager to measure transform duration.
        
        Args:
            feature_name: Name of the feature being transformed
            
        Yields:
            None
            
        Example:
            >>> with metrics.measure_transform("entropy"):
            ...     result = entropy(data)
        """
        if not PROMETHEUS_AVAILABLE:
            yield
            return
        
        start = time.time()
        self.active_transforms.labels(feature_name=feature_name).inc()
        
        try:
            yield
            duration = time.time() - start
            self.transform_latency.labels(feature_name=feature_name).observe(duration)
            self.transform_count.labels(
                feature_name=feature_name,
                status="success"
            ).inc()
        except Exception as e:
            duration = time.time() - start
            self.transform_latency.labels(feature_name=feature_name).observe(duration)
            self.transform_count.labels(
                feature_name=feature_name,
                status="error"
            ).inc()
            self.error_count.labels(
                feature_name=feature_name,
                error_type=type(e).__name__
            ).inc()
            raise
        finally:
            self.active_transforms.labels(feature_name=feature_name).dec()
    
    def record_success(self, feature_name: str, duration: float) -> None:
        """Record successful transformation.
        
        Args:
            feature_name: Name of the feature
            duration: Transform duration in seconds
        """
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.transform_latency.labels(feature_name=feature_name).observe(duration)
        self.transform_count.labels(
            feature_name=feature_name,
            status="success"
        ).inc()
    
    def record_error(
        self,
        feature_name: str,
        error_type: str,
        duration: float
    ) -> None:
        """Record failed transformation.
        
        Args:
            feature_name: Name of the feature
            error_type: Type of error that occurred
            duration: Transform duration in seconds
        """
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.transform_latency.labels(feature_name=feature_name).observe(duration)
        self.transform_count.labels(
            feature_name=feature_name,
            status="error"
        ).inc()
        self.error_count.labels(
            feature_name=feature_name,
            error_type=error_type
        ).inc()


# Global instances for convenience
_default_logger = StructuredLogger("tradepulse.indicators")
_default_metrics = IndicatorMetrics()


def get_logger(name: Optional[str] = None) -> StructuredLogger:
    """Get or create a structured logger.
    
    Args:
        name: Logger name (uses default if None)
        
    Returns:
        Structured logger instance
    """
    if name is None:
        return _default_logger
    return StructuredLogger(name)


def get_metrics() -> IndicatorMetrics:
    """Get the default metrics collector.
    
    Returns:
        Metrics collector instance
    """
    return _default_metrics


def with_observability(
    logger: Optional[StructuredLogger] = None,
    metrics: Optional[IndicatorMetrics] = None,
) -> Callable[[F], F]:
    """Decorator to add observability to transform functions.
    
    Automatically logs and measures any function that returns a FeatureResult.
    
    Args:
        logger: Optional logger (uses default if None)
        metrics: Optional metrics collector (uses default if None)
        
    Returns:
        Decorated function with observability
        
    Example:
        >>> @with_observability()
        ... def compute_indicator(data):
        ...     return FeatureResult(name="test", value=42)
    """
    _logger = logger or _default_logger
    _metrics = metrics or _default_metrics
    
    def decorator(func: F) -> F:
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> FeatureResult:
            feature_name = kwargs.get("name", func.__name__)
            
            # Extract trace_id if available in kwargs
            trace_id = kwargs.get("trace_id")
            
            _logger.info(
                "transform_start",
                trace_id=trace_id,
                feature=feature_name,
                function=func.__name__,
            )
            
            start = time.time()
            try:
                with _metrics.measure_transform(feature_name):
                    result = func(*args, **kwargs)
                
                duration = time.time() - start
                _logger.info(
                    "transform_complete",
                    trace_id=trace_id or result.trace_id,
                    feature=feature_name,
                    duration_seconds=duration,
                    status=result.status.value,
                )
                
                return result
            except Exception as e:
                duration = time.time() - start
                _logger.error(
                    "transform_error",
                    trace_id=trace_id,
                    feature=feature_name,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    duration_seconds=duration,
                )
                raise
        
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> FeatureResult:
            feature_name = kwargs.get("name", func.__name__)
            trace_id = kwargs.get("trace_id")
            
            _logger.info(
                "transform_start",
                trace_id=trace_id,
                feature=feature_name,
                function=func.__name__,
            )
            
            start = time.time()
            try:
                # Note: Can't use context manager with async, manually track
                if PROMETHEUS_AVAILABLE:
                    _metrics.active_transforms.labels(feature_name=feature_name).inc()
                
                result = await func(*args, **kwargs)
                
                duration = time.time() - start
                if PROMETHEUS_AVAILABLE:
                    _metrics.record_success(feature_name, duration)
                
                _logger.info(
                    "transform_complete",
                    trace_id=trace_id or result.trace_id,
                    feature=feature_name,
                    duration_seconds=duration,
                    status=result.status.value,
                )
                
                return result
            except Exception as e:
                duration = time.time() - start
                if PROMETHEUS_AVAILABLE:
                    _metrics.record_error(feature_name, type(e).__name__, duration)
                
                _logger.error(
                    "transform_error",
                    trace_id=trace_id,
                    feature=feature_name,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    duration_seconds=duration,
                )
                raise
            finally:
                if PROMETHEUS_AVAILABLE:
                    _metrics.active_transforms.labels(feature_name=feature_name).dec()
        
        # Return appropriate wrapper based on function type
        import inspect
        if inspect.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)
    
    return decorator


__all__ = [
    "StructuredLogger",
    "IndicatorMetrics",
    "get_logger",
    "get_metrics",
    "with_observability",
    "PROMETHEUS_AVAILABLE",
    "OTEL_AVAILABLE",
]
