# SPDX-License-Identifier: MIT
"""OpenTelemetry distributed tracing for TradePulse.

This module provides OpenTelemetry integration for distributed tracing,
allowing end-to-end observability of feature computations, backtests,
and data pipelines.
"""
from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional, TypeVar

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None  # type: ignore


# Type for decorated functions
F = TypeVar('F', bound=Callable[..., Any])


class TracingManager:
    """Manages OpenTelemetry tracing for TradePulse."""
    
    def __init__(self, service_name: str = "tradepulse"):
        """Initialize tracing manager.
        
        Args:
            service_name: Name of the service for tracing
        """
        self._enabled = OTEL_AVAILABLE
        self._service_name = service_name
        self._tracer: Optional[Any] = None
        
        if self._enabled:
            self._setup_tracer()
    
    def _setup_tracer(self) -> None:
        """Set up OpenTelemetry tracer."""
        if not OTEL_AVAILABLE:
            return
        
        # Create resource with service name
        resource = Resource.create({"service.name": self._service_name})
        
        # Create tracer provider
        provider = TracerProvider(resource=resource)
        
        # Add console exporter (can be replaced with OTLP, Jaeger, etc.)
        processor = BatchSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)
        
        # Set as global tracer provider
        trace.set_tracer_provider(provider)
        
        # Get tracer
        self._tracer = trace.get_tracer(__name__)
    
    @property
    def enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self._enabled
    
    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Any]:
        """Start a tracing span.
        
        Args:
            name: Name of the span
            attributes: Optional attributes to attach to span
            
        Yields:
            Span object (or None if tracing disabled)
        """
        if not self._enabled or self._tracer is None:
            yield None
            return
        
        with self._tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span
    
    def trace_function(self, name: Optional[str] = None) -> Callable[[F], F]:
        """Decorator to trace a function.
        
        Args:
            name: Optional custom span name (defaults to function name)
            
        Returns:
            Decorated function
        """
        def decorator(func: F) -> F:
            span_name = name or func.__name__
            
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.start_span(span_name):
                    return func(*args, **kwargs)
            
            return wrapper  # type: ignore
        
        return decorator
    
    def trace_method(self, name: Optional[str] = None) -> Callable[[F], F]:
        """Decorator to trace a class method.
        
        Args:
            name: Optional custom span name (defaults to method name)
            
        Returns:
            Decorated method
        """
        def decorator(func: F) -> F:
            span_name = name or func.__name__
            
            @functools.wraps(func)
            def wrapper(self_obj: Any, *args: Any, **kwargs: Any) -> Any:
                class_name = self_obj.__class__.__name__
                full_name = f"{class_name}.{span_name}"
                
                with _global_tracer.start_span(full_name):
                    return func(self_obj, *args, **kwargs)
            
            return wrapper  # type: ignore
        
        return decorator


# Global tracer instance
_global_tracer = TracingManager()


def get_tracer() -> TracingManager:
    """Get the global tracing manager.
    
    Returns:
        Global TracingManager instance
    """
    return _global_tracer


def configure_tracing(
    service_name: str = "tradepulse",
    enabled: bool = True,
) -> TracingManager:
    """Configure distributed tracing.
    
    Args:
        service_name: Name of the service
        enabled: Whether to enable tracing
        
    Returns:
        Configured TracingManager
    """
    global _global_tracer
    
    if enabled and OTEL_AVAILABLE:
        _global_tracer = TracingManager(service_name=service_name)
    
    return _global_tracer


@contextmanager
def trace_operation(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> Iterator[Any]:
    """Context manager for tracing an operation.
    
    Args:
        name: Name of the operation
        attributes: Optional attributes to attach
        
    Yields:
        Span object (or None if tracing disabled)
        
    Example:
        >>> with trace_operation("compute_indicator", {"indicator": "RSI"}):
        ...     result = compute_rsi(data)
    """
    with _global_tracer.start_span(name, attributes) as span:
        yield span


def trace_function(name: Optional[str] = None) -> Callable[[F], F]:
    """Decorator to trace a function.
    
    Args:
        name: Optional custom span name
        
    Returns:
        Decorated function
        
    Example:
        >>> @trace_function("compute_features")
        >>> def compute_all_features(data):
        ...     return features
    """
    return _global_tracer.trace_function(name)


def trace_method(name: Optional[str] = None) -> Callable[[F], F]:
    """Decorator to trace a class method.
    
    Args:
        name: Optional custom span name
        
    Returns:
        Decorated method
        
    Example:
        >>> class MyClass:
        ...     @trace_method()
        ...     def process(self, data):
        ...         return result
    """
    return _global_tracer.trace_method(name)


__all__ = [
    "OTEL_AVAILABLE",
    "TracingManager",
    "get_tracer",
    "configure_tracing",
    "trace_operation",
    "trace_function",
    "trace_method",
]
