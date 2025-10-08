# SPDX-License-Identifier: MIT
"""Structured JSON logging utilities for TradePulse.

This module provides structured logging with JSON formatting, correlation IDs,
and performance tracking capabilities.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional
from uuid import uuid4


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add correlation ID if present
        if hasattr(record, "correlation_id"):
            log_data["correlation_id"] = record.correlation_id
            
        # Add custom fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
            
        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)


class StructuredLogger:
    """Wrapper around standard logger with structured logging capabilities."""
    
    def __init__(self, name: str, correlation_id: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.correlation_id = correlation_id or str(uuid4())
        
    def _log(self, level: int, msg: str, **kwargs: Any) -> None:
        """Internal logging method with structured fields."""
        extra_data: Dict[str, Any] = {"correlation_id": self.correlation_id}
        if kwargs:
            extra_data["extra_fields"] = kwargs
        self.logger.log(level, msg, extra=extra_data)
        
    def debug(self, msg: str, **kwargs: Any) -> None:
        """Log debug message with structured fields."""
        self._log(logging.DEBUG, msg, **kwargs)
        
    def info(self, msg: str, **kwargs: Any) -> None:
        """Log info message with structured fields."""
        self._log(logging.INFO, msg, **kwargs)
        
    def warning(self, msg: str, **kwargs: Any) -> None:
        """Log warning message with structured fields."""
        self._log(logging.WARNING, msg, **kwargs)
        
    def error(self, msg: str, **kwargs: Any) -> None:
        """Log error message with structured fields."""
        self._log(logging.ERROR, msg, **kwargs)
        
    def critical(self, msg: str, **kwargs: Any) -> None:
        """Log critical message with structured fields."""
        self._log(logging.CRITICAL, msg, **kwargs)
        
    @contextmanager
    def operation(self, operation_name: str, **context: Any) -> Iterator[Dict[str, Any]]:
        """Context manager for tracking operation timing and status.
        
        Args:
            operation_name: Name of the operation being tracked
            **context: Additional context fields to log
            
        Yields:
            Dictionary to store operation results
            
        Example:
            >>> logger = StructuredLogger("myapp")
            >>> with logger.operation("compute_indicator", indicator="RSI") as op:
            ...     result = compute_rsi(prices)
            ...     op["result_value"] = result
        """
        start_time = time.time()
        op_context: Dict[str, Any] = {"operation": operation_name, **context}
        
        self.info(f"Starting operation: {operation_name}", **op_context)
        
        try:
            yield op_context
            duration = time.time() - start_time
            self.info(
                f"Completed operation: {operation_name}",
                duration_seconds=duration,
                status="success",
                **op_context
            )
        except Exception as e:
            duration = time.time() - start_time
            self.error(
                f"Failed operation: {operation_name}",
                duration_seconds=duration,
                status="failure",
                error_type=type(e).__name__,
                error_message=str(e),
                **op_context
            )
            raise


def configure_logging(
    level: str = "INFO",
    use_json: bool = True,
    stream: Any = None
) -> None:
    """Configure application-wide logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_json: Whether to use JSON formatting
        stream: Output stream (defaults to sys.stdout)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create handler
    handler = logging.StreamHandler(stream or sys.stdout)
    
    # Set formatter
    if use_json:
        json_formatter: logging.Formatter = JSONFormatter()
        handler.setFormatter(json_formatter)
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def get_logger(name: str, correlation_id: Optional[str] = None) -> StructuredLogger:
    """Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        correlation_id: Optional correlation ID for request tracking
        
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name, correlation_id)


__all__ = [
    "JSONFormatter",
    "StructuredLogger",
    "configure_logging",
    "get_logger",
]
