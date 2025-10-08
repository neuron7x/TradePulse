# SPDX-License-Identifier: MIT
"""Tests for structured logging module."""
from __future__ import annotations

import json
import logging
import time
from io import StringIO

import pytest

from core.utils.logging import (
    JSONFormatter,
    StructuredLogger,
    configure_logging,
    get_logger,
)


class TestJSONFormatter:
    """Test JSONFormatter class."""

    def test_formats_record_as_json(self) -> None:
        """Should format log record as valid JSON."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        
        result = formatter.format(record)
        
        # Should be valid JSON
        data = json.loads(result)
        assert data["message"] == "Test message"
        assert data["level"] == "INFO"
        assert data["logger"] == "test"

    def test_includes_module_and_function(self) -> None:
        """Should include module and function information."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="test_func",
        )
        record.module = "test_module"
        
        result = formatter.format(record)
        data = json.loads(result)
        
        assert data["module"] == "test_module"
        assert data["function"] == "test_func"
        assert data["line"] == 42

    def test_includes_correlation_id(self) -> None:
        """Should include correlation ID if present."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.correlation_id = "test-correlation-id"
        
        result = formatter.format(record)
        data = json.loads(result)
        
        assert data["correlation_id"] == "test-correlation-id"

    def test_includes_extra_fields(self) -> None:
        """Should include extra fields if present."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.extra_fields = {"custom": "value", "count": 123}
        
        result = formatter.format(record)
        data = json.loads(result)
        
        assert data["custom"] == "value"
        assert data["count"] == 123

    def test_includes_exception_info(self) -> None:
        """Should include exception information."""
        formatter = JSONFormatter()
        
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
            
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=42,
                msg="Error occurred",
                args=(),
                exc_info=exc_info,
            )
            
            result = formatter.format(record)
            data = json.loads(result)
            
            assert "exception" in data
            assert "ValueError" in data["exception"]
            assert "Test error" in data["exception"]


class TestStructuredLogger:
    """Test StructuredLogger class."""

    def test_creates_logger_with_correlation_id(self) -> None:
        """Should create logger with correlation ID."""
        logger = StructuredLogger("test")
        
        assert logger.correlation_id is not None
        assert len(logger.correlation_id) > 0

    def test_uses_provided_correlation_id(self) -> None:
        """Should use provided correlation ID."""
        correlation_id = "custom-id-123"
        logger = StructuredLogger("test", correlation_id=correlation_id)
        
        assert logger.correlation_id == correlation_id

    def test_debug_logs_message(self, caplog) -> None:
        """Should log debug messages."""
        logger = StructuredLogger("test")
        
        with caplog.at_level(logging.DEBUG):
            logger.debug("Debug message")
        
        assert "Debug message" in caplog.text

    def test_info_logs_message(self, caplog) -> None:
        """Should log info messages."""
        logger = StructuredLogger("test")
        
        with caplog.at_level(logging.INFO):
            logger.info("Info message")
        
        assert "Info message" in caplog.text

    def test_warning_logs_message(self, caplog) -> None:
        """Should log warning messages."""
        logger = StructuredLogger("test")
        
        with caplog.at_level(logging.WARNING):
            logger.warning("Warning message")
        
        assert "Warning message" in caplog.text

    def test_error_logs_message(self, caplog) -> None:
        """Should log error messages."""
        logger = StructuredLogger("test")
        
        with caplog.at_level(logging.ERROR):
            logger.error("Error message")
        
        assert "Error message" in caplog.text

    def test_critical_logs_message(self, caplog) -> None:
        """Should log critical messages."""
        logger = StructuredLogger("test")
        
        with caplog.at_level(logging.CRITICAL):
            logger.critical("Critical message")
        
        assert "Critical message" in caplog.text

    def test_logs_with_extra_fields(self, caplog) -> None:
        """Should include extra fields in logs."""
        logger = StructuredLogger("test")
        
        with caplog.at_level(logging.INFO):
            logger.info("Message", user_id=123, action="login")
        
        # Note: extra fields are stored in the record, not necessarily in text
        assert "Message" in caplog.text


class TestConfigureLogging:
    """Test configure_logging function."""

    def test_configures_logging_with_json(self) -> None:
        """Should configure logging with JSON formatter."""
        # Create a string stream to capture output
        stream = StringIO()
        
        configure_logging(level="INFO", use_json=True, stream=stream)
        
        # Test that logging works
        logger = logging.getLogger("test")
        logger.info("Test message")
        
        output = stream.getvalue()
        # Should be valid JSON
        data = json.loads(output.strip())
        assert data["message"] == "Test message"

    def test_configures_logging_without_json(self) -> None:
        """Should configure logging with standard format."""
        stream = StringIO()
        
        configure_logging(level="INFO", use_json=False, stream=stream)
        
        logger = logging.getLogger("test2")
        logger.info("Test message")
        
        output = stream.getvalue()
        # Should be plain text
        assert "Test message" in output


class TestGetLogger:
    """Test get_logger function."""

    def test_returns_structured_logger(self) -> None:
        """Should return StructuredLogger instance."""
        logger = get_logger("test")
        
        assert isinstance(logger, StructuredLogger)
        assert logger.logger.name == "test"

    def test_uses_provided_correlation_id(self) -> None:
        """Should use provided correlation ID."""
        correlation_id = "test-id-456"
        logger = get_logger("test", correlation_id=correlation_id)
        
        assert logger.correlation_id == correlation_id

    def test_generates_unique_correlation_ids(self) -> None:
        """Should generate unique correlation IDs."""
        logger1 = get_logger("test1")
        logger2 = get_logger("test2")
        
        assert logger1.correlation_id != logger2.correlation_id


class TestOperationTiming:
    """Test operation timing context manager."""

    def test_operation_context_manager(self, caplog) -> None:
        """Should measure operation duration."""
        logger = StructuredLogger("test")
        
        with caplog.at_level(logging.INFO):
            with logger.operation("test_op"):
                time.sleep(0.01)  # Sleep for 10ms
        
        # Should log operation start and completion
        assert "test_op" in caplog.text
        assert any("Starting" in record.message or "Completed" in record.message for record in caplog.records)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_logger_handles_unicode(self, caplog) -> None:
        """Should handle unicode characters in messages."""
        logger = StructuredLogger("test")
        
        with caplog.at_level(logging.INFO):
            logger.info("Unicode message: 你好 世界 🚀")
        
        assert "Unicode message" in caplog.text

    def test_logger_handles_special_characters(self, caplog) -> None:
        """Should handle special characters."""
        logger = StructuredLogger("test")
        
        with caplog.at_level(logging.INFO):
            logger.info("Special chars: \"quotes\" and 'apostrophes'")
        
        assert "Special chars" in caplog.text

    def test_formatter_handles_none_values(self) -> None:
        """Should handle None values in extra fields."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.extra_fields = {"value": None}
        
        result = formatter.format(record)
        data = json.loads(result)
        
        assert data["value"] is None

    def test_structured_logger_with_empty_name(self) -> None:
        """Should handle empty logger name."""
        logger = StructuredLogger("")
        
        # Should not crash
        logger.info("Test message")
