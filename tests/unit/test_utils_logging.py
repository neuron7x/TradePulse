# SPDX-License-Identifier: MIT
"""Tests for the structured logging utilities."""
from __future__ import annotations

import io
import json
import logging

import pytest

from core.utils.logging import JSONFormatter, StructuredLogger, configure_logging


def _make_record(**extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="tradepulse.tests",
        level=logging.ERROR,
        pathname=__file__,
        lineno=42,
        msg="problem occurred",
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_includes_extras_and_exception() -> None:
    formatter = JSONFormatter()

    try:
        raise ValueError("boom")
    except ValueError as exc:
        record = _make_record(
            correlation_id="cid-123",
            extra_fields={"action": "compute"},
            exc_info=(ValueError, exc, exc.__traceback__),
        )

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "ERROR"
    assert payload["correlation_id"] == "cid-123"
    assert payload["action"] == "compute"
    assert "ValueError: boom" in payload["exception"]


def test_structured_logger_operation_success_emits_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)

    logger = StructuredLogger("tradepulse.ops", correlation_id="cid-success")
    with logger.operation("sync_data", asset="BTC-USDT") as ctx:
        ctx["result_value"] = 42

    start_record, end_record = caplog.records[-2:]

    assert start_record.message == "Starting operation: sync_data"
    assert start_record.correlation_id == "cid-success"
    assert start_record.extra_fields["operation"] == "sync_data"
    assert start_record.extra_fields["asset"] == "BTC-USDT"

    assert end_record.levelno == logging.INFO
    assert end_record.message == "Completed operation: sync_data"
    assert end_record.correlation_id == "cid-success"
    assert end_record.extra_fields["status"] == "success"
    assert end_record.extra_fields["result_value"] == 42
    assert end_record.extra_fields["asset"] == "BTC-USDT"
    assert end_record.extra_fields["operation"] == "sync_data"
    assert "duration_seconds" in end_record.extra_fields


def test_structured_logger_operation_failure_logs_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)

    logger = StructuredLogger("tradepulse.ops", correlation_id="cid-failure")

    with pytest.raises(RuntimeError):
        with logger.operation("rebalance", portfolio="alpha"):
            raise RuntimeError("unable to rebalance")

    start_record, error_record = caplog.records[-2:]

    assert start_record.message == "Starting operation: rebalance"
    assert start_record.correlation_id == "cid-failure"

    assert error_record.levelno == logging.ERROR
    assert error_record.message == "Failed operation: rebalance"
    assert error_record.correlation_id == "cid-failure"
    assert error_record.extra_fields["status"] == "failure"
    assert error_record.extra_fields["error_type"] == "RuntimeError"
    assert error_record.extra_fields["error_message"] == "unable to rebalance"
    assert error_record.extra_fields["portfolio"] == "alpha"


def test_configure_logging_emits_json_payload() -> None:
    stream = io.StringIO()
    configure_logging(level="DEBUG", use_json=True, stream=stream)

    logging.getLogger("tradepulse.tests").info("hello world")

    output = stream.getvalue().strip()
    assert output

    payload = json.loads(output)

    assert payload["level"] == "INFO"
    assert payload["logger"] == "tradepulse.tests"
    assert payload["message"] == "hello world"
    assert "timestamp" in payload
