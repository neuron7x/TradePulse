from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from src.audit.audit_logger import AuditLogger, AuditRecord, HttpAuditSink

from tests.fixtures.conftest import SecretManagerHarness

AUDIT_SECRET_ID = "tests/unit/audit-secret"
AUDIT_SECRET_VALUE = "UnitTest-AuditSecret-987!"


def _fixed_clock() -> datetime:
    return datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def audit_logger_fixture(
    secret_manager_harness: SecretManagerHarness,
) -> tuple[AuditLogger, SecretManagerHarness]:
    harness = secret_manager_harness
    harness.provider.set_secret(AUDIT_SECRET_ID, AUDIT_SECRET_VALUE, ttl=timedelta(hours=1))
    logger = AuditLogger(
        secret_manager=harness.manager,
        secret_id=AUDIT_SECRET_ID,
        clock=_fixed_clock,
    )
    return logger, harness


def _make_record(audit_logger: AuditLogger) -> AuditRecord:
    return audit_logger.log_event(
        event_type="unit_test",
        actor="tester",
        ip_address="203.0.113.10",
        details={"action": "engage", "token": "sensitive"},
    )


def test_audit_logger_emits_signed_records(
    audit_logger_fixture: tuple[AuditLogger, SecretManagerHarness]
) -> None:
    audit_logger, harness = audit_logger_fixture
    records: list[AuditRecord] = []
    logger = AuditLogger(
        secret_manager=harness.manager,
        secret_id=AUDIT_SECRET_ID,
        sink=records.append,
        clock=_fixed_clock,
    )

    record = logger.log_event(
        event_type="kill_switch_engaged",
        actor="ops",
        ip_address="203.0.113.5",
        details={"reason": "manual"},
    )

    assert record.actor == "ops"
    assert record.details == {"reason": "manual"}
    assert logger.verify(record) is True
    assert records == [record]
    assert record.key_version


def test_audit_logger_detects_tampering(
    audit_logger_fixture: tuple[AuditLogger, SecretManagerHarness]
) -> None:
    audit_logger, _ = audit_logger_fixture
    record = audit_logger.log_event(
        event_type="kill_switch_engaged",
        actor="ops",
        ip_address="203.0.113.5",
        details={"reason": "manual"},
    )

    tampered = record.model_copy(update={"details": {"reason": "tampered"}})
    assert audit_logger.verify(record) is True
    assert audit_logger.verify(tampered) is False


def test_http_audit_sink_posts_payload(
    audit_logger_fixture: tuple[AuditLogger, SecretManagerHarness]
) -> None:
    audit_logger, _ = audit_logger_fixture
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(202)

    client = httpx.Client(base_url="https://audit.example.com", transport=httpx.MockTransport(handler))
    sink = HttpAuditSink("/ingest", http_client=client, timeout=1.0)
    try:
        sink(_make_record(audit_logger))
    finally:
        client.close()

    assert len(captured) == 1
    request = captured[0]
    assert request.method == "POST"
    assert request.url.path == "/ingest"
    payload = request.content.decode("utf-8")
    assert "unit_test" in payload


def test_http_audit_sink_logs_failures(
    audit_logger_fixture: tuple[AuditLogger, SecretManagerHarness],
    caplog: pytest.LogCaptureFixture,
) -> None:
    audit_logger, _ = audit_logger_fixture
    caplog.set_level(logging.ERROR, logger="tradepulse.audit.http_sink")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = httpx.Client(base_url="https://audit.example.com", transport=httpx.MockTransport(handler))
    sink = HttpAuditSink("/ingest", http_client=client, timeout=1.0)
    try:
        sink(_make_record(audit_logger))
    finally:
        client.close()

    assert "Failed to forward audit record" in caplog.text
