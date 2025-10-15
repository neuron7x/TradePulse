from __future__ import annotations

import logging
from datetime import timedelta
from typing import Callable

import pytest
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.testclient import TestClient

from execution.risk import RiskLimits, RiskManager
from src.admin.remote_control import (
    AdminIdentity,
    AdminRateLimiter,
    create_remote_control_router,
)
from src.audit.audit_logger import AuditLogger, AuditRecord
from src.risk.risk_manager import KillSwitchState, RiskManagerFacade
from src.security.token_authenticator import TokenAuthenticator

from tests.fixtures.conftest import SecretManagerHarness

AUDIT_SECRET_ID = "tests/admin/audit-secret"
AUDIT_SECRET_VALUE = "UnitTest-AuditSecret-123!"
ADMIN_TOKEN_ID = "tests/admin/token"
ADMIN_TOKEN_VALUE = "UnitTest-AdminToken-456!"

RemoteControlBundle = tuple[TestClient, RiskManager, list[AuditRecord], AuditLogger, SecretManagerHarness]


def _build_audit_logger(
    harness: SecretManagerHarness,
    *,
    sink: Callable[[AuditRecord], None] | None = None,
) -> AuditLogger:
    harness.provider.set_secret(AUDIT_SECRET_ID, AUDIT_SECRET_VALUE, ttl=timedelta(hours=1))
    return AuditLogger(
        secret_manager=harness.manager,
        secret_id=AUDIT_SECRET_ID,
        sink=sink,
        clock=lambda: harness.clock(),
    )


@pytest.fixture()
def remote_control_factory(secret_manager_harness: SecretManagerHarness) -> Callable[..., RemoteControlBundle]:
    clients: list[TestClient] = []

    def _factory(
        *,
        audit_ttl: timedelta = timedelta(hours=1),
        token_ttl: timedelta = timedelta(hours=1),
        rate_limiter: AdminRateLimiter | None = None,
    ) -> RemoteControlBundle:
        harness = secret_manager_harness
        harness.provider.set_secret(AUDIT_SECRET_ID, AUDIT_SECRET_VALUE, ttl=audit_ttl)
        harness.provider.set_secret(ADMIN_TOKEN_ID, ADMIN_TOKEN_VALUE, ttl=token_ttl)

        records: list[AuditRecord] = []
        audit_logger = AuditLogger(
            secret_manager=harness.manager,
            secret_id=AUDIT_SECRET_ID,
            sink=records.append,
            clock=lambda: harness.clock(),
        )
        token_authenticator = TokenAuthenticator(harness.manager, ADMIN_TOKEN_ID)
        risk_manager = RiskManager(RiskLimits())
        facade = RiskManagerFacade(risk_manager)

        async def identity_dependency(request: Request) -> AdminIdentity:
            header_subject = request.headers.get("X-Test-Admin-Subject")
            subject = header_subject or "unit-admin"
            return AdminIdentity(subject=subject)

        app = FastAPI()
        app.include_router(
            create_remote_control_router(
                facade,
                audit_logger,
                identity_dependency=identity_dependency,
                rate_limiter=rate_limiter,
                token_authenticator=token_authenticator,
            )
        )
        client = TestClient(app)
        client.headers.update({"X-Admin-Token": ADMIN_TOKEN_VALUE})
        clients.append(client)
        return client, risk_manager, records, audit_logger, harness

    try:
        yield _factory
    finally:
        for client in clients:
            client.close()


def test_kill_switch_endpoint_engages_kill_switch(
    remote_control_factory: Callable[..., RemoteControlBundle],
) -> None:
    client, risk_manager, records, audit_logger, _ = remote_control_factory()
    response = client.post(
        "/admin/kill-switch",
        headers={"X-Test-Admin-Subject": "root"},
        json={"reason": "manual intervention"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["kill_switch_engaged"] is True
    assert body["already_engaged"] is False
    assert risk_manager.kill_switch.is_triggered()
    assert len(records) == 1
    record = records[0]
    assert record.event_type == "kill_switch_engaged"
    assert record.details["reason"] == "manual intervention"
    assert record.actor == "root"
    assert audit_logger.verify(record)


def test_kill_switch_endpoint_uses_default_subject(
    remote_control_factory: Callable[..., RemoteControlBundle],
) -> None:
    client, risk_manager, records, audit_logger, _ = remote_control_factory()
    response = client.post(
        "/admin/kill-switch",
        json={"reason": "manual intervention"},
    )
    assert response.status_code == 200
    assert risk_manager.kill_switch.is_triggered()
    assert len(records) == 1
    record = records[0]
    assert record.actor == "unit-admin"
    assert audit_logger.verify(record)


def test_kill_switch_endpoint_reflects_facade_state(secret_manager_harness: SecretManagerHarness) -> None:
    class StubFacade:
        def __init__(self) -> None:
            self.reasons: list[str] = []

        def engage_kill_switch(self, reason: str) -> KillSwitchState:
            self.reasons.append(reason)
            return KillSwitchState(engaged=False, reason=reason, already_engaged=False)

    records: list[AuditRecord] = []
    audit_logger = _build_audit_logger(secret_manager_harness, sink=records.append)
    secret_manager_harness.provider.set_secret(ADMIN_TOKEN_ID, ADMIN_TOKEN_VALUE, ttl=timedelta(hours=1))
    facade = StubFacade()

    async def identity_dependency(request: Request) -> AdminIdentity:
        subject = request.headers.get("X-Test-Admin-Subject", "unit-admin")
        return AdminIdentity(subject=subject)

    app = FastAPI()
    app.include_router(
        create_remote_control_router(
            facade,
            audit_logger,
            identity_dependency=identity_dependency,
            token_authenticator=TokenAuthenticator(secret_manager_harness.manager, ADMIN_TOKEN_ID),
        )
    )
    client = TestClient(app)
    client.headers.update({"X-Admin-Token": ADMIN_TOKEN_VALUE})
    try:
        response = client.post(
            "/admin/kill-switch",
            headers={"X-Test-Admin-Subject": "ops"},
            json={"reason": "scheduled maintenance"},
        )
    finally:
        client.close()
    assert response.status_code == 200
    body = response.json()
    assert body["kill_switch_engaged"] is False
    assert body["reason"] == "scheduled maintenance"
    assert facade.reasons == ["scheduled maintenance"]


def test_kill_switch_rejects_whitespace_reason(
    remote_control_factory: Callable[..., RemoteControlBundle],
) -> None:
    client, risk_manager, records, _, _ = remote_control_factory()
    response = client.post(
        "/admin/kill-switch",
        json={"reason": "   \t\n"},
    )
    assert response.status_code == 422
    assert not risk_manager.kill_switch.is_triggered()
    assert records == []


def test_kill_switch_reaffirmation_is_audited(
    remote_control_factory: Callable[..., RemoteControlBundle],
) -> None:
    client, _, records, _, _ = remote_control_factory()
    headers = {"X-Test-Admin-Subject": "auditor"}
    first = client.post(
        "/admin/kill-switch", headers=headers, json={"reason": "initial"}
    )
    assert first.status_code == 200
    second = client.post(
        "/admin/kill-switch", headers=headers, json={"reason": "still engaged"}
    )
    assert second.status_code == 200
    body = second.json()
    assert body["already_engaged"] is True
    assert len(records) == 2
    reaffirmation = records[1]
    assert reaffirmation.event_type == "kill_switch_reaffirmed"
    assert reaffirmation.details["already_engaged"] is True
    assert reaffirmation.details["reason"] == "still engaged"


def test_admin_rate_limiter_blocks_excessive_attempts(
    remote_control_factory: Callable[..., RemoteControlBundle],
) -> None:
    limiter = AdminRateLimiter(max_attempts=1, interval_seconds=60.0)
    client, _, records, _, _ = remote_control_factory(rate_limiter=limiter)
    first = client.post(
        "/admin/kill-switch",
        json={"reason": "initial"},
    )
    assert first.status_code == 200

    second = client.post(
        "/admin/kill-switch",
        json={"reason": "repeat"},
    )
    assert second.status_code == 429
    assert len(records) == 1


def test_missing_admin_token_is_rejected(
    remote_control_factory: Callable[..., RemoteControlBundle],
) -> None:
    client, _, _, _, _ = remote_control_factory()
    response = client.post(
        "/admin/kill-switch",
        headers={"X-Admin-Token": "", "X-Test-Admin-Subject": "ops"},
        json={"reason": "manual"},
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_invalid_admin_token_is_rejected(
    remote_control_factory: Callable[..., RemoteControlBundle],
) -> None:
    client, _, _, _, _ = remote_control_factory()
    response = client.post(
        "/admin/kill-switch",
        headers={"X-Admin-Token": "wrong-token", "X-Test-Admin-Subject": "ops"},
        json={"reason": "manual"},
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_secret_metrics_and_logs_emitted(
    remote_control_factory: Callable[..., RemoteControlBundle], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="tradepulse.security.secrets")
    client, _, records, _, harness = remote_control_factory(audit_ttl=timedelta(seconds=0))
    response = client.post(
        "/admin/kill-switch",
        headers={"X-Test-Admin-Subject": "ops"},
        json={"reason": "manual"},
    )
    assert response.status_code == 200
    assert records
    metric_events = [event for event, _ in harness.metrics]
    assert "secret.access" in metric_events
    assert "secret.near_expiry" in metric_events
    assert any("secret.near_expiry" in record.message for record in caplog.records)
