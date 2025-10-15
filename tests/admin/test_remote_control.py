from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
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

RemoteControlBundle = tuple[TestClient, RiskManager, list[AuditRecord], AuditLogger]


@pytest.fixture()
def remote_control_fixture() -> Iterator[RemoteControlBundle]:
    records: list[AuditRecord] = []
    audit_logger = AuditLogger(
        secret="unit-test-secret",
        sink=records.append,
        clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
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
        )
    )
    client = TestClient(app)
    try:
        yield client, risk_manager, records, audit_logger
    finally:
        client.close()


def test_kill_switch_endpoint_engages_kill_switch(
    remote_control_fixture: RemoteControlBundle,
) -> None:
    client, risk_manager, records, audit_logger = remote_control_fixture
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
    remote_control_fixture: RemoteControlBundle,
) -> None:
    client, risk_manager, records, audit_logger = remote_control_fixture
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


def test_kill_switch_endpoint_reflects_facade_state() -> None:
    class StubFacade:
        def __init__(self) -> None:
            self.reasons: list[str] = []

        def engage_kill_switch(self, reason: str) -> KillSwitchState:
            self.reasons.append(reason)
            return KillSwitchState(engaged=False, reason=reason, already_engaged=False)

    records: list[AuditRecord] = []
    audit_logger = AuditLogger(
        secret="unit-test-secret",
        sink=records.append,
        clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    facade = StubFacade()

    async def identity_dependency(request: Request) -> AdminIdentity:
        subject = request.headers.get("X-Test-Admin-Subject", "unit-admin")
        return AdminIdentity(subject=subject)

    app = FastAPI()
    app.include_router(
        create_remote_control_router(facade, audit_logger, identity_dependency=identity_dependency)
    )
    client = TestClient(app)
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
    remote_control_fixture: RemoteControlBundle,
) -> None:
    client, risk_manager, records, _ = remote_control_fixture
    response = client.post(
        "/admin/kill-switch",
        json={"reason": "   \t\n"},
    )
    assert response.status_code == 422
    assert not risk_manager.kill_switch.is_triggered()
    assert records == []


def test_kill_switch_reaffirmation_is_audited(
    remote_control_fixture: RemoteControlBundle,
) -> None:
    client, _, records, _ = remote_control_fixture
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


def test_admin_rate_limiter_blocks_excessive_attempts() -> None:
    records: list[AuditRecord] = []
    audit_logger = AuditLogger(
        secret="unit-test-secret",
        sink=records.append,
        clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    risk_manager = RiskManager(RiskLimits())
    facade = RiskManagerFacade(risk_manager)

    async def identity_dependency(_: Request) -> AdminIdentity:
        return AdminIdentity(subject="unit-admin")

    limiter = AdminRateLimiter(max_attempts=1, interval_seconds=60.0)
    app = FastAPI()
    app.include_router(
        create_remote_control_router(
            facade,
            audit_logger,
            identity_dependency=identity_dependency,
            rate_limiter=limiter,
        )
    )
    client = TestClient(app)
    try:
        first = client.post(
            "/admin/kill-switch",
            json={"reason": "initial"},
        )
        assert first.status_code == 200

        second = client.post(
            "/admin/kill-switch",
            json={"reason": "repeat"},
        )
    finally:
        client.close()
    assert second.status_code == 429
    assert len(records) == 1


def test_identity_dependency_errors_are_propagated() -> None:
    audit_logger = AuditLogger(
        secret="unit-test-secret",
        sink=lambda record: None,
        clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    risk_manager = RiskManagerFacade(RiskManager(RiskLimits()))

    async def failing_identity(_: Request) -> AdminIdentity:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid cert")

    app = FastAPI()
    app.include_router(
        create_remote_control_router(
            risk_manager,
            audit_logger,
            identity_dependency=failing_identity,
        )
    )
    client = TestClient(app)
    response = client.post("/admin/kill-switch", json={"reason": "manual"})
    assert response.status_code == 401
