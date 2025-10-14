from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from execution.risk import RiskLimits, RiskManager
from src.admin.remote_control import (
    AdminRateLimiter,
    TokenAuthenticator,
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
    authenticator = TokenAuthenticator(token="s3cr3t-token", subject="unit-admin")
    app = FastAPI()
    app.include_router(
        create_remote_control_router(facade, audit_logger, authenticator)
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
        headers={"X-Admin-Token": "s3cr3t-token", "X-Admin-Subject": "root"},
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
        headers={"X-Admin-Token": "s3cr3t-token"},
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
    authenticator = TokenAuthenticator(token="s3cr3t-token", subject="unit-admin")
    app = FastAPI()
    app.include_router(
        create_remote_control_router(facade, audit_logger, authenticator)
    )
    client = TestClient(app)
    try:
        response = client.post(
            "/admin/kill-switch",
            headers={"X-Admin-Token": "s3cr3t-token", "X-Admin-Subject": "ops"},
            json={"reason": "scheduled maintenance"},
        )
    finally:
        client.close()
    assert response.status_code == 200
    body = response.json()
    assert body["kill_switch_engaged"] is False
    assert body["reason"] == "scheduled maintenance"
    assert facade.reasons == ["scheduled maintenance"]


def test_kill_switch_requires_valid_token(
    remote_control_fixture: RemoteControlBundle,
) -> None:
    client, risk_manager, records, _ = remote_control_fixture
    response = client.post(
        "/admin/kill-switch",
        headers={"X-Admin-Token": "invalid"},
        json={"reason": "manual intervention"},
    )
    assert response.status_code == 401
    assert not risk_manager.kill_switch.is_triggered()
    assert records == []


def test_kill_switch_rejects_whitespace_reason(
    remote_control_fixture: RemoteControlBundle,
) -> None:
    client, risk_manager, records, _ = remote_control_fixture
    response = client.post(
        "/admin/kill-switch",
        headers={"X-Admin-Token": "s3cr3t-token"},
        json={"reason": "   \t\n"},
    )
    assert response.status_code == 422
    assert not risk_manager.kill_switch.is_triggered()
    assert records == []


def test_kill_switch_reaffirmation_is_audited(
    remote_control_fixture: RemoteControlBundle,
) -> None:
    client, _, records, _ = remote_control_fixture
    headers = {"X-Admin-Token": "s3cr3t-token", "X-Admin-Subject": "auditor"}
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
    authenticator = TokenAuthenticator(token="s3cr3t-token", subject="unit-admin")
    limiter = AdminRateLimiter(max_attempts=1, interval_seconds=60.0)
    app = FastAPI()
    app.include_router(
        create_remote_control_router(
            facade,
            audit_logger,
            authenticator,
            rate_limiter=limiter,
        )
    )
    client = TestClient(app)
    try:
        first = client.post(
            "/admin/kill-switch",
            headers={"X-Admin-Token": "s3cr3t-token"},
            json={"reason": "initial"},
        )
        assert first.status_code == 200

        second = client.post(
            "/admin/kill-switch",
            headers={"X-Admin-Token": "s3cr3t-token"},
            json={"reason": "follow-up"},
        )
        assert second.status_code == 429
        assert "Too many administrative requests" in second.json()["detail"]
        # No additional audit event should be recorded once the limiter rejects the request.
        assert len(records) == 1
    finally:
        client.close()


def test_risk_manager_facade_preserves_reason_when_reaffirmed_without_message() -> None:
    risk_manager = RiskManager(RiskLimits())
    facade = RiskManagerFacade(risk_manager)

    initial_state = facade.engage_kill_switch("manual intervention required")
    assert initial_state.engaged is True
    assert initial_state.already_engaged is False
    assert initial_state.reason == "manual intervention required"

    reaffirmed = facade.engage_kill_switch("")
    assert reaffirmed.engaged is True
    assert reaffirmed.already_engaged is True
    assert reaffirmed.reason == "manual intervention required"
    assert risk_manager.kill_switch.reason == "manual intervention required"


def test_kill_switch_prefers_forwarded_for_header(
    remote_control_fixture: RemoteControlBundle,
) -> None:
    client, _, records, _ = remote_control_fixture
    headers = {
        "X-Admin-Token": "s3cr3t-token",
        "X-Admin-Subject": "auditor",
        "X-Forwarded-For": "203.0.113.10:443, 10.0.0.1",
    }
    response = client.post(
        "/admin/kill-switch", headers=headers, json={"reason": "initial"}
    )
    assert response.status_code == 200
    assert records[0].ip_address == "203.0.113.10"


def test_kill_switch_falls_back_to_real_ip(
    remote_control_fixture: RemoteControlBundle,
) -> None:
    client, _, records, _ = remote_control_fixture
    headers = {
        "X-Admin-Token": "s3cr3t-token",
        "X-Real-IP": "2001:db8::1%eth0",
        "X-Forwarded-For": "invalid-entry",
    }
    response = client.post(
        "/admin/kill-switch", headers=headers, json={"reason": "initial"}
    )
    assert response.status_code == 200
    assert records[0].ip_address == "2001:db8::1"


def test_kill_switch_state_endpoint_returns_state_and_audit(
    remote_control_fixture: RemoteControlBundle,
) -> None:
    client, risk_manager, records, audit_logger = remote_control_fixture
    headers = {"X-Admin-Token": "s3cr3t-token", "X-Admin-Subject": "observer"}

    response = client.get("/admin/kill-switch", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "status": "disengaged",
        "kill_switch_engaged": False,
        "reason": "",
        "already_engaged": False,
    }
    assert len(records) == 1
    view_record = records[0]
    assert view_record.event_type == "kill_switch_state_viewed"
    assert view_record.actor == "observer"
    assert audit_logger.verify(view_record)

    records.clear()
    risk_manager.kill_switch.trigger("manual override")
    response = client.get("/admin/kill-switch", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["kill_switch_engaged"] is True
    assert body["status"] == "engaged"
    assert body["reason"] == "manual override"
    assert len(records) == 1
    assert records[0].details["reason"] == "manual override"


def test_kill_switch_reset_endpoint_is_idempotent(
    remote_control_fixture: RemoteControlBundle,
) -> None:
    client, risk_manager, records, audit_logger = remote_control_fixture
    headers = {"X-Admin-Token": "s3cr3t-token", "X-Admin-Subject": "ops"}

    risk_manager.kill_switch.trigger("manual intervention")
    first = client.delete("/admin/kill-switch", headers=headers)
    assert first.status_code == 200
    body = first.json()
    assert body == {
        "status": "reset",
        "kill_switch_engaged": False,
        "reason": "manual intervention",
        "already_engaged": True,
    }
    assert not risk_manager.kill_switch.is_triggered()
    assert len(records) == 1
    reset_record = records[0]
    assert reset_record.event_type == "kill_switch_reset"
    assert reset_record.details["previously_engaged"] is True
    assert reset_record.details["reason"] == "manual intervention"
    assert audit_logger.verify(reset_record)

    second = client.delete("/admin/kill-switch", headers=headers)
    assert second.status_code == 200
    body = second.json()
    assert body == {
        "status": "already-clear",
        "kill_switch_engaged": False,
        "reason": "",
        "already_engaged": False,
    }
    assert len(records) == 2
    noop_record = records[1]
    assert noop_record.event_type == "kill_switch_reset_noop"
    assert noop_record.details["previously_engaged"] is False
    assert audit_logger.verify(noop_record)


def test_kill_switch_endpoints_are_documented(remote_control_fixture: RemoteControlBundle) -> None:
    client, _, _, _ = remote_control_fixture
    schema = client.app.openapi()
    path_item = schema["paths"]["/admin/kill-switch"]

    assert "description" in path_item["post"]
    assert "audit log" in path_item["post"]["description"].lower()
    assert "description" in path_item["get"]
    assert "audit" in path_item["get"]["description"].lower()
    assert "description" in path_item["delete"]
    assert "audit" in path_item["delete"]["description"].lower()
