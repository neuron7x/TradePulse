from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from execution.risk import RiskLimits, RiskManager
from src.admin.remote_control import (
    AdminAccessPolicyConfig,
    RateLimitSettings,
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
    app.include_router(create_remote_control_router(facade, audit_logger, authenticator))
    client = TestClient(app)
    try:
        yield client, risk_manager, records, audit_logger
    finally:
        client.close()


def test_kill_switch_endpoint_engages_kill_switch(remote_control_fixture: RemoteControlBundle) -> None:
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
    app.include_router(create_remote_control_router(facade, audit_logger, authenticator))
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


def test_kill_switch_requires_valid_token(remote_control_fixture: RemoteControlBundle) -> None:
    client, risk_manager, records, _ = remote_control_fixture
    response = client.post(
        "/admin/kill-switch",
        headers={"X-Admin-Token": "invalid"},
        json={"reason": "manual intervention"},
    )
    assert response.status_code == 401
    assert not risk_manager.kill_switch.is_triggered()
    assert records == []


def test_kill_switch_rejects_whitespace_reason(remote_control_fixture: RemoteControlBundle) -> None:
    client, risk_manager, records, _ = remote_control_fixture
    response = client.post(
        "/admin/kill-switch",
        headers={"X-Admin-Token": "s3cr3t-token"},
        json={"reason": "   \t\n"},
    )
    assert response.status_code == 422
    assert not risk_manager.kill_switch.is_triggered()
    assert records == []


def test_kill_switch_reaffirmation_is_audited(remote_control_fixture: RemoteControlBundle) -> None:
    client, _, records, _ = remote_control_fixture
    headers = {"X-Admin-Token": "s3cr3t-token", "X-Admin-Subject": "auditor"}
    first = client.post("/admin/kill-switch", headers=headers, json={"reason": "initial"})
    assert first.status_code == 200
    second = client.post("/admin/kill-switch", headers=headers, json={"reason": "still engaged"})
    assert second.status_code == 200
    body = second.json()
    assert body["already_engaged"] is True
    assert len(records) == 2
    reaffirmation = records[1]
    assert reaffirmation.event_type == "kill_switch_reaffirmed"
    assert reaffirmation.details["already_engaged"] is True
    assert reaffirmation.details["reason"] == "still engaged"


def test_kill_switch_rate_limits_are_audited() -> None:
    records: list[AuditRecord] = []
    audit_logger = AuditLogger(
        secret="unit-test-secret",
        sink=records.append,
        clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    risk_manager = RiskManager(RiskLimits())
    facade = RiskManagerFacade(risk_manager)
    authenticator = TokenAuthenticator(token="s3cr3t-token", subject="unit-admin")
    policy_config = AdminAccessPolicyConfig(
        subject_rate_limit=RateLimitSettings(max_attempts=1, window_seconds=3600),
        ip_rate_limit=RateLimitSettings(max_attempts=5, window_seconds=60),
    )
    app = FastAPI()
    app.include_router(
        create_remote_control_router(
            facade,
            audit_logger,
            authenticator,
            access_policy_config=policy_config,
        )
    )
    client = TestClient(app)
    try:
        headers = {"X-Admin-Token": "s3cr3t-token", "X-Admin-Subject": "subject-a"}
        first = client.post("/admin/kill-switch", headers=headers, json={"reason": "initial"})
        assert first.status_code == 200
        throttled = client.post("/admin/kill-switch", headers=headers, json={"reason": "retry"})
        assert throttled.status_code == 429
        assert "Retry-After" in throttled.headers
    finally:
        client.close()

    assert len(records) == 2
    assert records[1].event_type == "kill_switch_rate_limited"
    assert records[1].details["subject_retry_after"] > 0
    assert records[1].details["ip_retry_after"] >= 0


def test_kill_switch_enforces_cidr_allow_list() -> None:
    records: list[AuditRecord] = []
    audit_logger = AuditLogger(
        secret="unit-test-secret",
        sink=records.append,
        clock=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    risk_manager = RiskManager(RiskLimits())
    facade = RiskManagerFacade(risk_manager)
    authenticator = TokenAuthenticator(token="s3cr3t-token", subject="unit-admin")
    policy_config = AdminAccessPolicyConfig(
        allow_cidrs=("10.0.0.0/8",),
        subject_rate_limit=RateLimitSettings(max_attempts=5, window_seconds=60),
        ip_rate_limit=RateLimitSettings(max_attempts=5, window_seconds=60),
    )
    app = FastAPI()
    app.include_router(
        create_remote_control_router(
            facade,
            audit_logger,
            authenticator,
            access_policy_config=policy_config,
        )
    )
    client = TestClient(app)
    try:
        headers = {
            "X-Admin-Token": "s3cr3t-token",
            "X-Admin-Subject": "forbidden",
            "X-Forwarded-For": "203.0.113.42",
        }
        denied = client.post("/admin/kill-switch", headers=headers, json={"reason": "initial"})
        assert denied.status_code == 403
    finally:
        client.close()

    assert records
    assert records[0].event_type == "kill_switch_access_denied"
    assert records[0].details["allowed_cidrs"] == ["10.0.0.0/8"]
    assert not risk_manager.kill_switch.is_triggered()


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


def test_kill_switch_prefers_forwarded_for_header(remote_control_fixture: RemoteControlBundle) -> None:
    client, _, records, _ = remote_control_fixture
    headers = {
        "X-Admin-Token": "s3cr3t-token",
        "X-Admin-Subject": "auditor",
        "X-Forwarded-For": "203.0.113.10:443, 10.0.0.1",
    }
    response = client.post("/admin/kill-switch", headers=headers, json={"reason": "initial"})
    assert response.status_code == 200
    assert records[0].ip_address == "203.0.113.10"


def test_kill_switch_falls_back_to_real_ip(remote_control_fixture: RemoteControlBundle) -> None:
    client, _, records, _ = remote_control_fixture
    headers = {
        "X-Admin-Token": "s3cr3t-token",
        "X-Real-IP": "2001:db8::1%eth0",
        "X-Forwarded-For": "invalid-entry",
    }
    response = client.post("/admin/kill-switch", headers=headers, json={"reason": "initial"})
    assert response.status_code == 200
    assert records[0].ip_address == "2001:db8::1"
