from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from execution.risk import RiskLimits, RiskManager
from src.admin.remote_control import TokenAuthenticator, create_remote_control_router
from src.audit.audit_logger import AuditLogger, AuditRecord
from src.risk.risk_manager import RiskManagerFacade

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
