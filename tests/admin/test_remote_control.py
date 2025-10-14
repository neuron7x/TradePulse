from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from execution.risk import RiskLimits, RiskManager
from src.admin.remote_control import ShortLivedTokenVerifier, create_remote_control_router
from src.audit.audit_logger import AuditLogger, AuditRecord
from src.risk.risk_manager import KillSwitchState, RiskManagerFacade

SIGNING_SECRET_NAME = "remote-control-signing"
SIGNING_KEY_VALUE = "unit-test-signing-key"
REMOTE_AUDIENCE = "tradepulse.admin.kill-switch"
REQUIRED_SCOPE = "kill-switch:engage"


class StubSecretManager:
    def __init__(self) -> None:
        self._secrets = {SIGNING_SECRET_NAME: SIGNING_KEY_VALUE}

    def get_secret(self, name: str) -> str:
        try:
            return self._secrets[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"unknown secret requested: {name}") from exc


class MutableClock:
    def __init__(self, start: datetime) -> None:
        self._current = start

    def now(self) -> datetime:
        return self._current

    def advance(self, delta: timedelta) -> None:
        self._current += delta


@dataclass
class RemoteControlBundle:
    client: TestClient
    risk_manager: RiskManager
    records: list[AuditRecord]
    audit_logger: AuditLogger
    verifier: ShortLivedTokenVerifier
    clock: MutableClock


@pytest.fixture()
def remote_control_fixture() -> Iterator[RemoteControlBundle]:
    records: list[AuditRecord] = []
    clock = MutableClock(datetime(2025, 1, 1, tzinfo=timezone.utc))
    audit_logger = AuditLogger(
        secret="unit-test-secret",
        sink=records.append,
        clock=clock.now,
    )
    risk_manager = RiskManager(RiskLimits())
    facade = RiskManagerFacade(risk_manager)
    verifier = ShortLivedTokenVerifier(
        secret_manager=StubSecretManager(),
        secret_name=SIGNING_SECRET_NAME,
        clock=clock.now,
    )
    app = FastAPI()
    app.include_router(
        create_remote_control_router(
            facade,
            audit_logger,
            verifier,
            required_audience=REMOTE_AUDIENCE,
            required_scopes=(REQUIRED_SCOPE,),
        )
    )
    client = TestClient(app)
    try:
        yield RemoteControlBundle(client, risk_manager, records, audit_logger, verifier, clock)
    finally:
        client.close()


def _authorization_headers(
    bundle: RemoteControlBundle,
    *,
    subject: str = "unit-admin",
    scopes: Iterable[str] | None = None,
    audience: str = REMOTE_AUDIENCE,
    ttl_seconds: int = 300,
    thumbprint: str = "thumbprint-001",
) -> dict[str, str]:
    token = bundle.verifier.issue_token(
        subject=subject,
        scopes=scopes or {REQUIRED_SCOPE},
        audience=audience,
        ttl_seconds=ttl_seconds,
        cert_thumbprint=thumbprint,
    )
    return {
        "Authorization": f"Bearer {token}",
        "X-Client-Cert-Thumbprint": thumbprint,
    }


def test_kill_switch_endpoint_engages_kill_switch(remote_control_fixture: RemoteControlBundle) -> None:
    bundle = remote_control_fixture
    headers = _authorization_headers(bundle, subject="root", thumbprint="thumb-root")
    response = bundle.client.post(
        "/admin/kill-switch",
        headers=headers,
        json={"reason": "manual intervention"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["kill_switch_engaged"] is True
    assert body["already_engaged"] is False
    assert bundle.risk_manager.kill_switch.is_triggered()
    assert len(bundle.records) == 1
    record = bundle.records[0]
    assert record.event_type == "kill_switch_engaged"
    assert record.details["reason"] == "manual intervention"
    assert record.actor == "root"
    assert bundle.audit_logger.verify(record)


def test_kill_switch_endpoint_reflects_facade_state() -> None:
    class StubFacade:
        def __init__(self) -> None:
            self.reasons: list[str] = []

        def engage_kill_switch(self, reason: str) -> KillSwitchState:
            self.reasons.append(reason)
            return KillSwitchState(engaged=False, reason=reason, already_engaged=False)

    records: list[AuditRecord] = []
    clock = MutableClock(datetime(2025, 1, 1, tzinfo=timezone.utc))
    audit_logger = AuditLogger(
        secret="unit-test-secret",
        sink=records.append,
        clock=clock.now,
    )
    facade = StubFacade()
    verifier = ShortLivedTokenVerifier(
        secret_manager=StubSecretManager(),
        secret_name=SIGNING_SECRET_NAME,
        clock=clock.now,
    )
    app = FastAPI()
    app.include_router(
        create_remote_control_router(
            facade,
            audit_logger,
            verifier,
            required_audience=REMOTE_AUDIENCE,
            required_scopes=(REQUIRED_SCOPE,),
        )
    )
    client = TestClient(app)
    try:
        headers = {
            "Authorization": "Bearer "
            + verifier.issue_token(
                subject="ops",
                scopes={REQUIRED_SCOPE},
                audience=REMOTE_AUDIENCE,
                ttl_seconds=300,
                cert_thumbprint="thumb-ops",
            ),
            "X-Client-Cert-Thumbprint": "thumb-ops",
        }
        response = client.post(
            "/admin/kill-switch",
            headers=headers,
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
    bundle = remote_control_fixture
    response = bundle.client.post(
        "/admin/kill-switch",
        headers={
            "Authorization": "Bearer invalid",
            "X-Client-Cert-Thumbprint": "thumbprint-001",
        },
        json={"reason": "manual intervention"},
    )
    assert response.status_code == 401
    assert not bundle.risk_manager.kill_switch.is_triggered()
    assert bundle.records == []


def test_kill_switch_rejects_whitespace_reason(remote_control_fixture: RemoteControlBundle) -> None:
    bundle = remote_control_fixture
    response = bundle.client.post(
        "/admin/kill-switch",
        headers=_authorization_headers(bundle),
        json={"reason": "   \t\n"},
    )
    assert response.status_code == 422
    assert not bundle.risk_manager.kill_switch.is_triggered()
    assert bundle.records == []


def test_kill_switch_reaffirmation_is_audited(remote_control_fixture: RemoteControlBundle) -> None:
    bundle = remote_control_fixture
    first = bundle.client.post(
        "/admin/kill-switch",
        headers=_authorization_headers(bundle, subject="auditor", thumbprint="thumb-auditor"),
        json={"reason": "initial"},
    )
    assert first.status_code == 200
    second = bundle.client.post(
        "/admin/kill-switch",
        headers=_authorization_headers(bundle, subject="auditor", thumbprint="thumb-auditor"),
        json={"reason": "still engaged"},
    )
    assert second.status_code == 200
    body = second.json()
    assert body["already_engaged"] is True
    assert len(bundle.records) == 2
    reaffirmation = bundle.records[1]
    assert reaffirmation.event_type == "kill_switch_reaffirmed"
    assert reaffirmation.details["already_engaged"] is True
    assert reaffirmation.details["reason"] == "still engaged"


def test_kill_switch_rejects_expired_token(remote_control_fixture: RemoteControlBundle) -> None:
    bundle = remote_control_fixture
    headers = _authorization_headers(bundle, ttl_seconds=60)
    bundle.clock.advance(timedelta(seconds=120))
    response = bundle.client.post(
        "/admin/kill-switch",
        headers=headers,
        json={"reason": "manual intervention"},
    )
    assert response.status_code == 401
    assert not bundle.risk_manager.kill_switch.is_triggered()
    assert bundle.records == []


def test_kill_switch_rejects_missing_scope(remote_control_fixture: RemoteControlBundle) -> None:
    bundle = remote_control_fixture
    headers = _authorization_headers(bundle, scopes={"audit:read"})
    response = bundle.client.post(
        "/admin/kill-switch",
        headers=headers,
        json={"reason": "manual intervention"},
    )
    assert response.status_code == 403
    assert not bundle.risk_manager.kill_switch.is_triggered()
    assert bundle.records == []


def test_kill_switch_rejects_mtls_mismatch(remote_control_fixture: RemoteControlBundle) -> None:
    bundle = remote_control_fixture
    headers = _authorization_headers(bundle, thumbprint="thumb-correct")
    headers["X-Client-Cert-Thumbprint"] = "thumb-incorrect"
    response = bundle.client.post(
        "/admin/kill-switch",
        headers=headers,
        json={"reason": "manual intervention"},
    )
    assert response.status_code == 401
    assert not bundle.risk_manager.kill_switch.is_triggered()
    assert bundle.records == []


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
    bundle = remote_control_fixture
    headers = _authorization_headers(bundle, subject="auditor")
    headers["X-Forwarded-For"] = "203.0.113.10:443, 10.0.0.1"
    response = bundle.client.post(
        "/admin/kill-switch", headers=headers, json={"reason": "initial"}
    )
    assert response.status_code == 200
    assert bundle.records[0].ip_address == "203.0.113.10"


def test_kill_switch_falls_back_to_real_ip(remote_control_fixture: RemoteControlBundle) -> None:
    bundle = remote_control_fixture
    headers = _authorization_headers(bundle)
    headers.update(
        {
            "X-Real-IP": "2001:db8::1%eth0",
            "X-Forwarded-For": "invalid-entry",
        }
    )
    response = bundle.client.post(
        "/admin/kill-switch", headers=headers, json={"reason": "initial"}
    )
    assert response.status_code == 200
    assert bundle.records[0].ip_address == "2001:db8::1"
