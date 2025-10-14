"""Integration tests for the remote control router."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, List

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from src.admin.remote_control import (
    IdempotencyCache,
    InMemoryRateLimiter,
    RateLimitConfig,
    RemoteControlMetrics,
    RemoteControlContext,
    RetryConfig,
    SystemClock,
    TokenAuthenticator,
    create_remote_control_router,
)
from src.audit.audit_logger import AuditLogger
from src.risk.risk_manager import KillSwitchState


class _IntegrationRiskManager:
    def __init__(self) -> None:
        self.calls: List[str] = []

    def engage_kill_switch(self, reason: str) -> KillSwitchState:
        self.calls.append(reason)
        return KillSwitchState(engaged=True, reason=reason, already_engaged=len(self.calls) > 1)


@pytest.fixture()
def make_client() -> Callable[..., TestClient]:
    def _build(
        *,
        risk_manager: object,
        rate_limit_config: RateLimitConfig | None = None,
        rate_limiter: InMemoryRateLimiter | None = None,
        retry_config: RetryConfig | None = None,
    ) -> TestClient:
        clock = SystemClock()
        audit_logger = AuditLogger("secret", sink=lambda record: None, clock=lambda: datetime.now(timezone.utc))
        router = create_remote_control_router(
            risk_manager=risk_manager,
            audit_logger=audit_logger,
            authenticator=TokenAuthenticator("token"),
            rate_limit_config=rate_limit_config,
            retry_config=retry_config,
            rate_limiter=rate_limiter,
            idempotency_cache=IdempotencyCache(clock=clock),
            metrics=RemoteControlMetrics(registry=CollectorRegistry()),
            clock=clock,
        )
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    return _build


def test_successful_kill_switch_activation(make_client: Callable[..., TestClient]) -> None:
    integration_app = make_client(risk_manager=_IntegrationRiskManager())
    payload = {"reason": "halt trading"}
    headers = {
        "X-Admin-Token": "token",
        "X-Idempotency-Key": "idem-1",
        "X-Admin-Roles": "admin:kill-switch",
        "X-Correlation-ID": "corr-1",
    }

    response = integration_app.post("/admin/kill-switch", json=payload, headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["kill_switch_engaged"] is True
    assert body["idempotency_key"] == "idem-1"
    assert body["correlation_id"] == "corr-1"


def test_missing_idempotency_header_fails(make_client: Callable[..., TestClient]) -> None:
    integration_app = make_client(risk_manager=_IntegrationRiskManager())
    payload = {"reason": "halt trading"}
    headers = {
        "X-Admin-Token": "token",
        "X-Admin-Roles": "admin:kill-switch",
    }
    response = integration_app.post("/admin/kill-switch", json=payload, headers=headers)
    assert response.status_code == 400


def test_forbidden_without_role(make_client: Callable[..., TestClient]) -> None:
    integration_app = make_client(risk_manager=_IntegrationRiskManager())
    payload = {"reason": "halt trading"}
    headers = {
        "X-Admin-Token": "token",
        "X-Idempotency-Key": "idem-unauth",
        "X-Admin-Roles": "viewer",
    }
    response = integration_app.post("/admin/kill-switch", json=payload, headers=headers)
    assert response.status_code == 403


def test_rate_limit_returns_429(make_client: Callable[..., TestClient]) -> None:
    risk = _IntegrationRiskManager()
    clock = SystemClock()
    limiter = InMemoryRateLimiter(config=RateLimitConfig(max_requests=1, per_seconds=60), clock=clock)
    client = make_client(
        risk_manager=risk,
        rate_limit_config=RateLimitConfig(max_requests=1, per_seconds=60),
        rate_limiter=limiter,
    )
    headers = {
        "X-Admin-Token": "token",
        "X-Idempotency-Key": "idem-rl-1",
        "X-Admin-Roles": "admin:kill-switch",
    }
    payload = {"reason": "halt trading"}
    assert client.post("/admin/kill-switch", json=payload, headers=headers).status_code == 200
    headers["X-Idempotency-Key"] = "idem-rl-2"
    response = client.post("/admin/kill-switch", json=payload, headers=headers)
    assert response.status_code == 429


def test_remote_control_error_returns_503(make_client: Callable[..., TestClient]) -> None:
    class _FailingRisk:
        def engage_kill_switch(self, _: str) -> KillSwitchState:
            raise RuntimeError("boom")

    client = make_client(
        risk_manager=_FailingRisk(),
        retry_config=RetryConfig(max_attempts=1, timeout_seconds=0.01),
    )
    headers = {
        "X-Admin-Token": "token",
        "X-Idempotency-Key": "idem-fail",
        "X-Admin-Roles": "admin:kill-switch",
    }
    payload = {"reason": "halt trading"}
    response = client.post("/admin/kill-switch", json=payload, headers=headers)
    assert response.status_code == 503


def test_unexpected_error_returns_500(make_client: Callable[..., TestClient]) -> None:
    class _ExplodingLimiter(InMemoryRateLimiter):
        async def check(self, context: RemoteControlContext) -> None:  # type: ignore[override]
            raise RuntimeError("limiter exploded")

    clock = SystemClock()
    limiter = _ExplodingLimiter(config=RateLimitConfig(max_requests=1, per_seconds=60), clock=clock)
    client = make_client(
        risk_manager=_IntegrationRiskManager(),
        rate_limit_config=RateLimitConfig(max_requests=1, per_seconds=60),
        rate_limiter=limiter,
    )
    headers = {
        "X-Admin-Token": "token",
        "X-Idempotency-Key": "idem-err",
        "X-Admin-Roles": "admin:kill-switch",
    }
    payload = {"reason": "halt trading"}
    response = client.post("/admin/kill-switch", json=payload, headers=headers)
    assert response.status_code == 500
