"""Fuzz tests for the remote control FastAPI endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from prometheus_client import CollectorRegistry

from src.admin.remote_control import (
    IdempotencyCache,
    InMemoryRateLimiter,
    RateLimitConfig,
    RemoteControlMetrics,
    RetryConfig,
    SystemClock,
    TokenAuthenticator,
    create_remote_control_router,
)
from src.audit.audit_logger import AuditLogger
from src.risk.risk_manager import KillSwitchState


@dataclass
class _Sink:
    events: list[Dict[str, Any]]

    def __call__(self, record) -> None:  # type: ignore[override]
        self.events.append(record.details)


class _FuzzRiskManager:
    def __init__(self) -> None:
        self.counter = 0

    def engage_kill_switch(self, reason: str) -> KillSwitchState:
        self.counter += 1
        return KillSwitchState(engaged=True, reason=reason, already_engaged=self.counter > 1)


@pytest.fixture(scope="module")
def fuzz_client() -> TestClient:
    clock = SystemClock()
    risk = _FuzzRiskManager()
    sink = _Sink(events=[])
    audit_logger = AuditLogger("secret", sink=sink, clock=lambda: datetime.now(timezone.utc))
    rate_limiter = InMemoryRateLimiter(config=RateLimitConfig(max_requests=100, per_seconds=1), clock=clock)
    cache = IdempotencyCache(clock=clock)
    metrics = RemoteControlMetrics(registry=CollectorRegistry())
    retry = RetryConfig(max_attempts=2, timeout_seconds=0.5, initial_backoff_seconds=0.01)
    router = create_remote_control_router(
        risk_manager=risk,
        audit_logger=audit_logger,
        authenticator=TokenAuthenticator("token"),
        rate_limit_config=RateLimitConfig(max_requests=100, per_seconds=1),
        retry_config=retry,
        rate_limiter=rate_limiter,
        idempotency_cache=cache,
        metrics=metrics,
        clock=clock,
    )
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@given(
    reason=st.text(min_size=1, max_size=256),
    idem=st.uuids(),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_kill_switch_handles_varied_reasons(fuzz_client: TestClient, reason: str, idem) -> None:
    headers = {
        "X-Admin-Token": "token",
        "X-Admin-Roles": "admin:kill-switch",
        "X-Idempotency-Key": str(idem),
        "X-Correlation-ID": str(idem),
    }
    response = fuzz_client.post("/admin/kill-switch", json={"reason": reason}, headers=headers)
    if len(reason.strip()) >= 3:
        assert response.status_code == 200
        body = response.json()
        assert body["idempotency_key"] == str(idem)
    else:
        assert response.status_code == 422
