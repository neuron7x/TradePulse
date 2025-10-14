"""Unit tests for the hardened remote control service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, cast

import pytest
from fastapi import HTTPException
from prometheus_client import CollectorRegistry

from src.admin.remote_control import (
    AuthorizationError,
    IdempotencyCache,
    InMemoryRateLimiter,
    KillSwitchRequest,
    RemoteCommand,
    RateLimitConfig,
    RateLimitExceeded,
    RemoteControlContext,
    RemoteControlMetrics,
    RemoteControlService,
    RemoteControlError,
    RetryConfig,
    SystemClock,
    TokenAuthenticator,
    _correlation_id,
    _extract_idempotency_key,
    _resolve_ip,
)
from src.audit.audit_logger import AuditLogger
from src.risk.risk_manager import KillSwitchState


@dataclass
class _KillSwitchLog:
    """Simple structure to capture audit log invocations."""

    events: List[Dict[str, Any]]

    def __call__(self, record) -> None:  # type: ignore[override]
        self.events.append(record.details)


class _DummyRiskManager:
    """Risk manager stub that records invocations."""

    def __init__(self) -> None:
        self.calls: int = 0
        self.reason: str = ""
        self.already_engaged: bool = False

    def engage_kill_switch(self, reason: str) -> KillSwitchState:
        self.calls += 1
        already = self.already_engaged
        self.already_engaged = True
        self.reason = reason
        return KillSwitchState(engaged=True, reason=reason, already_engaged=already)


class _AlwaysDenyRateLimiter:
    """Rate limiter that consistently rejects invocations."""

    async def check(self, _: RemoteControlContext) -> None:  # pragma: no cover - trivial
        raise RateLimitExceeded("rate limited")


class _FakeClock:
    """Deterministic clock used for cache and rate-limiter tests."""

    def __init__(self) -> None:
        self._monotonic = 0.0

    def utcnow(self) -> datetime:
        return datetime.now(timezone.utc)

    def monotonic(self) -> float:
        return self._monotonic

    def advance(self, seconds: float) -> None:
        self._monotonic += seconds


@pytest.fixture(name="clock")
def fixture_clock() -> SystemClock:
    return SystemClock()


@pytest.fixture(name="audit_logger")
def fixture_audit_logger() -> AuditLogger:
    sink = _KillSwitchLog(events=[])
    return AuditLogger("secret", sink=sink, clock=lambda: datetime.now(timezone.utc))


@pytest.mark.asyncio
async def test_service_enforces_authorization(clock: SystemClock, audit_logger: AuditLogger) -> None:
    risk = _DummyRiskManager()
    cache = IdempotencyCache(clock=clock)
    limiter = InMemoryRateLimiter(config=RateLimitConfig(max_requests=5, per_seconds=10), clock=clock)
    service = RemoteControlService(
        risk_manager=risk,
        audit_logger=audit_logger,
        rate_limiter=limiter,
        idempotency_cache=cache,
        retry_config=RetryConfig(max_attempts=1, timeout_seconds=0.5),
        metrics=RemoteControlMetrics(registry=CollectorRegistry()),
        clock=clock,
    )

    context = RemoteControlContext(
        correlation_id="corr-1",
        subject="user",
        roles=("observer",),
        idempotency_key="idem-1",
        ip_address="127.0.0.1",
        received_at=clock.utcnow(),
    )

    with pytest.raises(AuthorizationError):
        await service.engage_kill_switch(context=context, payload=KillSwitchRequest(reason="halt"))

    assert risk.calls == 0


@pytest.mark.asyncio
async def test_service_enforces_rate_limit(clock: SystemClock, audit_logger: AuditLogger) -> None:
    risk = _DummyRiskManager()
    cache = IdempotencyCache(clock=clock)
    service = RemoteControlService(
        risk_manager=risk,
        audit_logger=audit_logger,
        rate_limiter=_AlwaysDenyRateLimiter(),
        idempotency_cache=cache,
        retry_config=RetryConfig(max_attempts=1, timeout_seconds=0.5),
        metrics=RemoteControlMetrics(registry=CollectorRegistry()),
        clock=clock,
    )

    context = RemoteControlContext(
        correlation_id="corr-2",
        subject="user",
        roles=("admin:kill-switch",),
        idempotency_key="idem-2",
        ip_address="127.0.0.1",
        received_at=clock.utcnow(),
    )

    with pytest.raises(RateLimitExceeded):
        await service.engage_kill_switch(context=context, payload=KillSwitchRequest(reason="halt"))

    assert risk.calls == 0


@pytest.mark.asyncio
async def test_service_caches_idempotent_response(clock: SystemClock, audit_logger: AuditLogger) -> None:
    risk = _DummyRiskManager()
    cache = IdempotencyCache(clock=clock)
    limiter = InMemoryRateLimiter(config=RateLimitConfig(max_requests=10, per_seconds=10), clock=clock)
    service = RemoteControlService(
        risk_manager=risk,
        audit_logger=audit_logger,
        rate_limiter=limiter,
        idempotency_cache=cache,
        retry_config=RetryConfig(max_attempts=2, timeout_seconds=0.5),
        metrics=RemoteControlMetrics(registry=CollectorRegistry()),
        clock=clock,
    )

    context = RemoteControlContext(
        correlation_id="corr-3",
        subject="user",
        roles=("admin:kill-switch",),
        idempotency_key="idem-3",
        ip_address="127.0.0.1",
        received_at=clock.utcnow(),
    )

    payload = KillSwitchRequest(reason="halt now")
    first = await service.engage_kill_switch(context=context, payload=payload)
    second = await service.engage_kill_switch(context=context, payload=payload)

    assert first == second
    assert risk.calls == 1


@pytest.mark.asyncio
async def test_retry_logic_retries_on_failure(clock: SystemClock, audit_logger: AuditLogger) -> None:
    failures: int = 0

    class _FlakyRisk:
        def engage_kill_switch(self, reason: str) -> KillSwitchState:
            nonlocal failures
            if failures < 1:
                failures += 1
                raise RuntimeError("transient error")
            return KillSwitchState(engaged=True, reason=reason, already_engaged=False)

    cache = IdempotencyCache(clock=clock)
    limiter = InMemoryRateLimiter(config=RateLimitConfig(max_requests=10, per_seconds=10), clock=clock)
    service = RemoteControlService(
        risk_manager=_FlakyRisk(),
        audit_logger=audit_logger,
        rate_limiter=limiter,
        idempotency_cache=cache,
        retry_config=RetryConfig(max_attempts=3, timeout_seconds=0.5, initial_backoff_seconds=0.01),
        metrics=RemoteControlMetrics(registry=CollectorRegistry()),
        clock=clock,
    )

    context = RemoteControlContext(
        correlation_id="corr-4",
        subject="user",
        roles=("admin:kill-switch",),
        idempotency_key="idem-4",
        ip_address="127.0.0.1",
        received_at=clock.utcnow(),
    )

    response = await service.engage_kill_switch(context=context, payload=KillSwitchRequest(reason="halt"))
    assert response.kill_switch_engaged is True
    assert failures == 1


def test_rate_limit_config_validation() -> None:
    with pytest.raises(ValueError):
        RateLimitConfig(max_requests=0)
    with pytest.raises(ValueError):
        RateLimitConfig(per_seconds=0)


def test_retry_config_validation() -> None:
    with pytest.raises(ValueError):
        RetryConfig(max_attempts=0)
    with pytest.raises(ValueError):
        RetryConfig(initial_backoff_seconds=0)
    with pytest.raises(ValueError):
        RetryConfig(initial_backoff_seconds=1, max_backoff_seconds=0.5)
    with pytest.raises(ValueError):
        RetryConfig(timeout_seconds=0)
    with pytest.raises(ValueError):
        RetryConfig(backoff_multiplier=0.5)


@pytest.mark.asyncio
async def test_idempotency_cache_expires_entries() -> None:
    clock = _FakeClock()
    cache = IdempotencyCache(clock=clock, ttl_seconds=0.5)
    await cache.set("key", {"value": 1})
    assert await cache.get("key") == {"value": 1}
    clock.advance(1.0)
    assert await cache.get("key") is None


@pytest.mark.asyncio
async def test_inmemory_rate_limiter_sliding_window() -> None:
    clock = _FakeClock()
    limiter = InMemoryRateLimiter(config=RateLimitConfig(max_requests=2, per_seconds=1), clock=clock)
    context = RemoteControlContext(
        correlation_id="corr-rl",
        subject="rate-user",
        roles=("admin:kill-switch",),
        idempotency_key="rate-key",
        ip_address="127.0.0.1",
        received_at=clock.utcnow(),
    )
    await limiter.check(context)
    await limiter.check(context)
    with pytest.raises(RateLimitExceeded):
        await limiter.check(context)
    clock.advance(1.5)
    await limiter.check(context)


@pytest.mark.asyncio
async def test_service_rejects_unsupported_command(clock: SystemClock, audit_logger: AuditLogger) -> None:
    risk = _DummyRiskManager()
    cache = IdempotencyCache(clock=clock)
    limiter = InMemoryRateLimiter(config=RateLimitConfig(max_requests=5, per_seconds=10), clock=clock)
    service = RemoteControlService(
        risk_manager=risk,
        audit_logger=audit_logger,
        rate_limiter=limiter,
        idempotency_cache=cache,
        retry_config=RetryConfig(max_attempts=1, timeout_seconds=0.5),
        metrics=RemoteControlMetrics(registry=CollectorRegistry()),
        clock=clock,
    )

    context = RemoteControlContext(
        correlation_id="corr-invalid",
        subject="user",
        roles=("admin:kill-switch",),
        idempotency_key="idem-invalid",
        ip_address="127.0.0.1",
        received_at=clock.utcnow(),
    )

    payload = KillSwitchRequest(reason="halt")
    payload.command = cast(RemoteCommand, cast(Any, "shutdown"))

    with pytest.raises(RemoteControlError):
        await service.engage_kill_switch(context=context, payload=payload)


@pytest.mark.asyncio
async def test_service_records_metrics_on_failure(clock: SystemClock, audit_logger: AuditLogger) -> None:
    class _FailRisk:
        def engage_kill_switch(self, _: str) -> KillSwitchState:
            raise RuntimeError("boom")

    metrics = RemoteControlMetrics(registry=CollectorRegistry())
    cache = IdempotencyCache(clock=clock)
    limiter = InMemoryRateLimiter(config=RateLimitConfig(max_requests=5, per_seconds=10), clock=clock)
    service = RemoteControlService(
        risk_manager=_FailRisk(),
        audit_logger=audit_logger,
        rate_limiter=limiter,
        idempotency_cache=cache,
        retry_config=RetryConfig(max_attempts=1, timeout_seconds=0.01),
        metrics=metrics,
        clock=clock,
    )

    context = RemoteControlContext(
        correlation_id="corr-fail",
        subject="user",
        roles=("admin:kill-switch",),
        idempotency_key="idem-fail",
        ip_address="127.0.0.1",
        received_at=clock.utcnow(),
    )

    with pytest.raises(RemoteControlError):
        await service.engage_kill_switch(context=context, payload=KillSwitchRequest(reason="halt"))

    assert metrics.errors._value.get() == 1  # type: ignore[attr-defined]


def test_token_authenticator_rejects_invalid_token() -> None:
    authenticator = TokenAuthenticator("secret")
    with pytest.raises(HTTPException) as exc:
        authenticator(provided="wrong", subject_override=None, roles_header=None)
    assert exc.value.status_code == 401


def test_token_authenticator_default_roles() -> None:
    authenticator = TokenAuthenticator("secret")
    identity = authenticator(provided="secret", subject_override=None, roles_header=None)
    assert identity.roles == ("admin:kill-switch",)


def test_token_authenticator_parses_roles() -> None:
    authenticator = TokenAuthenticator("secret")
    identity = authenticator(
        provided="secret",
        subject_override="alice",
        roles_header="admin:super, admin:ops",
    )
    assert identity.subject == "alice"
    assert identity.roles == ("admin:super", "admin:ops")


def test_resolve_ip_unknown() -> None:
    request = SimpleNamespace(client=None)
    assert _resolve_ip(request) == "unknown"


def test_extract_idempotency_key_requires_header() -> None:
    request = SimpleNamespace(headers={})
    with pytest.raises(HTTPException):
        _extract_idempotency_key(request)


def test_correlation_id_generation_and_passthrough() -> None:
    generated = _correlation_id(SimpleNamespace(headers={}))
    assert len(generated) == 36
    request = SimpleNamespace(headers={"X-Correlation-ID": "abc"})
    assert _correlation_id(request) == "abc"
