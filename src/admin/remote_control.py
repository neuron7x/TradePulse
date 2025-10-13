"""FastAPI router implementing hardened remote control operations."""

from __future__ import annotations

import asyncio
import hmac
import threading
import time
from contextlib import nullcontext
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Mapping, MutableMapping, Optional, Protocol, TypedDict
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.utils.logging import get_logger
from src.audit.audit_logger import AuditLogger
from src.risk.risk_manager import KillSwitchState, RiskManagerFacade

try:  # pragma: no cover - optional dependency
    from prometheus_client import Counter, Histogram
except Exception:  # pragma: no cover - degrade gracefully
    class Counter:  # type: ignore[too-many-ancestors]
        """Fallback no-op counter when Prometheus is not installed."""

        def __init__(self, *_: Any, **__: Any) -> None:
            return

        def inc(self, *_: Any, **__: Any) -> None:
            return

    class Histogram:  # type: ignore[too-many-ancestors]
        """Fallback no-op histogram when Prometheus is not installed."""

        def __init__(self, *_: Any, **__: Any) -> None:
            return

        def observe(self, *_: Any, **__: Any) -> None:
            return

try:  # pragma: no cover - optional dependency
    from opentelemetry import trace
except Exception:  # pragma: no cover - degrade gracefully
    trace = None  # type: ignore[assignment]

__all__ = [
    "AdminIdentity",
    "KillSwitchRequest",
    "KillSwitchResponse",
    "RateLimitConfig",
    "RetryConfig",
    "TokenAuthenticator",
    "create_remote_control_router",
]


class RemoteControlError(RuntimeError):
    """Base exception for remote control failures."""


class AuthorizationError(RemoteControlError):
    """Raised when an administrator lacks the required role."""


class RateLimitExceeded(RemoteControlError):
    """Raised when the rate limiter rejects an operation."""


class RemoteCommand(str, Enum):
    """White-listed remote control commands."""

    ENGAGE_KILL_SWITCH = "engage_kill_switch"


ALLOWED_COMMANDS = frozenset({RemoteCommand.ENGAGE_KILL_SWITCH})


class Clock(Protocol):
    """Abstraction over wall-clock and monotonic time providers."""

    def utcnow(self) -> datetime:
        """Return the current UTC time as a timezone-aware datetime."""

    def monotonic(self) -> float:
        """Return the current monotonic time in seconds."""


class RateLimiter(Protocol):
    """Protocol implemented by asynchronous rate limiters."""

    async def check(self, context: "RemoteControlContext") -> None:
        """Validate the request against the rate limiter, raising on violations."""


@dataclass(slots=True)
class SystemClock:
    """Clock implementation backed by :mod:`datetime` and :mod:`time`."""

    def utcnow(self) -> datetime:
        return datetime.now(timezone.utc)

    def monotonic(self) -> float:
        return time.monotonic()


@dataclass(slots=True)
class RateLimitConfig:
    """Configuration for remote control rate limiting."""

    max_requests: int = 5
    per_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.max_requests <= 0:
            raise ValueError("max_requests must be positive")
        if self.per_seconds <= 0:
            raise ValueError("per_seconds must be positive")


@dataclass(slots=True)
class RetryConfig:
    """Retry semantics for kill-switch engagement."""

    max_attempts: int = 3
    initial_backoff_seconds: float = 0.05
    max_backoff_seconds: float = 0.5
    timeout_seconds: float = 1.5
    backoff_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.initial_backoff_seconds <= 0:
            raise ValueError("initial_backoff_seconds must be positive")
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("max_backoff_seconds must be >= initial_backoff_seconds")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be >= 1")


@dataclass(slots=True)
class RemoteControlContext:
    """Security context extracted from the incoming request."""

    correlation_id: str
    subject: str
    roles: tuple[str, ...]
    idempotency_key: str
    ip_address: str
    received_at: datetime


@dataclass(slots=True)
class CachedKillSwitchState:
    """Cached kill-switch state snapshot."""

    expires_at: float
    state: KillSwitchState


@dataclass(slots=True)
class CachedResponse:
    """Stored idempotent response with an expiration."""

    expires_at: float
    payload: dict[str, Any]


class IdempotencyCache:
    """Cache ensuring idempotent processing across retries."""

    def __init__(
        self,
        *,
        clock: Clock,
        ttl_seconds: float = 300.0,
        storage: Optional[MutableMapping[str, CachedResponse]] = None,
    ) -> None:
        self._clock = clock
        self._ttl = ttl_seconds
        self._storage = storage if storage is not None else {}
        self._locks: MutableMapping[str, asyncio.Lock] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[dict[str, Any]]:
        """Return a cached response if it remains valid."""

        async with self._lock:
            cached = self._storage.get(key)
            if cached is None:
                return None
            now = self._clock.monotonic()
            if now >= cached.expires_at:
                self._storage.pop(key, None)
                return None
            return dict(cached.payload)

    async def set(self, key: str, payload: Mapping[str, Any]) -> None:
        """Persist a response in the cache."""

        async with self._lock:
            expires_at = self._clock.monotonic() + self._ttl
            self._storage[key] = CachedResponse(expires_at=expires_at, payload=dict(payload))

    async def acquire_lock(self, key: str) -> asyncio.Lock:
        """Return a per-key asynchronous lock for atomic operations."""

        async with self._lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            return lock


class InMemoryRateLimiter:
    """Simple asynchronous sliding-window rate limiter."""

    def __init__(
        self,
        *,
        config: RateLimitConfig,
        clock: Clock,
        storage: Optional[MutableMapping[str, deque[float]]] = None,
    ) -> None:
        self._config = config
        self._clock = clock
        self._storage = storage if storage is not None else defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, context: RemoteControlContext) -> None:
        """Enforce the configured rate limits for the administrator."""

        subject_key = context.subject
        now = self._clock.monotonic()
        window_start = now - self._config.per_seconds

        async with self._lock:
            timestamps = self._storage[subject_key]
            while timestamps and timestamps[0] < window_start:
                timestamps.popleft()
            if len(timestamps) >= self._config.max_requests:
                raise RateLimitExceeded(
                    f"Rate limit exceeded for subject '{context.subject}'"
                )
            timestamps.append(now)


@dataclass(slots=True)
class RemoteControlMetrics:
    """Prometheus-backed metrics for remote control operations."""

    registry: Any | None = None
    engagements: Counter = field(init=False)
    rate_limited: Counter = field(init=False)
    errors: Counter = field(init=False)
    latency_seconds: Histogram = field(init=False)

    def __post_init__(self) -> None:
        metric_kwargs: dict[str, Any]
        metric_kwargs = {"registry": self.registry} if self.registry is not None else {}
        self.engagements = Counter(
            "tradepulse_admin_kill_switch_total",
            "Total kill-switch engagement attempts",
            **metric_kwargs,
        )
        self.rate_limited = Counter(
            "tradepulse_admin_kill_switch_rate_limited_total",
            "Total kill-switch requests rejected by rate limiting",
            **metric_kwargs,
        )
        self.errors = Counter(
            "tradepulse_admin_kill_switch_errors_total",
            "Total kill-switch attempts that resulted in errors",
            **metric_kwargs,
        )
        self.latency_seconds = Histogram(
            "tradepulse_admin_kill_switch_latency_seconds",
            "Time spent executing kill-switch operations",
            **metric_kwargs,
        )

    def record_success(self, duration: float) -> None:
        self.engagements.inc()
        self.latency_seconds.observe(duration)

    def record_rate_limited(self) -> None:
        self.rate_limited.inc()

    def record_error(self) -> None:
        self.errors.inc()


class RemoteControlService:
    """Business logic for remote control requests."""

    def __init__(
        self,
        *,
        risk_manager: RiskManagerFacade,
        audit_logger: AuditLogger,
        rate_limiter: RateLimiter,
        idempotency_cache: IdempotencyCache,
        retry_config: RetryConfig,
        metrics: RemoteControlMetrics,
        clock: Clock,
        cache_ttl: float = 1.0,
    ) -> None:
        self._risk_manager = risk_manager
        self._audit_logger = audit_logger
        self._rate_limiter = rate_limiter
        self._idempotency_cache = idempotency_cache
        self._retry_config = retry_config
        self._metrics = metrics
        self._clock = clock
        self._state_cache: MutableMapping[str, CachedKillSwitchState] = {}
        self._state_lock = threading.Lock()
        self._cache_ttl = cache_ttl
        self._tracer = trace.get_tracer(__name__) if trace is not None else None

    async def engage_kill_switch(
        self,
        *,
        context: RemoteControlContext,
        payload: "KillSwitchRequest",
    ) -> "KillSwitchResponse":
        """Engage the kill-switch with retries, auditing, and telemetry."""

        if payload.command not in ALLOWED_COMMANDS:
            raise RemoteControlError(f"Unsupported command: {payload.command}")

        self._ensure_authorized(context)
        logger = get_logger("tradepulse.admin.remote_control", context.correlation_id)

        try:
            await self._rate_limiter.check(context)
        except RateLimitExceeded:
            self._metrics.record_rate_limited()
            raise

        cached_response = await self._idempotency_cache.get(context.idempotency_key)
        if cached_response is not None:
            return KillSwitchResponse(**cached_response)

        lock = await self._idempotency_cache.acquire_lock(context.idempotency_key)
        async with lock:
            cached_response = await self._idempotency_cache.get(context.idempotency_key)
            if cached_response is not None:
                return KillSwitchResponse(**cached_response)

            if self._tracer is not None:  # pragma: no branch - optional dependency
                span_cm = self._tracer.start_as_current_span("admin.kill_switch")
            else:
                span_cm = nullcontext()

            with span_cm as span:
                if span is not None:  # pragma: no branch - optional dependency
                    span.set_attribute("subject", context.subject)
                    span.set_attribute("correlation_id", context.correlation_id)

                start = self._clock.monotonic()
                try:
                    state = await self._execute_with_retries(payload.reason)
                except Exception as exc:  # noqa: BLE001 - propagate sanitized error
                    self._metrics.record_error()
                    logger.error(
                        "kill_switch_failure",
                        correlation_id=context.correlation_id,
                        subject=context.subject,
                        error=str(exc),
                    )
                    raise
                duration = self._clock.monotonic() - start
                self._metrics.record_success(duration)

            response = KillSwitchResponse(
                status="already-engaged" if state.already_engaged else "engaged",
                kill_switch_engaged=True,
                reason=state.reason,
                already_engaged=state.already_engaged,
                correlation_id=context.correlation_id,
                idempotency_key=context.idempotency_key,
                timestamp=context.received_at,
            )
            await self._idempotency_cache.set(context.idempotency_key, response.model_dump())

        self._audit_logger.log_event(
            event_type=(
                "kill_switch_reaffirmed" if state.already_engaged else "kill_switch_engaged"
            ),
            actor=context.subject,
            ip_address=context.ip_address,
            details={
                "reason": state.reason,
                "already_engaged": state.already_engaged,
                "correlation_id": context.correlation_id,
                "idempotency_key": context.idempotency_key,
                "received_at": context.received_at,
            },
        )
        return response

    def _ensure_authorized(self, context: RemoteControlContext) -> None:
        roles = {role.strip().lower() for role in context.roles}
        if "admin:kill-switch" not in roles and "admin:super" not in roles:
            raise AuthorizationError("Administrator lacks kill-switch privileges")

    async def _execute_with_retries(self, reason: str) -> KillSwitchState:
        attempt = 0
        delay = self._retry_config.initial_backoff_seconds
        last_error: Exception | None = None

        while attempt < self._retry_config.max_attempts:
            attempt += 1
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(self._engage_kill_switch, reason),
                    timeout=self._retry_config.timeout_seconds,
                )
            except Exception as exc:  # noqa: BLE001 - handled generically
                last_error = exc
                if attempt >= self._retry_config.max_attempts:
                    break
                await asyncio.sleep(delay)
                delay = min(
                    delay * self._retry_config.backoff_multiplier,
                    self._retry_config.max_backoff_seconds,
                )

        raise RemoteControlError("Unable to engage kill-switch") from last_error

    def _engage_kill_switch(self, reason: str) -> KillSwitchState:
        """Invoke the risk manager synchronously with caching."""

        now = self._clock.monotonic()
        cache_key = reason.strip().lower()

        with self._state_lock:
            cached_state = self._state_cache.get(cache_key)
            if cached_state is not None and cached_state.expires_at > now:
                return cached_state.state

            state = self._risk_manager.engage_kill_switch(reason)
            expires_at = now + self._cache_ttl
            self._state_cache[cache_key] = CachedKillSwitchState(
                expires_at=expires_at,
                state=state,
            )
            return state


class AdminIdentity(BaseModel):
    """Authenticated administrator identity extracted from the request."""

    subject: str = Field(..., description="Unique subject identifier for the administrator.")
    roles: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Roles assigned to the administrator used for authorization.",
    )

    model_config = ConfigDict(frozen=True)


class KillSwitchRequest(BaseModel):
    """Request payload for activating the kill-switch."""

    command: RemoteCommand = Field(
        default=RemoteCommand.ENGAGE_KILL_SWITCH,
        description="Name of the remote control command to execute.",
    )
    reason: str = Field(..., min_length=3, max_length=256, description="Human readable reason.")

    @field_validator("reason")
    @classmethod
    def _trim_and_validate_reason(cls, value: str) -> str:
        stripped = value.strip()
        if len(stripped) < 3:
            raise ValueError("reason must be at least 3 characters long")
        return stripped


class KillSwitchResponse(BaseModel):
    """Response payload describing the kill-switch state."""

    status: str = Field(..., description="Status message of the kill-switch operation.")
    kill_switch_engaged: bool = Field(..., description="Whether the kill-switch is active.")
    reason: str = Field(..., description="Reason supplied when the kill-switch was engaged.")
    already_engaged: bool = Field(..., description="True if the kill-switch was already active.")
    correlation_id: str = Field(..., description="Correlation identifier for tracing logs.")
    idempotency_key: str = Field(..., description="Idempotency key supplied with the request.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Normalized UTC timestamp when the request was processed.",
    )


class TokenAuthenticator:
    """Validate administrative requests using a static token."""

    def __init__(
        self,
        token: str,
        *,
        subject: str = "remote-admin",
    ) -> None:
        if not token:
            raise ValueError("token must be provided for remote control authentication")
        self._token = token
        self._subject = subject

    def __call__(
        self,
        provided: str = Header(..., alias="X-Admin-Token"),
        subject_override: str | None = Header(default=None, alias="X-Admin-Subject"),
        roles_header: str | None = Header(default=None, alias="X-Admin-Roles"),
    ) -> AdminIdentity:
        """Authenticate the incoming request and return the administrator identity."""

        if not provided or not hmac.compare_digest(provided, self._token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid administrative token",
            )
        subject = subject_override or self._subject
        roles: tuple[str, ...]
        if roles_header:
            roles = tuple(role.strip() for role in roles_header.split(",") if role.strip())
        else:
            roles = ("admin:kill-switch",)
        return AdminIdentity(subject=subject, roles=roles)


class RequestMetadata(TypedDict):
    """Structured request metadata for logging."""

    correlation_id: str
    idempotency_key: str
    ip_address: str
    subject: str


def _resolve_ip(request: Request) -> str:
    """Return the originating IP address for the request."""

    if request.client is None:
        return "unknown"
    return request.client.host


def _extract_idempotency_key(request: Request) -> str:
    """Return the request idempotency key, generating a deterministic value if absent."""

    key = request.headers.get("X-Idempotency-Key")
    if not key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-Idempotency-Key header is required",
        )
    return key.strip()


def _correlation_id(request: Request) -> str:
    """Return or generate a correlation identifier."""

    header = request.headers.get("X-Correlation-ID")
    return header.strip() if header else str(uuid4())


def create_remote_control_router(
    risk_manager: RiskManagerFacade,
    audit_logger: AuditLogger,
    authenticator: TokenAuthenticator,
    *,
    rate_limit_config: RateLimitConfig | None = None,
    retry_config: RetryConfig | None = None,
    rate_limiter: RateLimiter | None = None,
    idempotency_cache: IdempotencyCache | None = None,
    metrics: RemoteControlMetrics | None = None,
    clock: Clock | None = None,
) -> APIRouter:
    """Create a router exposing secure administrative endpoints."""

    real_clock = clock or SystemClock()
    rl_config = rate_limit_config or RateLimitConfig()
    retries = retry_config or RetryConfig()
    cache = idempotency_cache or IdempotencyCache(clock=real_clock)
    limiter = rate_limiter or InMemoryRateLimiter(config=rl_config, clock=real_clock)
    metric_set = metrics or RemoteControlMetrics()

    service = RemoteControlService(
        risk_manager=risk_manager,
        audit_logger=audit_logger,
        rate_limiter=limiter,
        idempotency_cache=cache,
        retry_config=retries,
        metrics=metric_set,
        clock=real_clock,
    )

    router = APIRouter(
        prefix="/admin",
        tags=["admin"],
        responses={401: {"description": "Unauthorized"}},
    )

    def get_service() -> RemoteControlService:
        return service

    @router.post(
        "/kill-switch",
        response_model=KillSwitchResponse,
        status_code=status.HTTP_200_OK,
        summary="Engage the global kill-switch",
    )
    async def engage_kill_switch(
        payload: KillSwitchRequest,
        request: Request,
        identity: AdminIdentity = Depends(authenticator),
        service: RemoteControlService = Depends(get_service),
    ) -> KillSwitchResponse:
        """Engage the risk manager kill-switch and log the operation."""

        correlation_id = _correlation_id(request)
        request.state.correlation_id = correlation_id
        idempotency_key = _extract_idempotency_key(request)
        ip_address = _resolve_ip(request)
        logger = get_logger("tradepulse.admin.remote_control", correlation_id)
        metadata: RequestMetadata = {
            "correlation_id": correlation_id,
            "idempotency_key": idempotency_key,
            "ip_address": ip_address,
            "subject": identity.subject,
        }
        logger.info("kill_switch_request_received", **metadata)

        context = RemoteControlContext(
            correlation_id=correlation_id,
            subject=identity.subject,
            roles=identity.roles,
            idempotency_key=idempotency_key,
            ip_address=ip_address,
            received_at=real_clock.utcnow(),
        )

        try:
            response = await service.engage_kill_switch(context=context, payload=payload)
        except AuthorizationError as exc:
            logger.warning("kill_switch_unauthorized", error=str(exc), **metadata)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Administrator lacks authorization for kill-switch",
            ) from exc
        except RateLimitExceeded as exc:
            logger.warning("kill_switch_rate_limited", error=str(exc), **metadata)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Kill-switch rate limit exceeded",
            ) from exc
        except RemoteControlError as exc:
            logger.error("kill_switch_remote_control_error", error=str(exc), **metadata)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Kill-switch unavailable",
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - final catch-all
            logger.error("kill_switch_unexpected_error", error=str(exc), **metadata)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unexpected kill-switch failure",
            ) from exc

        logger.info("kill_switch_request_completed", **metadata)
        return response

    return router
