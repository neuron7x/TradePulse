"""FastAPI router implementing secure remote control operations."""

import hmac
import ipaddress
import math
import time
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Sequence

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.utils.metrics import get_metrics_collector
from src.audit.audit_logger import AuditLogger
from src.risk.risk_manager import KillSwitchState, RiskManagerFacade

__all__ = [
    "AdminIdentity",
    "AdminAccessPolicy",
    "AdminAccessPolicyConfig",
    "KillSwitchRequest",
    "KillSwitchResponse",
    "RateLimitSettings",
    "TokenAuthenticator",
    "create_remote_control_router",
]


_metrics = get_metrics_collector()
Network = ipaddress.IPv4Network | ipaddress.IPv6Network


@dataclass(frozen=True)
class RateLimitSettings:
    """Configuration for a sliding window rate limit."""

    max_attempts: int
    window_seconds: float

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")

    @classmethod
    def parse(cls, value: str) -> "RateLimitSettings":
        """Parse a rate limit specification formatted as "attempts/window"."""

        parts = value.strip().split("/")
        if len(parts) != 2:
            raise ValueError("rate limit specification must be formatted as 'attempts/window'")
        try:
            attempts = int(parts[0])
        except ValueError as exc:  # pragma: no cover - defensive conversion
            raise ValueError("attempts component must be an integer") from exc
        try:
            window = float(parts[1])
        except ValueError as exc:  # pragma: no cover - defensive conversion
            raise ValueError("window component must be numeric") from exc
        return cls(max_attempts=attempts, window_seconds=window)


@dataclass(frozen=True)
class AdminAccessPolicyConfig:
    """Container describing administrative access control policy."""

    allow_cidrs: tuple[str, ...] = ()
    subject_rate_limit: RateLimitSettings = RateLimitSettings(max_attempts=5, window_seconds=60.0)
    ip_rate_limit: RateLimitSettings = RateLimitSettings(max_attempts=30, window_seconds=60.0)


class _SlidingWindowLimiter:
    """Thread-safe sliding window rate limiter."""

    def __init__(self, settings: RateLimitSettings) -> None:
        self._settings = settings
        self._records: dict[str, deque[float]] = {}
        self._lock = Lock()

    def allow(self, key: str, *, now: float | None = None) -> tuple[bool, float]:
        """Return whether the key is permitted and the suggested retry delay."""

        current = time.monotonic() if now is None else now
        with self._lock:
            bucket = self._records.get(key)
            if bucket is None:
                bucket = deque()
                self._records[key] = bucket
            window = self._settings.window_seconds
            limit = self._settings.max_attempts
            while bucket and current - bucket[0] >= window:
                bucket.popleft()
            if len(bucket) >= limit:
                retry_after = window - (current - bucket[0])
                return False, max(retry_after, 0.0)
            bucket.append(current)
            return True, 0.0


class AdminIdentity(BaseModel):
    """Authenticated administrator identity extracted from the request."""

    subject: str = Field(..., description="Unique subject identifier for the administrator.")

    model_config = ConfigDict(frozen=True)


class KillSwitchRequest(BaseModel):
    """Request payload for activating the kill-switch."""

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

    @property
    def default_subject(self) -> str:
        """Return the fallback subject applied when none is provided."""

        return self._subject

    def __call__(
        self,
        provided: str = Header(..., alias="X-Admin-Token"),
        subject_override: str | None = Header(default=None, alias="X-Admin-Subject"),
    ) -> AdminIdentity:
        """Authenticate the incoming request and return the administrator identity."""

        if not provided or not hmac.compare_digest(provided, self._token):
            _metrics.record_admin_remote_control_attempt("denied")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid administrative token",
            )
        subject = subject_override or self._subject
        return AdminIdentity(subject=subject)


class AdminAccessPolicy:
    """FastAPI dependency enforcing admin access restrictions before authentication."""

    def __init__(
        self,
        audit_logger: AuditLogger,
        *,
        authenticator: TokenAuthenticator,
        allow_cidrs: Sequence[str] | None = None,
        subject_rate_limit: RateLimitSettings | None = None,
        ip_rate_limit: RateLimitSettings | None = None,
    ) -> None:
        self._audit_logger = audit_logger
        self._authenticator = authenticator
        self._allowed_networks = self._compile_networks(allow_cidrs or ())
        self._subject_limiter = (
            _SlidingWindowLimiter(subject_rate_limit) if subject_rate_limit is not None else None
        )
        self._ip_limiter = _SlidingWindowLimiter(ip_rate_limit) if ip_rate_limit is not None else None

    @staticmethod
    def _compile_networks(candidates: Sequence[str]) -> tuple[Network, ...]:
        networks: list[Network] = []
        for cidr in candidates:
            stripped = cidr.strip()
            if not stripped:
                continue
            network = ipaddress.ip_network(stripped, strict=False)
            networks.append(network)
        return tuple(networks)

    def _ip_allowed(self, ip_address_value: str) -> bool:
        if not self._allowed_networks:
            return True
        try:
            ip_obj = ipaddress.ip_address(ip_address_value)
        except ValueError:
            return False
        return any(ip_obj in network for network in self._allowed_networks)

    def _rate_limit(
        self,
        limiter: _SlidingWindowLimiter | None,
        key: str,
    ) -> tuple[bool, float]:
        if limiter is None:
            return True, 0.0
        return limiter.allow(key)

    async def __call__(self, request: Request) -> None:
        ip_address_value = _resolve_ip(request)
        subject = request.headers.get("X-Admin-Subject") or self._authenticator.default_subject

        if not self._ip_allowed(ip_address_value):
            _metrics.record_admin_remote_control_attempt("denied")
            self._audit_logger.log_event(
                event_type="kill_switch_access_denied",
                actor=subject,
                ip_address=ip_address_value,
                details={
                    "path": str(request.url.path),
                    "reason": "ip_not_allowed",
                    "allowed_cidrs": [str(network) for network in self._allowed_networks],
                },
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Administrative access forbidden from this address",
            )

        subject_allowed, subject_retry = self._rate_limit(self._subject_limiter, subject)
        ip_allowed, ip_retry = self._rate_limit(self._ip_limiter, ip_address_value)
        if not subject_allowed or not ip_allowed:
            retry_after = max(subject_retry, ip_retry)
            _metrics.record_admin_remote_control_attempt("throttled")
            if retry_after > 0:
                _metrics.observe_admin_remote_control_throttle(retry_after)
            headers = {}
            if retry_after > 0:
                headers["Retry-After"] = str(int(math.ceil(retry_after)))
            self._audit_logger.log_event(
                event_type="kill_switch_rate_limited",
                actor=subject,
                ip_address=ip_address_value,
                details={
                    "path": str(request.url.path),
                    "subject_retry_after": subject_retry,
                    "ip_retry_after": ip_retry,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Administrative access temporarily rate limited",
                headers=headers,
            )


def _resolve_ip(request: Request) -> str:
    """Return the originating IP address for the request."""

    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        for part in forwarded_for.split(","):
            candidate = _normalize_ip(part)
            if candidate is not None:
                return candidate

    real_ip = _normalize_ip(request.headers.get("X-Real-IP"))
    if real_ip is not None:
        return real_ip

    if request.client is not None and request.client.host:
        return request.client.host
    return "unknown"


def _normalize_ip(raw: str | None) -> str | None:
    """Return a validated IP address extracted from header data."""

    if raw is None:
        return None

    candidate = raw.strip()
    if not candidate:
        return None

    # Mitigate header injection attempts such as "1.1.1.1   malicious".
    candidate = candidate.split()[0]

    # Remove square brackets that may wrap IPv6 addresses (e.g. "[::1]").
    candidate = candidate.strip("[]")

    # Drop a zone identifier (e.g. "fe80::1%eth0") as ipaddress cannot parse it.
    if "%" in candidate:
        candidate = candidate.split("%", 1)[0]

    # Handle IPv4 addresses that include a port component.
    if "." in candidate and candidate.count(":") == 1:
        host, _, port = candidate.rpartition(":")
        if port.isdigit():
            candidate = host

    try:
        ipaddress.ip_address(candidate)
    except ValueError:
        return None
    return candidate


def create_remote_control_router(
    risk_manager: RiskManagerFacade,
    audit_logger: AuditLogger,
    authenticator: TokenAuthenticator,
    *,
    access_policy: AdminAccessPolicy | None = None,
    access_policy_config: AdminAccessPolicyConfig | None = None,
) -> APIRouter:
    """Create a router exposing secure administrative endpoints."""

    router = APIRouter(
        prefix="/admin",
        tags=["admin"],
        responses={401: {"description": "Unauthorized"}},
    )

    if access_policy is None:
        config = access_policy_config or AdminAccessPolicyConfig()
        access_policy = AdminAccessPolicy(
            audit_logger,
            authenticator=authenticator,
            allow_cidrs=config.allow_cidrs,
            subject_rate_limit=config.subject_rate_limit,
            ip_rate_limit=config.ip_rate_limit,
        )

    def get_risk_manager() -> RiskManagerFacade:
        return risk_manager

    def get_audit_logger() -> AuditLogger:
        return audit_logger

    @router.post(
        "/kill-switch",
        response_model=KillSwitchResponse,
        status_code=status.HTTP_200_OK,
        summary="Engage the global kill-switch",
    )
    async def engage_kill_switch(
        payload: KillSwitchRequest,
        request: Request,
        _: None = Depends(access_policy),
        identity: AdminIdentity = Depends(authenticator),
        manager: RiskManagerFacade = Depends(get_risk_manager),
        logger: AuditLogger = Depends(get_audit_logger),
    ) -> KillSwitchResponse:
        """Engage the risk manager kill-switch and log the operation."""

        state: KillSwitchState = manager.engage_kill_switch(payload.reason)
        _metrics.record_admin_remote_control_attempt("success")
        event_type = "kill_switch_reaffirmed" if state.already_engaged else "kill_switch_engaged"
        logger.log_event(
            event_type=event_type,
            actor=identity.subject,
            ip_address=_resolve_ip(request),
            details={
                "reason": state.reason,
                "already_engaged": state.already_engaged,
                "path": str(request.url.path),
            },
        )
        status_message = "already-engaged" if state.already_engaged else "engaged"
        return KillSwitchResponse(
            status=status_message,
            kill_switch_engaged=state.engaged,
            reason=state.reason,
            already_engaged=state.already_engaged,
        )

    return router
