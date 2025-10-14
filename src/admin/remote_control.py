"""FastAPI router implementing secure remote control operations."""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Protocol, Sequence

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.audit.audit_logger import AuditLogger
from src.risk.risk_manager import KillSwitchState, RiskManagerFacade

__all__ = [
    "AdminIdentity",
    "KillSwitchRequest",
    "KillSwitchResponse",
    "RequestContext",
    "ShortLivedTokenVerifier",
    "create_remote_control_router",
]


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


class RequestContext(BaseModel):
    """Runtime context extracted from verified administrative credentials."""

    identity: AdminIdentity
    scopes: tuple[str, ...]
    audience: str
    expires_at: datetime
    client_thumbprint: str | None = None

    model_config = ConfigDict(frozen=True)


class SecretManager(Protocol):
    """Minimal protocol describing how secrets are retrieved for signing."""

    def get_secret(self, name: str) -> str:
        """Return the plaintext secret identified by ``name``."""


class ShortLivedTokenVerifier:
    """Issue and validate HMAC-backed short-lived credentials."""

    def __init__(
        self,
        *,
        secret_manager: SecretManager,
        secret_name: str,
        clock: Callable[[], datetime] | None = None,
        leeway_seconds: int = 30,
    ) -> None:
        if not secret_name:
            raise ValueError("secret_name must be provided for credential verification")
        self._secret_manager = secret_manager
        self._secret_name = secret_name
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(timezone.utc))
        self._leeway = timedelta(seconds=max(leeway_seconds, 0))

    def issue_token(
        self,
        *,
        subject: str,
        scopes: Iterable[str],
        audience: str,
        ttl_seconds: int,
        cert_thumbprint: str | None = None,
    ) -> str:
        """Create a signed credential for administrative requests."""

        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive for issued credentials")
        now = self._clock()
        expires_at = now + timedelta(seconds=ttl_seconds)
        payload = {
            "sub": subject,
            "aud": audience,
            "scopes": sorted(set(scopes)),
            "exp": int(expires_at.timestamp()),
            "iat": int(now.timestamp()),
        }
        if cert_thumbprint:
            payload["cert_thumbprint"] = cert_thumbprint
        return self._encode_payload(payload)

    def dependency(
        self,
        *,
        audience: str,
        required_scopes: Sequence[str] = (),
    ) -> Callable[[Request, str, str | None], RequestContext]:
        """Return a FastAPI dependency that validates credentials and scope."""

        required_scope_set = frozenset(required_scopes)

        def _dependency(
            request: Request,
            authorization: str = Header(..., alias="Authorization"),
            client_thumbprint: str | None = Header(None, alias="X-Client-Cert-Thumbprint"),
        ) -> RequestContext:
            payload = self._decode_and_validate(
                authorization,
                expected_audience=audience,
                required_scopes=required_scope_set,
                provided_thumbprint=client_thumbprint,
            )
            expires_at = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
            scopes = tuple(payload.get("scopes", ()))
            identity = AdminIdentity(subject=payload["sub"])
            context = RequestContext(
                identity=identity,
                scopes=scopes,
                audience=payload["aud"],
                expires_at=expires_at,
                client_thumbprint=payload.get("cert_thumbprint"),
            )
            request.state.admin_context = context
            return context

        return _dependency

    def _encode_payload(self, payload: dict[str, object]) -> str:
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        body_segment = _urlsafe_b64encode(body)
        secret = self._secret_manager.get_secret(self._secret_name).encode("utf-8")
        digest = _urlsafe_b64encode(_compute_hmac(secret, body_segment, hashlib.sha256))
        return f"{body_segment}.{digest}"

    def _decode_and_validate(
        self,
        authorization_header: str,
        *,
        expected_audience: str,
        required_scopes: Iterable[str],
        provided_thumbprint: str | None,
    ) -> dict[str, object]:
        token = self._extract_bearer_token(authorization_header)
        body_segment, signature_segment = self._split_token(token)
        secret = self._secret_manager.get_secret(self._secret_name).encode("utf-8")
        expected_signature = _urlsafe_b64encode(
            _compute_hmac(secret, body_segment, hashlib.sha256)
        )
        if not hmac.compare_digest(signature_segment, expected_signature):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Credential signature mismatch",
            )

        payload_bytes = _urlsafe_b64decode(body_segment)
        try:
            payload: dict[str, object] = json.loads(payload_bytes)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Credential payload invalid",
            ) from exc

        now = self._clock()
        exp_value = payload.get("exp")
        if not isinstance(exp_value, int):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Credential missing expiry",
            )
        expires_at = datetime.fromtimestamp(exp_value, tz=timezone.utc)
        if now - self._leeway > expires_at:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Credential has expired",
            )

        subject = payload.get("sub")
        audience = payload.get("aud")
        scopes = payload.get("scopes", [])
        if not isinstance(subject, str) or not subject:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Credential missing subject",
            )
        if audience != expected_audience:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Credential audience mismatch",
            )
        if not isinstance(scopes, list):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Credential scopes malformed",
            )
        scope_set = set(scope for scope in scopes if isinstance(scope, str))
        missing_scopes = [scope for scope in required_scopes if scope not in scope_set]
        if missing_scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Credential missing required scope",
            )

        expected_thumbprint = payload.get("cert_thumbprint")
        if expected_thumbprint:
            if provided_thumbprint is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="mTLS client certificate not provided",
                )
            if expected_thumbprint != provided_thumbprint:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="mTLS client certificate mismatch",
                )

        payload["scopes"] = list(scope_set)
        payload["aud"] = audience
        payload["sub"] = subject
        return payload

    @staticmethod
    def _extract_bearer_token(header: str) -> str:
        if not header:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header missing",
            )
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unsupported authorization scheme",
            )
        return token.strip()

    @staticmethod
    def _split_token(token: str) -> tuple[str, str]:
        try:
            body_segment, signature_segment = token.split(".", 1)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Credential format invalid",
            ) from exc
        if not body_segment or not signature_segment:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Credential format invalid",
            )
        return body_segment, signature_segment


def _compute_hmac(secret: bytes, message_segment: str, digestmod: Callable[..., object]) -> bytes:
    """Compute a keyed HMAC over the message segment."""

    mac = hmac.new(secret, message_segment.encode("utf-8"), digestmod=digestmod)  # type: ignore[arg-type]
    return mac.digest()


def _urlsafe_b64encode(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")
    return encoded


def _urlsafe_b64decode(segment: str) -> bytes:
    padding = "=" * ((4 - len(segment) % 4) % 4)
    return base64.urlsafe_b64decode(segment + padding)


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
    authenticator: ShortLivedTokenVerifier,
    *,
    required_audience: str = "tradepulse.admin.kill-switch",
    required_scopes: Sequence[str] = ("kill-switch:engage",),
) -> APIRouter:
    """Create a router exposing secure administrative endpoints."""

    router = APIRouter(
        prefix="/admin",
        tags=["admin"],
        responses={401: {"description": "Unauthorized"}},
    )

    def get_risk_manager() -> RiskManagerFacade:
        return risk_manager

    def get_audit_logger() -> AuditLogger:
        return audit_logger

    request_context_dependency = authenticator.dependency(
        audience=required_audience,
        required_scopes=required_scopes,
    )

    @router.post(
        "/kill-switch",
        response_model=KillSwitchResponse,
        status_code=status.HTTP_200_OK,
        summary="Engage the global kill-switch",
    )
    async def engage_kill_switch(
        payload: KillSwitchRequest,
        request: Request,
        context: RequestContext = Depends(request_context_dependency),
        manager: RiskManagerFacade = Depends(get_risk_manager),
        logger: AuditLogger = Depends(get_audit_logger),
    ) -> KillSwitchResponse:
        """Engage the risk manager kill-switch and log the operation."""

        state: KillSwitchState = manager.engage_kill_switch(payload.reason)
        event_type = "kill_switch_reaffirmed" if state.already_engaged else "kill_switch_engaged"
        logger.log_event(
            event_type=event_type,
            actor=context.identity.subject,
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
