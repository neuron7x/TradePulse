"""FastAPI router implementing secure remote control operations."""

from __future__ import annotations

import hmac

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.audit.audit_logger import AuditLogger
from src.risk.risk_manager import KillSwitchState, RiskManagerFacade

__all__ = [
    "AdminIdentity",
    "KillSwitchRequest",
    "KillSwitchResponse",
    "TokenAuthenticator",
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
    ) -> AdminIdentity:
        """Authenticate the incoming request and return the administrator identity."""

        if not provided or not hmac.compare_digest(provided, self._token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid administrative token",
            )
        subject = subject_override or self._subject
        return AdminIdentity(subject=subject)


def _resolve_ip(request: Request) -> str:
    """Return the originating IP address for the request."""

    if request.client is None:
        return "unknown"
    return request.client.host


def create_remote_control_router(
    risk_manager: RiskManagerFacade,
    audit_logger: AuditLogger,
    authenticator: TokenAuthenticator,
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
        manager: RiskManagerFacade = Depends(get_risk_manager),
        logger: AuditLogger = Depends(get_audit_logger),
    ) -> KillSwitchResponse:
        """Engage the risk manager kill-switch and log the operation."""

        state: KillSwitchState = manager.engage_kill_switch(payload.reason)
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
            kill_switch_engaged=True,
            reason=state.reason,
            already_engaged=state.already_engaged,
        )

    return router
