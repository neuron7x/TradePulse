"""Structured audit logging utilities for administrative actions."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["AuditLogger", "AuditRecord", "AuditSink"]


def _ensure_utc(timestamp: datetime) -> datetime:
    """Return a timezone-aware timestamp in UTC."""

    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialize a payload into canonical JSON for signing."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda value: value.isoformat() if isinstance(value, datetime) else value,
    )


class AuditRecord(BaseModel):
    """Structured representation of an audit log entry."""

    event_type: str = Field(..., description="Machine-readable event category.")
    actor: str = Field(..., min_length=1, description="Actor triggering the event.")
    ip_address: str = Field(..., description="Source IP address of the request.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the event was recorded.",
    )
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional structured context for the event."
    )
    signature: str = Field(..., description="HMAC-SHA256 signature of the event payload.")

    model_config = ConfigDict(frozen=True)


class AuditSink(Protocol):
    """Callable sink used to persist audit records."""

    def __call__(self, record: AuditRecord) -> None:
        """Persist the provided audit record."""


class AuditLogger:
    """Generate signed audit records and forward them to configured sinks."""

    def __init__(
        self,
        secret: str,
        *,
        logger: logging.Logger | None = None,
        sink: AuditSink | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not secret:
            raise ValueError("secret must be provided for audit logging")
        self._key = secret.encode("utf-8")
        self._logger = logger or logging.getLogger("tradepulse.audit")
        self._sink = sink
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def _sign(self, payload: Mapping[str, Any]) -> str:
        """Return an HMAC signature for the payload."""

        message = _canonical_json(payload).encode("utf-8")
        return hmac.new(self._key, message, hashlib.sha256).hexdigest()

    def log_event(
        self,
        *,
        event_type: str,
        actor: str,
        ip_address: str,
        details: Mapping[str, Any] | None = None,
    ) -> AuditRecord:
        """Create and persist a signed audit record."""

        timestamp = _ensure_utc(self._clock())
        payload: dict[str, Any] = {
            "event_type": event_type,
            "actor": actor,
            "ip_address": ip_address,
            "timestamp": timestamp,
            "details": dict(details or {}),
        }
        signature = self._sign(payload)
        record = AuditRecord(**payload, signature=signature)
        self._logger.info("audit.event", extra={"audit": record.model_dump()})
        if self._sink is not None:
            self._sink(record)
        return record

    def verify(self, record: AuditRecord) -> bool:
        """Validate the signature of an existing record."""

        payload = {
            "event_type": record.event_type,
            "actor": record.actor,
            "ip_address": record.ip_address,
            "timestamp": _ensure_utc(record.timestamp),
            "details": dict(record.details),
        }
        expected = self._sign(payload)
        return hmac.compare_digest(expected, record.signature)
