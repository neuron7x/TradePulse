"""Structured audit logging utilities for administrative actions."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from threading import RLock
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field

from src.security.secret_manager import SecretManager, SecretValue

__all__ = ["AuditLogger", "AuditRecord", "AuditSink", "HttpAuditSink"]


_REDACTED_VALUE = "[REDACTED]"
_SENSITIVE_KEYWORDS = ("token", "secret", "password", "key", "credential")


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


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(keyword in lowered for keyword in _SENSITIVE_KEYWORDS)


def _redact_sensitive_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of *payload* with sensitive keys redacted for logging."""

    def _redact(value: Any, *, parent_key: str | None = None) -> Any:
        if isinstance(value, Mapping):
            return {inner_key: _redact(inner_value, parent_key=inner_key) for inner_key, inner_value in value.items()}
        if isinstance(value, list):
            return [_redact(item, parent_key=parent_key) for item in value]
        if parent_key is not None and _is_sensitive_key(parent_key):
            return _REDACTED_VALUE
        return value

    return _redact(dict(payload))  # type: ignore[arg-type]


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
    key_version: str = Field(..., description="Identifier for the signing secret version.")

    model_config = ConfigDict(frozen=True)


class AuditSink(Protocol):
    """Callable sink used to persist audit records."""

    def __call__(self, record: AuditRecord) -> None:
        """Persist the provided audit record."""


class AuditLogger:
    """Generate signed audit records, sign them, and forward to configured sinks."""

    def __init__(
        self,
        secret_manager: SecretManager,
        *,
        secret_id: str,
        logger: logging.Logger | None = None,
        sink: AuditSink | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not secret_id:
            raise ValueError("secret_id must be provided for audit logging")
        self._secret_manager = secret_manager
        self._secret_id = secret_id
        self._logger = logger or logging.getLogger("tradepulse.audit")
        self._sink = sink
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = RLock()
        self._keys_by_version: dict[str, bytes] = {}
        self._current_version: str | None = None
        self._load_secret()
        self._secret_manager.subscribe(secret_id, self._on_secret_rotation)

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
        key, version = self._current_key()
        signature = hmac.new(key, _canonical_json(payload).encode("utf-8"), hashlib.sha256).hexdigest()
        record = AuditRecord(**payload, signature=signature, key_version=version)
        self._log_record(record)
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
        key = self._key_for_version(record.key_version)
        if key is None:
            return False
        message = _canonical_json(payload).encode("utf-8")
        expected = hmac.new(key, message, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, record.signature)

    def _log_record(self, record: AuditRecord) -> None:
        audit_payload = record.model_dump()
        audit_payload["details"] = _redact_sensitive_payload(audit_payload["details"])
        self._logger.info("audit.event", extra={"audit": audit_payload})

    def _current_key(self) -> tuple[bytes, str]:
        with self._lock:
            version = self._current_version
            key = self._keys_by_version.get(version or "")
        if version is None or key is None:
            secret = self._load_secret()
            with self._lock:
                version = self._current_version
                key = self._keys_by_version[version]
        return key, version  # type: ignore[arg-type]

    def _key_for_version(self, version: str) -> bytes | None:
        with self._lock:
            key = self._keys_by_version.get(version)
        return key

    def _load_secret(self) -> SecretValue:
        secret = self._secret_manager.get_secret(self._secret_id)
        self._store_secret(secret)
        return secret

    def _store_secret(self, secret: SecretValue) -> None:
        key_bytes = secret.value.encode("utf-8")
        with self._lock:
            self._keys_by_version[secret.version] = key_bytes
            self._current_version = secret.version
        self._logger.info(
            "audit.secret.loaded",
            extra={
                "audit_secret": {
                    "id": self._secret_id,
                    "version": secret.version,
                    "expires_at": secret.expires_at.isoformat(),
                }
            },
        )

    def _on_secret_rotation(self, secret: SecretValue) -> None:
        self._store_secret(secret)


class HttpAuditSink:
    """Forward audit records to an external HTTP endpoint."""

    def __init__(
        self,
        endpoint: str,
        *,
        http_client: httpx.Client | None = None,
        timeout: float = 5.0,
        logger: logging.Logger | None = None,
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint must be provided for the audit sink")
        self._endpoint = endpoint
        self._client = http_client
        self._timeout = timeout
        self._logger = logger or logging.getLogger("tradepulse.audit.http_sink")

    def __call__(self, record: AuditRecord) -> None:
        payload = record.model_dump(mode="json")
        response: httpx.Response | None = None
        try:
            if self._client is not None:
                response = self._client.post(
                    self._endpoint,
                    json=payload,
                    timeout=self._timeout,
                )
            else:
                response = httpx.post(
                    self._endpoint,
                    json=payload,
                    timeout=self._timeout,
                )
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - logging side effect
            status_code = None
            if isinstance(exc, httpx.HTTPStatusError):
                status_code = exc.response.status_code
            elif response is not None:
                status_code = response.status_code
            self._logger.error(
                "Failed to forward audit record",
                exc_info=exc,
                extra={
                    "audit_sink": {
                        "endpoint": self._endpoint,
                        "status_code": status_code,
                    }
                },
            )
