"""Structured audit logging utilities for administrative actions."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, PriorityQueue
from typing import Any, Callable, Mapping, Protocol
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .stores import AuditRecordStore

__all__ = [
    "AuditLogger",
    "AuditRecord",
    "AuditSink",
    "HttpAuditSink",
    "SiemAuditSink",
]


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

    model_config = ConfigDict(frozen=True)


class AuditSink(Protocol):
    """Callable sink used to persist audit records."""

    def __call__(self, record: AuditRecord) -> None:
        """Persist the provided audit record."""


class AuditLogger:
    """Generate signed audit records, sign them, and forward to configured sinks."""

    def __init__(
        self,
        secret: str,
        *,
        logger: logging.Logger | None = None,
        sink: AuditSink | None = None,
        store: AuditRecordStore | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not secret:
            raise ValueError("secret must be provided for audit logging")
        self._key = secret.encode("utf-8")
        self._logger = logger or logging.getLogger("tradepulse.audit")
        self._sink = sink
        self._store = store
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
        self._log_record(record)
        self._persist_record(record)
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

    def _log_record(self, record: AuditRecord) -> None:
        audit_payload = record.model_dump()
        audit_payload["details"] = _redact_sensitive_payload(audit_payload["details"])
        self._logger.info("audit.event", extra={"audit": audit_payload})

    def _persist_record(self, record: AuditRecord) -> None:
        if self._store is None:
            return
        try:
            self._store.append(record)
        except Exception as exc:  # pragma: no cover - failure is logged for SRE triage
            self._logger.error(
                "Failed to persist signed audit record",
                exc_info=exc,
                extra={
                    "audit": {
                        "event_type": record.event_type,
                        "signature": record.signature,
                    }
                },
            )


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


class SiemAuditSink:
    """Durably forward audit records to a SIEM endpoint with retries."""

    def __init__(
        self,
        endpoint: str,
        spool_dir: Path,
        *,
        http_client: httpx.Client | None = None,
        timeout: float = 5.0,
        max_retries: int = 5,
        base_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 60.0,
        dead_letter_dir: Path | None = None,
        logger: logging.Logger | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint must be provided for the SIEM sink")
        self._endpoint = endpoint
        self._spool_dir = spool_dir
        self._spool_dir.mkdir(parents=True, exist_ok=True)
        self._dead_letter_dir = dead_letter_dir or (self._spool_dir / "dead-letter")
        self._dead_letter_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logger or logging.getLogger("tradepulse.audit.siem_sink")
        self._timeout = timeout
        self._max_retries = max(0, max_retries)
        self._base_backoff = max(0.0, base_backoff_seconds)
        self._max_backoff = max_backoff_seconds
        self._sleep = sleep or time.sleep
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client()
        self._queue: PriorityQueue[tuple[float, Path | None]] = PriorityQueue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._load_existing_spool()
        self._worker = threading.Thread(target=self._run, name="siem-sink", daemon=True)
        self._worker.start()

    def __call__(self, record: AuditRecord) -> None:
        envelope = {
            "record": record.model_dump(mode="json"),
            "attempts": 0,
        }
        path = self._write_envelope(envelope)
        self._schedule(path, ready_at=time.monotonic())

    def close(self) -> None:
        self._stop.set()
        self._queue.put((time.monotonic(), None))
        self._worker.join()
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "SiemAuditSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _load_existing_spool(self) -> None:
        for path in sorted(self._spool_dir.glob("*.json")):
            self._schedule(path, ready_at=time.monotonic())

    def _schedule(self, path: Path, *, ready_at: float) -> None:
        self._queue.put((ready_at, path))

    def _write_envelope(self, envelope: Mapping[str, Any]) -> Path:
        identifier = uuid4().hex
        tmp_path = self._spool_dir / f".{identifier}.json.tmp"
        final_path = self._spool_dir / f"{identifier}.json"
        payload = json.dumps(envelope, sort_keys=True)
        with self._lock:
            with tmp_path.open("w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, final_path)
        return final_path

    def _run(self) -> None:
        while True:
            try:
                ready_at, path = self._queue.get(timeout=0.5)
            except Empty:
                if self._stop.is_set():
                    continue
                continue
            if path is None:
                break
            delay = max(0.0, ready_at - time.monotonic())
            if delay:
                self._sleep(delay)
            if self._stop.is_set():
                break
            try:
                envelope = self._read_envelope(path)
            except Exception as exc:  # pragma: no cover - defensive logging path
                self._logger.error(
                    "Failed to load persisted audit envelope", exc_info=exc, extra={"envelope_path": str(path)}
                )
                self._move_to_dead_letter(path, reason="invalid-envelope")
                continue
            record = envelope.get("record")
            attempts = int(envelope.get("attempts", 0))
            try:
                model = AuditRecord.model_validate(record)
                self._send(model)
            except Exception as exc:  # pragma: no cover - retry path exercises
                self._handle_failure(path, envelope, attempts, exc)
            else:
                self._acknowledge(path)

    def _read_envelope(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _send(self, record: AuditRecord) -> None:
        payload = record.model_dump(mode="json")
        response: httpx.Response | None = None
        try:
            response = self._client.post(self._endpoint, json=payload, timeout=self._timeout)
            response.raise_for_status()
        except Exception as exc:
            status_code = None
            if isinstance(exc, httpx.HTTPStatusError):
                status_code = exc.response.status_code
            elif response is not None:
                status_code = response.status_code
            raise RuntimeError(
                f"SIEM delivery failed (status={status_code})"
            ) from exc

    def _acknowledge(self, path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:  # pragma: no cover - already removed
            return

    def _handle_failure(
        self,
        path: Path,
        envelope: Mapping[str, Any],
        attempts: int,
        exc: Exception,
    ) -> None:
        attempts += 1
        updated = dict(envelope)
        updated["attempts"] = attempts
        self._logger.warning(
            "SIEM delivery attempt failed",
            exc_info=exc,
            extra={
                "siem_sink": {
                    "path": str(path),
                    "attempt": attempts,
                }
            },
        )
        if attempts > self._max_retries:
            self._move_to_dead_letter(path, reason="max-retries", envelope=updated)
            return
        retry_delay = self._compute_backoff(attempts)
        self._rewrite_envelope(path, updated)
        self._schedule(path, ready_at=time.monotonic() + retry_delay)

    def _rewrite_envelope(self, path: Path, envelope: Mapping[str, Any]) -> None:
        payload = json.dumps(envelope, sort_keys=True)
        with self._lock:
            with path.open("w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())

    def _move_to_dead_letter(
        self,
        path: Path,
        *,
        reason: str,
        envelope: Mapping[str, Any] | None = None,
    ) -> None:
        target = self._dead_letter_dir / path.name
        if envelope is not None:
            self._rewrite_envelope(path, envelope)
        try:
            os.replace(path, target)
        except FileNotFoundError:  # pragma: no cover - already moved
            return
        self._logger.error(
            "Moved audit record to SIEM dead-letter store",
            extra={"siem_sink": {"path": str(target), "reason": reason}},
        )

    def _compute_backoff(self, attempts: int) -> float:
        if self._base_backoff == 0:
            return 0.0
        exponent = max(0, attempts - 1)
        delay = self._base_backoff * (2**exponent)
        return min(delay, self._max_backoff)
