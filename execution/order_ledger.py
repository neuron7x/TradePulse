"""Append-only event ledger for order state reconstruction and auditing."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping

__all__ = ["OrderLedgerEvent", "OrderLedger"]


def _coerce(value: Any) -> Any:
    """Convert ``value`` into a JSON-serialisable representation."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Mapping):
        return {str(key): _coerce(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_coerce(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _canonical_dumps(payload: Mapping[str, Any]) -> str:
    """Return a canonical JSON representation with stable key ordering."""

    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


@dataclass(frozen=True, slots=True)
class OrderLedgerEvent:
    """Single append-only entry within the order ledger."""

    sequence: int
    event: str
    timestamp: str
    order_id: str | None
    correlation_id: str | None
    metadata: Mapping[str, Any]
    order_snapshot: MutableMapping[str, Any] | None
    state_snapshot: MutableMapping[str, Any] | None
    state_hash: str | None
    previous_digest: str | None
    digest: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the event."""

        return {
            "sequence": self.sequence,
            "event": self.event,
            "timestamp": self.timestamp,
            "order_id": self.order_id,
            "correlation_id": self.correlation_id,
            "metadata": dict(self.metadata),
            "order_snapshot": self.order_snapshot,
            "state_snapshot": self.state_snapshot,
            "state_hash": self.state_hash,
            "previous_digest": self.previous_digest,
            "digest": self.digest,
        }


class OrderLedger:
    """Append-only ledger capturing every order mutation and state snapshot."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._next_sequence, self._tail_digest = self._load_tail()

    @property
    def path(self) -> Path:
        """Return the file backing the ledger."""

        return self._path

    def append(
        self,
        event: str,
        *,
        order: Mapping[str, Any] | None = None,
        correlation_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        state_snapshot: Mapping[str, Any] | None = None,
    ) -> OrderLedgerEvent:
        """Append a new entry to the ledger and return the structured event."""

        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock:
            sequence = self._next_sequence
            previous_digest = self._tail_digest
            payload: dict[str, Any] = {
                "sequence": sequence,
                "event": str(event),
                "timestamp": timestamp,
                "order_id": None,
                "correlation_id": correlation_id,
                "metadata": _coerce(metadata or {}),
                "order_snapshot": None,
                "state_snapshot": None,
                "state_hash": None,
                "previous_digest": previous_digest,
            }
            if order is not None:
                order_snapshot = _coerce(order)
                payload["order_snapshot"] = order_snapshot
                payload["order_id"] = _coerce(order.get("order_id"))
            if state_snapshot is not None:
                coerced_state = _coerce(state_snapshot)
                payload["state_snapshot"] = coerced_state
                payload["state_hash"] = sha256(
                    _canonical_dumps(coerced_state).encode("utf-8")
                ).hexdigest()

            digest_source = dict(payload)
            digest = sha256(_canonical_dumps(digest_source).encode("utf-8")).hexdigest()
            payload["digest"] = digest

            record = json.dumps(payload, sort_keys=True, ensure_ascii=False)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(record + "\n")
            event_obj = OrderLedgerEvent(
                sequence=sequence,
                event=str(event),
                timestamp=timestamp,
                order_id=payload["order_id"],
                correlation_id=correlation_id,
                metadata=payload["metadata"],
                order_snapshot=payload["order_snapshot"],
                state_snapshot=payload["state_snapshot"],
                state_hash=payload["state_hash"],
                previous_digest=previous_digest,
                digest=digest,
            )
            self._next_sequence = sequence + 1
            self._tail_digest = digest
        return event_obj

    def replay(self, *, verify: bool = True) -> Iterator[OrderLedgerEvent]:
        """Iterate through recorded events in chronological order."""

        if not self._path.exists():
            return
        previous_digest: str | None = None
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                digest = str(payload["digest"])
                if verify:
                    content = dict(payload)
                    del content["digest"]
                    computed = sha256(
                        _canonical_dumps(content).encode("utf-8")
                    ).hexdigest()
                    if computed != digest:
                        raise ValueError(
                            "Ledger integrity check failed: digest mismatch"
                        )
                    if content.get("previous_digest") != previous_digest:
                        raise ValueError(
                            "Ledger integrity check failed: broken digest chain"
                        )
                event = OrderLedgerEvent(
                    sequence=int(payload["sequence"]),
                    event=str(payload["event"]),
                    timestamp=str(payload["timestamp"]),
                    order_id=payload.get("order_id"),
                    correlation_id=payload.get("correlation_id"),
                    metadata=payload.get("metadata", {}),
                    order_snapshot=payload.get("order_snapshot"),
                    state_snapshot=payload.get("state_snapshot"),
                    state_hash=payload.get("state_hash"),
                    previous_digest=payload.get("previous_digest"),
                    digest=digest,
                )
                previous_digest = digest
                yield event

    def latest_event(self, *, verify: bool = True) -> OrderLedgerEvent | None:
        """Return the last event recorded in the ledger."""

        last: OrderLedgerEvent | None = None
        for event in self.replay(verify=verify):
            last = event
        return last

    def latest_state(self, *, verify: bool = True) -> MutableMapping[str, Any] | None:
        """Return the most recent state snapshot if available."""

        event = self.latest_event(verify=verify)
        if event is None:
            return None
        snapshot = event.state_snapshot
        if snapshot is None:
            return None
        return snapshot

    def verify(self) -> None:
        """Validate ledger integrity by replaying the digest chain."""

        for _ in self.replay(verify=True):
            continue

    def _load_tail(self) -> tuple[int, str | None]:
        if not self._path.exists():
            return 1, None
        last_line = ""
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last_line = line
        if not last_line:
            return 1, None
        payload = json.loads(last_line)
        sequence = int(payload.get("sequence", 0)) + 1
        digest = str(payload.get("digest")) if payload.get("digest") else None
        return max(sequence, 1), digest
