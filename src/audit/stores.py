"""Append-only persistence primitives for audit records."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .audit_logger import AuditRecord

__all__ = ["AuditRecordStore", "JsonLinesAuditStore"]


class AuditRecordStore(Protocol):
    """Protocol describing append-only persistence for :class:`AuditRecord`."""

    def append(self, record: "AuditRecord") -> None:
        """Persist *record* using an append-only strategy."""


class JsonLinesAuditStore:
    """Append audit records to a JSON Lines file with durability guarantees."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: "AuditRecord") -> None:
        payload = record.model_dump(mode="json")
        line = json.dumps(payload, sort_keys=True)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
