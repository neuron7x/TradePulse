# SPDX-License-Identifier: MIT
"""Indicator caching primitives with fingerprinting and incremental backfill.

This module provides a production-grade caching layer tailored for quantitative
indicators.  It supports deterministic cache keys (fingerprints) that blend the
indicator name, configuration parameters, input data hash and the running code
version.  Results are partitioned per timeframe so multi-resolution analytics
can reuse cached computations independently.

Beyond simple memoization, the cache records coverage metadata (time span,
record counts) and exposes an incremental backfill protocol.  Downstream
pipelines can persist the last processed timestamp for every timeframe and
append only the missing data on the next run—mirroring the workflow used in
large research/replay systems (QuantConnect, Backtrader, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import functools
import hashlib
import json
import os
import pickle
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping, MutableMapping, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd

from core.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - used only for type checking
    from .multiscale_kuramoto import TimeFrame


_logger = get_logger(__name__)


def _normalize_json(value: Any) -> Any:
    """Convert arbitrary Python objects into JSON-friendly structures."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):  # numpy scalar
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_json(item) for item in value]
    if hasattr(value, "name") and hasattr(value, "value"):
        # Enum support – record both name and value for readability
        return {"__enum__": value.__class__.__name__, "name": value.name, "value": value.value}
    return repr(value)


def make_fingerprint(
    indicator_name: str,
    params: Mapping[str, Any],
    data_hash: str,
    code_version: str,
) -> str:
    """Create a deterministic fingerprint for an indicator execution."""

    payload = {
        "indicator": indicator_name,
        "params": _normalize_json(params),
        "data_hash": data_hash,
        "code_version": code_version,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def hash_input_data(data: Any) -> str:
    """Hash arbitrary indicator input data."""

    if isinstance(data, pd.DataFrame):
        hashed = pd.util.hash_pandas_object(data, index=True).values
        return hashlib.sha256(hashed.tobytes()).hexdigest()
    if isinstance(data, pd.Series):
        hashed = pd.util.hash_pandas_object(data, index=True).values
        return hashlib.sha256(hashed.tobytes()).hexdigest()
    if isinstance(data, np.ndarray):
        arr = np.ascontiguousarray(data)
        return hashlib.sha256(arr.tobytes()).hexdigest()
    if isinstance(data, Mapping):
        normalized = _normalize_json(data)
        raw = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        normalized = [_normalize_json(item) for item in data]
        raw = json.dumps(normalized, sort_keys=False, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return hashlib.sha256(pickle.dumps(data)).hexdigest()


def _resolve_code_version() -> str:
    """Best-effort resolution of the current code version."""

    git_dir = Path(__file__).resolve().parents[2]
    try:
        head = (git_dir / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1]
            ref_path = git_dir / ".git" / ref
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()
        if head:
            return head
    except Exception:  # pragma: no cover - gitless environments
        pass

    version_file = git_dir / "VERSION"
    if version_file.exists():
        return version_file.read_text(encoding="utf-8").strip()

    return "0.0.0"


@dataclass(slots=True)
class CacheRecord:
    """Materialized cache entry."""

    value: Any
    metadata: Mapping[str, Any]
    fingerprint: str
    coverage_start: datetime | None
    coverage_end: datetime | None
    stored_at: datetime


@dataclass(slots=True)
class BackfillState:
    """Book-keeping for incremental backfill per timeframe."""

    timeframe: str
    last_timestamp: datetime | None
    fingerprint: str | None
    updated_at: datetime
    extras: Mapping[str, Any]


class FileSystemIndicatorCache:
    """Disk-backed cache that stores indicator outputs per timeframe."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        code_version: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.code_version = code_version or _resolve_code_version()
        _logger.debug(
            "indicator_cache_initialized",
            root=str(self.root),
            code_version=self.code_version,
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _timeframe_key(timeframe: TimeFrame | str | None) -> str:
        if timeframe is None:
            return "_global"
        if hasattr(timeframe, "name"):
            return str(getattr(timeframe, "name"))
        return str(timeframe)

    def _entry_dir(self, timeframe: TimeFrame | str | None, fingerprint: str) -> Path:
        timeframe_dir = self.root / self._timeframe_key(timeframe)
        return timeframe_dir / fingerprint

    # ---------------------------------------------------------------- serialization
    def _serialize(self, directory: Path, value: Any) -> tuple[str, str]:
        data_path = directory / "payload"

        # pandas structures
        if isinstance(value, pd.DataFrame):
            try:
                file_path = data_path.with_suffix(".parquet")
                value.to_parquet(file_path)
                return file_path.name, "parquet"
            except Exception:  # pragma: no cover - optional dependency
                file_path = data_path.with_suffix(".pkl")
                value.to_pickle(file_path)
                return file_path.name, "pickle"
        if isinstance(value, pd.Series):
            try:
                file_path = data_path.with_suffix(".parquet")
                value.to_frame(name=value.name).to_parquet(file_path)
                return file_path.name, "parquet"
            except Exception:  # pragma: no cover
                file_path = data_path.with_suffix(".pkl")
                value.to_pickle(file_path)
                return file_path.name, "pickle"
        if isinstance(value, np.ndarray):
            file_path = data_path.with_suffix(".npy")
            np.save(file_path, value)
            return file_path.name, "numpy"

        file_path = data_path.with_suffix(".pkl")
        with file_path.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return file_path.name, "pickle"

    @staticmethod
    def _deserialize(path: Path, fmt: str) -> Any:
        if fmt == "parquet":
            return pd.read_parquet(path)
        if fmt == "numpy":
            return np.load(path, allow_pickle=False)
        if fmt == "pickle":
            with path.open("rb") as handle:
                return pickle.load(handle)
        raise ValueError(f"Unsupported cache format '{fmt}'")

    # ---------------------------------------------------------------- fingerprint
    def fingerprint(
        self,
        indicator_name: str,
        params: Mapping[str, Any],
        data_hash: str,
        *,
        code_version: str | None = None,
    ) -> str:
        return make_fingerprint(
            indicator_name,
            params,
            data_hash,
            code_version or self.code_version,
        )

    # ------------------------------------------------------------------- mutation
    def store(
        self,
        *,
        indicator_name: str,
        params: Mapping[str, Any],
        data_hash: str,
        value: Any,
        timeframe: TimeFrame | str | None = None,
        coverage_start: datetime | str | None = None,
        coverage_end: datetime | str | None = None,
        metadata: Mapping[str, Any] | None = None,
        code_version: str | None = None,
    ) -> str:
        """Persist a cache entry and return its fingerprint."""

        fingerprint = self.fingerprint(
            indicator_name,
            params,
            data_hash,
            code_version=code_version,
        )
        directory = self._entry_dir(timeframe, fingerprint)
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

        data_file, data_format = self._serialize(directory, value)

        payload: MutableMapping[str, Any] = {
            "fingerprint": fingerprint,
            "indicator": indicator_name,
            "params": _normalize_json(params),
            "data_hash": data_hash,
            "code_version": code_version or self.code_version,
            "timeframe": self._timeframe_key(timeframe),
            "data_file": data_file,
            "data_format": data_format,
            "stored_at": datetime.now(UTC).isoformat(),
            "coverage_start": (
                coverage_start.isoformat() if isinstance(coverage_start, datetime) else coverage_start
            ),
            "coverage_end": (
                coverage_end.isoformat() if isinstance(coverage_end, datetime) else coverage_end
            ),
            "metadata": _normalize_json(metadata or {}),
        }

        with (directory / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)

        _logger.debug(
            "indicator_cache_store",
            indicator=indicator_name,
            timeframe=payload["timeframe"],
            fingerprint=fingerprint,
            coverage_end=payload.get("coverage_end"),
        )
        return fingerprint

    # -------------------------------------------------------------------- retrieval
    def load(
        self,
        *,
        indicator_name: str,
        params: Mapping[str, Any],
        data_hash: str,
        timeframe: TimeFrame | str | None = None,
        code_version: str | None = None,
    ) -> CacheRecord | None:
        fingerprint = self.fingerprint(
            indicator_name,
            params,
            data_hash,
            code_version=code_version,
        )
        directory = self._entry_dir(timeframe, fingerprint)
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            return None

        try:
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

        data_path = directory / meta["data_file"]
        if not data_path.exists():
            return None

        value = self._deserialize(data_path, meta["data_format"])

        coverage_start = (
            datetime.fromisoformat(meta["coverage_start"]) if meta.get("coverage_start") else None
        )
        coverage_end = (
            datetime.fromisoformat(meta["coverage_end"]) if meta.get("coverage_end") else None
        )
        stored_at = datetime.fromisoformat(meta["stored_at"])

        return CacheRecord(
            value=value,
            metadata=meta.get("metadata", {}),
            fingerprint=fingerprint,
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            stored_at=stored_at,
        )

    # ---------------------------------------------------------------- backfill API
    def _backfill_path(self, timeframe: TimeFrame | str) -> Path:
        return self.root / self._timeframe_key(timeframe) / "backfill.json"

    def get_backfill_state(self, timeframe: TimeFrame | str) -> BackfillState | None:
        path = self._backfill_path(timeframe)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

        last_ts = payload.get("last_timestamp")
        return BackfillState(
            timeframe=self._timeframe_key(timeframe),
            last_timestamp=datetime.fromisoformat(last_ts) if last_ts else None,
            fingerprint=payload.get("fingerprint"),
            updated_at=datetime.fromisoformat(payload["updated_at"]),
            extras=payload.get("extras", {}),
        )

    def update_backfill_state(
        self,
        timeframe: TimeFrame | str,
        *,
        last_timestamp: datetime | str,
        fingerprint: str | None,
        extras: Mapping[str, Any] | None = None,
    ) -> None:
        timestamp = last_timestamp.isoformat() if isinstance(last_timestamp, datetime) else str(last_timestamp)
        payload = {
            "timeframe": self._timeframe_key(timeframe),
            "last_timestamp": timestamp,
            "fingerprint": fingerprint,
            "updated_at": datetime.now(UTC).isoformat(),
            "extras": _normalize_json(extras or {}),
        }
        path = self._backfill_path(timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

        _logger.debug(
            "indicator_cache_backfill_update",
            timeframe=payload["timeframe"],
            last_timestamp=payload["last_timestamp"],
        )


def cache_indicator(
    cache: FileSystemIndicatorCache,
    *,
    indicator_name: str | None = None,
    timeframe: TimeFrame | str | None = None,
    params_fn: Callable[..., Mapping[str, Any]] | None = None,
    data_fn: Callable[..., Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that transparently caches indicator outputs."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        resolved_name = indicator_name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            params = params_fn(*args, **kwargs) if params_fn else kwargs
            data = data_fn(*args, **kwargs) if data_fn else (args[0] if args else kwargs.get("data"))
            if data is None:
                raise ValueError("cache_indicator requires data argument to compute fingerprint")

            data_hash = hash_input_data(data)
            record = cache.load(
                indicator_name=resolved_name,
                params=params,
                data_hash=data_hash,
                timeframe=timeframe,
            )
            if record is not None:
                _logger.debug(
                    "indicator_cache_hit",
                    indicator=resolved_name,
                    timeframe=cache._timeframe_key(timeframe),
                )
                return record.value

            value = fn(*args, **kwargs)
            cache.store(
                indicator_name=resolved_name,
                params=params,
                data_hash=data_hash,
                value=value,
                timeframe=timeframe,
            )
            _logger.debug(
                "indicator_cache_miss",
                indicator=resolved_name,
                timeframe=cache._timeframe_key(timeframe),
            )
            return value

        return wrapper

    return decorator


__all__ = [
    "BackfillState",
    "CacheRecord",
    "FileSystemIndicatorCache",
    "cache_indicator",
    "hash_input_data",
    "make_fingerprint",
]

