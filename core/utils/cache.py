# SPDX-License-Identifier: MIT
"""Deterministic caching primitives for indicator computations.

The cache couples indicator parameters, input data fingerprints and the
application code version to guarantee correctness.  Fingerprints are
constructed from a stable hash of indicator configuration, the input data and
current code version so that any change invalidates previous entries.

The cache is designed around timeframe partitions and supports incremental
backfill.  When new market data extends beyond the last cached timestamp, a
backfill plan is produced so callers can process only the missing increments.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
from typing import Any, Generic, Mapping, MutableMapping, Sequence, TypeVar, cast

import numpy as np
import pandas as pd

T_co = TypeVar("T_co", covariant=True)


def _json_safe(value: Any) -> Any:
    """Return a JSON-serialisable representation of ``value``."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "name") and isinstance(getattr(value, "name"), str):
        # Enums expose a stable ``name`` attribute
        return cast(Any, value).name
    return repr(value)


def _read_version_file() -> str | None:
    try:
        return Path("VERSION").read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def resolve_code_version() -> str:
    """Best effort resolution of the current code version.

    Prefer the git commit hash when available, otherwise fall back to the
    project ``VERSION`` file or ``"unknown"``.
    """

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        version = _read_version_file()
        return version if version else "unknown"
    commit = result.stdout.strip()
    return commit or "unknown"


@dataclass(slots=True)
class IndicatorCacheKey:
    """Identifier for a cached indicator partition."""

    indicator: str
    timeframe: str | None = None


@dataclass(slots=True)
class CacheMetadata:
    """Metadata persisted next to every cached payload."""

    indicator: str
    timeframe: str | None
    fingerprint: str
    data_hash: str
    code_version: str
    params_hash: str
    latest_timestamp: str | None
    row_count: int
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "indicator": self.indicator,
            "timeframe": self.timeframe,
            "fingerprint": self.fingerprint,
            "data_hash": self.data_hash,
            "code_version": self.code_version,
            "params_hash": self.params_hash,
            "latest_timestamp": self.latest_timestamp,
            "row_count": self.row_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CacheMetadata":
        created = datetime.fromisoformat(str(raw["created_at"]))
        updated = datetime.fromisoformat(str(raw["updated_at"]))
        latest_str = raw.get("latest_timestamp")
        latest = str(latest_str) if latest_str else None
        return cls(
            indicator=str(raw["indicator"]),
            timeframe=cast(str | None, raw.get("timeframe")),
            fingerprint=str(raw["fingerprint"]),
            data_hash=str(raw["data_hash"]),
            code_version=str(raw["code_version"]),
            params_hash=str(raw["params_hash"]),
            latest_timestamp=latest,
            row_count=int(raw.get("row_count", 0)),
            created_at=created,
            updated_at=updated,
        )

    def latest_timestamp_pd(self) -> pd.Timestamp | None:
        if self.latest_timestamp is None:
            return None
        return pd.Timestamp(self.latest_timestamp)


@dataclass(slots=True)
class IndicatorCacheEntry(Generic[T_co]):
    """Container returned when a cache payload is successfully loaded."""

    metadata: CacheMetadata
    payload: T_co


@dataclass(slots=True)
class BackfillPlan:
    """Represents the required work to bring a cache entry up to date."""

    fingerprint: str
    cache_hit: bool
    needs_update: bool
    incremental: bool
    start_timestamp: pd.Timestamp | None


class IndicatorCache:
    """Durable cache for expensive indicator computations."""

    def __init__(self, root: Path | str, *, code_version: str | None = None) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._code_version = code_version or resolve_code_version()

    @property
    def code_version(self) -> str:
        return self._code_version

    def _entry_dir(self, key: IndicatorCacheKey, *, create: bool) -> Path:
        directory = self._root / key.indicator
        if key.timeframe is not None:
            directory = directory / key.timeframe
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _metadata_path(self, key: IndicatorCacheKey) -> Path:
        return self._entry_dir(key, create=False) / "metadata.json"

    def _payload_path(self, key: IndicatorCacheKey) -> Path:
        return self._entry_dir(key, create=False) / "payload.pkl"

    def load_entry(self, key: IndicatorCacheKey) -> IndicatorCacheEntry[Any] | None:
        meta_path = self._metadata_path(key)
        payload_path = self._payload_path(key)
        if not meta_path.exists() or not payload_path.exists():
            return None
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            metadata = CacheMetadata.from_dict(raw)
        except Exception:
            return None
        if metadata.code_version != self._code_version:
            return None
        try:
            with payload_path.open("rb") as fh:
                payload = pickle.load(fh)
        except Exception:
            return None
        return IndicatorCacheEntry(metadata=metadata, payload=payload)

    def _read_metadata(self, key: IndicatorCacheKey) -> CacheMetadata | None:
        meta_path = self._metadata_path(key)
        if not meta_path.exists():
            return None
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            metadata = CacheMetadata.from_dict(raw)
        except Exception:
            return None
        if metadata.code_version != self._code_version:
            return None
        return metadata

    def store_entry(
        self,
        key: IndicatorCacheKey,
        *,
        fingerprint: str,
        data_hash: str,
        params_hash: str,
        latest_timestamp: pd.Timestamp | None,
        row_count: int,
        payload: Any,
    ) -> None:
        directory = self._entry_dir(key, create=True)
        existing = self._read_metadata(key)
        now = datetime.now(timezone.utc)
        latest_iso = None
        if latest_timestamp is not None:
            latest_iso = pd.Timestamp(latest_timestamp).isoformat()
        metadata = CacheMetadata(
            indicator=key.indicator,
            timeframe=key.timeframe,
            fingerprint=fingerprint,
            data_hash=data_hash,
            code_version=self._code_version,
            params_hash=params_hash,
            latest_timestamp=latest_iso,
            row_count=int(row_count),
            created_at=existing.created_at if existing is not None else now,
            updated_at=now,
        )
        payload_path = directory / "payload.pkl"
        tmp_payload = directory / "payload.pkl.tmp"
        with tmp_payload.open("wb") as fh:
            pickle.dump(payload, fh)
        os.replace(tmp_payload, payload_path)

        meta_path = directory / "metadata.json"
        tmp_meta = directory / "metadata.json.tmp"
        with tmp_meta.open("w", encoding="utf-8") as fh:
            json.dump(metadata.to_dict(), fh, sort_keys=True)
        os.replace(tmp_meta, meta_path)

    @staticmethod
    def hash_params(params: Mapping[str, Any]) -> str:
        canonical: MutableMapping[str, Any] = {}
        for key, value in params.items():
            canonical[str(key)] = _json_safe(value)
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def hash_series(series: pd.Series) -> str:
        hashed = pd.util.hash_pandas_object(series, index=True)
        as_uint = hashed.to_numpy(dtype=np.uint64, copy=False)
        return hashlib.sha256(as_uint.tobytes()).hexdigest()

    @staticmethod
    def hash_dataframe(frame: pd.DataFrame) -> str:
        hashed = pd.util.hash_pandas_object(frame, index=True)
        as_uint = hashed.to_numpy(dtype=np.uint64, copy=False)
        return hashlib.sha256(as_uint.tobytes()).hexdigest()

    @staticmethod
    def hash_sequence(values: Sequence[Any]) -> str:
        canonical = [_json_safe(value) for value in values]
        encoded = json.dumps(canonical, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def make_fingerprint(
        *,
        indicator: str,
        params: Mapping[str, Any],
        data_hash: str,
        code_version: str,
        timeframe: str | None,
    ) -> tuple[str, str]:
        params_hash = IndicatorCache.hash_params(params)
        payload = "|".join(
            [
                indicator,
                timeframe or "",
                params_hash,
                data_hash,
                code_version,
            ]
        )
        fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return fingerprint, params_hash

    def plan_backfill(
        self,
        entry: IndicatorCacheEntry[Any] | None,
        *,
        fingerprint: str,
        latest_timestamp: pd.Timestamp | None,
    ) -> BackfillPlan:
        if entry is None:
            return BackfillPlan(
                fingerprint=fingerprint,
                cache_hit=False,
                needs_update=True,
                incremental=False,
                start_timestamp=None,
            )
        metadata = entry.metadata
        if metadata.fingerprint == fingerprint:
            return BackfillPlan(
                fingerprint=fingerprint,
                cache_hit=True,
                needs_update=False,
                incremental=False,
                start_timestamp=None,
            )
        previous_ts = metadata.latest_timestamp_pd()
        if latest_timestamp is None or previous_ts is None:
            return BackfillPlan(
                fingerprint=fingerprint,
                cache_hit=True,
                needs_update=True,
                incremental=False,
                start_timestamp=None,
            )
        if latest_timestamp <= previous_ts:
            # Data changed but not advanced in time – force full recompute.
            return BackfillPlan(
                fingerprint=fingerprint,
                cache_hit=True,
                needs_update=True,
                incremental=False,
                start_timestamp=None,
            )
        return BackfillPlan(
            fingerprint=fingerprint,
            cache_hit=True,
            needs_update=True,
            incremental=True,
            start_timestamp=previous_ts,
        )


__all__ = [
    "BackfillPlan",
    "CacheMetadata",
    "IndicatorCache",
    "IndicatorCacheEntry",
    "IndicatorCacheKey",
    "resolve_code_version",
]
