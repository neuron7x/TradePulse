"""Online feature store helpers with integrity guards and retention policies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import hmac
import json
import sqlite3
from pathlib import Path
from threading import RLock
from typing import Callable, Dict, Iterable, Literal, Mapping, Protocol

import pandas as pd

from core.utils.dataframe_io import (
    purge_dataframe_artifacts,
    read_dataframe,
    write_dataframe,
)


class FeatureStoreIntegrityError(RuntimeError):
    """Raised when integrity invariants fail for the online feature store."""


@dataclass(frozen=True)
class IntegritySnapshot:
    """Compact representation of a dataset used for integrity comparisons."""

    row_count: int
    data_hash: str


@dataclass(frozen=True)
class IntegrityReport:
    """Integrity comparison between the offline payload and persisted store."""

    feature_view: str
    offline_rows: int
    online_rows: int
    row_count_diff: int
    offline_hash: str
    online_hash: str
    hash_differs: bool

    def ensure_valid(self) -> None:
        """Raise :class:`FeatureStoreIntegrityError` when invariants are violated."""

        if self.row_count_diff != 0:
            raise FeatureStoreIntegrityError(
                f"Row count mismatch for {self.feature_view!r}: "
                f"offline={self.offline_rows}, online={self.online_rows}"
            )
        if self.hash_differs:
            raise FeatureStoreIntegrityError(
                f"Hash mismatch for {self.feature_view!r}: "
                f"offline={self.offline_hash}, online={self.online_hash}"
            )


@dataclass(frozen=True)
class RetentionPolicy:
    """Retention settings applied after each write."""

    ttl: timedelta | pd.Timedelta | None = None
    max_rows: int | None = None
    max_versions: int | None = None

    def __post_init__(self) -> None:
        if self.ttl is not None and pd.to_timedelta(self.ttl) <= pd.Timedelta(0):
            raise ValueError("ttl must be positive when provided")
        if self.max_rows is not None and self.max_rows <= 0:
            raise ValueError("max_rows must be positive when provided")
        if self.max_versions is not None and self.max_versions <= 0:
            raise ValueError("max_versions must be positive when provided")

    def apply(
        self,
        frame: pd.DataFrame,
        *,
        timestamp_column: str,
        now: pd.Timestamp | datetime | None = None,
    ) -> pd.DataFrame:
        """Return ``frame`` after enforcing the retention constraints."""

        result = frame.copy(deep=True)
        if self.ttl is not None and not result.empty:
            if timestamp_column not in result.columns:
                raise ValueError(
                    "Retention policy requires timestamp column "
                    f"{timestamp_column!r} to be present"
                )

            if now is None:
                reference = pd.Timestamp.now(tz=UTC)
            else:
                reference = pd.Timestamp(now)
                if reference.tzinfo is None:
                    reference = reference.tz_localize(UTC)
                else:
                    reference = reference.tz_convert(UTC)

            cutoff = reference - pd.to_timedelta(self.ttl)
            timestamps = pd.to_datetime(result[timestamp_column], utc=True, errors="coerce")
            mask = timestamps >= cutoff
            result = result.loc[mask]
        if self.max_versions is not None and not result.empty:
            required = {"entity_id", timestamp_column}
            missing = required - set(result.columns)
            if missing:
                joined = ", ".join(sorted(missing))
                raise KeyError(
                    "Retention policy with max_versions requires columns: "
                    f"{joined}"
                )
            ordered = result.sort_values(
                by=["entity_id", timestamp_column],
                kind="mergesort",
            )
            limited = ordered.groupby("entity_id", as_index=False).tail(self.max_versions)
            result = limited.sort_values(
                by=["entity_id", timestamp_column],
                kind="mergesort",
            )
        if self.max_rows is not None and result.shape[0] > self.max_rows:
            result = result.iloc[-self.max_rows :]
        return result.reset_index(drop=True)


class FeatureStoreBackend(Protocol):
    """Minimal backend protocol used by :class:`OnlineFeatureStore`."""

    def load(self, feature_view: str) -> pd.DataFrame:
        ...

    def write(
        self,
        feature_view: str,
        frame: pd.DataFrame,
        *,
        mode: Literal["append", "overwrite"],
    ) -> None:
        ...

    def purge(self, feature_view: str) -> None:
        ...


class FilesystemFeatureStoreBackend:
    """Persist feature views as parquet/json payloads on disk."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, feature_view: str) -> Path:
        safe_name = feature_view.replace("/", "__").replace(".", "__")
        return self._root / safe_name

    def load(self, feature_view: str) -> pd.DataFrame:
        path = self._resolve_path(feature_view)
        return read_dataframe(path, allow_json_fallback=True)

    def write(
        self,
        feature_view: str,
        frame: pd.DataFrame,
        *,
        mode: Literal["append", "overwrite"],
    ) -> None:
        # ``mode`` is ignored because the caller passes the full dataset.
        path = self._resolve_path(feature_view)
        prepared = frame.reset_index(drop=True)
        write_dataframe(prepared, path, index=False, allow_json_fallback=True)

    def purge(self, feature_view: str) -> None:
        path = self._resolve_path(feature_view)
        purge_dataframe_artifacts(path)


class SQLiteFeatureStoreBackend:
    """SQLite-backed feature storage with transparent schema management."""

    def __init__(self, database_path: Path) -> None:
        self._path = Path(database_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _table_name(self, feature_view: str) -> str:
        safe = feature_view.replace("/", "_").replace(".", "_")
        return f"fv_{safe}"

    def load(self, feature_view: str) -> pd.DataFrame:
        table = self._table_name(feature_view)
        with sqlite3.connect(self._path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            )
            if cursor.fetchone() is None:
                return pd.DataFrame()
            frame = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        for column in frame.columns:
            try:
                converted = pd.to_datetime(frame[column], utc=True)
            except (TypeError, ValueError):
                continue
            if converted.notna().any():
                frame[column] = converted
        return frame

    def write(
        self,
        feature_view: str,
        frame: pd.DataFrame,
        *,
        mode: Literal["append", "overwrite"],
    ) -> None:
        table = self._table_name(feature_view)
        prepared = frame.copy()

        def _serialize(value: object) -> object:
            if isinstance(value, (pd.Timestamp, datetime)):
                ts = pd.Timestamp(value)
                if ts.tzinfo is None:
                    ts = ts.tz_localize(UTC)
                else:
                    ts = ts.tz_convert(UTC)
                return ts.isoformat().replace("+00:00", "Z")
            return value

        for column in prepared.columns:
            prepared[column] = prepared[column].map(_serialize)
        with sqlite3.connect(self._path) as conn:
            if mode == "overwrite":
                conn.execute(f"DROP TABLE IF EXISTS {table}")
            prepared.to_sql(table, conn, if_exists="append", index=False)

    def purge(self, feature_view: str) -> None:
        table = self._table_name(feature_view)
        with sqlite3.connect(self._path) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {table}")


class _RedisLikeClient(Protocol):
    """Subset of redis-py used by :class:`RedisFeatureStoreBackend`."""

    def get(self, key: str) -> bytes | None:
        ...

    def set(self, key: str, value: bytes) -> None:
        ...

    def delete(self, key: str) -> None:
        ...


class RedisFeatureStoreBackend:
    """Redis-backed feature storage using JSON payloads."""

    def __init__(self, client: _RedisLikeClient, namespace: str = "feature_store") -> None:
        self._client = client
        self._namespace = namespace

    def _key(self, feature_view: str) -> str:
        return f"{self._namespace}:{feature_view}"

    def load(self, feature_view: str) -> pd.DataFrame:
        raw = self._client.get(self._key(feature_view))
        if not raw:
            return pd.DataFrame()
        payload = json.loads(raw.decode("utf-8"))
        frame = pd.DataFrame(payload)
        for column in frame.columns:
            try:
                converted = pd.to_datetime(frame[column], utc=True)
            except (TypeError, ValueError):
                continue
            if converted.notna().any():
                frame[column] = converted
        return frame

    def write(
        self,
        feature_view: str,
        frame: pd.DataFrame,
        *,
        mode: Literal["append", "overwrite"],
    ) -> None:
        prepared = frame.reset_index(drop=True)

        def _serialize(value: object) -> object:
            if isinstance(value, (pd.Timestamp, datetime)):
                ts = pd.Timestamp(value)
                if ts.tzinfo is None:
                    ts = ts.tz_localize(UTC)
                else:
                    ts = ts.tz_convert(UTC)
                return ts.isoformat().replace("+00:00", "Z")
            return value

        converted = prepared.copy()
        for column in converted.columns:
            converted[column] = converted[column].map(_serialize)
        payload = converted.to_dict(orient="list")
        self._client.set(self._key(feature_view), json.dumps(payload).encode("utf-8"))

    def purge(self, feature_view: str) -> None:
        self._client.delete(self._key(feature_view))


@dataclass(frozen=True)
class OfflineSourceConfig:
    """Configuration describing the offline source of truth for a feature view."""

    format: Literal["delta", "iceberg"]
    path: Path


class OfflineSourceRegistry:
    """Registry that maps feature views to offline Delta/Iceberg datasets."""

    def __init__(
        self,
        *,
        loaders: Mapping[str, Callable[[Path], pd.DataFrame]] | None = None,
    ) -> None:
        default_loaders: Dict[str, Callable[[Path], pd.DataFrame]] = {
            "delta": lambda path: read_dataframe(path, allow_json_fallback=True),
            "iceberg": lambda path: read_dataframe(path, allow_json_fallback=True),
        }
        self._loaders: Dict[str, Callable[[Path], pd.DataFrame]] = (
            {**default_loaders, **(loaders or {})}
        )
        self._sources: Dict[str, OfflineSourceConfig] = {}

    def register(self, feature_view: str, config: OfflineSourceConfig) -> None:
        self._sources[feature_view] = config

    def load(self, feature_view: str) -> pd.DataFrame:
        config = self._sources.get(feature_view)
        if config is None:
            raise KeyError(f"No offline source registered for {feature_view!r}")
        loader = self._loaders.get(config.format)
        if loader is None:
            raise ValueError(f"No loader registered for format {config.format!r}")
        return loader(config.path)


class ValidationSchedule:
    """Track validation cadence for feature views."""

    def __init__(self, interval: timedelta) -> None:
        self._interval = interval
        self._last_run: Dict[str, datetime] = {}

    def due(self, feature_view: str, *, now: datetime | None = None) -> bool:
        reference = now or datetime.now(tz=UTC)
        last_run = self._last_run.get(feature_view)
        if last_run is None:
            return True
        return reference - last_run >= self._interval

    def mark(self, feature_view: str, *, now: datetime | None = None) -> None:
        reference = now or datetime.now(tz=UTC)
        self._last_run[feature_view] = reference


class PeriodicOfflineValidator:
    """Run periodic integrity checks against offline Delta/Iceberg sources."""

    def __init__(
        self,
        store: OnlineFeatureStore,
        registry: OfflineSourceRegistry,
        schedule: ValidationSchedule,
    ) -> None:
        self._store = store
        self._registry = registry
        self._schedule = schedule

    def validate(
        self,
        feature_view: str,
        *,
        ensure_valid: bool = True,
        now: datetime | None = None,
    ) -> IntegrityReport | None:
        if not self._schedule.due(feature_view, now=now):
            return None
        report = self._store.validate_against_offline(
            feature_view,
            self._registry.load,
            ensure_valid=ensure_valid,
        )
        self._schedule.mark(feature_view, now=now)
        return report


class OnlineFeatureStore:
    """Online feature store with pluggable backends and retention policies."""

    def __init__(
        self,
        root: Path,
        *,
        backend: FeatureStoreBackend | None = None,
        retention: RetentionPolicy | None = None,
        timestamp_column: str = "timestamp",
        dedup_keys: Iterable[str] | None = None,
    ) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._backend = backend or FilesystemFeatureStoreBackend(self._root)
        self._retention = retention
        self._timestamp_column = timestamp_column
        self._dedup_keys = list(dedup_keys) if dedup_keys else []
        self._maintenance_lock = RLock()
        self._maintenance_in_progress = False

    def purge(self, feature_view: str) -> None:
        """Remove persisted artefacts for ``feature_view`` if they exist."""

        self._backend.purge(feature_view)

    def load(self, feature_view: str) -> pd.DataFrame:
        """Load the persisted dataframe for ``feature_view``."""

        return self._backend.load(feature_view)

    def _resolve_path(self, feature_view: str) -> Path:
        """Expose filesystem paths for legacy tests and tooling."""

        if isinstance(self._backend, FilesystemFeatureStoreBackend):
            return self._backend._resolve_path(feature_view)
        raise AttributeError("Filesystem backend not configured for this store")

    def sync(
        self,
        feature_view: str,
        frame: pd.DataFrame,
        *,
        mode: Literal["append", "overwrite"] = "append",
        validate: bool = True,
    ) -> IntegrityReport:
        """Persist ``frame`` and return an integrity report."""

        if mode not in {"append", "overwrite"}:
            raise ValueError("mode must be either 'append' or 'overwrite'")

        offline_frame = frame.copy(deep=True)

        if mode == "overwrite":
            self.purge(feature_view)
            combined = offline_frame
            online_delta = combined.reset_index(drop=True)
        else:
            existing = self.load(feature_view)
            if not existing.empty:
                missing = set(existing.columns) ^ set(offline_frame.columns)
                if missing:
                    raise ValueError(
                        "Cannot append payload with mismatched columns: "
                        f"{sorted(missing)}"
                    )
                offline_frame = offline_frame[existing.columns]
            combined = self._append_frames(existing, offline_frame)
            delta_rows = offline_frame.shape[0]
            if delta_rows:
                online_delta = combined.tail(delta_rows).reset_index(drop=True)
            else:
                online_delta = combined.iloc[0:0]

        self._persist(feature_view, combined, mode="overwrite")

        if mode == "overwrite":
            stored = combined.reset_index(drop=True)
            report = self._build_report(feature_view, offline_frame, stored)
        else:
            report = self._build_report(feature_view, offline_frame, online_delta)

        if validate:
            report.ensure_valid()
        return report

    def compute_integrity(self, feature_view: str, frame: pd.DataFrame) -> IntegrityReport:
        """Compare ``frame`` against the currently persisted dataset."""

        online = self.load(feature_view)
        return self._build_report(feature_view, frame.copy(deep=True), online)

    @staticmethod
    def _append_frames(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
        if existing.empty:
            return incoming.reset_index(drop=True)
        combined = pd.concat(
            [existing.reset_index(drop=True), incoming.reset_index(drop=True)],
            ignore_index=True,
        )
        return combined

    def _persist(
        self,
        feature_view: str,
        frame: pd.DataFrame,
        *,
        mode: Literal["append", "overwrite"],
    ) -> None:
        with self._maintenance_lock:
            self._backend.write(feature_view, frame.reset_index(drop=True), mode=mode)
            self._apply_post_write_maintenance_locked(feature_view)

    def _apply_post_write_maintenance_locked(self, feature_view: str) -> None:
        if self._maintenance_in_progress:
            return
        self._maintenance_in_progress = True
        try:
            current = self._backend.load(feature_view)
            updated = current
            if self._dedup_keys:
                updated = updated.drop_duplicates(subset=self._dedup_keys, keep="last")
            if self._retention is not None:
                updated = self._retention.apply(
                    updated,
                    timestamp_column=self._timestamp_column,
                    now=pd.Timestamp.now(tz=UTC),
                )
            if not updated.equals(current):
                self._backend.write(feature_view, updated, mode="overwrite")
        finally:
            self._maintenance_in_progress = False

    def validate_against_offline(
        self,
        feature_view: str,
        offline_loader: Callable[[str], pd.DataFrame],
        *,
        ensure_valid: bool = True,
    ) -> IntegrityReport:
        """Compare the current view against an offline source of truth."""

        offline_frame = offline_loader(feature_view)
        report = self.compute_integrity(feature_view, offline_frame)
        if ensure_valid:
            report.ensure_valid()
        return report

    def _build_report(
        self,
        feature_view: str,
        offline_frame: pd.DataFrame,
        online_frame: pd.DataFrame,
    ) -> IntegrityReport:
        offline_snapshot = self._snapshot(offline_frame)
        online_snapshot = self._snapshot(online_frame)
        hash_differs = not hmac.compare_digest(
            offline_snapshot.data_hash, online_snapshot.data_hash
        )
        return IntegrityReport(
            feature_view=feature_view,
            offline_rows=offline_snapshot.row_count,
            online_rows=online_snapshot.row_count,
            row_count_diff=online_snapshot.row_count - offline_snapshot.row_count,
            offline_hash=offline_snapshot.data_hash,
            online_hash=online_snapshot.data_hash,
            hash_differs=hash_differs,
        )

    @staticmethod
    def _snapshot(frame: pd.DataFrame) -> IntegritySnapshot:
        canonical = OnlineFeatureStore._canonicalize(frame)
        payload = canonical.to_json(
            orient="split",
            index=False,
            date_format="iso",
            date_unit="ns",
            double_precision=15,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return IntegritySnapshot(row_count=int(canonical.shape[0]), data_hash=digest)

    @staticmethod
    def _canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        columns = sorted(frame.columns)
        canonical = frame.loc[:, columns].copy()
        if columns:
            canonical = canonical.sort_values(by=columns, kind="mergesort")
        return canonical.reset_index(drop=True)


__all__ = [
    "OfflineSourceConfig",
    "OfflineSourceRegistry",
    "PeriodicOfflineValidator",
    "ValidationSchedule",
    "FeatureStoreBackend",
    "FeatureStoreIntegrityError",
    "FilesystemFeatureStoreBackend",
    "IntegrityReport",
    "IntegritySnapshot",
    "OnlineFeatureStore",
    "RedisFeatureStoreBackend",
    "RetentionPolicy",
    "SQLiteFeatureStoreBackend",
]
