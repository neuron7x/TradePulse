"""Online feature store helpers with integrity and retention utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
import hashlib
import hmac
from io import BytesIO
import sqlite3
from pathlib import Path
from typing import Callable, Literal, Protocol

import pandas as pd
from pandas.api import types as pd_types

from core.utils.dataframe_io import (
    purge_dataframe_artifacts,
    read_dataframe,
    write_dataframe,
)


class FeatureStoreIntegrityError(RuntimeError):
    """Raised when integrity invariants fail for feature store payloads."""


@dataclass(frozen=True)
class RetentionPolicy:
    """Configuration for expiring historical feature values."""

    ttl: pd.Timedelta | None = None
    max_versions: int | None = None

    def __post_init__(self) -> None:
        if self.ttl is not None and self.ttl <= pd.Timedelta(0):
            raise ValueError("ttl must be positive when provided")
        if self.max_versions is not None and self.max_versions <= 0:
            raise ValueError("max_versions must be positive when provided")


class _RetentionManager:
    """Apply retention rules to pandas dataframes."""

    def __init__(
        self,
        policy: RetentionPolicy | None,
        *,
        clock: Callable[[], pd.Timestamp] | None = None,
    ) -> None:
        self._policy = policy
        self._clock = clock or (lambda: pd.Timestamp.now(tz=UTC))

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or self._policy is None:
            return frame.copy()

        result = frame.copy()

        if self._policy.ttl is not None:
            if "ts" not in result.columns:
                raise KeyError("Retention policy with ttl requires a 'ts' column")
            cutoff = self._clock() - self._policy.ttl
            result = result[result["ts"] >= cutoff]

        if self._policy.max_versions is not None:
            missing = {"entity_id", "ts"} - set(result.columns)
            if missing:
                joined = ", ".join(sorted(missing))
                raise KeyError(
                    "Retention policy with max_versions requires columns: "
                    f"{joined}"
                )
            ordered = result.sort_values(by=["entity_id", "ts"], kind="mergesort")
            limited = ordered.groupby("entity_id", as_index=False).tail(self._policy.max_versions)
            result = limited.sort_values(by=["entity_id", "ts"], kind="mergesort")

        return result.reset_index(drop=True)


class KeyValueClient(Protocol):
    """Minimal protocol for key-value stores such as Redis."""

    def get(self, key: str) -> bytes | None:
        ...

    def set(self, key: str, value: bytes) -> None:
        ...

    def delete(self, key: str) -> None:
        ...


class InMemoryKeyValueClient:
    """In-memory key-value client used for tests and local development."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:  # pragma: no cover - trivial
        return self._store.get(key)

    def set(self, key: str, value: bytes) -> None:
        self._store[key] = value

    def delete(self, key: str) -> None:
        self._store.pop(key, None)


def _serialize_frame(frame: pd.DataFrame) -> bytes:
    """Serialize a dataframe to bytes using a safe JSON representation."""

    payload = frame.to_json(orient="table", index=False, date_unit="ns")
    return payload.encode("utf-8")


def _deserialize_frame(payload: bytes) -> pd.DataFrame:
    """Deserialize a dataframe from the JSON representation used in storage."""

    if not payload:
        return pd.DataFrame()
    return pd.read_json(BytesIO(payload), orient="table")


class RedisOnlineFeatureStore:
    """Redis-backed feature store with TTL-aware retention policies."""

    def __init__(
        self,
        client: KeyValueClient | None = None,
        *,
        retention_policy: RetentionPolicy | None = None,
        clock: Callable[[], pd.Timestamp] | None = None,
    ) -> None:
        self._client = client or InMemoryKeyValueClient()
        self._retention = _RetentionManager(retention_policy, clock=clock)

    def purge(self, feature_view: str) -> None:
        self._client.delete(feature_view)

    def load(self, feature_view: str) -> pd.DataFrame:
        payload = self._client.get(feature_view)
        if payload is None:
            return pd.DataFrame()
        frame = _deserialize_frame(payload)
        retained = self._retention.apply(frame)
        if not retained.equals(frame):
            self._client.set(feature_view, _serialize_frame(retained))
        return retained

    def sync(
        self,
        feature_view: str,
        frame: pd.DataFrame,
        *,
        mode: Literal["append", "overwrite"] = "append",
        validate: bool = True,
    ) -> "IntegrityReport":
        if mode not in {"append", "overwrite"}:
            raise ValueError("mode must be either 'append' or 'overwrite'")

        offline_frame = frame.copy(deep=True)

        if mode == "overwrite":
            stored = self._write(feature_view, offline_frame)
            report = OnlineFeatureStore._build_report(feature_view, offline_frame, stored)
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
                combined = OnlineFeatureStore._append_frames(existing, offline_frame)
            else:
                combined = offline_frame.reset_index(drop=True)
            stored = self._write(feature_view, combined)
            delta_rows = offline_frame.shape[0]
            if delta_rows:
                online_delta = stored.tail(delta_rows).reset_index(drop=True)
            else:
                online_delta = stored.iloc[0:0]
            report = OnlineFeatureStore._build_report(feature_view, offline_frame, online_delta)

        if validate:
            report.ensure_valid()
        return report

    def _write(self, feature_view: str, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = self._retention.apply(frame)
        self._client.set(feature_view, _serialize_frame(prepared))
        return prepared.reset_index(drop=True)


class SQLiteOnlineFeatureStore:
    """SQLite-backed online feature store with retention controls."""

    def __init__(
        self,
        path: Path,
        *,
        retention_policy: RetentionPolicy | None = None,
        clock: Callable[[], pd.Timestamp] | None = None,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._path)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS feature_views (name TEXT PRIMARY KEY, payload BLOB)"
        )
        self._connection.commit()
        self._retention = _RetentionManager(retention_policy, clock=clock)

    def purge(self, feature_view: str) -> None:
        with self._connection:
            self._connection.execute("DELETE FROM feature_views WHERE name = ?", (feature_view,))

    def load(self, feature_view: str) -> pd.DataFrame:
        cursor = self._connection.execute(
            "SELECT payload FROM feature_views WHERE name = ?", (feature_view,)
        )
        row = cursor.fetchone()
        if row is None:
            return pd.DataFrame()
        frame = _deserialize_frame(row[0])
        retained = self._retention.apply(frame)
        if not retained.equals(frame):
            self._persist(feature_view, retained)
        return retained

    def sync(
        self,
        feature_view: str,
        frame: pd.DataFrame,
        *,
        mode: Literal["append", "overwrite"] = "append",
        validate: bool = True,
    ) -> "IntegrityReport":
        if mode not in {"append", "overwrite"}:
            raise ValueError("mode must be either 'append' or 'overwrite'")

        offline_frame = frame.copy(deep=True)

        if mode == "overwrite":
            stored = self._write(feature_view, offline_frame)
            report = OnlineFeatureStore._build_report(feature_view, offline_frame, stored)
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
                combined = OnlineFeatureStore._append_frames(existing, offline_frame)
            else:
                combined = offline_frame.reset_index(drop=True)
            stored = self._write(feature_view, combined)
            delta_rows = offline_frame.shape[0]
            if delta_rows:
                online_delta = stored.tail(delta_rows).reset_index(drop=True)
            else:
                online_delta = stored.iloc[0:0]
            report = OnlineFeatureStore._build_report(feature_view, offline_frame, online_delta)

        if validate:
            report.ensure_valid()
        return report

    def _write(self, feature_view: str, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = self._retention.apply(frame)
        self._persist(feature_view, prepared)
        return prepared.reset_index(drop=True)

    def _persist(self, feature_view: str, frame: pd.DataFrame) -> None:
        payload = _serialize_frame(frame)
        with self._connection:
            self._connection.execute(
                "REPLACE INTO feature_views (name, payload) VALUES (?, ?)",
                (feature_view, payload),
            )


class OfflineTableSource(Protocol):
    """Protocol describing offline tables used as the source of truth."""

    def load(self) -> pd.DataFrame:
        ...


@dataclass(frozen=True)
class DeltaLakeSource(OfflineTableSource):
    """Offline source backed by a Delta Lake table."""

    path: Path

    def load(self) -> pd.DataFrame:
        return read_dataframe(self.path, allow_json_fallback=True)


@dataclass(frozen=True)
class IcebergSource(OfflineTableSource):
    """Offline source backed by an Apache Iceberg table."""

    path: Path

    def load(self) -> pd.DataFrame:
        return read_dataframe(self.path, allow_json_fallback=True)


class OfflineStoreValidator:
    """Periodically compare online materialisations with the offline source of truth."""

    def __init__(
        self,
        feature_view: str,
        offline_source: OfflineTableSource,
        online_loader: Callable[[str], pd.DataFrame],
        *,
        interval: pd.Timedelta = pd.Timedelta(hours=1),
        clock: Callable[[], pd.Timestamp] | None = None,
    ) -> None:
        if interval <= pd.Timedelta(0):
            raise ValueError("interval must be positive")
        self._feature_view = feature_view
        self._offline_source = offline_source
        self._online_loader = online_loader
        self._interval = interval
        self._clock = clock or (lambda: pd.Timestamp.now(tz=UTC))
        self._last_run: pd.Timestamp | None = None

    def should_run(self) -> bool:
        if self._last_run is None:
            return True
        return self._clock() - self._last_run >= self._interval

    def run(self, *, enforce: bool = True) -> "IntegrityReport":
        offline_frame = self._offline_source.load().copy(deep=True)
        online_frame = self._online_loader(self._feature_view).copy(deep=True)
        report = OnlineFeatureStore._build_report(
            self._feature_view,
            offline_frame,
            online_frame,
        )
        if enforce:
            report.ensure_valid()
        self._last_run = self._clock()
        return report


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


class OnlineFeatureStore:
    """Simple parquet-backed store providing overwrite/append semantics."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, feature_view: str) -> Path:
        safe_name = feature_view.replace("/", "__").replace(".", "__")
        return self._root / safe_name

    def purge(self, feature_view: str) -> None:
        """Remove persisted artefacts for ``feature_view`` if they exist."""

        path = self._resolve_path(feature_view)
        purge_dataframe_artifacts(path)

    def load(self, feature_view: str) -> pd.DataFrame:
        """Load the persisted dataframe for ``feature_view``."""

        path = self._resolve_path(feature_view)
        return read_dataframe(path, allow_json_fallback=True)

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
        path = self._resolve_path(feature_view)

        if mode == "overwrite":
            self.purge(feature_view)
            stored = self._write_frame(path, offline_frame)
            report = self._build_report(feature_view, offline_frame, stored)
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
            stored = self._append_frames(existing, offline_frame)
            self._write_frame(path, stored)
            delta_rows = offline_frame.shape[0]
            if delta_rows:
                online_delta = stored.tail(delta_rows).reset_index(drop=True)
            else:
                online_delta = stored.iloc[0:0]
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

    def _write_frame(self, path: Path, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.reset_index(drop=True)
        write_dataframe(prepared, path, index=False, allow_json_fallback=True)
        return prepared

    @staticmethod
    def _build_report(
        feature_view: str,
        offline_frame: pd.DataFrame,
        online_frame: pd.DataFrame,
    ) -> IntegrityReport:
        offline_snapshot = OnlineFeatureStore._snapshot(offline_frame)
        online_snapshot = OnlineFeatureStore._snapshot(online_frame)
        hash_differs = not hmac.compare_digest(
            offline_snapshot.data_hash,
            online_snapshot.data_hash,
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
        normalized = OnlineFeatureStore._normalize_for_hash(canonical)
        payload = normalized.to_json(
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

    @staticmethod
    def _normalize_for_hash(frame: pd.DataFrame) -> pd.DataFrame:
        """Coerce values into stable types prior to hashing."""

        if frame.empty:
            return frame.copy()

        normalized = frame.copy()

        for column in normalized.columns:
            series = normalized[column]

            if pd_types.is_datetime64_any_dtype(series):
                normalized[column] = pd.to_datetime(series, utc=True)
                continue

            if pd_types.is_timedelta64_dtype(series):
                # Represent timedeltas using ISO-8601 duration strings for stability.
                normalized[column] = series.astype("timedelta64[ns]").astype(str)
                continue

            if series.dtype == object:
                # Attempt to coerce datetime-like payloads first.
                try:
                    coerced_datetime = pd.to_datetime(series, utc=True)
                except (TypeError, ValueError):
                    coerced_datetime = None
                if coerced_datetime is not None and pd_types.is_datetime64_any_dtype(coerced_datetime):
                    normalized[column] = coerced_datetime
                    continue

                numeric = pd.to_numeric(series, errors="coerce")
                if pd_types.is_numeric_dtype(numeric):
                    if numeric.isna().equals(pd.isna(series)):
                        normalized[column] = numeric
                        continue

                normalized[column] = series.astype(str)

        return normalized


__all__ = [
    "DeltaLakeSource",
    "FeatureStoreIntegrityError",
    "IcebergSource",
    "InMemoryKeyValueClient",
    "IntegrityReport",
    "IntegritySnapshot",
    "OfflineStoreValidator",
    "OnlineFeatureStore",
    "RedisOnlineFeatureStore",
    "RetentionPolicy",
    "SQLiteOnlineFeatureStore",
]
