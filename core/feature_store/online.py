"""Online feature store implementations."""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Optional, TYPE_CHECKING

import pandas as pd

from .config import OnlineStoreConfig
from .models import FeatureSet

if TYPE_CHECKING:  # pragma: no cover
    from redis import Redis


class OnlineStore(ABC):
    """Shared interface implemented by Redis and SQLite stores."""

    def __init__(self, config: OnlineStoreConfig) -> None:
        self.config = config

    @abstractmethod
    def write(self, feature_set: FeatureSet) -> None:
        """Persist features to the online store."""

    @abstractmethod
    def read(self, feature_set_name: str, entity_filter: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Fetch feature rows for the provided entities (or all rows when omitted)."""

    @abstractmethod
    def latest_timestamp(self, feature_set_name: str, timestamp_column: str) -> Optional[pd.Timestamp]:
        """Return the most recent timestamp stored for a feature set."""

    @abstractmethod
    def purge(self, feature_set_name: str) -> None:
        """Remove all materialized rows for the provided feature set."""


class SQLiteOnlineStore(OnlineStore):
    """SQLite-backed online store suited for lightweight serving."""

    def __init__(self, config: OnlineStoreConfig) -> None:
        if config.backend != "sqlite":
            raise ValueError("SQLiteOnlineStore requires backend='sqlite'")
        if config.sqlite_path is None:
            raise ValueError("sqlite_path must be configured for SQLiteOnlineStore")
        super().__init__(config)
        self._db_path = Path(config.sqlite_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._db_path)
        self._connection.execute("PRAGMA journal_mode=WAL;")

    def _table(self, feature_set_name: str) -> str:
        return f"feature__{feature_set_name}"

    def _ensure_table(self, feature_set: FeatureSet) -> None:
        table = self._table(feature_set.name)
        with self._connection:
            self._connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    entity_key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    event_timestamp TEXT NOT NULL
                )
                """
            )

    def write(self, feature_set: FeatureSet) -> None:
        self._ensure_table(feature_set)
        table = self._table(feature_set.name)
        rows = []
        for key, record in zip(feature_set.entity_keys(), feature_set.to_records(), strict=True):
            event_ts = pd.to_datetime(record[feature_set.timestamp_column], utc=True)
            record[feature_set.timestamp_column] = event_ts.isoformat()
            rows.append((key, json.dumps(record, default=str), record[feature_set.timestamp_column]))
        with self._connection:
            self._connection.executemany(
                f"""
                INSERT INTO {table} (entity_key, payload, event_timestamp)
                VALUES (?, ?, ?)
                ON CONFLICT(entity_key) DO UPDATE SET
                    payload=excluded.payload,
                    event_timestamp=excluded.event_timestamp
                """,
                rows,
            )

    def read(self, feature_set_name: str, entity_filter: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        table = self._table(feature_set_name)
        if not self._table_exists(table):
            return pd.DataFrame()
        query = f"SELECT payload FROM {table}"
        parameters: list[str] = []
        if entity_filter is not None and not entity_filter.empty:
            entity_keys = entity_filter.astype(str).agg("|".join, axis=1).tolist()
            placeholders = ",".join("?" for _ in entity_keys)
            query = f"SELECT payload FROM {table} WHERE entity_key IN ({placeholders})"
            parameters = entity_keys
        cursor = self._connection.execute(query, parameters)
        payloads = [json.loads(row[0]) for row in cursor.fetchall()]
        if not payloads:
            return pd.DataFrame()
        frame = pd.DataFrame(payloads)
        if "event_timestamp" in frame.columns:
            frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"], utc=True, errors="coerce")
        return frame

    def latest_timestamp(self, feature_set_name: str, timestamp_column: str) -> Optional[pd.Timestamp]:
        table = self._table(feature_set_name)
        if not self._table_exists(table):
            return None
        cursor = self._connection.execute(
            f"SELECT event_timestamp FROM {table} ORDER BY event_timestamp DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return pd.to_datetime(row[0], utc=True)

    def purge(self, feature_set_name: str) -> None:
        table = self._table(feature_set_name)
        if not self._table_exists(table):
            return
        with self._connection:
            self._connection.execute(f"DELETE FROM {table}")

    def _table_exists(self, table: str) -> bool:
        cursor = self._connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        return cursor.fetchone() is not None


class RedisOnlineStore(OnlineStore):
    """Redis-backed online store optimised for low-latency feature serving."""

    def __init__(self, config: OnlineStoreConfig, client: Optional["Redis"] = None) -> None:
        if config.backend != "redis":
            raise ValueError("RedisOnlineStore requires backend='redis'")
        super().__init__(config)
        if client is not None:
            self._client = client
        else:  # pragma: no cover - exercised in integration tests
            try:
                from redis import Redis
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "Redis support requires the 'redis' extra (pip install tradepulse[feature-store])"
                ) from exc
            self._client = Redis(
                host=config.redis_host,
                port=config.redis_port,
                db=config.redis_db,
                username=config.redis_username,
                password=config.redis_password,
                ssl=config.redis_ssl,
                decode_responses=True,
            )
        self._ttl_seconds = config.redis_ttl_seconds

    def _key(self, feature_set_name: str, entity_key: str) -> str:
        return f"tradepulse:{feature_set_name}:{entity_key}"

    def write(self, feature_set: FeatureSet) -> None:
        pipeline = self._client.pipeline()
        for entity_key, record in zip(feature_set.entity_keys(), feature_set.to_records(), strict=True):
            event_ts = pd.to_datetime(record[feature_set.timestamp_column], utc=True)
            record[feature_set.timestamp_column] = event_ts.isoformat()
            payload = json.dumps(record, default=str)
            key = self._key(feature_set.name, entity_key)
            pipeline.hset(
                key,
                mapping={"payload": payload, "event_timestamp": record[feature_set.timestamp_column]},
            )
            if self._ttl_seconds:
                pipeline.expire(key, int(self._ttl_seconds))
        pipeline.execute()

    def read(self, feature_set_name: str, entity_filter: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        keys: Iterable[str]
        if entity_filter is not None and not entity_filter.empty:
            entity_keys = entity_filter.astype(str).agg("|".join, axis=1).tolist()
            keys = [self._key(feature_set_name, key) for key in entity_keys]
        else:
            pattern = self._key(feature_set_name, "*")
            keys = list(self._client.scan_iter(match=pattern))
        if not keys:
            return pd.DataFrame()
        pipeline = self._client.pipeline()
        for key in keys:
            pipeline.hgetall(key)
        rows = pipeline.execute()
        payloads = [json.loads(row.get("payload", "{}")) for row in rows if row]
        if not payloads:
            return pd.DataFrame()
        frame = pd.DataFrame(payloads)
        if "event_timestamp" in frame.columns:
            frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"], utc=True, errors="coerce")
        return frame

    def latest_timestamp(self, feature_set_name: str, timestamp_column: str) -> Optional[pd.Timestamp]:
        pattern = self._key(feature_set_name, "*")
        latest: Optional[pd.Timestamp] = None
        for key in self._client.scan_iter(match=pattern):
            payload = self._client.hget(key, "event_timestamp")
            if payload is None:
                continue
            ts = pd.to_datetime(payload, utc=True)
            if latest is None or ts > latest:
                latest = ts
        return latest

    def purge(self, feature_set_name: str) -> None:
        pattern = self._key(feature_set_name, "*")
        keys = list(self._client.scan_iter(match=pattern))
        if keys:
            self._client.delete(*keys)


__all__ = ["OnlineStore", "RedisOnlineStore", "SQLiteOnlineStore"]
