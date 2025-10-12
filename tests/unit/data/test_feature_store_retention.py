from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from core.data.feature_store import (
    OfflineSourceConfig,
    OfflineSourceRegistry,
    OnlineFeatureStore,
    PeriodicOfflineValidator,
    RedisFeatureStoreBackend,
    RetentionPolicy,
    SQLiteFeatureStoreBackend,
    ValidationSchedule,
)
from core.utils.dataframe_io import write_dataframe


def _frame(rows: list[tuple[str, datetime, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": [row[0] for row in rows],
            "timestamp": [row[1] for row in rows],
            "value": [row[2] for row in rows],
        }
    )


class _FakeRedisClient:
    def __init__(self) -> None:
        self._payload: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self._payload.get(key)

    def set(self, key: str, value: bytes) -> None:
        self._payload[key] = value

    def delete(self, key: str) -> None:
        self._payload.pop(key, None)


def test_retention_policy_evicts_stale_rows(tmp_path) -> None:
    retention = RetentionPolicy(ttl=timedelta(days=1))
    store = OnlineFeatureStore(
        tmp_path,
        retention=retention,
        dedup_keys=["entity_id", "timestamp"],
    )

    stale_ts = datetime(2020, 1, 1, tzinfo=UTC)
    recent_ts = datetime.now(tz=UTC)

    store.sync("features.prices", _frame([("A", stale_ts, 1.0)]), mode="overwrite")
    assert store.load("features.prices").empty

    store.sync("features.prices", _frame([("A", recent_ts, 2.0)]), mode="append")
    stored = store.load("features.prices")
    assert stored.shape[0] == 1
    assert stored.iloc[0]["value"] == 2.0


def test_retention_policy_cap_rows(tmp_path) -> None:
    retention = RetentionPolicy(max_rows=2)
    store = OnlineFeatureStore(
        tmp_path,
        retention=retention,
        dedup_keys=["entity_id", "timestamp"],
    )

    base = datetime(2023, 1, 1, tzinfo=UTC)
    payload = _frame(
        [
            ("A", base, 1.0),
            ("B", base + timedelta(days=1), 2.0),
            ("C", base + timedelta(days=2), 3.0),
        ]
    )

    store.sync("features.prices", payload.iloc[:2], mode="overwrite")
    store.sync("features.prices", payload.iloc[2:], mode="append")

    stored = store.load("features.prices")
    assert stored.shape[0] == 2
    assert set(stored["entity_id"]) == {"B", "C"}


def test_sqlite_backend_respects_retention(tmp_path) -> None:
    backend = SQLiteFeatureStoreBackend(tmp_path / "features.db")
    retention = RetentionPolicy(max_rows=1)
    store = OnlineFeatureStore(
        tmp_path,
        backend=backend,
        retention=retention,
        dedup_keys=["entity_id", "timestamp"],
    )

    base = datetime(2024, 3, 1, tzinfo=UTC)
    store.sync("signals.slow", _frame([("X", base, 1.0)]), mode="overwrite")
    store.sync("signals.slow", _frame([("Y", base + timedelta(days=1), 2.0)]), mode="append")

    stored = store.load("signals.slow")
    assert stored.shape[0] == 1
    assert stored.iloc[0]["entity_id"] == "Y"


def test_redis_backend_roundtrip(tmp_path) -> None:
    backend = RedisFeatureStoreBackend(_FakeRedisClient())
    store = OnlineFeatureStore(
        tmp_path,
        backend=backend,
        dedup_keys=["entity_id", "timestamp"],
    )

    base = datetime(2024, 4, 1, tzinfo=UTC)
    store.sync("stream.features", _frame([("A", base, 1.0)]), mode="overwrite")
    store.sync(
        "stream.features",
        _frame([("A", base + timedelta(minutes=1), 2.0)]),
        mode="append",
    )

    stored = store.load("stream.features")
    assert stored.shape[0] == 2
    assert stored.iloc[-1]["value"] == 2.0


def test_periodic_validator_reads_offline_sources(tmp_path) -> None:
    store = OnlineFeatureStore(tmp_path, dedup_keys=["entity_id", "timestamp"])
    base = datetime(2024, 5, 1, tzinfo=UTC)
    offline_frame = _frame([("A", base, 1.0), ("B", base + timedelta(hours=1), 2.0)])
    store.sync("features.live", offline_frame, mode="overwrite")

    offline_path = write_dataframe(
        offline_frame,
        tmp_path / "offline_dataset.delta",
        allow_json_fallback=True,
    )

    registry = OfflineSourceRegistry()
    registry.register(
        "features.live",
        OfflineSourceConfig(format="delta", path=offline_path),
    )
    schedule = ValidationSchedule(interval=timedelta(minutes=30))
    validator = PeriodicOfflineValidator(store, registry, schedule)

    reference_time = datetime(2024, 5, 2, tzinfo=UTC)

    first = validator.validate("features.live", now=reference_time)
    assert first is not None
    assert first.row_count_diff == 0

    skipped = validator.validate("features.live", now=reference_time + timedelta(minutes=15))
    assert skipped is None

    rerun = validator.validate(
        "features.live",
        now=reference_time + timedelta(minutes=30),
    )
    assert rerun is not None
    assert rerun.row_count_diff == 0
