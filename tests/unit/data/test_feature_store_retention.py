from __future__ import annotations

from datetime import UTC

import pandas as pd
import pytest

from core.data.feature_store import (
    DeltaLakeSource,
    FeatureStoreIntegrityError,
    OfflineStoreValidator,
    RedisOnlineFeatureStore,
    RetentionPolicy,
    SQLiteOnlineFeatureStore,
)
from core.utils.dataframe_io import write_dataframe


class _MutableClock:
    def __init__(self, now: pd.Timestamp) -> None:
        self._now = now

    def now(self) -> pd.Timestamp:
        return self._now

    def advance(self, delta: pd.Timedelta) -> None:
        self._now = self._now + delta


class _DictClient:
    def __init__(self) -> None:
        self.payloads: dict[str, bytes] = {}

    def get(self, key: str):  # pragma: no cover - trivial
        return self.payloads.get(key)

    def set(self, key: str, value: bytes) -> None:
        self.payloads[key] = value

    def delete(self, key: str) -> None:  # pragma: no cover - defensive
        self.payloads.pop(key, None)


@pytest.fixture
def base_frame() -> pd.DataFrame:
    ts = pd.Timestamp("2024-01-01 00:00:00", tz=UTC)
    return pd.DataFrame(
        {
            "entity_id": ["a", "a", "b"],
            "ts": [ts - pd.Timedelta(hours=2), ts - pd.Timedelta(minutes=30), ts - pd.Timedelta(minutes=5)],
            "value": [1.0, 2.0, 3.0],
        }
    )


def test_redis_ttl_retention(base_frame: pd.DataFrame) -> None:
    clock = _MutableClock(pd.Timestamp("2024-01-01 00:00:00", tz=UTC))
    policy = RetentionPolicy(ttl=pd.Timedelta(hours=1))
    store = RedisOnlineFeatureStore(client=_DictClient(), retention_policy=policy, clock=clock.now)

    store.sync("demo.fv", base_frame, mode="overwrite", validate=False)
    stored = store.load("demo.fv")
    assert stored.shape[0] == 2
    assert stored["ts"].min() >= clock.now() - pd.Timedelta(hours=1)


def test_sqlite_max_versions(tmp_path, base_frame: pd.DataFrame) -> None:
    policy = RetentionPolicy(max_versions=1)
    store = SQLiteOnlineFeatureStore(tmp_path / "store.db", retention_policy=policy)

    store.sync("demo.fv", base_frame, mode="overwrite", validate=False)
    newer = base_frame.copy()
    newer.loc[:, "value"] = [10.0, 11.0, 12.0]
    newer.loc[:, "ts"] = newer["ts"] + pd.Timedelta(minutes=10)
    store.sync("demo.fv", newer, mode="append", validate=False)

    stored = store.load("demo.fv")
    assert stored.shape[0] == 2
    assert stored["ts"].is_monotonic_increasing


def test_offline_validator_detects_mismatch(tmp_path, base_frame: pd.DataFrame) -> None:
    offline_path = tmp_path / "delta"
    write_dataframe(base_frame, offline_path, allow_json_fallback=True)
    source = DeltaLakeSource(offline_path)

    # Drop the latest row to trigger an integrity failure.
    online_payload = base_frame.iloc[:-1]
    loader = lambda _name: online_payload

    validator = OfflineStoreValidator(
        "demo.fv",
        source,
        loader,
        interval=pd.Timedelta(minutes=5),
        clock=lambda: pd.Timestamp("2024-01-01 00:00:00", tz=UTC),
    )

    with pytest.raises(FeatureStoreIntegrityError):
        validator.run()


def test_offline_validator_respects_interval(tmp_path, base_frame: pd.DataFrame) -> None:
    offline_path = tmp_path / "delta"
    write_dataframe(base_frame, offline_path, allow_json_fallback=True)
    source = DeltaLakeSource(offline_path)

    clock = _MutableClock(pd.Timestamp("2024-01-01 00:00:00", tz=UTC))
    validator = OfflineStoreValidator(
        "demo.fv",
        source,
        lambda _name: base_frame,
        interval=pd.Timedelta(hours=1),
        clock=clock.now,
    )

    assert validator.should_run() is True
    validator.run()
    assert validator.should_run() is False
    clock.advance(pd.Timedelta(minutes=59))
    assert validator.should_run() is False
    clock.advance(pd.Timedelta(minutes=1))
    assert validator.should_run() is True
