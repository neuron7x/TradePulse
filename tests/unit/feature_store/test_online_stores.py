import warnings

import pandas as pd
import pytest

from core.feature_store import FeatureSet, OnlineStoreConfig, RedisOnlineStore

pytest.importorskip("fakeredis")

import fakeredis  # noqa: E402


@pytest.fixture()
def fake_redis():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message="Call to '__init__' function with deprecated usage of input argument/s 'retry_on_timeout'",
        )
        return fakeredis.FakeRedis(decode_responses=True)


def test_redis_online_store_round_trip(fake_redis, tmp_path):
    config = OnlineStoreConfig(backend="redis")
    store = RedisOnlineStore(config, client=fake_redis)
    frame = pd.DataFrame(
        {
            "account_id": ["abc", "xyz"],
            "event_timestamp": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
            "balance": [100.0, 200.0],
        }
    )
    feature_set = FeatureSet(
        name="account_features",
        dataframe=frame,
        entity_columns=["account_id"],
        timestamp_column="event_timestamp",
    )

    store.write(feature_set)

    loaded = store.read("account_features")
    assert len(loaded) == 2
    assert set(loaded["account_id"]) == {"abc", "xyz"}

    latest = store.latest_timestamp("account_features", "event_timestamp")
    assert latest == pd.Timestamp("2024-01-02T00:00:00Z", tz="UTC")
