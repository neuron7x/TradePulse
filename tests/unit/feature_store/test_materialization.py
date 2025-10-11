import pandas as pd
import pytest

from core.feature_store import (
    FeatureMaterializer,
    FeatureSet,
    FeatureStoreConfig,
    MaterializationConfig,
    OfflineStoreConfig,
    OnlineStoreConfig,
    OfflineStoreFactory,
    SQLiteOnlineStore,
)

pytest.importorskip("pyarrow")


@pytest.fixture
def sample_config(tmp_path):
    offline_config = OfflineStoreConfig(format="parquet", base_path=tmp_path / "offline")
    online_config = OnlineStoreConfig(backend="sqlite", sqlite_path=tmp_path / "online.db")
    materialization_config = MaterializationConfig(
        entity_columns=["account_id"],
        timestamp_column="event_timestamp",
        allow_overwrite=False,
    )
    return FeatureStoreConfig(
        offline=offline_config,
        online=online_config,
        materialization=materialization_config,
    )


def test_materialize_parquet_sqlite(sample_config):
    offline_store = OfflineStoreFactory(sample_config.offline).create()
    sqlite_store = SQLiteOnlineStore(sample_config.online)
    materializer = FeatureMaterializer(sample_config, offline_store=offline_store, online_store=sqlite_store)

    frame = pd.DataFrame(
        {
            "account_id": ["abc", "xyz"],
            "balance": [100.0, 200.5],
            "event_timestamp": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
        }
    )

    result = materializer.materialize(frame, "account_features", mode="overwrite")

    assert result.rows_written_offline == 2
    assert result.rows_written_online == 2
    assert (sample_config.offline.base_path / "account_features").exists()

    offline_loaded = offline_store.read("account_features")
    assert len(offline_loaded) == 2

    online_loaded = sqlite_store.read("account_features")
    assert len(online_loaded) == 2
    assert materializer.validate_consistency("account_features")


def test_materialize_overwrite_purges_stale_rows(sample_config):
    offline_store = OfflineStoreFactory(sample_config.offline).create()
    sqlite_store = SQLiteOnlineStore(sample_config.online)
    materializer = FeatureMaterializer(sample_config, offline_store=offline_store, online_store=sqlite_store)

    initial_frame = pd.DataFrame(
        {
            "account_id": ["abc", "xyz"],
            "balance": [100.0, 200.5],
            "event_timestamp": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
        }
    )
    materializer.materialize(initial_frame, "account_features", mode="overwrite")

    updated_frame = pd.DataFrame(
        {
            "account_id": ["xyz"],
            "balance": [250.75],
            "event_timestamp": ["2024-01-03T00:00:00Z"],
        }
    )

    result = materializer.materialize(updated_frame, "account_features", mode="overwrite")

    assert result.rows_written_online == 1
    offline_loaded = offline_store.read("account_features")
    assert len(offline_loaded) == 1
    online_loaded = sqlite_store.read("account_features")
    assert set(online_loaded["account_id"]) == {"xyz"}
    assert materializer.validate_consistency("account_features")


def test_materialize_rejects_regressions(sample_config):
    offline_store = OfflineStoreFactory(sample_config.offline).create()
    sqlite_store = SQLiteOnlineStore(sample_config.online)
    materializer = FeatureMaterializer(sample_config, offline_store=offline_store, online_store=sqlite_store)

    first_frame = pd.DataFrame(
        {
            "account_id": ["abc"],
            "balance": [100.0],
            "event_timestamp": ["2024-01-01T00:00:00Z"],
        }
    )
    materializer.materialize(first_frame, "account_features", mode="overwrite")

    older_frame = pd.DataFrame(
        {
            "account_id": ["abc"],
            "balance": [80.0],
            "event_timestamp": ["2023-12-31T00:00:00Z"],
        }
    )

    with pytest.raises(ValueError):
        materializer.materialize(older_frame, "account_features")


def test_sync_online_from_offline_incremental(tmp_path):
    offline_config = OfflineStoreConfig(format="parquet", base_path=tmp_path / "offline")
    online_config = OnlineStoreConfig(backend="sqlite", sqlite_path=tmp_path / "online.db")
    materialization_config = MaterializationConfig(
        entity_columns=["account_id"],
        timestamp_column="event_timestamp",
        allow_overwrite=True,
    )
    feature_config = FeatureStoreConfig(
        offline=offline_config,
        online=online_config,
        materialization=materialization_config,
    )

    offline_store = OfflineStoreFactory(offline_config).create()
    sqlite_store = SQLiteOnlineStore(online_config)
    materializer = FeatureMaterializer(feature_config, offline_store=offline_store, online_store=sqlite_store)

    initial_frame = pd.DataFrame(
        {
            "account_id": ["abc"],
            "balance": [100.0],
            "event_timestamp": ["2024-01-01T00:00:00Z"],
        }
    )
    materializer.materialize(initial_frame, "account_features", mode="overwrite")

    new_frame = pd.DataFrame(
        {
            "account_id": ["abc", "xyz"],
            "balance": [120.0, 300.0],
            "event_timestamp": ["2024-01-03T00:00:00Z", "2024-01-04T00:00:00Z"],
        }
    )
    feature_set = FeatureSet(
        name="account_features",
        dataframe=new_frame,
        entity_columns=["account_id"],
        timestamp_column="event_timestamp",
    )
    offline_store.write(feature_set, mode="append")
    sqlite_store.purge("account_features")

    result = materializer.sync_online_from_offline(
        "account_features", since=pd.Timestamp("2024-01-01T00:00:00Z", tz="UTC")
    )

    assert result.rows_written_online == 2
    online_loaded = sqlite_store.read("account_features")
    assert set(online_loaded["account_id"]) == {"abc", "xyz"}
