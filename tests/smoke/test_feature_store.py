import pandas as pd
import pytest

from core.feature_store import (
    FeatureMaterializer,
    FeatureStoreConfig,
    MaterializationConfig,
    OfflineStoreConfig,
    OnlineStoreConfig,
)

pytest.importorskip("pyarrow")


def test_feature_store_smoke(tmp_path):
    config = FeatureStoreConfig(
        offline=OfflineStoreConfig(format="parquet", base_path=tmp_path / "offline"),
        online=OnlineStoreConfig(backend="sqlite", sqlite_path=tmp_path / "online.db"),
        materialization=MaterializationConfig(entity_columns=["id"], timestamp_column="event_timestamp"),
    )
    materializer = FeatureMaterializer(config)
    frame = pd.DataFrame(
        {
            "id": ["a"],
            "value": [1.0],
            "event_timestamp": ["2024-01-01T00:00:00Z"],
        }
    )
    result = materializer.materialize(frame, "test_features", mode="overwrite")
    assert result.rows_written_online == 1
    assert materializer.validate_consistency("test_features")
