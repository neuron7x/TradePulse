from datetime import UTC

import pandas as pd
import pytest

from core.data.feature_store import OnlineFeatureStore
from core.data.parity import (
    FeatureParityCoordinator,
    FeatureParityReport,
    FeatureParitySpec,
    FeatureTimeSkewError,
    FeatureUpdateBlocked,
)


@pytest.fixture
def parity_spec() -> FeatureParitySpec:
    return FeatureParitySpec(
        feature_view="prices",
        timestamp_granularity="1min",
        numeric_tolerance=0.0,
        max_clock_skew=pd.Timedelta(minutes=5),
    )


def _make_frame(ts_values: list[str], values: list[float]) -> pd.DataFrame:
    timestamps = [pd.Timestamp(item, tz=UTC) for item in ts_values]
    return pd.DataFrame({"entity_id": ["A"] * len(values), "ts": timestamps, "value": values})


def test_parity_coordinator_validates_empty_frames(tmp_path) -> None:
    store = OnlineFeatureStore(tmp_path)
    coordinator = FeatureParityCoordinator(store)
    spec = FeatureParitySpec(feature_view="prices", allow_schema_evolution=True)

    empty = pd.DataFrame()

    with pytest.raises(KeyError, match="missing required columns"):
        coordinator.synchronize(spec, empty, mode="overwrite")


def test_parity_coordinator_overwrite_success(tmp_path, parity_spec) -> None:
    store = OnlineFeatureStore(tmp_path)
    coordinator = FeatureParityCoordinator(store)

    offline = _make_frame(
        ["2024-01-01T00:00:30Z", "2024-01-01T00:01:15Z"],
        [1.0, 2.0],
    )

    report = coordinator.synchronize(parity_spec, offline, mode="overwrite")

    assert isinstance(report, FeatureParityReport)
    assert report.inserted_rows == 2
    assert report.updated_rows == 0
    assert report.dropped_rows == 0
    assert report.integrity.hash_differs is False

    stored = store.load("prices")
    assert list(stored.columns) == ["entity_id", "ts", "value"]
    assert stored.shape[0] == 2
    assert set(stored["ts"]) == {
        pd.Timestamp("2024-01-01T00:00:00Z", tz=UTC),
        pd.Timestamp("2024-01-01T00:01:00Z", tz=UTC),
    }


def test_parity_coordinator_blocks_excessive_clock_skew(tmp_path, parity_spec) -> None:
    store = OnlineFeatureStore(tmp_path)
    coordinator = FeatureParityCoordinator(store)

    initial = _make_frame(["2024-01-01T00:00:00Z"], [1.0])
    coordinator.synchronize(parity_spec, initial, mode="overwrite")

    skewed = _make_frame(["2024-01-01T00:10:00Z"], [1.5])
    spec = FeatureParitySpec(
        feature_view="prices",
        timestamp_granularity="1min",
        numeric_tolerance=0.0,
        max_clock_skew=pd.Timedelta(minutes=2),
    )

    with pytest.raises(FeatureTimeSkewError):
        coordinator.synchronize(spec, skewed, mode="append")


def test_parity_coordinator_blocks_feature_drift(tmp_path, parity_spec) -> None:
    store = OnlineFeatureStore(tmp_path)
    coordinator = FeatureParityCoordinator(store)

    baseline = _make_frame(["2024-01-01T00:00:00Z"], [1.0])
    coordinator.synchronize(parity_spec, baseline, mode="overwrite")

    drifted = _make_frame(["2024-01-01T00:00:30Z"], [2.0])

    with pytest.raises(FeatureUpdateBlocked, match="drift exceeds"):
        coordinator.synchronize(parity_spec, drifted, mode="overwrite")


def test_parity_coordinator_allows_schema_evolution_on_overwrite(tmp_path) -> None:
    store = OnlineFeatureStore(tmp_path)
    coordinator = FeatureParityCoordinator(store)

    baseline = _make_frame(["2024-01-01T00:00:00Z"], [1.0])
    spec = FeatureParitySpec(
        feature_view="prices",
        timestamp_granularity="1min",
        numeric_tolerance=0.0,
        max_clock_skew=pd.Timedelta(minutes=5),
    )
    coordinator.synchronize(spec, baseline, mode="overwrite")

    evolved = _make_frame(["2024-01-01T00:05:00Z"], [1.5])
    evolved["confidence"] = [0.9]

    evolving_spec = FeatureParitySpec(
        feature_view="prices",
        timestamp_granularity="1min",
        numeric_tolerance=0.0,
        max_clock_skew=pd.Timedelta(minutes=5),
        allow_schema_evolution=True,
    )

    report = coordinator.synchronize(evolving_spec, evolved, mode="overwrite")
    assert report.columns_added == ("confidence",)
    assert store.load("prices").columns.tolist() == ["entity_id", "ts", "value", "confidence"]


def test_parity_coordinator_append_skips_duplicate_rows(tmp_path, parity_spec) -> None:
    store = OnlineFeatureStore(tmp_path)
    coordinator = FeatureParityCoordinator(store)

    initial = _make_frame(["2024-01-01T00:00:00Z"], [1.0])
    coordinator.synchronize(parity_spec, initial, mode="overwrite")

    duplicate = _make_frame(["2024-01-01T00:00:15Z"], [1.0])
    report = coordinator.synchronize(parity_spec, duplicate, mode="append")

    assert report.inserted_rows == 0
    assert store.load("prices").shape[0] == 1


def test_parity_coordinator_rejects_invalid_mode(tmp_path, parity_spec) -> None:
    store = OnlineFeatureStore(tmp_path)
    coordinator = FeatureParityCoordinator(store)
    frame = _make_frame(["2024-01-01T00:00:00Z"], [1.0])
    with pytest.raises(ValueError):
        coordinator.synchronize(parity_spec, frame, mode="invalid")


def test_parity_coordinator_blocks_schema_changes_without_flag(tmp_path, parity_spec) -> None:
    store = OnlineFeatureStore(tmp_path)
    coordinator = FeatureParityCoordinator(store)
    baseline = _make_frame(["2024-01-01T00:00:00Z"], [1.0])
    coordinator.synchronize(parity_spec, baseline, mode="overwrite")

    evolved = baseline.copy()
    evolved["extra"] = [0.5]
    with pytest.raises(FeatureUpdateBlocked, match="Schema change"):
        coordinator.synchronize(parity_spec, evolved, mode="overwrite")


def test_parity_coordinator_append_detects_updates(tmp_path, parity_spec) -> None:
    store = OnlineFeatureStore(tmp_path)
    coordinator = FeatureParityCoordinator(store)
    baseline = _make_frame(["2024-01-01T00:00:00Z"], [1.0])
    coordinator.synchronize(parity_spec, baseline, mode="overwrite")

    modified = _make_frame(["2024-01-01T00:00:30Z"], [1.5])
    with pytest.raises(FeatureUpdateBlocked, match="Append mode cannot modify"):
        coordinator.synchronize(parity_spec, modified, mode="append")


def test_parity_coordinator_respects_numeric_tolerance(tmp_path) -> None:
    store = OnlineFeatureStore(tmp_path)
    coordinator = FeatureParityCoordinator(store)
    spec = FeatureParitySpec(
        feature_view="prices",
        timestamp_granularity="1min",
        numeric_tolerance=0.01,
        max_clock_skew=pd.Timedelta(minutes=5),
    )
    baseline = _make_frame(["2024-01-01T00:00:00Z"], [1.0])
    coordinator.synchronize(spec, baseline, mode="overwrite")

    slight_drift = _make_frame(["2024-01-01T00:00:30Z"], [1.005])
    report = coordinator.synchronize(spec, slight_drift, mode="overwrite")
    assert report.max_value_drift is not None
    assert report.max_value_drift <= spec.numeric_tolerance


def test_parity_coordinator_floors_timestamps_to_granularity(tmp_path) -> None:
    store = OnlineFeatureStore(tmp_path)
    coordinator = FeatureParityCoordinator(store)
    spec = FeatureParitySpec(
        feature_view="prices",
        timestamp_granularity="1min",
        allow_schema_evolution=True,
    )
    frame = _make_frame(["2024-01-01T00:00:30Z"], [1.0])
    report = coordinator.synchronize(spec, frame, mode="overwrite")
    stored = store.load("prices")
    assert stored["ts"].iloc[0] == pd.Timestamp("2024-01-01T00:00:00Z", tz=UTC)
    assert report.inserted_rows == 1


def test_parity_resolve_value_columns_uses_explicit_spec(tmp_path) -> None:
    store = OnlineFeatureStore(tmp_path)
    coordinator = FeatureParityCoordinator(store)
    spec = FeatureParitySpec(
        feature_view="prices",
        timestamp_granularity="1min",
        value_columns=("value",),
    )
    offline = _make_frame(["2024-01-01T00:00:00Z"], [1.0])
    online = offline.copy()
    offline_prepared = coordinator._prepare_frame(offline, spec)
    online_prepared = coordinator._prepare_frame(online, spec, require_columns=False)
    keys = list(spec.entity_columns) + [spec.timestamp_column]
    resolved = coordinator._resolve_value_columns(offline_prepared, online_prepared, keys, spec)
    assert resolved == ("value",)


def test_parity_prepare_frame_returns_empty_copy_when_columns_missing(tmp_path) -> None:
    store = OnlineFeatureStore(tmp_path)
    coordinator = FeatureParityCoordinator(store)
    spec = FeatureParitySpec(feature_view="prices")
    prepared = coordinator._prepare_frame(pd.DataFrame(), spec, require_columns=False)
    assert prepared.empty


def test_parity_compute_value_drift_with_strings(tmp_path) -> None:
    store = OnlineFeatureStore(tmp_path)
    coordinator = FeatureParityCoordinator(store)
    spec = FeatureParitySpec(
        feature_view="prices",
        timestamp_granularity="1min",
        value_columns=("label",),
    )
    offline = _make_frame(["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z"], [1.0, 2.0])
    offline["label"] = ["A", "B"]
    online = _make_frame(["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z"], [1.0, 2.0])
    online["label"] = ["A", "C"]
    offline_prepared = coordinator._prepare_frame(offline, spec)
    online_prepared = coordinator._prepare_frame(online, spec, require_columns=False)
    keys = list(spec.entity_columns) + [spec.timestamp_column]
    drift, updated = coordinator._compute_value_drift(
        offline_prepared, online_prepared, keys, ("label",), spec
    )
    assert drift == 1.0
    assert updated


def test_parity_resolve_value_columns_filters_missing_online_columns(tmp_path) -> None:
    store = OnlineFeatureStore(tmp_path)
    coordinator = FeatureParityCoordinator(store)
    spec = FeatureParitySpec(feature_view="prices")
    offline = _make_frame(["2024-01-01T00:00:00Z"], [1.0])
    offline["extra"] = [5.0]
    online = _make_frame(["2024-01-01T00:00:00Z"], [1.0])
    offline_prepared = coordinator._prepare_frame(offline, spec)
    online_prepared = coordinator._prepare_frame(online, spec, require_columns=False)
    keys = list(spec.entity_columns) + [spec.timestamp_column]
    resolved = coordinator._resolve_value_columns(offline_prepared, online_prepared, keys, spec)
    assert resolved == ("value",)


def test_parity_prepare_frame_floors_timestamps(tmp_path) -> None:
    store = OnlineFeatureStore(tmp_path)
    coordinator = FeatureParityCoordinator(store)
    spec = FeatureParitySpec(feature_view="prices", timestamp_granularity="1min")
    frame = _make_frame(["2024-01-01T00:00:45Z", "2024-01-01T00:01:15Z"], [1.0, 2.0])
    prepared = coordinator._prepare_frame(frame, spec)
    assert all(ts.second == 0 for ts in prepared[spec.timestamp_column])


def test_parity_compute_clock_skew_handles_missing_timestamps(tmp_path, parity_spec) -> None:
    store = OnlineFeatureStore(tmp_path)
    coordinator = FeatureParityCoordinator(store)
    offline = _make_frame(["2024-01-01T00:00:00Z"], [1.0])
    offline.loc[0, "ts"] = pd.NaT
    online = _make_frame(["2024-01-01T00:00:00Z"], [1.0])
    clock_skew = coordinator._compute_clock_skew(offline, online, parity_spec)
    assert clock_skew is None


def test_parity_compute_value_drift_with_numeric_values(tmp_path) -> None:
    store = OnlineFeatureStore(tmp_path)
    coordinator = FeatureParityCoordinator(store)
    spec = FeatureParitySpec(
        feature_view="prices",
        timestamp_granularity="1min",
        numeric_tolerance=0.5,
    )
    offline = _make_frame(["2024-01-01T00:00:00Z"], [2.0])
    online = _make_frame(["2024-01-01T00:00:00Z"], [1.0])
    offline_prepared = coordinator._prepare_frame(offline, spec)
    online_prepared = coordinator._prepare_frame(online, spec, require_columns=False)
    keys = list(spec.entity_columns) + [spec.timestamp_column]
    drift, updated = coordinator._compute_value_drift(
        offline_prepared, online_prepared, keys, ("value",), spec
    )
    assert drift == 1.0
    assert updated


def test_parity_compute_value_drift_with_infinite_tolerance(tmp_path) -> None:
    store = OnlineFeatureStore(tmp_path)
    coordinator = FeatureParityCoordinator(store)
    spec = FeatureParitySpec(
        feature_view="prices",
        timestamp_granularity="1min",
        numeric_tolerance=None,
    )
    offline = _make_frame(["2024-01-01T00:00:00Z"], [3.0])
    online = _make_frame(["2024-01-01T00:00:00Z"], [1.0])
    offline_prepared = coordinator._prepare_frame(offline, spec)
    online_prepared = coordinator._prepare_frame(online, spec, require_columns=False)
    keys = list(spec.entity_columns) + [spec.timestamp_column]
    drift, updated = coordinator._compute_value_drift(
        offline_prepared, online_prepared, keys, ("value",), spec
    )
    assert drift == 2.0
    assert updated == set()


def test_parity_compute_value_drift_handles_empty_inputs(tmp_path, parity_spec) -> None:
    store = OnlineFeatureStore(tmp_path)
    coordinator = FeatureParityCoordinator(store)
    keys = list(parity_spec.entity_columns) + [parity_spec.timestamp_column]
    drift, updated = coordinator._compute_value_drift(
        pd.DataFrame(), pd.DataFrame(), keys, ("value",), parity_spec
    )
    assert drift is None
    assert updated == set()


def test_parity_compute_value_drift_merged_empty(tmp_path, parity_spec) -> None:
    store = OnlineFeatureStore(tmp_path)
    coordinator = FeatureParityCoordinator(store)
    offline = _make_frame(["2024-01-01T00:00:00Z"], [1.0])
    online = _make_frame(["2024-01-01T00:01:00Z"], [1.0])
    keys = list(parity_spec.entity_columns) + [parity_spec.timestamp_column]
    drift, updated = coordinator._compute_value_drift(
        offline, online, keys, ("value",), parity_spec
    )
    assert drift is None
    assert updated == set()
