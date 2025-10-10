# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.utils.cache import IndicatorCache, IndicatorCacheKey


def test_indicator_cache_roundtrip(tmp_path: Path) -> None:
    cache = IndicatorCache(tmp_path / "cache")
    key = IndicatorCacheKey("demo", "M1")
    series = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.date_range("2024-01-01", periods=3, freq="1min"),
    )
    data_hash = cache.hash_series(series)
    fingerprint, params_hash = cache.make_fingerprint(
        indicator="demo",
        params={"window": 12, "mode": "test"},
        data_hash=data_hash,
        code_version=cache.code_version,
        timeframe="M1",
    )
    cache.store_entry(
        key,
        fingerprint=fingerprint,
        data_hash=data_hash,
        params_hash=params_hash,
        latest_timestamp=series.index[-1],
        row_count=len(series),
        payload={"value": 42},
    )

    entry = cache.load_entry(key)
    assert entry is not None
    assert entry.metadata.fingerprint == fingerprint
    assert entry.metadata.row_count == len(series)
    assert entry.metadata.latest_timestamp_pd() == series.index[-1]
    assert entry.payload == {"value": 42}


def test_indicator_cache_plan_backfill(tmp_path: Path) -> None:
    cache = IndicatorCache(tmp_path / "cache")
    key = IndicatorCacheKey("demo", "M5")
    series = pd.Series(
        [10.0, 11.0, 12.0, 13.0],
        index=pd.date_range("2024-01-01", periods=4, freq="5min"),
    )
    params = {"alpha": 0.25}
    data_hash = cache.hash_series(series)
    fingerprint, params_hash = cache.make_fingerprint(
        indicator="demo",
        params=params,
        data_hash=data_hash,
        code_version=cache.code_version,
        timeframe="M5",
    )
    cache.store_entry(
        key,
        fingerprint=fingerprint,
        data_hash=data_hash,
        params_hash=params_hash,
        latest_timestamp=series.index[-1],
        row_count=len(series),
        payload={"values": series.tolist()},
    )

    entry = cache.load_entry(key)
    assert entry is not None

    up_to_date = cache.plan_backfill(
        entry,
        fingerprint=fingerprint,
        latest_timestamp=series.index[-1],
    )
    assert up_to_date.cache_hit is True
    assert up_to_date.needs_update is False
    assert up_to_date.incremental is False

    extended_index = pd.date_range(series.index[-1] + pd.Timedelta(minutes=5), periods=2, freq="5min")
    extended_series = pd.Series([14.0, 15.0], index=extended_index)
    new_hash = cache.hash_series(pd.concat([series, extended_series]))
    next_fingerprint, _ = cache.make_fingerprint(
        indicator="demo",
        params=params,
        data_hash=new_hash,
        code_version=cache.code_version,
        timeframe="M5",
    )
    incremental = cache.plan_backfill(
        entry,
        fingerprint=next_fingerprint,
        latest_timestamp=extended_series.index[-1],
    )
    assert incremental.cache_hit is True
    assert incremental.needs_update is True
    assert incremental.incremental is True
    assert incremental.start_timestamp == series.index[-1]
