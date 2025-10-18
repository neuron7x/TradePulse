from __future__ import annotations

import pandas as pd
import pytest

from core.data.normalization_pipeline import (
    MarketNormalizationConfig,
    _ensure_ohlcv_columns,
    _extract_metadata,
    _fill_gaps,
    _from_ticks,
    _prepare_frame,
    normalize_market_data,
)


def test_normalize_ticks_resamples_to_ohlcv() -> None:
    timestamps = pd.to_datetime(
        [
            "2024-02-01T09:30:00Z",
            "2024-02-01T09:30:30Z",
            "2024-02-01T09:31:00Z",
            "2024-02-01T09:33:00Z",
            "2024-02-01T09:33:00Z",  # duplicate entry should be removed
            "2024-02-01T09:34:30Z",
        ]
    )
    frame = pd.DataFrame(
        {
            "price": [100.0, 101.0, 102.0, 101.5, 101.5, 103.0],
            "volume": [5.0, 1.0, 1.5, 2.0, 2.0, 3.0],
            "symbol": "ETHUSD",
            "venue": "coinbase",
        },
        index=timestamps,
    )

    config = MarketNormalizationConfig(kind="tick", frequency="1min")
    result = normalize_market_data(frame, config=config)

    assert list(result.frame.columns) == ["open", "high", "low", "close", "volume"]
    expected_index = pd.date_range(
        "2024-02-01T09:30:00Z", "2024-02-01T09:34:00Z", freq="1min", tz="UTC"
    )
    pd.testing.assert_index_equal(result.frame.index, expected_index)

    # Missing intervals should be forward-filled for prices while volume defaults to 0.
    assert result.frame.loc["2024-02-01T09:32:00+00:00", "close"] == 102.0
    assert result.frame.loc["2024-02-01T09:32:00+00:00", "volume"] == 0.0

    metadata = result.metadata
    assert metadata.kind == "tick"
    assert metadata.frequency == "1min"
    assert metadata.duplicates_dropped == 1
    assert metadata.missing_intervals == 1
    assert metadata.filled_intervals == 1
    assert metadata.metadata["symbol"] == "ETHUSD"
    assert metadata.metadata["venue"] == "coinbase"


def test_normalize_ticks_handles_nan_only_bins(monkeypatch) -> None:
    timestamps = pd.to_datetime([
        "2024-02-01T09:30:00Z",
        "2024-02-01T09:31:00Z",
    ])
    frame = pd.DataFrame({"price": [float("nan"), float("nan")], "volume": [1.0, 2.0]}, index=timestamps)
    config = MarketNormalizationConfig(kind="tick", frequency="1min")
    empty = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"],
        index=pd.DatetimeIndex([], tz="UTC"),
    )
    monkeypatch.setattr(
        "core.data.normalization_pipeline._from_ticks",
        lambda *_args, **_kwargs: empty,
    )
    result = normalize_market_data(frame, config=config)
    assert result.frame.empty
    assert result.metadata.rows == 0
    assert result.metadata.missing_intervals == 0


def test_normalize_ohlcv_interpolates_prices() -> None:
    index = pd.to_datetime(
        [
            "2024-02-01T10:00:00Z",
            "2024-02-01T10:01:00Z",
            "2024-02-01T10:03:00Z",
        ]
    )
    frame = pd.DataFrame(
        {
            "open": [200.0, 201.0, 203.0],
            "high": [201.0, 202.0, 204.0],
            "low": [199.0, 200.5, 202.5],
            "close": [200.5, 201.5, 203.5],
            "volume": [10.0, 12.0, 14.0],
            "instrument_type": "spot",
        },
        index=index,
    )

    config = MarketNormalizationConfig(
        kind="ohlcv", frequency="1min", fill_method="interpolate"
    )
    result = normalize_market_data(frame, config=config)

    expected_index = pd.date_range(
        "2024-02-01T10:00:00Z", "2024-02-01T10:03:00Z", freq="1min", tz="UTC"
    )
    pd.testing.assert_index_equal(result.frame.index, expected_index)

    # Interpolated bar should linearly interpolate prices and reset volume to zero.
    interpolated = result.frame.loc["2024-02-01T10:02:00+00:00"]
    assert interpolated["volume"] == 0.0
    assert interpolated["close"] == 202.5

    metadata = result.metadata
    assert metadata.kind == "ohlcv"
    assert metadata.frequency == "1min"
    assert metadata.missing_intervals == 1
    assert metadata.metadata["instrument_type"] == "spot"


def test_normalize_allows_empty_when_configured() -> None:
    config = MarketNormalizationConfig(allow_empty=True)
    empty = pd.DataFrame()
    result = normalize_market_data(empty, config=config)
    assert result.frame.empty
    assert result.metadata.rows == 0


def test_normalize_rejects_empty_when_not_allowed() -> None:
    config = MarketNormalizationConfig()
    with pytest.raises(ValueError):
        normalize_market_data(pd.DataFrame(), config=config)


def test_normalize_requires_timestamp_column_when_missing() -> None:
    frame = pd.DataFrame({"price": [1.0, 2.0]})
    config = MarketNormalizationConfig()
    with pytest.raises(KeyError):
        normalize_market_data(frame, config=config)


def test_normalize_requires_frequency_when_inference_fails() -> None:
    index = pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T00:00:05Z", "2024-01-01T00:00:17Z"])
    frame = pd.DataFrame({"price": [1.0, 2.0, 3.0]}, index=index)
    config = MarketNormalizationConfig(kind="tick", frequency=None)
    with pytest.raises(ValueError):
        normalize_market_data(frame, config=config)


def test_fill_gaps_none_returns_frame() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="1min")
    frame = pd.DataFrame({"open": [1.0, 2.0]}, index=index)
    filled = _fill_gaps(frame, "none")
    pd.testing.assert_frame_equal(filled, frame)


def test_from_ticks_requires_price_column() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="1min")
    frame = pd.DataFrame({"volume": [1.0, 2.0]}, index=index)
    config = MarketNormalizationConfig(kind="tick", price_col="price", volume_col="volume")
    with pytest.raises(KeyError):
        _from_ticks(frame, config, "1min")


def test_from_ticks_defaults_missing_volume_to_zero() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="1min")
    frame = pd.DataFrame({"price": [100.0, 101.0, 102.0]}, index=index)
    config = MarketNormalizationConfig(kind="tick", price_col="price", volume_col="volume")
    ohlcv = _from_ticks(frame, config, "1min")
    assert "volume" in ohlcv.columns
    assert ohlcv["volume"].iloc[0] == 0.0


def test_ensure_ohlcv_columns_adds_missing_fields() -> None:
    index = pd.date_range("2024-01-01", periods=1, freq="1min", tz="UTC")
    frame = pd.DataFrame({"close": [100.0]}, index=index)
    aligned = _ensure_ohlcv_columns(frame)
    assert list(aligned.columns) == ["open", "high", "low", "close", "volume"]
    assert aligned["volume"].iloc[0] == 0.0


def test_extract_metadata_from_mapping() -> None:
    payload = {"symbol": "ETHUSD", "venue": "coinbase"}
    result = _extract_metadata(payload, ["symbol", "venue", "instrument_type"])
    assert result == {"symbol": "ETHUSD", "venue": "coinbase", "instrument_type": None}


def test_prepare_frame_handles_timestamp_column_without_utc_conversion() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00:00", "2024-01-01T00:01:00"],
            "price": [1.0, 2.0],
        }
    )
    config = MarketNormalizationConfig(expect_utc=False)
    prepared = _prepare_frame(frame, config)
    assert prepared.index.tz is None
    assert list(prepared.index) == list(pd.to_datetime(frame["timestamp"]))


def test_prepare_frame_converts_index_to_utc() -> None:
    index = pd.date_range("2024-01-01T00:00:00", periods=2, freq="1min", tz="Europe/Paris")
    frame = pd.DataFrame({"open": [1.0, 2.0]}, index=index)
    config = MarketNormalizationConfig()
    prepared = _prepare_frame(frame, config)
    assert str(prepared.index.tz) == "UTC"


def test_normalize_without_deduplication_reports_zero_duplicates() -> None:
    index = pd.date_range("2024-01-01T00:00:00Z", periods=3, freq="1min")
    frame = pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.05, 1.15, 1.25],
            "volume": [10.0, 12.0, 14.0],
        },
        index=index,
    )
    config = MarketNormalizationConfig(kind="ohlcv", deduplicate=False)
    result = normalize_market_data(frame, config=config)
    assert result.metadata.duplicates_dropped == 0
