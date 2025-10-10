# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib
import importlib.util
import sys
import warnings
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from pydantic import ValidationError

from core.data.models import (
    PYDANTIC_V2,
    AggregateMetric,
    DataKind,
    InstrumentType,
    MarketMetadata,
    OHLCVBar,
    PriceTick,
    _FrozenModel,
)


def test_price_tick_normalises_timestamp_and_decimal_conversion() -> None:
    naive = datetime(2024, 1, 1, 12, 0, 0)
    tick = PriceTick.create(
        symbol="BTCUSD",
        venue="BINANCE",
        price="100.5",
        volume="0.25",
        timestamp=naive,
        instrument_type=InstrumentType.FUTURES,
        trade_id=" 12345 ",
    )

    assert tick.kind is DataKind.TICK
    assert tick.timestamp.tzinfo is timezone.utc
    assert tick.instrument_type is InstrumentType.FUTURES
    assert isinstance(tick.price, Decimal)
    assert tick.price == Decimal("100.5")
    assert tick.volume == Decimal("0.25")
    assert tick.trade_id == "12345"


def test_price_tick_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        PriceTick.create(
            symbol="BTCUSD",
            venue="BINANCE",
            price=-1,
            volume=1,
            timestamp=datetime.now(timezone.utc),
        )

    with pytest.raises(ValidationError):
        PriceTick.create(
            symbol="BTCUSD",
            venue="BINANCE",
            price=True,  # type: ignore[arg-type]
            volume=1,
            timestamp=datetime.now(timezone.utc),
        )


def test_price_tick_supports_decimal_inputs_and_timestamp_conversion() -> None:
    tick = PriceTick.create(
        symbol="ETHUSD",
        venue="COINBASE",
        price=Decimal("1200.5"),
        volume=Decimal("1.25"),
        timestamp=1_700_000_000,
        trade_id=" tid ",
    )

    assert tick.kind is DataKind.TICK
    assert tick.ts == pytest.approx(1_700_000_000.0)
    assert tick.trade_id == "tid"


def test_market_metadata_rejects_blank_inputs() -> None:
    with pytest.raises(ValidationError):
        MarketMetadata(symbol=" ", venue="BINANCE")

    with pytest.raises(ValidationError):
        MarketMetadata(symbol="BTCUSD", venue=" ")


def test_ohlcv_bar_enforces_bounds() -> None:
    bar = OHLCVBar(
        metadata=MarketMetadata(symbol="ETHUSD", venue="BINANCE"),
        timestamp=datetime.now(timezone.utc),
        open="1000",
        high="1100",
        low="900",
        close="950",
        volume="12.5",
        interval_seconds=60,
    )

    assert bar.kind is DataKind.OHLCV
    assert bar.volume == Decimal("12.5")

    with pytest.raises(ValueError):
        OHLCVBar(
            metadata=bar.metadata,
            timestamp=bar.timestamp,
            open="1000",
            high="850",
            low="900",
            close="950",
            volume="12",
            interval_seconds=60,
        )

    with pytest.raises(ValueError):
        OHLCVBar(
            metadata=bar.metadata,
            timestamp=bar.timestamp,
            open="1200",
            high="1300",
            low="900",
            close="850",
            volume="1",
            interval_seconds=60,
        )


def test_aggregate_metric_validates_inputs() -> None:
    metric = AggregateMetric(
        metadata=MarketMetadata(symbol="SOLUSD", venue="BINANCE"),
        timestamp=datetime.now(timezone.utc),
        metric="  vwap  ",
        value="123.45",
        window_seconds=300,
    )

    assert metric.metric == "vwap"
    assert metric.kind is DataKind.AGGREGATE
    assert metric.value == Decimal("123.45")

    with pytest.raises(ValueError):
        AggregateMetric(
            metadata=metric.metadata,
            timestamp=metric.timestamp,
            metric=" ",
            value="10",
            window_seconds=60,
        )


def test_aggregate_metric_coerces_kind_and_epoch_timestamp() -> None:
    metric = AggregateMetric(
        metadata=MarketMetadata(symbol="DOTUSD", venue="KRAKEN"),
        timestamp=1_690_000_000,
        metric="spread",
        value="1.23",
        window_seconds=120,
        kind="aggregate",
    )

    assert metric.kind is DataKind.AGGREGATE
    assert metric.timestamp.tzinfo is timezone.utc
    assert metric.ts == pytest.approx(1_690_000_000.0)


def test_price_tick_factory_supports_forced_v1_compatibility(monkeypatch) -> None:
    import core.data.models as models
    from pydantic.warnings import PydanticDeprecatedSince20

    monkeypatch.setenv("TRADEPULSE_FORCE_PYDANTIC_V1", "1")
    module_name = "core.data.models_forced_v1"
    spec = importlib.util.spec_from_file_location(module_name, models.__file__)
    assert spec and spec.loader
    forced_module = importlib.util.module_from_spec(spec)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PydanticDeprecatedSince20)
        sys.modules[module_name] = forced_module
        spec.loader.exec_module(forced_module)

    try:
        tick = forced_module.PriceTick.create(
            symbol="ADAUSD",
            venue="BINANCE",
            price=1.23,
            volume=0.5,
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            trade_id=" tid ",
        )
        assert tick.trade_id == "tid"
        assert tick.volume == Decimal("0.5")
        assert tick.price == Decimal("1.23")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PydanticDeprecatedSince20)
            assert "1.23" in tick.json()

        bar = forced_module.OHLCVBar(
            metadata=forced_module.MarketMetadata(symbol="SOLUSD", venue="BINANCE"),
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=1,
            high=2,
            low=1,
            close=1.5,
            volume=3,
            interval_seconds=60,
            kind="ohlcv",
        )
        assert bar.kind is forced_module.DataKind.OHLCV

        metric = forced_module.AggregateMetric(
            metadata=bar.metadata,
            timestamp=bar.timestamp,
            metric=" vw ",
            value=2.5,
            window_seconds=30,
            kind="aggregate",
        )
        assert metric.metric == "vw"
        assert metric.kind is forced_module.DataKind.AGGREGATE
    finally:
        monkeypatch.delenv("TRADEPULSE_FORCE_PYDANTIC_V1", raising=False)
        sys.modules.pop(module_name, None)


def test_frozen_model_serializer_converts_decimals_to_strings() -> None:
    assert _FrozenModel._serialize_decimal(_FrozenModel, Decimal("42")) == "42"
    assert _FrozenModel._serialize_decimal(_FrozenModel, "noop") == "noop"
