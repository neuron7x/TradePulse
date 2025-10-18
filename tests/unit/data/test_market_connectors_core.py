from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from core.data.connectors.market import (
    BaseMarketDataConnector,
    PolygonMarketDataConnector,
    _decimal_to_float,
    _decimal_to_int,
    _timestamp_to_micros,
)
from core.events import TickEvent


@dataclass
class _Tick:
    symbol: str
    price: Decimal | float | int | None
    volume: Decimal | float | int | None
    timestamp: datetime


class _DummySchemaInfo:
    version_str = "1.0.0"

    def load(self) -> dict[str, object]:
        return {
            "fields": [
                {"name": "event_id", "type": "string"},
                {"name": "schema_version", "type": "string"},
                {"name": "symbol", "type": "string"},
                {"name": "timestamp", "type": "long"},
                {"name": "bid_price", "type": "double"},
                {"name": "ask_price", "type": "double"},
                {"name": "last_price", "type": ["null", "double"], "default": None},
                {"name": "volume", "type": ["null", "long"], "default": None},
            ]
        }


class _DummyRegistry:
    def latest(self, event_type: str, fmt: object) -> _DummySchemaInfo:  # pragma: no cover - simple delegation
        return _DummySchemaInfo()


class _DummyAdapter:
    def __init__(self) -> None:
        self.fetch_payload = []
        self._stream_attempt = 0
        self._closed = False

    async def fetch(self, **kwargs):
        return list(self.fetch_payload)

    async def stream(self, **kwargs):
        self._stream_attempt += 1
        tick_ok = _Tick(
            symbol="ETH-USD",
            price=Decimal("101.5"),
            volume=Decimal("2"),
            timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
        )
        tick_bad = _Tick(
            symbol="ETH-USD",
            price=None,
            volume=Decimal("2"),
            timestamp=datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
        )
        if self._stream_attempt == 1:
            yield tick_ok
            yield tick_bad
            raise RuntimeError("transient failure")
        else:
            yield tick_ok

    async def aclose(self) -> None:
        self._closed = True


def _connector(adapter: _DummyAdapter) -> BaseMarketDataConnector:
    return BaseMarketDataConnector(adapter, schema_registry=_DummyRegistry())


def test_decimal_helpers_normalise_values() -> None:
    assert _decimal_to_float(Decimal("1.5")) == pytest.approx(1.5)
    assert _decimal_to_int(Decimal("5")) == 5
    assert _decimal_to_int(None) is None
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert _timestamp_to_micros(ts) == 1_704_067_200_000_000


@pytest.mark.asyncio
async def test_base_connector_fetch_snapshot_filters_invalid_ticks() -> None:
    adapter = _DummyAdapter()
    adapter.fetch_payload = [
        _Tick(
            symbol="ETH-USD",
            price=Decimal("100.0"),
            volume=Decimal("1"),
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
        _Tick(
            symbol="ETH-USD",
            price=None,
            volume=Decimal("1"),
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
    ]
    connector = _connector(adapter)
    events = await connector.fetch_snapshot()
    assert len(events) == 1
    assert events[0].symbol == "ETH-USD"
    assert connector.dead_letter_queue.peek()
    await connector.aclose()


@pytest.mark.asyncio
async def test_polygon_connector_fetch_aggregates() -> None:
    adapter = _DummyAdapter()
    adapter.fetch_payload = [
        _Tick(
            symbol="BTC-USD",
            price=Decimal("25000.0"),
            volume=Decimal("5"),
            timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
        ),
        _Tick(
            symbol="BTC-USD",
            price=None,
            volume=Decimal("1"),
            timestamp=datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
        ),
    ]
    connector = PolygonMarketDataConnector(
        api_key="dummy",
        adapter=adapter,
        schema_registry=_DummyRegistry(),
    )
    events = await connector.fetch_aggregates(symbol="BTC-USD", start="2024-01-01", end="2024-01-02")
    assert events and events[0].symbol == "BTC-USD"
    assert connector.dead_letter_queue.peek()  # invalid tick is captured
    await connector.aclose()
    assert adapter._closed is True


@pytest.mark.asyncio
async def test_base_connector_stream_ticks_retries_and_records_dead_letters(monkeypatch) -> None:
    adapter = _DummyAdapter()
    connector = _connector(adapter)

    async def _sleep(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _sleep)

    stream = connector.stream_ticks()
    first = await stream.__anext__()
    assert isinstance(first, TickEvent)
    second = await stream.__anext__()
    assert isinstance(second, TickEvent)
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()
    # The invalid tick and the transient failure should both be captured.
    assert len(connector.dead_letter_queue.peek()) >= 1
    await connector.aclose()
