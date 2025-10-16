from __future__ import annotations

from collections.abc import Iterator

import pytest
from prometheus_client import CollectorRegistry

from core.utils import metrics as metrics_module
from core.utils.metrics import MetricsCollector
from domain import Order, OrderSide, OrderType
from execution.adapters.base import RESTWebSocketConnector
from execution.adapters.coinbase import CoinbaseRESTConnector


class DummyConnector(RESTWebSocketConnector):
    """Minimal connector for exercising trade latency instrumentation."""

    def __init__(self) -> None:
        super().__init__(name="dummy", base_url="https://example.com", sandbox=True)

    def _resolve_credentials(self, credentials):  # type: ignore[override]
        return {}

    def _sign_request(self, method, path, *, params, json_payload, headers):  # type: ignore[override]
        return params, json_payload, headers

    def _order_endpoint(self) -> str:  # type: ignore[override]
        return "/orders"

    def _build_place_payload(self, order: Order, idempotency_key: str | None) -> dict:  # type: ignore[override]
        return {"symbol": order.symbol, "side": order.side.value, "type": order.order_type.value}

    def _parse_order(self, payload, *, original: Order | None = None) -> Order:  # type: ignore[override]
        assert original is not None
        return Order(
            symbol=original.symbol,
            side=original.side,
            quantity=original.quantity,
            price=original.price,
            order_type=original.order_type,
            order_id="dummy-1",
        )

    def _request(self, method, path, *, params=None, json_payload=None, signed=False, weight=1):  # type: ignore[override]
        return {"orderId": "dummy-1"}

    def _cancel_endpoint(self, order_id: str):  # type: ignore[override]
        return "/orders", {"orderId": order_id}

    def _fetch_endpoint(self, order_id: str):  # type: ignore[override]
        return "/orders", {"orderId": order_id}

    def _open_orders_endpoint(self):  # type: ignore[override]
        return "/orders", {}

    def _positions_endpoint(self):  # type: ignore[override]
        return "/positions", {}

    def _parse_positions(self, payload):  # type: ignore[override]
        return []


def _patch_perf_counter(monkeypatch: pytest.MonkeyPatch, target: str, values: Iterator[float]) -> None:
    monkeypatch.setattr(target, lambda: next(values))


def test_rest_connector_emits_trade_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = CollectorRegistry()
    collector = MetricsCollector(registry)
    monkeypatch.setattr(metrics_module, "_collector", collector, raising=False)
    observed: list[dict] = []
    original = collector.observe_trade_latency_ms

    def capture(**kwargs):
        observed.append(kwargs)
        original(**kwargs)

    monkeypatch.setattr(collector, "observe_trade_latency_ms", capture)
    _patch_perf_counter(monkeypatch, "execution.adapters.base.time.perf_counter", iter([100.0, 100.25]))

    connector = DummyConnector()
    order = Order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=1.0,
        price=25_000.0,
        order_type=OrderType.LIMIT,
    )

    submitted = connector.place_order(order)

    assert submitted.order_id == "dummy-1"
    assert observed, "trade latency should be recorded"
    assert observed[0]["latency_ms"] == pytest.approx(250.0)

    latency_sum = registry.get_sample_value(
        "trade_latency_ms_sum",
        {"exchange": "dummy", "adapter": "DummyConnector", "symbol": "BTCUSDT", "order_type": "limit"},
    )
    assert latency_sum == pytest.approx(250.0)


def test_coinbase_connector_emits_trade_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = CollectorRegistry()
    collector = MetricsCollector(registry)
    monkeypatch.setattr(metrics_module, "_collector", collector, raising=False)
    observed: list[dict] = []
    original = collector.observe_trade_latency_ms

    def capture(**kwargs):
        observed.append(kwargs)
        original(**kwargs)

    monkeypatch.setattr(collector, "observe_trade_latency_ms", capture)
    _patch_perf_counter(monkeypatch, "execution.adapters.coinbase.time.perf_counter", iter([200.0, 200.12]))

    connector = CoinbaseRESTConnector()

    def fake_request(*args, **kwargs):
        return {
            "order": {
                "order_id": "cb-1",
                "product_id": "BTC-USD",
                "side": "BUY",
                "size": "1.0",
                "price": "25000",
                "status": "OPEN",
                "order_type": "limit",
            }
        }

    monkeypatch.setattr(connector, "_request", fake_request)

    order = Order(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        quantity=1.0,
        price=25_000.0,
        order_type=OrderType.LIMIT,
    )

    submitted = connector.place_order(order)

    assert submitted.order_id == "cb-1"
    assert observed, "trade latency should be recorded"
    assert observed[0]["latency_ms"] == pytest.approx(120.0)

    latency_sum = registry.get_sample_value(
        "trade_latency_ms_sum",
        {"exchange": "coinbase", "adapter": "CoinbaseRESTConnector", "symbol": "BTC-USD", "order_type": "limit"},
    )
    assert latency_sum == pytest.approx(120.0)
