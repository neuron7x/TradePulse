from __future__ import annotations

import httpx

from domain import Order, OrderType
from interfaces.execution.binance import BinanceExecutionConnector


def test_binance_stop_order_payload_and_parsing() -> None:
    captured_params: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/v3/order":
            captured_params.update(dict(request.url.params))
            return httpx.Response(
                200,
                json={"orderId": 8001, "status": "NEW", "executedQty": "0"},
            )
        if request.method == "GET" and request.url.path == "/api/v3/order":
            return httpx.Response(
                200,
                json={
                    "symbol": "BTCUSDT",
                    "orderId": 8001,
                    "status": "NEW",
                    "type": "STOP_LOSS",
                    "origQty": "1",
                    "stopPrice": "30000",
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    connector = BinanceExecutionConnector(
        sandbox=True,
        enable_stream=False,
        transport=httpx.MockTransport(handler),
    )
    connector.connect(credentials={"API_KEY": "key", "API_SECRET": "secret"})

    stop_order = Order(
        symbol="BTCUSDT",
        side="buy",
        quantity=1.0,
        order_type=OrderType.STOP,
        stop_price=30000.0,
    )

    placed = connector.place_order(stop_order)
    assert placed.order_type is OrderType.STOP
    assert captured_params["type"] == "STOP_LOSS"
    assert captured_params["stopPrice"] == "30000"

    fetched = connector.fetch_order("8001")
    assert fetched.order_type is OrderType.STOP
    assert fetched.stop_price == 30000.0
    assert fetched.price is None


def test_binance_stop_limit_order_payload_and_parsing() -> None:
    captured_params: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/v3/order":
            captured_params.update(dict(request.url.params))
            return httpx.Response(
                200,
                json={"orderId": 8002, "status": "NEW", "executedQty": "0"},
            )
        if request.method == "GET" and request.url.path == "/api/v3/order":
            return httpx.Response(
                200,
                json={
                    "symbol": "BTCUSDT",
                    "orderId": 8002,
                    "status": "NEW",
                    "type": "STOP_LOSS_LIMIT",
                    "origQty": "2",
                    "price": "30500",
                    "stopPrice": "30000",
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    connector = BinanceExecutionConnector(
        sandbox=True,
        enable_stream=False,
        transport=httpx.MockTransport(handler),
    )
    connector.connect(credentials={"API_KEY": "key", "API_SECRET": "secret"})

    stop_limit_order = Order(
        symbol="BTCUSDT",
        side="sell",
        quantity=2.0,
        order_type=OrderType.STOP_LIMIT,
        price=30500.0,
        stop_price=30000.0,
    )

    placed = connector.place_order(stop_limit_order)
    assert placed.order_type is OrderType.STOP_LIMIT
    assert captured_params["type"] == "STOP_LOSS_LIMIT"
    assert captured_params["stopPrice"] == "30000"
    assert captured_params["price"] == "30500"
    assert captured_params["timeInForce"] == "GTC"

    fetched = connector.fetch_order("8002")
    assert fetched.order_type is OrderType.STOP_LIMIT
    assert fetched.stop_price == 30000.0
    assert fetched.price == 30500.0
