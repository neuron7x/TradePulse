from __future__ import annotations

from domain import Order

from execution.adapters import BinanceFuturesRESTConnector, BybitRESTConnector
from execution.adapters import bybit as bybit_module


def test_binance_futures_stream_url(monkeypatch):
    connector = BinanceFuturesRESTConnector()
    order = Order(symbol="BTCUSDT", side="buy", quantity=0.1, order_type="limit", price=100.0)
    payload = connector._build_place_payload(order, "abc123")
    assert payload["newOrderRespType"] == "RESULT"

    monkeypatch.setattr(connector, "_request", lambda method, path, params=None, signed=False: {"listenKey": "listen"})
    stream_url = connector._stream_url()
    assert stream_url is not None
    assert stream_url.endswith("/listen")


def test_bybit_signing_and_parsing(monkeypatch):
    connector = BybitRESTConnector()
    credentials = connector._resolve_credentials({"api_key": "key", "api_secret": "secret", "recv_window": "5000"})
    connector._credentials = credentials

    monkeypatch.setattr(bybit_module.time, "time", lambda: 1_700_000_000)
    params, payload, headers = connector._sign_request(
        "POST",
        connector._order_endpoint(),
        params={"category": "linear"},
        json_payload={"symbol": "BTCUSDT"},
        headers={},
    )
    assert headers["X-BAPI-API-KEY"] == "key"
    assert "X-BAPI-SIGN" in headers

    sample = {
        "orderId": "123",
        "symbol": "BTCUSDT",
        "side": "Buy",
        "orderType": "Limit",
        "qty": "0.1",
        "price": "100.0",
        "orderStatus": "PartiallyFilled",
        "avgPrice": "100.0",
    }
    order = connector._parse_order(sample)
    assert order.order_id == "123"
    assert order.status.value == "partially_filled"
    assert order.side.value == "buy"
