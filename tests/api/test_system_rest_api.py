from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from application.api.system_access import create_system_app
from application.system import ExchangeAdapterConfig, TradePulseSystem, TradePulseSystemConfig
from domain import Order
from execution.connectors import SimulatedExchangeConnector


class _EagerFillConnector(SimulatedExchangeConnector):
    """Connector that immediately fills orders and exposes static positions."""

    def __init__(self, *, positions: list[dict[str, Any]] | None = None) -> None:
        super().__init__()
        self._positions = positions or [
            {
                "symbol": "BTCUSD",
                "quantity": 0.5,
                "entry_price": 48000.0,
                "current_price": 50000.0,
                "unrealized_pnl": 1000.0,
            }
        ]

    def get_positions(self) -> list[dict[str, Any]]:
        return list(self._positions)

    def place_order(self, order: Order, *, idempotency_key: str | None = None) -> Order:
        placed = super().place_order(order, idempotency_key=idempotency_key)
        fill_price = order.price or 50000.0
        placed.record_fill(order.quantity, fill_price)
        return placed


@pytest.fixture()
def system() -> TradePulseSystem:
    connector = _EagerFillConnector()
    config = TradePulseSystemConfig(
        venues=(ExchangeAdapterConfig(name="dummy", connector=connector),)
    )
    return TradePulseSystem(config)


@pytest.fixture()
def client(system: TradePulseSystem) -> TestClient:
    app = create_system_app(system)
    return TestClient(app)


def test_status_endpoint_reports_running_state(client: TestClient) -> None:
    response = client.get("/api/v1/status")
    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "running"
    assert isinstance(payload["uptime_seconds"], (int, float))
    assert payload["uptime_seconds"] >= 0
    assert isinstance(payload["version"], str) and payload["version"]


def test_positions_endpoint_returns_normalised_payload(client: TestClient) -> None:
    response = client.get("/api/v1/positions")
    payload = response.json()

    assert response.status_code == 200
    assert "positions" in payload
    assert payload["positions"], "expected positions to be returned"
    position = payload["positions"][0]
    assert position["symbol"] == "BTCUSD"
    assert pytest.approx(position["quantity"], rel=1e-6) == 0.5
    assert pytest.approx(position["entry_price"], rel=1e-6) == 48000.0
    assert pytest.approx(position["current_price"], rel=1e-6) == 50000.0
    assert pytest.approx(position["unrealized_pnl"], rel=1e-6) == 1000.0


def test_order_submission_returns_filled_order(client: TestClient) -> None:
    response = client.post(
        "/api/v1/orders",
        json={
            "symbol": "ETHUSD",
            "side": "buy",
            "order_type": "limit",
            "quantity": 1.0,
            "price": 1800.0,
        },
    )

    payload = response.json()
    assert response.status_code == 201
    assert payload["status"] == "filled"
    assert pytest.approx(payload["filled_quantity"], rel=1e-6) == 1.0
    assert pytest.approx(payload["average_price"], rel=1e-6) == 1800.0
    assert payload["order_id"]


def test_limit_order_requires_price_validation(client: TestClient) -> None:
    response = client.post(
        "/api/v1/orders",
        json={
            "symbol": "ETHUSD",
            "side": "buy",
            "order_type": "limit",
            "quantity": 1.0,
        },
    )

    assert response.status_code == 422
    detail = response.json()
    assert detail["detail"][0]["msg"].startswith("Value error")


def test_market_order_requires_reference_price(client: TestClient) -> None:
    response = client.post(
        "/api/v1/orders",
        json={
            "symbol": "ETHUSD",
            "side": "buy",
            "quantity": 0.5,
        },
    )

    assert response.status_code == 422
    detail = response.json()
    assert any(
        "reference_price" in error.get("msg", "") for error in detail.get("detail", [])
    )


def test_market_order_uses_reference_price_for_risk_validation(
    client: TestClient, system: TradePulseSystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, float] = {}

    def _capture(symbol: str, side: str, qty: float, price: float) -> None:
        captured.update(symbol=symbol, side=side, qty=qty, price=price)

    monkeypatch.setattr(system.risk_manager, "validate_order", _capture)

    response = client.post(
        "/api/v1/orders",
        json={
            "symbol": "ETHUSD",
            "side": "buy",
            "quantity": 0.25,
            "reference_price": 1900.0,
        },
    )

    assert response.status_code == 201
    assert captured["price"] == 1900.0

