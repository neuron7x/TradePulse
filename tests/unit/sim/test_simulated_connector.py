# SPDX-License-Identifier: MIT
"""Integration tests for the simulated execution connector."""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import count

from core.sim import CancelEvent, FillEvent, LimitOrderBookSimulator
from domain import Order, OrderSide, OrderType
from execution.adapters import SimulatedConnector


def _clock_factory():
    ticks = count()

    def _clock() -> datetime:
        return datetime.fromtimestamp(next(ticks), timezone.utc)

    return _clock


def _limit_order(symbol: str, side: OrderSide, quantity: float, price: float) -> Order:
    return Order(
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        order_type=OrderType.LIMIT,
    )


def test_simulated_connector_routes_orders() -> None:
    simulator = LimitOrderBookSimulator(clock=_clock_factory())
    connector = SimulatedConnector(simulator)

    ask = _limit_order("BTCUSDT", OrderSide.SELL, 1, 101)
    connector.place_order(ask)

    bid = _limit_order("BTCUSDT", OrderSide.BUY, 1, 102)
    connector.place_order(bid)

    events = connector.drain_events()
    assert len(events) == 2
    assert all(isinstance(event, FillEvent) for event in events)


def test_simulated_connector_supports_idempotency() -> None:
    connector = SimulatedConnector(LimitOrderBookSimulator(clock=_clock_factory()))
    order = _limit_order("ETHUSDT", OrderSide.BUY, 0.5, 1800)

    first = connector.place_order(order, idempotency_key="client-1")
    second = connector.place_order(order, idempotency_key="client-1")

    assert first is second


def test_simulated_connector_cancel_emits_event() -> None:
    simulator = LimitOrderBookSimulator(clock=_clock_factory())
    connector = SimulatedConnector(simulator)

    ask = _limit_order("SOLUSDT", OrderSide.SELL, 1, 35)
    connector.place_order(ask)

    assert connector.cancel_order(ask.order_id or "") is True
    cancel_events = [event for event in connector.drain_events() if isinstance(event, CancelEvent)]
    assert cancel_events and cancel_events[0].order_id == ask.order_id
