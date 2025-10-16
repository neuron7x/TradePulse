# SPDX-License-Identifier: MIT
"""Unit tests for the in-memory exchange simulator."""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import count

import pytest

from core.sim import FillEvent, LimitOrderBookSimulator
from domain import Order, OrderSide, OrderStatus, OrderType


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


def test_limit_orders_match_immediately() -> None:
    simulator = LimitOrderBookSimulator(clock=_clock_factory())
    ask = _limit_order("BTCUSDT", OrderSide.SELL, 1, 101)
    simulator.submit(ask)

    bid = _limit_order("BTCUSDT", OrderSide.BUY, 1, 102)
    simulator.submit(bid)

    events = simulator.drain_events()
    assert {type(event) for event in events} == {FillEvent}
    taker = [event for event in events if event.order_id == bid.order_id][0]
    maker = [event for event in events if event.order_id == ask.order_id][0]

    assert taker.liquidity == "taker"
    assert maker.liquidity == "maker"
    assert taker.price == pytest.approx(101)

    assert ask.status is OrderStatus.FILLED
    assert bid.status is OrderStatus.FILLED
    assert simulator.open_orders(symbol="BTCUSDT") == []


def test_partial_fill_leaves_resting_order() -> None:
    simulator = LimitOrderBookSimulator(clock=_clock_factory())
    ask = _limit_order("ETHUSDT", OrderSide.SELL, 2, 100)
    simulator.submit(ask)

    bid = _limit_order("ETHUSDT", OrderSide.BUY, 1, 100)
    simulator.submit(bid)

    events = simulator.drain_events()
    assert len(events) == 2
    assert all(isinstance(event, FillEvent) for event in events)

    assert ask.status is OrderStatus.PARTIALLY_FILLED
    assert pytest.approx(1.0) == ask.remaining_quantity
    assert bid.status is OrderStatus.FILLED

    open_orders = simulator.open_orders(symbol="ETHUSDT")
    assert [order.order_id for order in open_orders] == [ask.order_id]


def test_deterministic_seed_and_clock() -> None:
    def run_scenario() -> tuple[list[str], list[tuple]]:
        simulator = LimitOrderBookSimulator(seed=42, clock=_clock_factory())
        ask_one = _limit_order("SOLUSDT", OrderSide.SELL, 1, 25)
        ask_two = _limit_order("SOLUSDT", OrderSide.SELL, 1, 26)
        bid = _limit_order("SOLUSDT", OrderSide.BUY, 2, 30)

        simulator.submit(ask_one)
        simulator.submit(ask_two)
        simulator.submit(bid)

        events = simulator.drain_events()
        ids = [ask_one.order_id or "", ask_two.order_id or "", bid.order_id or ""]
        serialized = []
        for event in events:
            if isinstance(event, FillEvent):
                payload = (
                    "fill",
                    event.order_id,
                    event.counterparty_id,
                    event.quantity,
                    event.price,
                    event.liquidity,
                    event.timestamp.isoformat(),
                )
            else:
                payload = (
                    "cancel",
                    event.order_id,
                    event.reason,
                    event.timestamp.isoformat(),
                )
            serialized.append(payload)
        return ids, serialized

    ids_one, events_one = run_scenario()
    ids_two, events_two = run_scenario()

    assert ids_one == ids_two
    assert events_one == events_two
