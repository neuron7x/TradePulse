# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from domain import Order, OrderType
from execution.order import position_sizing
from execution.risk import (
    IdempotentRetryExecutor,
    LimitViolation,
    OrderRateExceeded,
    RiskLimits,
    RiskManager,
    RiskError,
    portfolio_heat,
)


def test_order_defaults_to_market_type() -> None:
    order = Order(symbol="BTCUSD", side="buy", quantity=1.0)
    assert order.order_type == OrderType.MARKET
    assert order.price is None


def test_position_sizing_never_exceeds_balance() -> None:
    balance = 1000.0
    risk = 0.1
    price = 50.0
    size = position_sizing(balance, risk, price)
    assert size <= balance / price
    assert size >= 0.0


def test_portfolio_heat_sums_absolute_exposure() -> None:
    positions = [
        {"qty": 2.0, "price": 100.0},
        {"qty": -1.0, "price": 50.0},
    ]
    heat = portfolio_heat(positions)
    expected = abs(2.0 * 100.0) + abs(-1.0 * 50.0)
    assert heat == pytest.approx(expected, rel=1e-12)


class _TimeStub:
    def __init__(self) -> None:
        self._now = 0.0

    def advance(self, delta: float) -> None:
        self._now += delta

    def __call__(self) -> float:
        return self._now


def test_risk_manager_enforces_position_and_notional_caps() -> None:
    clock = _TimeStub()
    limits = RiskLimits(max_notional=100.0, max_position=5.0, max_orders_per_interval=5, interval_seconds=1.0)
    manager = RiskManager(limits, time_source=clock)

    manager.validate_order("BTC", "buy", qty=2.0, price=20.0)
    manager.register_fill("BTC", "buy", qty=2.0, price=20.0)
    assert manager.current_position("BTC") == pytest.approx(2.0)

    with pytest.raises(LimitViolation):
        manager.validate_order("BTC", "buy", qty=5.0, price=25.0)

    with pytest.raises(LimitViolation):
        manager.validate_order("BTC", "sell", qty=8.0, price=40.0)


def test_risk_manager_rate_limiter_blocks_excess_orders() -> None:
    clock = _TimeStub()
    limits = RiskLimits(max_notional=1_000.0, max_position=100.0, max_orders_per_interval=2, interval_seconds=1.0)
    manager = RiskManager(limits, time_source=clock)

    manager.validate_order("ETH", "buy", qty=1.0, price=10.0)
    manager.validate_order("ETH", "buy", qty=1.0, price=10.0)
    with pytest.raises(OrderRateExceeded):
        manager.validate_order("ETH", "buy", qty=1.0, price=10.0)

    clock.advance(1.1)
    manager.validate_order("ETH", "buy", qty=1.0, price=10.0)


def test_kill_switch_blocks_all_orders() -> None:
    manager = RiskManager(RiskLimits(max_notional=100.0, max_position=10.0))
    manager.kill_switch.trigger("test")
    with pytest.raises(RiskError):
        manager.validate_order("BTC", "buy", qty=1.0, price=10.0)


def test_idempotent_retry_executor_retries_and_caches() -> None:
    executor = IdempotentRetryExecutor()
    attempts: list[int] = []

    def flaky(attempt: int) -> str:
        attempts.append(attempt)
        if attempt < 2:
            raise RuntimeError("boom")
        return "ok"

    result = executor.run("order-1", flaky, retries=3, retry_exceptions=(RuntimeError,))
    assert result == "ok"
    assert attempts == [1, 2]
    # Second call should be cached and not invoke callable
    assert executor.run("order-1", flaky, retries=3) == "ok"
