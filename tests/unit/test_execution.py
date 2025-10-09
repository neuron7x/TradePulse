# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from execution.order import Order, position_sizing
from execution.risk import portfolio_heat


def test_order_defaults_to_market_type() -> None:
    order = Order(side="buy", qty=1.0)
    assert order.type == "market"
    assert order.price is None


def test_position_sizing_never_exceeds_balance() -> None:
    balance = 1000.0
    risk = 0.1
    price = 50.0
    size = position_sizing(balance, risk, price)
    assert size <= balance / price
    assert size >= 0.0


def test_position_sizing_handles_denormal_budget() -> None:
    balance = 1.0
    risk = 0.25
    price = 1e308
    size = position_sizing(balance, risk, price)

    assert size * price <= balance * risk


def test_position_sizing_rejects_invalid_leverage() -> None:
    with pytest.raises(ValueError, match="max_leverage must be positive"):
        position_sizing(1000.0, 0.1, 50.0, max_leverage=0.0)


def test_position_sizing_treats_nan_inputs_as_zero() -> None:
    size = position_sizing(float("nan"), float("nan"), 100.0)
    assert size == 0.0


def test_portfolio_heat_sums_absolute_exposure() -> None:
    positions = [
        {"qty": 2.0, "price": 100.0},
        {"qty": -1.0, "price": 50.0},
    ]
    heat = portfolio_heat(positions)
    expected = abs(2.0 * 100.0) + abs(-1.0 * 50.0)
    assert heat == pytest.approx(expected, rel=1e-12)
