# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from domain import Order, OrderSide, OrderType
from execution.adapters import BrokerAdapter, ThrottleConfig
from execution.connectors import SimulatedExchangeConnector
from execution.risk import OrderRateExceeded, RiskError


def _limit_order() -> Order:
    return Order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=0.01,
        price=20_000.0,
        order_type=OrderType.LIMIT,
    )


def test_broker_adapter_switches_between_paper_and_live() -> None:
    paper = SimulatedExchangeConnector()
    live = SimulatedExchangeConnector(sandbox=False)
    adapter = BrokerAdapter(paper=paper, live=live)

    first = adapter.place_order(_limit_order())
    assert first.order_id is not None
    assert first.order_id in paper._orders  # noqa: SLF001 - inspect sandbox state for verification
    assert adapter.sandbox is True
    adapter.ensure_paper()

    adapter.promote_to_live()
    second = adapter.place_order(_limit_order())
    assert second.order_id is not None
    assert second.order_id in live._orders  # noqa: SLF001
    assert adapter.sandbox is False
    adapter.ensure_live()

    with pytest.raises(RiskError):
        adapter.ensure_paper()


def test_broker_adapter_enforces_throttle() -> None:
    paper = SimulatedExchangeConnector()
    live = SimulatedExchangeConnector(sandbox=False)
    adapter = BrokerAdapter(paper=paper, live=live)
    adapter.configure_throttle(ThrottleConfig(max_orders=1, interval_seconds=10.0))

    adapter.place_order(_limit_order())
    with pytest.raises(OrderRateExceeded):
        adapter.place_order(_limit_order())


def test_broker_adapter_kill_switch_blocks_orders() -> None:
    paper = SimulatedExchangeConnector()
    live = SimulatedExchangeConnector(sandbox=False)
    adapter = BrokerAdapter(paper=paper, live=live)
    adapter.kill_switch.trigger("maintenance")

    with pytest.raises(RiskError):
        adapter.place_order(_limit_order())
