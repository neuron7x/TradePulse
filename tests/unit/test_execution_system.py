# SPDX-License-Identifier: MIT
"""Tests for the execution order management stack."""

from __future__ import annotations

from datetime import timedelta

import pytest

from domain import Order, OrderSide, OrderType
from execution.algorithms import POVAlgorithm, TWAPAlgorithm, VWAPAlgorithm, aggregate_fills
from execution.connectors import BinanceConnector
from execution.normalization import NormalizationError, SymbolNormalizer, SymbolSpecification
from execution.oms import OMSConfig, OrderManagementSystem
from execution.risk import RiskLimits, RiskManager


@pytest.fixture()
def risk_manager() -> RiskManager:
    limits = RiskLimits(max_notional=1_000_000, max_position=100)
    return RiskManager(limits)


def test_oms_idempotent_submission_and_recovery(tmp_path, risk_manager: RiskManager) -> None:
    state_path = tmp_path / "oms_state.json"
    config = OMSConfig(state_path=state_path)
    connector = BinanceConnector()
    oms = OrderManagementSystem(connector, risk_manager, config)

    order = Order(symbol="BTCUSDT", side=OrderSide.BUY, quantity=1.0, price=20_000, order_type=OrderType.LIMIT)

    first = oms.submit(order, correlation_id="abc123")
    assert first is order
    processed = oms.process_next()
    assert processed.order_id is not None

    second = oms.submit(order, correlation_id="abc123")
    assert second.order_id == processed.order_id
    assert not oms._queue  # noqa: SLF001 - validate internal queue is untouched

    # Simulate restart and recover state
    oms_reload = OrderManagementSystem(connector, risk_manager, config)
    assert processed.order_id in {o.order_id for o in oms_reload.outstanding()}


def test_oms_register_fill_updates_risk(tmp_path, risk_manager: RiskManager) -> None:
    state_path = tmp_path / "fills_state.json"
    config = OMSConfig(state_path=state_path)
    connector = BinanceConnector()
    oms = OrderManagementSystem(connector, risk_manager, config)

    order = Order(symbol="BTCUSDT", side=OrderSide.BUY, quantity=2.0, price=25_000, order_type=OrderType.LIMIT)
    oms.submit(order, correlation_id="fill-1")
    placed = oms.process_next()
    assert placed.order_id is not None

    updated = oms.register_fill(placed.order_id, 1.0, 25_000)
    assert updated.filled_quantity == pytest.approx(1.0)
    assert updated.status.name == "PARTIALLY_FILLED"
    assert risk_manager.current_position("BTCUSDT") == pytest.approx(1.0)

    updated = oms.register_fill(placed.order_id, 1.0, 25_100)
    assert updated.status.name == "FILLED"
    assert risk_manager.current_position("BTCUSDT") == pytest.approx(2.0)


def test_execution_algorithms_split_quantities() -> None:
    parent = Order(symbol="BTCUSDT", side=OrderSide.SELL, quantity=4.0, price=21_500, order_type=OrderType.LIMIT)
    twap = TWAPAlgorithm(duration=timedelta(minutes=4), slices=4)
    children = twap.schedule(parent)
    assert len(children) == 4
    assert sum(child.order.quantity for child in children) == pytest.approx(parent.quantity)

    vwap = VWAPAlgorithm(volume_profile=[1, 2, 1], duration=timedelta(minutes=3))
    vwap_children = vwap.schedule(parent)
    assert len(vwap_children) == 3
    assert sum(child.order.quantity for child in vwap_children) == pytest.approx(parent.quantity)

    pov = POVAlgorithm(participation=0.25, forecast_volume=[4, 4, 8], duration=timedelta(minutes=3))
    pov_children = pov.schedule(parent)
    assert len(pov_children) == 3
    assert sum(child.order.quantity for child in pov_children) == pytest.approx(parent.quantity)

    for child in pov_children:
        child.order.record_fill(child.order.quantity, parent.price)
    assert aggregate_fills(pov_children) == pytest.approx(parent.quantity)


def test_symbol_normalizer_enforces_constraints() -> None:
    specs = {
        "BTCUSDT": SymbolSpecification("BTCUSDT", min_qty=0.001, min_notional=10, step_size=0.001, tick_size=0.1)
    }
    normalizer = SymbolNormalizer(specifications=specs)

    rounded_qty = normalizer.round_quantity("BTCUSDT", 0.0014)
    assert rounded_qty == pytest.approx(0.001)

    rounded_price = normalizer.round_price("BTCUSDT", 20000.123)
    assert rounded_price == pytest.approx(20000.1)

    normalizer.validate("BTCUSDT", 0.01, 20_000)

    with pytest.raises(NormalizationError):
        normalizer.validate("BTCUSDT", 0.0001, 20_000)

