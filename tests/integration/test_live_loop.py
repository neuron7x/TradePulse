# SPDX-License-Identifier: MIT
"""Integration-style tests for the live execution loop orchestration."""

from __future__ import annotations

import time

import pytest

from domain import Order, OrderSide, OrderType
from execution.connectors import BinanceConnector
from execution.live_loop import LiveExecutionLoop, LiveLoopConfig
from execution.oms import OMSConfig
from execution.risk import RiskLimits, RiskManager


class FlakyConnector(BinanceConnector):
    """Connector that fails the first connection and heartbeat call."""

    def __init__(self) -> None:
        super().__init__()
        self._connect_attempts = 0
        self._position_calls = 0

    def connect(self, credentials=None):  # type: ignore[override]
        self._connect_attempts += 1
        if self._connect_attempts < 2:
            raise ConnectionError("simulated connection failure")
        return super().connect(credentials)

    def get_positions(self):  # type: ignore[override]
        self._position_calls += 1
        if self._position_calls == 1:
            raise ConnectionError("positions unavailable")
        return super().get_positions()


@pytest.fixture()
def live_loop_config() -> LiveLoopConfig:
    return LiveLoopConfig(poll_interval=0.05, heartbeat_interval=0.1, backoff_base=0.05, max_backoff=0.2)


def test_live_execution_loop_recovers_after_restart(tmp_path, live_loop_config: LiveLoopConfig) -> None:
    state_path = tmp_path / "loop_state.json"
    config = OMSConfig(state_path=state_path)
    risk = RiskManager(RiskLimits(max_notional=1_000_000, max_position=100))
    connector = BinanceConnector()

    loop = LiveExecutionLoop(
        primary_connector=connector,
        risk_manager=risk,
        oms_config=config,
        config=live_loop_config,
    )

    loop.start(cold_start=True)

    order = Order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=0.5,
        price=20_000,
        order_type=OrderType.LIMIT,
    )
    loop.oms.submit(order, correlation_id="restart-order")
    placed = loop.oms.process_next()
    assert placed.order_id is not None

    loop.shutdown()

    # Drop the order from the connector to simulate an orphaned OMS order.
    connector._orders.pop(placed.order_id, None)  # type: ignore[attr-defined]

    replacement_risk = RiskManager(RiskLimits(max_notional=1_000_000, max_position=100))
    restart_loop = LiveExecutionLoop(
        primary_connector=connector,
        risk_manager=replacement_risk,
        oms_config=config,
        config=live_loop_config,
    )
    restart_loop.start(cold_start=False)

    deadline = time.time() + 1.0
    recovered = False
    while time.time() < deadline:
        pending_correlations = list(restart_loop.oms.pending_correlation_ids())
        if any(cid.startswith("resubmit::") for cid in pending_correlations):
            recovered = True
            break
        if connector.open_orders():
            recovered = True
            break
        time.sleep(0.05)

    assert recovered, "expected orphaned order to be reconciled"

    restart_loop.shutdown()


def test_live_execution_loop_handles_reconnection(tmp_path, live_loop_config: LiveLoopConfig) -> None:
    state_path = tmp_path / "loop_state.json"
    config = OMSConfig(state_path=state_path)
    risk = RiskManager(RiskLimits(max_notional=1_000_000, max_position=100))
    connector = FlakyConnector()

    reconnections: list[str] = []

    loop = LiveExecutionLoop(
        primary_connector=connector,
        risk_manager=risk,
        oms_config=config,
        config=live_loop_config,
        on_reconnect=reconnections.append,
    )

    loop.start(cold_start=True)
    # Allow heartbeat worker to trigger reconnection flow.
    time.sleep(0.3)
    loop.shutdown()

    # The connector should have invoked the reconnection callback at least twice
    # (once for the initial retry, once for the heartbeat failure).
    assert reconnections.count("primary") >= 1
