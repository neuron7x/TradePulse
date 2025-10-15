# SPDX-License-Identifier: MIT
"""Integration tests for the live execution loop orchestrator."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from datetime import datetime, timezone

from domain import Order, OrderSide, OrderStatus, OrderType
from execution.connectors import BinanceConnector, OrderError
from execution.live_loop import LiveExecutionLoop, LiveLoopConfig
from execution.risk import RiskLimits, RiskManager


class RecoveryConnector(BinanceConnector):
    def __init__(self) -> None:
        super().__init__()
        self.connected = False
        self.placements = 0

    def connect(self, credentials=None) -> None:  # type: ignore[override]
        self.connected = True

    def disconnect(self) -> None:  # type: ignore[override]
        self.connected = False

    def place_order(self, order: Order, *, idempotency_key: str | None = None) -> Order:  # type: ignore[override]
        self.placements += 1
        return super().place_order(order, idempotency_key=idempotency_key)

    def drop_order(self, order_id: str) -> None:
        self._orders.pop(order_id, None)


class FlakyConnector(BinanceConnector):
    def __init__(self) -> None:
        super().__init__()
        self._failures_remaining = 1
        self.reconnects = 0

    def connect(self, credentials=None) -> None:  # type: ignore[override]
        self.reconnects += 1

    def get_positions(self):  # type: ignore[override]
        if self._failures_remaining > 0:
            self._failures_remaining -= 1
            raise ConnectionError("simulated heartbeat disconnect")
        return super().get_positions()


class TerminalSyncConnector(BinanceConnector):
    def __init__(self) -> None:
        super().__init__()
        self._terminal_states: dict[str, tuple[OrderStatus, float | None, float | None, str | None]] = {}

    def mark_terminal(
        self,
        order_id: str,
        *,
        status: OrderStatus,
        filled_quantity: float | None = None,
        average_price: float | None = None,
        rejection: str | None = None,
    ) -> None:
        self._terminal_states[order_id] = (status, filled_quantity, average_price, rejection)

    def fetch_order(self, order_id: str) -> Order:  # type: ignore[override]
        order = self._orders.get(order_id)
        if order is None:
            raise OrderError(f"Unknown order_id: {order_id}")

        override = self._terminal_states.get(order_id)
        status = order.status
        filled = order.filled_quantity
        average_price = order.average_price
        rejection_reason = order.rejection_reason

        if override is not None:
            status, filled_override, average_override, rejection = override
            if filled_override is not None:
                filled = filled_override
            if average_override is not None:
                average_price = average_override
            if rejection is not None:
                rejection_reason = rejection

        remote = Order(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=order.price,
            order_type=order.order_type,
            stop_price=order.stop_price,
            order_id=order.order_id,
            status=status,
            filled_quantity=filled,
            average_price=average_price,
            rejection_reason=rejection_reason,
            created_at=order.created_at,
        )
        object.__setattr__(remote, "updated_at", datetime.now(timezone.utc))
        return remote


@pytest.fixture()
def live_loop_config(tmp_path: Path) -> LiveLoopConfig:
    return LiveLoopConfig(
        state_dir=tmp_path / "state",
        submission_interval=0.05,
        fill_poll_interval=0.05,
        heartbeat_interval=0.1,
        max_backoff=0.2,
    )


def test_live_loop_recovers_and_requeues_orders(live_loop_config: LiveLoopConfig) -> None:
    connector = RecoveryConnector()
    risk_manager = RiskManager(RiskLimits(max_notional=1_000_000, max_position=100))
    loop = LiveExecutionLoop({"binance": connector}, risk_manager, config=live_loop_config)

    loop.start(cold_start=True)
    order = Order(symbol="BTCUSDT", side=OrderSide.BUY, quantity=0.2, price=20_000, order_type=OrderType.LIMIT)
    loop.submit_order("binance", order, correlation_id="ord-1")

    order_id: str | None = None
    for _ in range(50):
        pending = [o for o in loop._contexts["binance"].oms.outstanding() if o.order_id]
        if pending:
            order_id = pending[0].order_id
            break
        time.sleep(0.05)
    assert order_id is not None

    loop.shutdown()

    assert connector.placements == 1
    connector.drop_order(order_id)
    stray = connector.place_order(
        Order(symbol="BTCUSDT", side=OrderSide.SELL, quantity=0.1, price=20_100, order_type=OrderType.LIMIT)
    )

    restart_risk = RiskManager(RiskLimits(max_notional=1_000_000, max_position=100))
    loop_restart = LiveExecutionLoop({"binance": connector}, restart_risk, config=live_loop_config)
    loop_restart.start(cold_start=False)

    for _ in range(50):
        if connector.placements >= 2:
            break
        time.sleep(0.05)
    assert connector.placements >= 2

    adopted_ids = [o.order_id for o in loop_restart._contexts["binance"].oms.outstanding() if o.order_id]
    assert stray.order_id in adopted_ids

    loop_restart.shutdown()


def test_live_loop_emits_reconnect_on_heartbeat_failure(live_loop_config: LiveLoopConfig) -> None:
    connector = FlakyConnector()
    risk_manager = RiskManager(RiskLimits(max_notional=1_000_000, max_position=100))
    loop = LiveExecutionLoop({"binance": connector}, risk_manager, config=live_loop_config)

    reconnect_events: list[tuple[str, int]] = []

    def on_reconnect(venue: str, attempt: int, delay: float, exc: Exception | None) -> None:
        reconnect_events.append((venue, attempt))

    loop.on_reconnect.connect(on_reconnect)
    loop.start(cold_start=True)

    for _ in range(50):
        if reconnect_events:
            break
        time.sleep(0.05)

    loop.shutdown()

    assert reconnect_events, "Expected at least one reconnect event to be emitted"
    venue, attempt = reconnect_events[0]
    assert venue == "binance"
    assert attempt >= 1
    assert connector.reconnects >= 1


@pytest.mark.parametrize(
    "status,rejection",
    [
        (OrderStatus.CANCELLED, None),
        (OrderStatus.REJECTED, "remote reject"),
    ],
)
def test_live_loop_syncs_remote_terminal_state(live_loop_config: LiveLoopConfig, status: OrderStatus, rejection: str | None) -> None:
    connector = TerminalSyncConnector()
    risk_manager = RiskManager(RiskLimits(max_notional=1_000_000, max_position=100))
    loop = LiveExecutionLoop({"binance": connector}, risk_manager, config=live_loop_config)

    try:
        loop.start(cold_start=True)
        order = Order(symbol="BTCUSDT", side=OrderSide.BUY, quantity=0.3, price=20_250, order_type=OrderType.LIMIT)
        loop.submit_order("binance", order, correlation_id=f"terminal-{status.value}")

        tracked_order_id: str | None = None
        for _ in range(100):
            outstanding = [o for o in loop._contexts["binance"].oms.outstanding() if o.order_id]
            if outstanding:
                tracked_order_id = outstanding[0].order_id
                break
            time.sleep(0.05)

        assert tracked_order_id is not None

        connector.mark_terminal(tracked_order_id, status=status, filled_quantity=None, average_price=None, rejection=rejection)

        for _ in range(100):
            stored = loop._contexts["binance"].oms._orders.get(tracked_order_id)
            if stored is not None and stored.status is status:
                break
            time.sleep(0.05)

        stored = loop._contexts["binance"].oms._orders.get(tracked_order_id)
        assert stored is not None
        assert stored.status is status
        assert stored.rejection_reason == rejection
        assert all(o.order_id != tracked_order_id for o in loop._contexts["binance"].oms.outstanding())
        assert tracked_order_id not in loop._order_connector
        assert tracked_order_id not in loop._last_reported_fill
    finally:
        loop.shutdown()
