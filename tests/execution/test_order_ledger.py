from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from domain import Order, OrderSide, OrderStatus, OrderType
from execution.oms import OMSConfig, OrderManagementSystem
from execution.order_ledger import OrderLedger


class DummyConnector:
    """Minimal connector implementing the execution interface used in tests."""

    name = "dummy"

    def __init__(self) -> None:
        self._counter = 0
        self.placed: dict[str, Order] = {}
        self.cancelled: set[str] = set()

    def place_order(self, order: Order, *, idempotency_key: str) -> Order:
        submitted = replace(order)
        submitted.mark_submitted(f"order-{self._counter}")
        self._counter += 1
        self.placed[submitted.order_id] = submitted
        return submitted

    def cancel_order(self, order_id: str) -> bool:
        self.cancelled.add(order_id)
        return True


class DummyRiskController:
    def validate_order(
        self, symbol: str, side: str, quantity: float, price: float | None
    ) -> None:
        return None

    def register_fill(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> None:
        return None


@pytest.fixture()
def simple_order() -> Order:
    return Order(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        quantity=1.0,
        price=100.0,
        order_type=OrderType.LIMIT,
    )


def test_order_ledger_appends_and_replays(tmp_path: Path, simple_order: Order) -> None:
    ledger_path = tmp_path / "order-ledger.jsonl"
    ledger = OrderLedger(ledger_path)

    bootstrap = ledger.append(
        "bootstrap",
        state_snapshot={
            "orders": [],
            "queue": [],
            "processed": {},
            "correlations": {},
        },
        metadata={"comment": "initial"},
    )
    assert bootstrap.sequence == 1
    assert bootstrap.digest

    order_dict = simple_order.to_dict()
    event = ledger.append(
        "order_recorded",
        order=order_dict,
        state_snapshot={
            "orders": [order_dict],
            "queue": [],
            "processed": {"abc": order_dict["order_id"]},
            "correlations": {order_dict["order_id"] or "temp": "abc"},
        },
    )
    assert event.sequence == 2
    assert event.order_snapshot["symbol"] == "BTC-USD"

    events = list(ledger.replay())
    assert [e.sequence for e in events] == [1, 2]

    # Integrity verification should succeed and latest state must match the append payload
    ledger.verify()
    latest_state = ledger.latest_state()
    assert latest_state is not None
    assert latest_state["orders"][0]["symbol"] == "BTC-USD"


def test_oms_writes_and_recovers_from_order_ledger(tmp_path: Path, simple_order: Order) -> None:
    state_path = tmp_path / "oms-state.json"
    ledger_path = tmp_path / "oms-ledger.jsonl"
    config = OMSConfig(state_path=state_path, ledger_path=ledger_path)
    connector = DummyConnector()
    risk = DummyRiskController()
    oms = OrderManagementSystem(connector, risk, config)

    oms.submit(simple_order, correlation_id="corr-1")
    submitted = oms.process_next()
    assert submitted.order_id is not None

    oms.register_fill(submitted.order_id, submitted.quantity, submitted.price or 100.0)

    events = list(oms._ledger.replay())  # type: ignore[attr-defined]
    assert [event.event for event in events] == [
        "order_queued",
        "order_acknowledged",
        "order_fill_recorded",
    ]

    last_state = events[-1].state_snapshot
    assert last_state is not None
    assert last_state["orders"][0]["status"] == OrderStatus.FILLED.value

    # Remove the persisted state to force ledger-driven recovery.
    state_path.unlink()
    connector2 = DummyConnector()
    risk2 = DummyRiskController()
    oms_recovered = OrderManagementSystem(connector2, risk2, config)

    assert len(oms_recovered._orders) == 1  # type: ignore[attr-defined]
    recovered_order = next(iter(oms_recovered._orders.values()))  # type: ignore[attr-defined]
    assert recovered_order.status is OrderStatus.FILLED
    assert recovered_order.order_id == submitted.order_id

    replayed_events = list(oms_recovered._ledger.replay())  # type: ignore[attr-defined]
    assert replayed_events[-1].event == "state_restored"
    assert replayed_events[-1].metadata["source"] == "ledger"


def test_default_ledger_path_scoped_to_instance(tmp_path: Path) -> None:
    state_dir = tmp_path / "instance"
    state_dir.mkdir()
    state_path = state_dir / "oms-state.json"
    config = OMSConfig(state_path=state_path)
    connector = DummyConnector()
    risk = DummyRiskController()

    oms = OrderManagementSystem(connector, risk, config)

    assert oms._ledger is not None  # type: ignore[attr-defined]
    expected_ledger = state_dir / "oms-state" / "order-ledger.jsonl"
    assert oms._ledger.path == expected_ledger  # type: ignore[attr-defined]
    assert expected_ledger.parent.is_dir()
