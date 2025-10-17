from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from domain import Order, OrderSide, OrderType

_EXECUTION_ROOT = Path(__file__).resolve().parents[2] / "execution"

if "execution" not in sys.modules:
    package = types.ModuleType("execution")
    package.__path__ = [str(_EXECUTION_ROOT)]
    package.__spec__ = importlib.machinery.ModuleSpec(
        "execution", loader=None, is_package=True
    )
    sys.modules["execution"] = package


def _load_execution_module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    module_path = _EXECUTION_ROOT / f"{name.split('.')[-1]}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise ImportError(f"Unable to load module {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_oms_module = _load_execution_module("execution.oms")
_recon_module = _load_execution_module("execution.reconciliation")

OMSConfig = _oms_module.OMSConfig
OrderManagementSystem = _oms_module.OrderManagementSystem
FillRecord = _recon_module.FillRecord


class DummyRiskController:
    """Simple risk controller capturing fill registrations for assertions."""

    def __init__(self) -> None:
        self.submissions: list[tuple[str, str, float, float]] = []

    def validate_order(self, symbol: str, side: str, qty: float, price: float) -> None:  # noqa: D401 - trivial stub
        return

    def register_fill(self, symbol: str, side: str, qty: float, price: float) -> None:
        self.submissions.append((symbol, side, qty, price))

    def current_position(self, symbol: str) -> float:  # noqa: D401 - trivial stub
        return 0.0

    def current_notional(self, symbol: str) -> float:  # noqa: D401 - trivial stub
        return 0.0

    @property
    def kill_switch(self) -> None:  # noqa: D401 - trivial stub
        return None


@pytest.fixture()
def dummy_risk() -> DummyRiskController:
    return DummyRiskController()


class StubConnector:
    """Deterministic in-memory execution connector for tests."""

    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}
        self._idempotency: dict[str, Order] = {}
        self._next_id = 1
        self.name = "stub"

    def place_order(self, order: Order, *, idempotency_key: str | None = None) -> Order:
        if idempotency_key is not None and idempotency_key in self._idempotency:
            return self._idempotency[idempotency_key]
        submitted = replace(order)
        submitted.mark_submitted(f"stub-{self._next_id:08d}")
        self._next_id += 1
        self._orders[submitted.order_id] = submitted
        if idempotency_key is not None:
            self._idempotency[idempotency_key] = submitted
        return submitted

    def cancel_order(self, order_id: str) -> bool:
        return False

    def fetch_order(self, order_id: str) -> Order:
        return self._orders[order_id]

    def open_orders(self) -> list[Order]:
        return [order for order in self._orders.values() if order.is_active]


@pytest.fixture()
def configured_oms(tmp_path, dummy_risk: DummyRiskController) -> OrderManagementSystem:
    config = OMSConfig(state_path=tmp_path / "oms_state.json")
    connector = StubConnector()
    return OrderManagementSystem(connector, dummy_risk, config)


def _place_order(oms: OrderManagementSystem) -> str:
    order = Order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=2.0,
        price=20_000.0,
        order_type=OrderType.LIMIT,
    )
    oms.submit(order, correlation_id="corr-1")
    placed = oms.process_next()
    assert placed.order_id is not None
    return placed.order_id


def test_reconcile_applies_missing_fills(configured_oms: OrderManagementSystem, dummy_risk: DummyRiskController) -> None:
    order_id = _place_order(configured_oms)
    configured_oms.register_fill(order_id, 0.5, 20_000.0)

    fills = [
        FillRecord(
            order_id=order_id,
            symbol="BTCUSDT",
            quantity=0.7,
            average_price=20_000.0,
            executed_at=datetime.now(timezone.utc),
        ),
        FillRecord(
            order_id=order_id,
            symbol="BTCUSDT",
            quantity=0.8,
            average_price=20_020.0,
            executed_at=datetime.now(timezone.utc),
        ),
    ]

    report = configured_oms.reconcile_with_exchange_fills(fills)

    order = configured_oms._orders[order_id]  # noqa: SLF001 - verify reconciliation updated state
    assert order.filled_quantity == pytest.approx(1.5)
    expected_price = ((0.7 * 20_000.0) + (0.8 * 20_020.0)) / 1.5
    assert order.average_price == pytest.approx(expected_price)
    assert report.corrected == 1
    assert report.matched == 0
    assert not report.missing_in_exchange
    assert not report.missing_in_oms
    assert {item.field for item in report.discrepancies} == {"filled_quantity", "average_price"}
    assert dummy_risk.submissions[-1] == (
        "BTCUSDT",
        "buy",
        pytest.approx(1.0),
        pytest.approx(expected_price),
    )


def test_reconcile_reports_missing_exchange_records(
    configured_oms: OrderManagementSystem,
) -> None:
    order_id = _place_order(configured_oms)
    configured_oms.register_fill(order_id, 1.0, 19_500.0)

    report = configured_oms.reconcile_with_exchange_fills([])

    assert report.missing_in_exchange == [order_id]
    assert not report.corrected
    assert configured_oms._orders[order_id].filled_quantity == pytest.approx(1.0)  # noqa: SLF001


def test_reconcile_flags_unexpected_exchange_fill(configured_oms: OrderManagementSystem) -> None:
    unexpected = FillRecord(
        order_id="Binance-99999999",
        symbol="BTCUSDT",
        quantity=0.25,
        average_price=21_000.0,
    )

    report = configured_oms.reconcile_with_exchange_fills([unexpected])

    assert report.missing_in_oms and report.missing_in_oms[0].order_id == unexpected.order_id
    assert report.corrected == 0
    assert report.matched == 0


def test_reconcile_does_not_reduce_reported_quantity(configured_oms: OrderManagementSystem, dummy_risk: DummyRiskController) -> None:
    order_id = _place_order(configured_oms)
    configured_oms.register_fill(order_id, 1.2, 21_000.0)
    prior_calls = list(dummy_risk.submissions)

    fills = [
        FillRecord(
            order_id=order_id,
            symbol="BTCUSDT",
            quantity=0.7,
            average_price=21_000.0,
        )
    ]

    report = configured_oms.reconcile_with_exchange_fills(fills)

    order = configured_oms._orders[order_id]  # noqa: SLF001 - inspect internal state
    assert order.filled_quantity == pytest.approx(1.2)
    assert any(item.field == "filled_quantity" and not item.corrected for item in report.discrepancies)
    assert dummy_risk.submissions == prior_calls
