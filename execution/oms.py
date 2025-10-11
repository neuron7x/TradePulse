# SPDX-License-Identifier: MIT
"""Order management system with persistence and idempotent queues."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Iterable, MutableMapping

from domain import Order
from interfaces.execution import RiskController

from .connectors import ExecutionConnector, OrderError


@dataclass(slots=True)
class QueuedOrder:
    """Order request paired with its idempotency key."""

    correlation_id: str
    order: Order


@dataclass(slots=True)
class OMSConfig:
    """Configuration for :class:`OrderManagementSystem`."""

    state_path: Path
    auto_persist: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.state_path, Path):
            object.__setattr__(self, "state_path", Path(self.state_path))


class OrderManagementSystem:
    """Queue-based OMS with state recovery."""

    def __init__(
        self,
        connector: ExecutionConnector,
        risk_controller: RiskController,
        config: OMSConfig,
    ) -> None:
        self.connector = connector
        self.risk = risk_controller
        self.config = config
        self._queue: Deque[QueuedOrder] = deque()
        self._orders: MutableMapping[str, Order] = {}
        self._processed: Dict[str, str] = {}
        self._load_state()

    # ------------------------------------------------------------------
    # Persistence helpers
    def _state_payload(self) -> dict:
        return {
            "orders": [self._serialize_order(order) for order in self._orders.values()],
            "queue": [
                {
                    "correlation_id": item.correlation_id,
                    "order": self._serialize_order(item.order),
                }
                for item in self._queue
            ],
            "processed": self._processed,
        }

    def _persist_state(self) -> None:
        if not self.config.auto_persist:
            return
        path = self.config.state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._state_payload(), indent=2, sort_keys=True))

    def _load_state(self) -> None:
        path = self.config.state_path
        if not path.exists():
            return
        payload = json.loads(path.read_text())
        self._orders = {
            order.order_id: order
            for order in self._restore_orders(payload.get("orders", []))
            if order.order_id
        }
        self._queue = deque(
            QueuedOrder(item["correlation_id"], self._restore_order(item["order"]))
            for item in payload.get("queue", [])
        )
        self._processed = {str(k): str(v) for k, v in payload.get("processed", {}).items()}

    # ------------------------------------------------------------------
    # Serialization helpers
    @staticmethod
    def _serialize_order(order: Order) -> dict:
        data = order.to_dict()
        data["side"] = order.side.value
        data["order_type"] = order.order_type.value
        data["status"] = order.status.value
        return data

    @staticmethod
    def _restore_orders(serialized: Iterable[dict]) -> Iterable[Order]:
        return [OrderManagementSystem._restore_order(item) for item in serialized]

    @staticmethod
    def _restore_order(data: MutableMapping[str, object]) -> Order:
        created_at = datetime.fromisoformat(str(data["created_at"]))
        order = Order(
            symbol=str(data["symbol"]),
            side=str(data["side"]),
            quantity=float(data["quantity"]),
            price=float(data["price"]) if data.get("price") is not None else None,
            order_type=str(data.get("order_type", "market")),
            stop_price=float(data["stop_price"]) if data.get("stop_price") is not None else None,
            order_id=str(data.get("order_id")) if data.get("order_id") else None,
            status=str(data.get("status", "pending")),
            filled_quantity=float(data.get("filled_quantity", 0.0)),
            average_price=float(data["average_price"]) if data.get("average_price") is not None else None,
            rejection_reason=str(data.get("rejection_reason")) if data.get("rejection_reason") else None,
            created_at=created_at,
        )
        if data.get("updated_at"):
            object.__setattr__(order, "updated_at", datetime.fromisoformat(str(data["updated_at"])) )
        return order

    # ------------------------------------------------------------------
    # Queue operations
    def submit(self, order: Order, *, correlation_id: str) -> Order:
        """Submit an order, enforcing idempotency with correlation IDs."""

        if correlation_id in self._processed:
            order_id = self._processed[correlation_id]
            return self._orders[order_id]

        reference_price = order.price if order.price is not None else max(order.average_price or 0.0, 1.0)
        self.risk.validate_order(order.symbol, order.side.value, order.quantity, reference_price)

        queued_order = QueuedOrder(correlation_id, order)
        self._queue.append(queued_order)
        self._persist_state()
        return order

    def process_next(self) -> Order:
        if not self._queue:
            raise LookupError("No orders pending")
        item = self._queue.popleft()
        try:
            submitted = self.connector.place_order(item.order)
        except OrderError as exc:
            item.order.reject(str(exc))
            self._persist_state()
            return item.order
        if submitted.order_id is None:
            raise RuntimeError("Connector returned order without ID")
        self._orders[submitted.order_id] = submitted
        self._processed[item.correlation_id] = submitted.order_id
        self._persist_state()
        return submitted

    def process_all(self) -> None:
        while self._queue:
            self.process_next()

    # ------------------------------------------------------------------
    # Lifecycle helpers
    def cancel(self, order_id: str) -> bool:
        if order_id not in self._orders:
            return False
        cancelled = self.connector.cancel_order(order_id)
        if cancelled:
            self._orders[order_id].cancel()
            self._persist_state()
        return cancelled

    def register_fill(self, order_id: str, quantity: float, price: float) -> Order:
        order = self._orders[order_id]
        order.record_fill(quantity, price)
        self.risk.register_fill(order.symbol, order.side.value, quantity, price)
        self._persist_state()
        return order

    def reload(self) -> None:
        """Reload state from disk (used after restart)."""

        self._queue.clear()
        self._orders.clear()
        self._processed.clear()
        self._load_state()

    def outstanding(self) -> Iterable[Order]:
        return list(self._orders.values())


__all__ = [
    "QueuedOrder",
    "OMSConfig",
    "OrderManagementSystem",
]
