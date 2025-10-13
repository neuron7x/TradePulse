# SPDX-License-Identifier: MIT
"""Order management system with persistence and idempotent queues."""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Iterable, MutableMapping

from domain import Order
from interfaces.execution import RiskController

from core.utils.metrics import get_metrics_collector
from .connectors import ExecutionConnector, OrderError, TransientOrderError


@dataclass(slots=True)
class QueuedOrder:
    """Order request paired with its idempotency key."""

    correlation_id: str
    order: Order
    attempts: int = 0
    last_error: str | None = None


@dataclass(slots=True)
class OMSConfig:
    """Configuration for :class:`OrderManagementSystem`."""

    state_path: Path
    auto_persist: bool = True
    max_retries: int = 3
    backoff_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.state_path, Path):
            object.__setattr__(self, "state_path", Path(self.state_path))
        if self.max_retries < 1:
            object.__setattr__(self, "max_retries", 1)
        if self.backoff_seconds < 0.0:
            object.__setattr__(self, "backoff_seconds", 0.0)


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
        self._metrics = get_metrics_collector()
        self._ack_timestamps: Dict[str, datetime] = {}
        self._pending: Dict[str, Order] = {}
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
                    "attempts": item.attempts,
                    "last_error": item.last_error,
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
            QueuedOrder(
                item["correlation_id"],
                self._restore_order(item["order"]),
                int(item.get("attempts", 0)),
                str(item.get("last_error")) if item.get("last_error") is not None else None,
            )
            for item in payload.get("queue", [])
        )
        self._processed = {str(k): str(v) for k, v in payload.get("processed", {}).items()}
        self._pending = {item.correlation_id: item.order for item in self._queue}

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
        if correlation_id in self._pending:
            return self._pending[correlation_id]

        reference_price = order.price if order.price is not None else max(order.average_price or 0.0, 1.0)
        self.risk.validate_order(order.symbol, order.side.value, order.quantity, reference_price)

        queued_order = QueuedOrder(correlation_id, order)
        self._queue.append(queued_order)
        self._pending[correlation_id] = order
        self._persist_state()
        return order

    def process_next(self) -> Order:
        if not self._queue:
            raise LookupError("No orders pending")
        item = self._queue[0]
        retryable: tuple[type[Exception], ...] = (
            TransientOrderError,
            TimeoutError,
            ConnectionError,
        )
        max_retries = max(1, int(self.config.max_retries))
        while True:
            item.attempts += 1
            try:
                start = time.perf_counter()
                submitted = self.connector.place_order(item.order, idempotency_key=item.correlation_id)
                ack_latency = time.perf_counter() - start
            except retryable as exc:
                item.last_error = str(exc)
                if item.attempts >= max_retries:
                    self._queue.popleft()
                    self._pending.pop(item.correlation_id, None)
                    item.order.reject(str(exc))
                    self._persist_state()
                    return item.order
                backoff = max(0.0, float(self.config.backoff_seconds))
                self._persist_state()
                if backoff:
                    time.sleep(backoff * item.attempts)
                continue
            except OrderError as exc:
                self._queue.popleft()
                self._pending.pop(item.correlation_id, None)
                item.order.reject(str(exc))
                self._persist_state()
                return item.order
            break
        self._queue.popleft()
        self._pending.pop(item.correlation_id, None)
        if submitted.order_id is None:
            raise RuntimeError("Connector returned order without ID")
        self._orders[submitted.order_id] = submitted
        self._processed[item.correlation_id] = submitted.order_id
        if self._metrics.enabled:
            exchange = getattr(self.connector, "name", self.connector.__class__.__name__.lower())
            self._metrics.record_order_ack_latency(exchange, submitted.symbol, max(0.0, ack_latency))
            self._ack_timestamps[submitted.order_id] = datetime.now(timezone.utc)
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
            self._ack_timestamps.pop(order_id, None)
            self._persist_state()
        return cancelled

    def register_fill(self, order_id: str, quantity: float, price: float) -> Order:
        order = self._orders[order_id]
        order.record_fill(quantity, price)
        self.risk.register_fill(order.symbol, order.side.value, quantity, price)
        if self._metrics.enabled:
            exchange = getattr(self.connector, "name", self.connector.__class__.__name__.lower())
            now = datetime.now(timezone.utc)
            ack_ts = self._ack_timestamps.get(order_id)
            if ack_ts is not None:
                latency = max(0.0, (now - ack_ts).total_seconds())
                self._metrics.record_order_fill_latency(exchange, order.symbol, latency)
            signal_origin = getattr(order, "created_at", None)
            signal_latency = None
            if isinstance(signal_origin, datetime):
                if signal_origin.tzinfo is None:
                    signal_origin = signal_origin.replace(tzinfo=timezone.utc)
                signal_latency = max(0.0, (now - signal_origin).total_seconds())
            if signal_latency is not None:
                metadata = getattr(order, "metadata", None)
                strategy = "unspecified"
                if isinstance(metadata, dict):
                    strategy = str(metadata.get("strategy") or strategy)
                else:
                    strategy = str(getattr(order, "strategy", strategy))
                self._metrics.record_signal_to_fill_latency(
                    strategy,
                    exchange,
                    order.symbol,
                    signal_latency,
                )
            self._ack_timestamps.pop(order_id, None)
        self._persist_state()
        return order

    def reload(self) -> None:
        """Reload state from disk (used after restart)."""

        self._queue.clear()
        self._orders.clear()
        self._processed.clear()
        self._ack_timestamps.clear()
        self._pending.clear()
        self._load_state()

    def outstanding(self) -> Iterable[Order]:
        return list(self._orders.values())


__all__ = [
    "QueuedOrder",
    "OMSConfig",
    "OrderManagementSystem",
]
