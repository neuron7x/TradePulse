# SPDX-License-Identifier: MIT
"""Order management system with persistence and idempotent queues."""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Iterable, Mapping, MutableMapping

from core.utils.metrics import get_metrics_collector
from observability.tracing import pipeline_span, record_span_event
from domain import Order, OrderStatus
from interfaces.execution import RiskController

from .audit import ExecutionAuditLogger, get_execution_audit_logger
from .compliance import ComplianceMonitor, ComplianceReport, ComplianceViolation
from .connectors import ExecutionConnector, OrderError, TransientOrderError
from .order_ledger import OrderLedger


DEFAULT_LEDGER_PATH = Path("observability/audit/order-ledger.jsonl")


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
    ledger_path: Path | None = DEFAULT_LEDGER_PATH

    def __post_init__(self) -> None:
        if not isinstance(self.state_path, Path):
            object.__setattr__(self, "state_path", Path(self.state_path))
        if self.max_retries < 1:
            object.__setattr__(self, "max_retries", 1)
        if self.backoff_seconds < 0.0:
            object.__setattr__(self, "backoff_seconds", 0.0)
        ledger_path = self.ledger_path
        if ledger_path is not None and not isinstance(ledger_path, Path):
            ledger_path = Path(ledger_path)
        if ledger_path == DEFAULT_LEDGER_PATH:
            ledger_path = self.state_path.parent / f"{self.state_path.stem}_ledger.jsonl"
        if ledger_path is not None:
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, "ledger_path", ledger_path)


class OrderManagementSystem:
    """Queue-based OMS with state recovery."""

    def __init__(
        self,
        connector: ExecutionConnector,
        risk_controller: RiskController,
        config: OMSConfig,
        *,
        compliance_monitor: ComplianceMonitor | None = None,
        audit_logger: ExecutionAuditLogger | None = None,
    ) -> None:
        self.connector = connector
        self.risk = risk_controller
        self.config = config
        self._compliance = compliance_monitor
        self._queue: Deque[QueuedOrder] = deque()
        self._orders: MutableMapping[str, Order] = {}
        self._processed: Dict[str, str] = {}
        self._correlations: Dict[str, str] = {}
        self._metrics = get_metrics_collector()
        self._ack_timestamps: Dict[str, datetime] = {}
        self._pending: Dict[str, Order] = {}
        self._audit = audit_logger or get_execution_audit_logger()
        ledger_path = self.config.ledger_path
        self._ledger = OrderLedger(ledger_path) if ledger_path is not None else None
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
            "correlations": self._correlations,
        }

    def _persist_state(self) -> None:
        if not self.config.auto_persist:
            return
        path = self.config.state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._state_payload(), indent=2, sort_keys=True))

    def _load_state(self) -> None:
        path = self.config.state_path
        payload: Mapping[str, object] | None = None
        source = "state_file"
        if path.exists():
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError:
                payload = None
        if payload is None and self._ledger is not None:
            payload = self._ledger.latest_state()
            source = "ledger"
        if payload is None:
            return
        self._apply_state_snapshot(payload, source=source)

    def _apply_state_snapshot(
        self, payload: Mapping[str, object], *, source: str = "manual"
    ) -> None:
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
                (
                    str(item.get("last_error"))
                    if item.get("last_error") is not None
                    else None
                ),
            )
            for item in payload.get("queue", [])
        )
        self._processed = {
            str(k): str(v) for k, v in payload.get("processed", {}).items()
        }
        self._correlations = {
            str(k): str(v) for k, v in payload.get("correlations", {}).items()
        }
        self._pending = {item.correlation_id: item.order for item in self._queue}
        if self._ledger is not None:
            self._record_ledger_event("state_restored", metadata={"source": source})

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
            stop_price=(
                float(data["stop_price"])
                if data.get("stop_price") is not None
                else None
            ),
            order_id=str(data.get("order_id")) if data.get("order_id") else None,
            status=str(data.get("status", "pending")),
            filled_quantity=float(data.get("filled_quantity", 0.0)),
            average_price=(
                float(data["average_price"])
                if data.get("average_price") is not None
                else None
            ),
            rejection_reason=(
                str(data.get("rejection_reason"))
                if data.get("rejection_reason")
                else None
            ),
            created_at=created_at,
        )
        if data.get("updated_at"):
            object.__setattr__(
                order, "updated_at", datetime.fromisoformat(str(data["updated_at"]))
            )
        return order

    # ------------------------------------------------------------------
    # Queue operations
    def submit(self, order: Order, *, correlation_id: str) -> Order:
        """Submit an order, enforcing idempotency with correlation IDs."""

        venue = getattr(
            self.connector, "name", self.connector.__class__.__name__.lower()
        )
        span_attrs = {
            "orders.venue": venue,
            "orders.symbol": order.symbol,
            "orders.side": order.side.value,
            "orders.type": order.order_type.value,
            "orders.correlation_id": correlation_id,
            "orders.quantity": float(order.quantity),
        }
        if order.price is not None:
            span_attrs["orders.price"] = float(order.price)

        with pipeline_span("orders.submit", **span_attrs) as span:
            if correlation_id in self._processed:
                order_id = self._processed[correlation_id]
                existing = self._orders[order_id]
                record_span_event(
                    span,
                    "orders.submit.deduplicated",
                    {"orders.order_id": order_id},
                )
                return existing
            if correlation_id in self._pending:
                record_span_event(
                    span,
                    "orders.submit.pending",
                    {"orders.order_id": self._pending[correlation_id].order_id},
                )
                return self._pending[correlation_id]

            if span is not None and order.order_id:
                span.set_attribute("orders.order_id", order.order_id)

            if self._compliance is not None:
                report = None
                try:
                    report = self._compliance.check(
                        order.symbol, order.quantity, order.price
                    )
                except ComplianceViolation as exc:
                    report = exc.report
                    self._metrics.record_compliance_check(
                        order.symbol,
                        "blocked",
                        () if report is None else report.violations,
                    )
                    self._emit_compliance_audit(order, correlation_id, report, str(exc))
                    self._record_ledger_event(
                        "compliance_blocked",
                        order=order,
                        correlation_id=correlation_id,
                        metadata={
                            "violations": []
                            if report is None
                            else report.violations,
                            "error": str(exc),
                        },
                    )
                    record_span_event(
                        span,
                        "orders.compliance.blocked",
                        {
                            "orders.compliance.status": "blocked",
                            "orders.compliance.error": str(exc),
                        },
                    )
                    raise
                status = "passed" if report is None or report.is_clean() else "warning"
                if span is not None:
                    span.set_attribute("orders.compliance.status", status)
                if report is not None:
                    self._metrics.record_compliance_check(
                        order.symbol,
                        "blocked" if report.blocked else status,
                        report.violations,
                    )
                    self._emit_compliance_audit(order, correlation_id, report, None)
                    if report.blocked:
                        self._record_ledger_event(
                            "compliance_blocked",
                            order=order,
                            correlation_id=correlation_id,
                            metadata={
                                "violations": report.violations,
                                "blocked": True,
                            },
                        )
                        record_span_event(
                            span,
                            "orders.compliance.blocked",
                            {
                                "orders.compliance.status": "blocked",
                                "orders.compliance.error": "policy_blocked",
                            },
                        )
                        raise ComplianceViolation(
                            "Compliance check blocked order", report=report
                        )

            reference_price = (
                order.price
                if order.price is not None
                else max(order.average_price or 0.0, 1.0)
            )
            self.risk.validate_order(
                order.symbol, order.side.value, order.quantity, reference_price
            )
            record_span_event(
                span,
                "orders.risk.validated",
                {"orders.reference_price": float(reference_price)},
            )

            queued_order = QueuedOrder(correlation_id, order)
            self._queue.append(queued_order)
            self._pending[correlation_id] = order
            self._persist_state()
            self._record_ledger_event(
                "order_queued",
                order=order,
                correlation_id=correlation_id,
            )
            record_span_event(
                span,
                "orders.submit.queued",
                {
                    "orders.queue_depth": len(self._queue),
                    "orders.retries": queued_order.attempts,
                },
            )
            return order

    def _emit_compliance_audit(
        self,
        order: Order,
        correlation_id: str,
        report: ComplianceReport | None,
        error: str | None,
    ) -> None:
        payload = {
            "event": "compliance_check",
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": float(order.quantity),
            "price": None if order.price is None else float(order.price),
            "correlation_id": correlation_id,
            "error": error,
        }
        if report is not None:
            payload["report"] = report.to_dict()
            if report.blocked:
                status = "blocked"
            elif report.is_clean():
                status = "passed"
            else:
                status = "warning"
        else:
            payload["report"] = None
            status = "blocked" if error else "passed"
        payload["status"] = status
        self._audit.emit(payload)

    def process_next(self) -> Order:
        if not self._queue:
            raise LookupError("No orders pending")
        item = self._queue[0]
        venue = getattr(
            self.connector, "name", self.connector.__class__.__name__.lower()
        )
        span_attrs = {
            "orders.venue": venue,
            "orders.symbol": item.order.symbol,
            "orders.side": item.order.side.value,
            "orders.correlation_id": item.correlation_id,
        }
        retryable: tuple[type[Exception], ...] = (
            TransientOrderError,
            TimeoutError,
            ConnectionError,
        )
        max_retries = max(1, int(self.config.max_retries))
        with pipeline_span("orders.dispatch", **span_attrs) as span:
            while True:
                item.attempts += 1
                try:
                    start = time.perf_counter()
                    submitted = self.connector.place_order(
                        item.order, idempotency_key=item.correlation_id
                    )
                    ack_latency = time.perf_counter() - start
                except retryable as exc:
                    item.last_error = str(exc)
                    if item.attempts >= max_retries:
                        self._queue.popleft()
                        self._pending.pop(item.correlation_id, None)
                        item.order.reject(str(exc))
                        self._persist_state()
                        self._record_ledger_event(
                            "order_rejected",
                            order=item.order,
                            correlation_id=item.correlation_id,
                            metadata={
                                "reason": str(exc),
                                "attempts": item.attempts,
                                "transient": True,
                            },
                        )
                        record_span_event(
                            span,
                            "orders.dispatch.failure",
                            {
                                "orders.error": str(exc),
                                "orders.attempts": item.attempts,
                                "orders.retry_exhausted": True,
                            },
                        )
                        return item.order
                    backoff = max(0.0, float(self.config.backoff_seconds))
                    self._persist_state()
                    self._record_ledger_event(
                        "order_retry_scheduled",
                        order=item.order,
                        correlation_id=item.correlation_id,
                        metadata={
                            "attempts": item.attempts,
                            "error": str(exc),
                            "backoff_seconds": backoff * item.attempts,
                        },
                    )
                    record_span_event(
                        span,
                        "orders.dispatch.retry",
                        {
                            "orders.error": str(exc),
                            "orders.attempts": item.attempts,
                            "orders.backoff_seconds": backoff * item.attempts,
                        },
                    )
                    if backoff:
                        time.sleep(backoff * item.attempts)
                    continue
                except OrderError as exc:
                    self._queue.popleft()
                    self._pending.pop(item.correlation_id, None)
                    item.order.reject(str(exc))
                    self._persist_state()
                    self._record_ledger_event(
                        "order_rejected",
                        order=item.order,
                        correlation_id=item.correlation_id,
                        metadata={"reason": str(exc)},
                    )
                    record_span_event(
                        span,
                        "orders.dispatch.failure",
                        {
                            "orders.error": str(exc),
                            "orders.attempts": item.attempts,
                            "orders.retry_exhausted": False,
                        },
                    )
                    return item.order
                break
            self._queue.popleft()
            self._pending.pop(item.correlation_id, None)
            if submitted.order_id is None:
                raise RuntimeError("Connector returned order without ID")
            self._orders[submitted.order_id] = submitted
            self._processed[item.correlation_id] = submitted.order_id
            self._correlations[submitted.order_id] = item.correlation_id
            if self._metrics.enabled:
                self._metrics.record_order_ack_latency(
                    venue, submitted.symbol, max(0.0, ack_latency)
                )
                self._ack_timestamps[submitted.order_id] = datetime.now(timezone.utc)
            self._persist_state()
            self._record_ledger_event(
                "order_acknowledged",
                order=submitted,
                correlation_id=item.correlation_id,
                metadata={"attempts": item.attempts, "ack_latency": ack_latency},
            )
            record_span_event(
                span,
                "orders.dispatch.ack",
                {
                    "orders.order_id": submitted.order_id,
                    "orders.ack_latency_ms": ack_latency * 1000.0,
                    "orders.attempts": item.attempts,
                },
            )
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
            self._record_ledger_event(
                "order_cancelled",
                order=self._orders[order_id],
                correlation_id=self._correlations.get(order_id),
            )
        return cancelled

    def register_fill(self, order_id: str, quantity: float, price: float) -> Order:
        order = self._orders[order_id]
        venue = getattr(
            self.connector, "name", self.connector.__class__.__name__.lower()
        )
        span_attrs = {
            "orders.venue": venue,
            "orders.symbol": order.symbol,
            "orders.order_id": order_id,
            "orders.fill_quantity": float(quantity),
            "orders.fill_price": float(price),
        }
        with pipeline_span("orders.fill", **span_attrs) as span:
            order.record_fill(quantity, price)
            self.risk.register_fill(order.symbol, order.side.value, quantity, price)
            if self._metrics.enabled:
                now = datetime.now(timezone.utc)
                ack_ts = self._ack_timestamps.get(order_id)
                if ack_ts is not None:
                    latency = max(0.0, (now - ack_ts).total_seconds())
                    self._metrics.record_order_fill_latency(venue, order.symbol, latency)
                    record_span_event(
                        span,
                        "orders.fill.latency",
                        {"orders.fill_latency_ms": latency * 1000.0},
                    )
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
                        venue,
                        order.symbol,
                        signal_latency,
                    )
                    record_span_event(
                        span,
                        "orders.fill.signal_to_fill",
                        {
                            "orders.signal_strategy": strategy,
                            "orders.signal_to_fill_latency_ms": signal_latency * 1000.0,
                        },
                    )
                self._ack_timestamps.pop(order_id, None)
            self._persist_state()
            self._record_ledger_event(
                "order_fill_recorded",
                order=order,
                correlation_id=self._correlations.get(order_id),
                metadata={"fill_quantity": quantity, "fill_price": price},
            )
            record_span_event(
                span,
                "orders.fill.recorded",
                {
                    "orders.position_after_fill": float(order.filled_quantity),
                },
            )
            return order

    def sync_remote_state(self, order: Order) -> Order:
        """Synchronize terminal state reported by the venue without reissuing API calls."""

        if order.order_id is None:
            raise ValueError("order must include an order_id to sync state")

        stored = self._orders.get(order.order_id)
        if stored is None:
            raise LookupError(f"Unknown order_id: {order.order_id}")

        stored.status = OrderStatus(order.status)
        stored.filled_quantity = float(order.filled_quantity)
        stored.average_price = (
            float(order.average_price) if order.average_price is not None else None
        )
        stored.rejection_reason = order.rejection_reason
        stored.updated_at = getattr(order, "updated_at", stored.updated_at)

        if not stored.is_active:
            self._ack_timestamps.pop(order.order_id, None)

        self._persist_state()
        self._record_ledger_event(
            "order_state_synced",
            order=stored,
            correlation_id=self._correlations.get(order.order_id),
        )
        return stored

    def reload(self) -> None:
        """Reload state from disk (used after restart)."""

        self._queue.clear()
        self._orders.clear()
        self._processed.clear()
        self._ack_timestamps.clear()
        self._pending.clear()
        self._correlations.clear()
        self._load_state()
        self._record_ledger_event("state_reloaded", metadata={"source": "reload"})

    def _record_ledger_event(
        self,
        event: str,
        *,
        order: Order | Mapping[str, object] | None = None,
        correlation_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if self._ledger is None:
            return
        order_payload: Mapping[str, object] | None
        if isinstance(order, Order):
            order_payload = order.to_dict()
        else:
            order_payload = order
        self._ledger.append(
            event,
            order=order_payload,
            correlation_id=correlation_id,
            metadata=metadata,
            state_snapshot=self._state_payload(),
        )

    # ------------------------------------------------------------------
    # Recovery helpers
    def correlation_for(self, order_id: str) -> str | None:
        """Return the correlation ID originally associated with *order_id*."""

        return self._correlations.get(order_id)

    def adopt_open_order(
        self, order: Order, *, correlation_id: str | None = None
    ) -> None:
        """Adopt an externally recovered order into the OMS state."""

        if order.order_id is None:
            raise ValueError("order must have an order_id to be adopted")
        correlation = correlation_id or f"recovered-{order.order_id}"
        self._orders[order.order_id] = order
        self._processed[correlation] = order.order_id
        self._correlations[order.order_id] = correlation
        self._persist_state()

    def requeue_order(self, order_id: str, *, correlation_id: str | None = None) -> str:
        """Re-enqueue an order whose venue state was lost or invalidated."""

        if order_id not in self._orders:
            raise LookupError(f"Unknown order_id: {order_id}")
        original = self._orders.pop(order_id)
        correlation = correlation_id or self._correlations.pop(order_id, None)
        if correlation is None:
            correlation = f"requeue-{order_id}"
        self._processed.pop(correlation, None)
        resubmittable = replace(
            original,
            order_id=None,
            status=OrderStatus.PENDING,
            filled_quantity=0.0,
            average_price=None,
            rejection_reason=None,
        )
        queued = QueuedOrder(correlation, resubmittable)
        self._queue.appendleft(queued)
        self._pending[correlation] = resubmittable
        self._ack_timestamps.pop(order_id, None)
        self._persist_state()
        return correlation

    def outstanding(self) -> Iterable[Order]:
        return [order for order in self._orders.values() if order.is_active]


__all__ = [
    "QueuedOrder",
    "OMSConfig",
    "OrderManagementSystem",
]
