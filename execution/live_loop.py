# SPDX-License-Identifier: MIT
"""Long-running orchestration loop for live order execution."""

from __future__ import annotations

import logging
import threading
from contextlib import suppress
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping

from core.utils.metrics import get_metrics_collector
from domain import Order, OrderSide, OrderStatus, OrderType

from .connectors import ExecutionConnector, OrderError, TransientOrderError
from .oms import OMSConfig, OrderManagementSystem
from .risk import RiskManager


@dataclass(slots=True)
class LiveLoopConfig:
    """Configuration toggles for :class:`LiveExecutionLoop`."""

    poll_interval: float = 1.0
    heartbeat_interval: float = 15.0
    max_backoff: float = 30.0
    backoff_base: float = 1.0

    def __post_init__(self) -> None:
        if self.poll_interval <= 0:
            self.poll_interval = 0.5
        if self.heartbeat_interval <= 0:
            self.heartbeat_interval = 5.0
        if self.max_backoff <= 0:
            self.max_backoff = 30.0
        if self.backoff_base <= 0:
            self.backoff_base = 1.0


class LiveExecutionLoop:
    """Coordinate connectors, OMS, and risk checks for live execution."""

    def __init__(
        self,
        *,
        primary_connector: ExecutionConnector,
        risk_manager: RiskManager,
        oms_config: OMSConfig,
        connectors: Mapping[str, ExecutionConnector] | None = None,
        config: LiveLoopConfig | None = None,
        logger: logging.Logger | None = None,
        on_kill_switch: Callable[[str], None] | None = None,
        on_reconnect: Callable[[str], None] | None = None,
        on_position_snapshot: Callable[[str, Iterable[Mapping[str, float]]], None] | None = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._metrics = get_metrics_collector()
        self._config = config or LiveLoopConfig()
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._running = False
        self._kill_switch_announced = False
        self._fill_cache: Dict[str, float] = {}

        self.primary_connector = primary_connector
        self.risk_manager = risk_manager
        self.oms = OrderManagementSystem(primary_connector, risk_manager, oms_config)

        connector_map: Dict[str, ExecutionConnector]
        if connectors is None:
            connector_map = {"primary": primary_connector}
        else:
            connector_map = dict(connectors)
            # Ensure the primary connector is addressable.
            if primary_connector not in connector_map.values():
                connector_map = {**connector_map, "primary": primary_connector}
        self._connectors: Dict[str, ExecutionConnector] = connector_map
        self._primary_name = next(
            (name for name, connector in connector_map.items() if connector is primary_connector),
            "primary",
        )

        self._on_kill_switch = on_kill_switch
        self._on_reconnect = on_reconnect
        self._on_position_snapshot = on_position_snapshot

    # ------------------------------------------------------------------
    # Public lifecycle
    def start(self, cold_start: bool) -> None:
        """Start background workers and perform state reconciliation."""

        with self._lock:
            if self._running:
                self._logger.info("live_execution_loop_already_running", extra={"event": "loop_start"})
                return
            self._logger.info(
                "live_execution_loop_starting",
                extra={"event": "loop_start", "cold_start": cold_start},
            )
            self._stop_event.clear()
            self._running = True

        self._connect_all()

        if cold_start:
            self._handle_cold_start()
        else:
            self._handle_warm_start()

        self._spawn_workers()

    def shutdown(self) -> None:
        """Stop all workers and disconnect connectors."""

        with self._lock:
            if not self._running:
                return
            self._running = False
            self._stop_event.set()

        for thread in self._threads:
            thread.join(timeout=self._config.heartbeat_interval * 2)
        self._threads.clear()

        for name, connector in self._connectors.items():
            with suppress(Exception):
                connector.disconnect()
            self._logger.info(
                "live_execution_loop_disconnected", extra={"event": "disconnect", "connector": name}
            )

        self._logger.info("live_execution_loop_shutdown", extra={"event": "loop_shutdown"})

    # ------------------------------------------------------------------
    # Internal lifecycle helpers
    def _connect_all(self) -> None:
        for name, connector in self._connectors.items():
            self._connect_with_retry(name, connector)

    def _connect_with_retry(self, name: str, connector: ExecutionConnector) -> None:
        attempts = 0
        while not self._stop_event.is_set():
            attempts += 1
            try:
                connector.connect()
            except Exception as exc:  # pragma: no cover - defensive guard
                delay = min(self._config.max_backoff, self._config.backoff_base * (2 ** (attempts - 1)))
                self._logger.warning(
                    "live_execution_loop_connect_failed",
                    extra={"event": "connect_retry", "connector": name, "attempt": attempts, "error": str(exc)},
                )
                self._stop_event.wait(delay)
                continue
            self._logger.info(
                "live_execution_loop_connected",
                extra={"event": "connect", "connector": name, "attempt": attempts},
            )
            if attempts > 1 and self._on_reconnect is not None:
                self._on_reconnect(name)
            return

    def _handle_cold_start(self) -> None:
        self.oms.purge_state()
        self.risk_manager.kill_switch.reset()
        self._fill_cache.clear()
        self._kill_switch_announced = False
        self._logger.info("live_execution_loop_cold_start", extra={"event": "cold_start"})

    def _handle_warm_start(self) -> None:
        self.oms.reload()
        self._fill_cache.clear()
        self._kill_switch_announced = False
        self._reconcile_orders()
        self._logger.info("live_execution_loop_warm_start", extra={"event": "warm_start"})

    def _spawn_workers(self) -> None:
        workers = [
            threading.Thread(target=self._order_submission_worker, name="order-submission", daemon=True),
            threading.Thread(target=self._fill_polling_worker, name="fill-polling", daemon=True),
            threading.Thread(target=self._heartbeat_worker, name="heartbeat", daemon=True),
        ]
        self._threads.extend(workers)
        for worker in workers:
            worker.start()

    # ------------------------------------------------------------------
    # Worker implementations
    def _order_submission_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.oms.process_next()
                continue
            except LookupError:
                # No pending orders – back off using configured interval.
                pass
            except Exception as exc:  # pragma: no cover - defensive guard
                reason = f"order_submission_failure: {exc}"
                self._logger.exception(
                    "live_execution_loop_order_worker_error",
                    extra={"event": "order_submission_error", "error": str(exc)},
                )
                self._trigger_kill_switch(reason)
            self._stop_event.wait(self._config.poll_interval)

    def _fill_polling_worker(self) -> None:
        poll_interval = max(self._config.poll_interval, 0.5)
        while not self._stop_event.is_set():
            try:
                self._poll_fills()
            except Exception as exc:  # pragma: no cover - defensive guard
                self._logger.exception(
                    "live_execution_loop_fill_poll_error",
                    extra={"event": "fill_poll_error", "error": str(exc)},
                )
            self._stop_event.wait(poll_interval)

    def _heartbeat_worker(self) -> None:
        interval = max(self._config.heartbeat_interval, 1.0)
        while not self._stop_event.is_set():
            self._check_kill_switch()
            for name, connector in self._connectors.items():
                try:
                    positions = connector.get_positions()
                    self._handle_positions(name, positions)
                except (TransientOrderError, ConnectionError) as exc:
                    self._logger.warning(
                        "live_execution_loop_heartbeat_disconnected",
                        extra={"event": "heartbeat_disconnect", "connector": name, "error": str(exc)},
                    )
                    self._handle_disconnect(name, connector)
                except Exception as exc:  # pragma: no cover - defensive guard
                    self._logger.exception(
                        "live_execution_loop_heartbeat_error",
                        extra={"event": "heartbeat_error", "connector": name, "error": str(exc)},
                    )
            self._stop_event.wait(interval)

    # ------------------------------------------------------------------
    # Reconciliation helpers
    def _reconcile_orders(self) -> None:
        outstanding = {
            order.order_id: order
            for order in self.oms.outstanding()
            if order.order_id and order.is_active
        }
        connector_orders: Dict[str, Order] = {}
        for name, connector in self._connectors.items():
            for order in connector.open_orders():
                if not order.order_id:
                    continue
                connector_orders[order.order_id] = order
                if order.order_id not in outstanding:
                    self.oms.adopt_order(order)
                    self._logger.info(
                        "live_execution_loop_adopted_order",
                        extra={"event": "order_adopted", "connector": name, "order_id": order.order_id},
                    )

        pending_order_ids = set(self.oms.pending_order_ids())
        for order_id, stored_order in outstanding.items():
            if order_id in connector_orders or order_id in pending_order_ids:
                continue
            remaining = max(stored_order.remaining_quantity, 0.0)
            if remaining <= 0:
                continue
            cloned = Order(
                symbol=stored_order.symbol,
                side=OrderSide(stored_order.side.value),
                quantity=remaining,
                price=stored_order.price,
                order_type=OrderType(stored_order.order_type.value),
                stop_price=stored_order.stop_price,
            )
            correlation_id = f"resubmit::{order_id}"
            try:
                self.oms.enqueue_for_resubmission(cloned, correlation_id=correlation_id)
                self._logger.warning(
                    "live_execution_loop_reenqueued_order",
                    extra={"event": "order_reenqueued", "order_id": order_id, "correlation_id": correlation_id},
                )
            except Exception as exc:
                self._logger.error(
                    "live_execution_loop_reenqueued_order_failed",
                    extra={"event": "order_reenqueued_error", "order_id": order_id, "error": str(exc)},
                )
                self._trigger_kill_switch(f"failed to re-enqueue order {order_id}: {exc}")

    def _poll_fills(self) -> None:
        for order in self.oms.outstanding():
            order_id = order.order_id
            if not order_id:
                continue
            try:
                venue_order = self.primary_connector.fetch_order(order_id)
            except OrderError:
                continue
            self._process_venue_order(venue_order)

    def _process_venue_order(self, venue_order: Order) -> None:
        order_id = venue_order.order_id
        if not order_id:
            return
        previous_fill = self._fill_cache.get(order_id, 0.0)
        current_fill = max(float(venue_order.filled_quantity), 0.0)
        incremental = current_fill - previous_fill
        if incremental > 1e-9:
            price = venue_order.average_price or venue_order.price or 0.0
            if price <= 0:
                self._logger.warning(
                    "live_execution_loop_missing_price",
                    extra={"event": "fill_missing_price", "order_id": order_id},
                )
            else:
                self.oms.register_fill(order_id, incremental, price)
                self._logger.info(
                    "live_execution_loop_registered_fill",
                    extra={"event": "fill_registered", "order_id": order_id, "quantity": incremental, "price": price},
                )
            self._fill_cache[order_id] = current_fill
        if venue_order.status in {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}:
            self._fill_cache.pop(order_id, None)

    def _handle_positions(self, connector_name: str, positions: Iterable[Mapping[str, float]]) -> None:
        positions_list = list(positions)
        if self._on_position_snapshot is not None:
            self._on_position_snapshot(connector_name, positions_list)
        if not self._metrics.enabled:
            return
        for position in positions_list:
            symbol = str(position.get("symbol") or position.get("instrument") or "").strip()
            if not symbol:
                continue
            qty = float(position.get("qty", 0.0))
            self._metrics.set_open_positions(connector_name, symbol, qty)

    # ------------------------------------------------------------------
    # Fault handling
    def _handle_disconnect(self, name: str, connector: ExecutionConnector) -> None:
        attempts = 0
        while not self._stop_event.is_set():
            attempts += 1
            with suppress(Exception):
                connector.disconnect()
            try:
                connector.connect()
            except Exception as exc:  # pragma: no cover - defensive guard
                delay = min(self._config.max_backoff, self._config.backoff_base * (2 ** (attempts - 1)))
                self._logger.warning(
                    "live_execution_loop_reconnect_failed",
                    extra={
                        "event": "reconnect_retry",
                        "connector": name,
                        "attempt": attempts,
                        "error": str(exc),
                    },
                )
                self._stop_event.wait(delay)
                continue
            self._logger.info(
                "live_execution_loop_reconnected",
                extra={"event": "reconnect", "connector": name, "attempt": attempts},
            )
            if self._on_reconnect is not None:
                self._on_reconnect(name)
            return

    def _trigger_kill_switch(self, reason: str) -> None:
        if self.risk_manager.kill_switch.is_triggered():
            self.risk_manager.kill_switch.reset()
        self.risk_manager.kill_switch.trigger(reason)
        self._announce_kill_switch()

    def _check_kill_switch(self) -> None:
        if self.risk_manager.kill_switch.is_triggered():
            self._announce_kill_switch()

    def _announce_kill_switch(self) -> None:
        if self._kill_switch_announced:
            return
        reason = self.risk_manager.kill_switch.reason
        self._logger.error(
            "live_execution_loop_kill_switch",
            extra={"event": "kill_switch", "reason": reason or "unspecified"},
        )
        self._kill_switch_announced = True
        if self._on_kill_switch is not None:
            self._on_kill_switch(reason)


__all__ = ["LiveExecutionLoop", "LiveLoopConfig"]
