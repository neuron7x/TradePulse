# SPDX-License-Identifier: MIT
"""Execution connector backed by the in-memory simulator."""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, Mapping, MutableMapping

from domain import Order

from core.sim import CancelEvent, FillEvent, LimitOrderBookSimulator
from execution.connectors import ExecutionConnector, OrderError


class SimulatedConnector(ExecutionConnector):
    """Adapter exposing :class:`LimitOrderBookSimulator` as a connector."""

    def __init__(
        self,
        simulator: LimitOrderBookSimulator | None = None,
        *,
        sandbox: bool = True,
    ) -> None:
        super().__init__(sandbox=sandbox)
        self._simulator = simulator or LimitOrderBookSimulator()
        self._idempotency: MutableMapping[str, Order] = {}
        self._event_buffer: Deque[FillEvent | CancelEvent] = deque()

    # ------------------------------------------------------------------
    # ExecutionConnector API
    def connect(self, credentials: Mapping[str, str] | None = None) -> None:  # type: ignore[override]
        """No-op; the simulator is always available."""
        del credentials

    def disconnect(self) -> None:  # type: ignore[override]
        """No-op for the in-memory simulator."""

    def place_order(self, order: Order, *, idempotency_key: str | None = None) -> Order:  # type: ignore[override]
        if idempotency_key is not None and idempotency_key in self._idempotency:
            return self._idempotency[idempotency_key]
        submitted = self._simulator.submit(order)
        if idempotency_key is not None:
            self._idempotency[idempotency_key] = submitted
        self._buffer_events()
        return submitted

    def cancel_order(self, order_id: str) -> bool:  # type: ignore[override]
        cancelled = self._simulator.cancel(order_id)
        self._buffer_events()
        return cancelled

    def fetch_order(self, order_id: str) -> Order:  # type: ignore[override]
        try:
            return self._simulator.get(order_id)
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise OrderError(str(exc)) from exc

    def open_orders(self) -> Iterable[Order]:  # type: ignore[override]
        return self._simulator.open_orders()

    def get_positions(self) -> list[dict]:  # type: ignore[override]
        return []

    # ------------------------------------------------------------------
    # Simulator helpers
    def drain_events(self) -> list[FillEvent | CancelEvent]:
        """Return accumulated fill/cancel events since the last call."""

        events = list(self._event_buffer)
        self._event_buffer.clear()
        return events

    def _buffer_events(self) -> None:
        self._event_buffer.extend(self._simulator.drain_events())
