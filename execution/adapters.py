# SPDX-License-Identifier: MIT
"""Broker adapters bridging paper trading and live execution workflows."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Deque, Iterable, Mapping

from domain import Order

from .connectors import ExecutionConnector
from .risk import KillSwitch, OrderRateExceeded, RiskError


class BrokerMode(str, Enum):
    """Operating mode for :class:`BrokerAdapter`."""

    PAPER = "paper"
    LIVE = "live"


@dataclass(slots=True)
class ThrottleConfig:
    """Configuration for rate limiting order submissions."""

    max_orders: int
    interval_seconds: float

    def __post_init__(self) -> None:
        if self.max_orders < 0:
            object.__setattr__(self, "max_orders", 0)
        if self.interval_seconds < 0:
            object.__setattr__(self, "interval_seconds", 0.0)


class OrderThrottle:
    """Lightweight rate limiter shared by broker adapters."""

    def __init__(
        self,
        config: ThrottleConfig,
        *,
        time_source: Callable[[], float] | None = None,
    ) -> None:
        self.config = config
        self._submissions: Deque[float] = deque()
        self._time = time_source or time.time

    def check(self) -> None:
        """Ensure the next submission is allowed within the configured budget."""

        if self.config.max_orders <= 0:
            return
        now = float(self._time())
        window = max(self.config.interval_seconds, 0.0)
        while self._submissions and now - self._submissions[0] > window:
            self._submissions.popleft()
        if len(self._submissions) >= self.config.max_orders:
            raise OrderRateExceeded(
                f"Order throttle exceeded: {len(self._submissions)} submissions in {window}s"
            )
        self._submissions.append(now)


class BrokerAdapter(ExecutionConnector):
    """Adapter that can seamlessly switch between paper and live connectors."""

    def __init__(
        self,
        *,
        paper: ExecutionConnector,
        live: ExecutionConnector,
        start_in: BrokerMode = BrokerMode.PAPER,
        throttle: OrderThrottle | None = None,
        kill_switch: KillSwitch | None = None,
    ) -> None:
        super().__init__(sandbox=paper.sandbox)
        self._paper = paper
        self._live = live
        self._mode = start_in
        self._kill_switch = kill_switch or KillSwitch()
        self._throttle = throttle
        self._sync_sandbox_flag()

    # ------------------------------------------------------------------
    # Lifecycle helpers
    def connect(
        self,
        credentials: Mapping[str, Mapping[str, str]] | Mapping[str, str] | None = None,
    ) -> None:  # type: ignore[override]
        """Connect both venues, accepting combined or per-venue credentials."""

        paper_creds: Mapping[str, str] | None = None
        live_creds: Mapping[str, str] | None = None

        if isinstance(credentials, Mapping):
            if any(key in credentials for key in ("paper", "live")):
                paper_val = credentials.get("paper")
                live_val = credentials.get("live")
                if isinstance(paper_val, Mapping):
                    paper_creds = paper_val  # type: ignore[assignment]
                if isinstance(live_val, Mapping):
                    live_creds = live_val  # type: ignore[assignment]
            if paper_creds is None and live_creds is None:
                paper_creds = credentials  # type: ignore[assignment]
                live_creds = credentials  # type: ignore[assignment]

        self._paper.connect(paper_creds)
        self._live.connect(live_creds)

    def disconnect(self) -> None:  # type: ignore[override]
        self._paper.disconnect()
        self._live.disconnect()

    # ------------------------------------------------------------------
    # Mode management
    @property
    def mode(self) -> BrokerMode:
        return self._mode

    def promote_to_live(self) -> None:
        self._mode = BrokerMode.LIVE
        self._sync_sandbox_flag()

    def demote_to_paper(self) -> None:
        self._mode = BrokerMode.PAPER
        self._sync_sandbox_flag()

    @property
    def kill_switch(self) -> KillSwitch:
        return self._kill_switch

    def configure_throttle(self, config: ThrottleConfig | None) -> None:
        if config is None:
            self._throttle = None
            return
        self._throttle = OrderThrottle(config)

    # ------------------------------------------------------------------
    # Connector interface
    def _active(self) -> ExecutionConnector:
        return self._live if self._mode is BrokerMode.LIVE else self._paper

    def _sync_sandbox_flag(self) -> None:
        """Mirror the active connector's sandbox flag."""

        self.sandbox = self._active().sandbox

    def place_order(self, order: Order) -> Order:  # type: ignore[override]
        self._kill_switch.guard()
        if self._throttle is not None:
            self._throttle.check()
        return self._active().place_order(order)

    def cancel_order(self, order_id: str) -> bool:  # type: ignore[override]
        return self._active().cancel_order(order_id)

    def fetch_order(self, order_id: str) -> Order:  # type: ignore[override]
        return self._active().fetch_order(order_id)

    def open_orders(self) -> Iterable[Order]:  # type: ignore[override]
        return self._active().open_orders()

    def get_positions(self) -> list[dict]:  # type: ignore[override]
        return list(self._active().get_positions())

    # Convenience helpers -------------------------------------------------
    def ensure_live(self) -> None:
        """Guard that the adapter is currently operating against the live venue."""

        if self._mode is not BrokerMode.LIVE:
            raise RiskError("Adapter is not in live mode")

    def ensure_paper(self) -> None:
        """Guard that the adapter is currently operating in paper trading mode."""

        if self._mode is not BrokerMode.PAPER:
            raise RiskError("Adapter is not in paper mode")


__all__ = ["BrokerAdapter", "BrokerMode", "OrderThrottle", "ThrottleConfig"]
