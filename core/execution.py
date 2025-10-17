"""Execution layer with simplified market/limit order simulation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict


@dataclass(slots=True)
class Order:
    ts_utc: datetime
    side: str
    quantity: float
    order_type: str
    price: float
    slippage_bps: float
    venue: str


class ExecutionModel:
    def __init__(self, venue: str, order_type: str = "market", slippage_bps: float = 3.0) -> None:
        self.venue = venue
        self.order_type = order_type
        self.slippage_bps = slippage_bps

    def route(self, signal: Dict[str, float], size: float, exit_signal: Dict[str, float] | None, bar: Dict[str, float]) -> Order:
        price = bar["close"]
        side = signal.get("side", "flat")
        if exit_signal and exit_signal.get("reason"):
            side = "sell" if size > 0 else "buy"
        direction = 1 if side == "buy" else -1 if side == "sell" else 0
        slip_multiplier = 1 + (self.slippage_bps / 10_000) * direction
        exec_price = price * slip_multiplier
        ts_raw = signal.get("ts_utc")
        if isinstance(ts_raw, str):
            ts_utc = datetime.fromisoformat(ts_raw)
        elif isinstance(ts_raw, datetime):
            ts_utc = ts_raw
        else:
            ts_utc = datetime.utcnow()
        return Order(
            ts_utc=ts_utc,
            side=side,
            quantity=size,
            order_type=self.order_type,
            price=exec_price,
            slippage_bps=self.slippage_bps,
            venue=self.venue,
        )

