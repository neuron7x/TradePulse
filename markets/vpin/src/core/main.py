# SPDX-License-Identifier: MIT
"""Volume-Synchronised Probability of Informed Trading (VPIN) core primitives.

This module converts raw trade prints into volume-balanced buckets and keeps a
rolling estimate of the VPIN metric.  The implementation deliberately avoids any
third-party dependencies so that it can be reused both inside pipeline tasks
and lightweight research notebooks.  The public API is centred around
``VPINCalculator`` which accepts streaming ``TradeTick`` events and computes the
metric incrementally.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Deque, Iterable, List, Optional, Sequence

__all__ = [
    "TradeTick",
    "VPINBucket",
    "VPINResult",
    "VPINCalculator",
    "compute_vpin_series",
]


@dataclass(slots=True)
class TradeTick:
    """Lightweight trade representation used by the VPIN calculator."""

    timestamp: datetime
    price: float
    volume: float
    side: str

    def __post_init__(self) -> None:
        normalized_side = self.side.lower()
        if normalized_side not in {"buy", "sell"}:
            msg = "side must be either 'buy' or 'sell'"
            raise ValueError(msg)
        if self.volume <= 0:
            raise ValueError("volume must be positive")
        if self.price <= 0:
            raise ValueError("price must be positive")
        self.side = normalized_side


@dataclass(slots=True)
class VPINBucket:
    """Aggregated buy/sell volume over a fixed bucket notional."""

    start: datetime
    end: datetime
    buy_volume: float = 0.0
    sell_volume: float = 0.0

    def imbalance(self) -> float:
        """Return the absolute buy/sell imbalance for the bucket."""

        return abs(self.buy_volume - self.sell_volume)

    def total_volume(self) -> float:
        """Return the total executed volume in the bucket."""

        return self.buy_volume + self.sell_volume

    def add_trade(self, trade: TradeTick) -> None:
        """Accumulate a trade into the bucket."""

        if trade.side == "buy":
            self.buy_volume += trade.volume
        else:
            self.sell_volume += trade.volume
        self.end = trade.timestamp


@dataclass(slots=True)
class VPINResult:
    """Container describing a VPIN measurement."""

    as_of: datetime
    value: float
    buckets: Sequence[VPINBucket]


class VPINCalculator:
    """Streaming VPIN calculator based on Easley, López de Prado & O'Hara (2011).

    Args:
        bucket_volume: Target notional volume for each VPIN bucket.
        window: Number of completed buckets to use for the rolling VPIN average.
        staleness: Optional timedelta after which the current bucket is flushed
            even if the full volume was not completed.  This guards against
            markets with sporadic trades.
    """

    def __init__(
        self,
        *,
        bucket_volume: float,
        window: int = 50,
        staleness: Optional[timedelta] = timedelta(minutes=5),
    ) -> None:
        if bucket_volume <= 0:
            raise ValueError("bucket_volume must be positive")
        if window <= 0:
            raise ValueError("window must be positive")
        self.bucket_volume = bucket_volume
        self.window = window
        self.staleness = staleness
        self._active_bucket: Optional[VPINBucket] = None
        self._completed: Deque[VPINBucket] = deque(maxlen=window)

    @property
    def buckets(self) -> Sequence[VPINBucket]:
        """Return a snapshot of the completed buckets."""

        return tuple(self._completed)

    def add_trade(self, trade: TradeTick) -> Optional[VPINResult]:
        """Process a trade and update the VPIN estimate.

        Returns a :class:`VPINResult` once the rolling window is fully
        populated.  ``None`` is returned until enough buckets have been
        collected.
        """

        bucket = self._ensure_bucket(trade.timestamp)
        bucket.add_trade(trade)

        if bucket.total_volume() >= self.bucket_volume:
            self._completed.append(bucket)
            self._active_bucket = None
        elif (
            self.staleness
            and trade.timestamp - bucket.start >= self.staleness
            and bucket.total_volume() > 0
        ):
            self._completed.append(bucket)
            self._active_bucket = None

        if len(self._completed) < self.window:
            return None

        vpin_value = self._compute_vpin()
        return VPINResult(as_of=trade.timestamp, value=vpin_value, buckets=self.buckets)

    def flush(self) -> Optional[VPINResult]:
        """Force the current bucket to be closed if it contains volume."""

        if self._active_bucket and self._active_bucket.total_volume() > 0:
            self._completed.append(self._active_bucket)
            self._active_bucket = None
        if len(self._completed) < self.window:
            return None
        value = self._compute_vpin()
        end_time = self._completed[-1].end
        return VPINResult(as_of=end_time, value=value, buckets=self.buckets)

    def _compute_vpin(self) -> float:
        total_volume = sum(bucket.total_volume() for bucket in self._completed)
        if total_volume == 0:
            return 0.0
        imbalance = sum(bucket.imbalance() for bucket in self._completed)
        return imbalance / total_volume

    def _ensure_bucket(self, timestamp: datetime) -> VPINBucket:
        bucket = self._active_bucket
        if bucket is None:
            bucket = VPINBucket(start=timestamp, end=timestamp)
            self._active_bucket = bucket
            return bucket
        if self.staleness and timestamp - bucket.start >= self.staleness and bucket.total_volume() > 0:
            self._completed.append(bucket)
            self._active_bucket = VPINBucket(start=timestamp, end=timestamp)
            return self._active_bucket
        return bucket


def compute_vpin_series(
    trades: Iterable[TradeTick],
    *,
    bucket_volume: float,
    window: int = 50,
    staleness: Optional[timedelta] = timedelta(minutes=5),
) -> List[VPINResult]:
    """Compute VPIN over a finite set of trades and return each emission."""

    calculator = VPINCalculator(bucket_volume=bucket_volume, window=window, staleness=staleness)
    results: List[VPINResult] = []
    for trade in trades:
        result = calculator.add_trade(trade)
        if result is not None:
            results.append(result)
    flushed = calculator.flush()
    if flushed is not None and flushed not in results:
        results.append(flushed)
    return results
