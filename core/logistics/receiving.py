"""Warehouse receiving utilities.

These helpers keep the scope intentionally small so the module can be used by
batch import jobs or interactive tooling alike.  No external persistence layer
is assumed; callers can serialise the resulting data structures as needed.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Dict, Iterable, List, MutableMapping, Optional


class QualityStatus(str, Enum):
    """Quality outcome for a received batch of items."""

    ACCEPTED = "accepted"
    DAMAGED = "damaged"
    EXPIRED = "expired"
    ON_HOLD = "on_hold"
    MISSING_INFO = "missing_info"


@dataclass(frozen=True)
class ReceivedLine:
    """Metadata about a received SKU batch.

    The dataclass remains immutable so instances can be safely cached or passed
    between threads without additional locking.
    """

    sku: str
    quantity: int
    quality: QualityStatus = QualityStatus.ACCEPTED
    lot: Optional[str] = None
    expiration: Optional[date] = None
    notes: Optional[str] = None

    def __post_init__(self) -> None:  # type: ignore[override]
        if not self.sku:
            raise ValueError("sku must be a non-empty string")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")


class ReceivingSession:
    """Tracks the lifecycle of a goods-receiving operation."""

    def __init__(
        self,
        expected_manifest: Optional[MutableMapping[str, int]] = None,
    ) -> None:
        self._expected: Dict[str, int] = {
            sku: int(qty)
            for sku, qty in (expected_manifest or {}).items()
            if int(qty) >= 0
        }
        self._accepted: Dict[str, List[ReceivedLine]] = defaultdict(list)
        self._rejected: Dict[str, List[ReceivedLine]] = defaultdict(list)
        self._unplanned: Dict[str, int] = defaultdict(int)

    def record_delivery(self, line: ReceivedLine) -> None:
        """Register a received batch.

        The method is deliberately strict about the input data: any non-positive
        quantities are rejected by :class:`ReceivedLine`.  Additional validation
        covers whether the SKU was expected.
        """

        if line.quality == QualityStatus.ACCEPTED:
            self._accepted[line.sku].append(line)
            if line.sku not in self._expected:
                self._unplanned[line.sku] += line.quantity
        else:
            self._rejected[line.sku].append(line)

    def accept(
        self,
        sku: str,
        quantity: int,
        *,
        lot: Optional[str] = None,
        expiration: Optional[date] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Convenience wrapper to record an accepted batch."""

        self.record_delivery(
            ReceivedLine(
                sku=sku,
                quantity=quantity,
                quality=QualityStatus.ACCEPTED,
                lot=lot,
                expiration=expiration,
                notes=notes,
            )
        )

    def reject(
        self,
        sku: str,
        quantity: int,
        *,
        reason: QualityStatus,
        lot: Optional[str] = None,
        expiration: Optional[date] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Convenience wrapper to register a rejected batch."""

        if reason == QualityStatus.ACCEPTED:
            raise ValueError("use accept() for accepted batches")
        self.record_delivery(
            ReceivedLine(
                sku=sku,
                quantity=quantity,
                quality=reason,
                lot=lot,
                expiration=expiration,
                notes=notes,
            )
        )

    def accepted_items(self) -> Dict[str, int]:
        """Aggregated quantity per SKU that passed inspection."""

        return {
            sku: sum(line.quantity for line in lines)
            for sku, lines in self._accepted.items()
        }

    def rejected_items(self) -> Dict[str, Dict[QualityStatus, int]]:
        """Aggregated rejections broken down by quality status."""

        breakdown: Dict[str, Dict[QualityStatus, int]] = {}
        for sku, lines in self._rejected.items():
            status_counts: Dict[QualityStatus, int] = defaultdict(int)
            for line in lines:
                status_counts[line.quality] += line.quantity
            breakdown[sku] = dict(status_counts)
        return breakdown

    def shortages(self) -> Dict[str, int]:
        """Expected minus accepted quantities when short."""

        shortages: Dict[str, int] = {}
        accepted = self.accepted_items()
        for sku, expected in self._expected.items():
            delta = expected - accepted.get(sku, 0)
            if delta > 0:
                shortages[sku] = delta
        return shortages

    def overages(self) -> Dict[str, int]:
        """Accepted quantities above expectations."""

        over: Dict[str, int] = {}
        accepted = self.accepted_items()
        for sku, qty in accepted.items():
            if sku not in self._expected:
                continue
            expected = self._expected.get(sku, 0)
            delta = qty - expected
            if delta > 0:
                over[sku] = delta
        return over

    def pending(self) -> Dict[str, int]:
        """Alias for :meth:`shortages` to aid UI code."""

        return self.shortages()

    def summary(self) -> Dict[str, object]:
        """Return a human-friendly snapshot of the session state."""

        shortages = self.shortages()
        summary = {
            "accepted": self.accepted_items(),
            "rejected": self.rejected_items(),
            "shortages": shortages,
            "overages": self.overages(),
            "unplanned": dict(self._unplanned),
        }
        summary["complete"] = not shortages
        return summary

    def iter_all_lines(self) -> Iterable[ReceivedLine]:
        """Iterate over every recorded batch in insertion order."""

        for lines in self._accepted.values():
            yield from lines
        for lines in self._rejected.values():
            yield from lines


class WarehouseInventory:
    """Simple in-memory representation of warehouse stock levels."""

    def __init__(self, initial_stock: Optional[MutableMapping[str, int]] = None) -> None:
        self._stock: Dict[str, int] = {
            sku: int(qty)
            for sku, qty in (initial_stock or {}).items()
            if int(qty) >= 0
        }

    def apply_receiving(self, session: ReceivingSession) -> None:
        """Apply accepted batches from ``session`` to the inventory."""

        for sku, quantity in session.accepted_items().items():
            self._stock[sku] = self._stock.get(sku, 0) + quantity

    def quantity(self, sku: str) -> int:
        """Return on-hand quantity for ``sku`` (defaults to zero)."""

        return self._stock.get(sku, 0)

    def snapshot(self) -> Dict[str, int]:
        """Return a copy of the current stock map."""

        return dict(self._stock)
