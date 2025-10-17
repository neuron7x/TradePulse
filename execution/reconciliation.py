# SPDX-License-Identifier: MIT
"""Utilities for reconciling OMS state with venue execution data."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping, MutableMapping

from domain import OrderStatus

__all__ = [
    "FillRecord",
    "ReconciliationDiscrepancy",
    "ReconciliationReport",
]


@dataclass(slots=True, frozen=True)
class FillRecord:
    """Normalized venue fill aggregated by order identifier."""

    order_id: str
    symbol: str
    quantity: float
    average_price: float | None
    executed_at: datetime | None = None
    venue: str | None = None
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.order_id:
            raise ValueError("order_id must be provided")
        if not self.symbol:
            raise ValueError("symbol must be provided")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.average_price is not None and self.average_price <= 0:
            raise ValueError("average_price must be positive when provided")
        if self.executed_at is not None:
            timestamp = self.executed_at
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            else:
                timestamp = timestamp.astimezone(timezone.utc)
            object.__setattr__(self, "executed_at", timestamp)
        if self.metadata is not None:
            object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def notional(self) -> float:
        """Return the notional value associated with the fill."""

        if self.average_price is None:
            return 0.0
        return self.quantity * self.average_price

    def merge(self, other: "FillRecord") -> "FillRecord":
        """Combine two fills for the same order into a single aggregate."""

        if self.order_id != other.order_id:
            raise ValueError("cannot merge fills for different orders")
        if self.symbol != other.symbol:
            raise ValueError("cannot merge fills with mismatched symbols")

        total_quantity = self.quantity + other.quantity
        notional = self.notional + other.notional
        if total_quantity <= 0:
            raise ValueError("aggregate quantity must be positive")
        if notional > 0:
            average_price: float | None = notional / total_quantity
        else:
            average_price = self.average_price or other.average_price

        executed_at = self.executed_at
        if executed_at is None or (
            other.executed_at is not None
            and (executed_at is None or other.executed_at > executed_at)
        ):
            executed_at = other.executed_at

        venue = other.venue or self.venue

        metadata: Mapping[str, object] | None
        if self.metadata or other.metadata:
            merged: MutableMapping[str, object] = {}
            if self.metadata:
                merged.update(self.metadata)
            if other.metadata:
                merged.update(other.metadata)
            metadata = dict(merged)
        else:
            metadata = None

        return FillRecord(
            order_id=self.order_id,
            symbol=self.symbol,
            quantity=total_quantity,
            average_price=average_price,
            executed_at=executed_at,
            venue=venue,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the fill record into a JSON-friendly payload."""

        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "quantity": self.quantity,
            "average_price": self.average_price,
            "executed_at": self.executed_at.isoformat()
            if self.executed_at is not None
            else None,
            "venue": self.venue,
            "metadata": dict(self.metadata) if self.metadata is not None else None,
        }


@dataclass(slots=True, frozen=True)
class ReconciliationDiscrepancy:
    """Detailed mismatch detected during reconciliation."""

    order_id: str
    field: str
    expected: float | str | None
    actual: float | str | None
    delta: float | None = None
    corrected: bool = False
    message: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "order_id": self.order_id,
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
            "delta": self.delta,
            "corrected": self.corrected,
            "message": self.message,
        }


@dataclass(slots=True)
class ReconciliationReport:
    """Structured reconciliation results with operational context."""

    generated_at: datetime
    total_orders: int
    total_fills: int
    matched: int = 0
    corrected: int = 0
    missing_in_exchange: list[str] = field(default_factory=list)
    missing_in_oms: list[FillRecord] = field(default_factory=list)
    discrepancies: list[ReconciliationDiscrepancy] = field(default_factory=list)
    total_notional_delta: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "total_orders": self.total_orders,
            "total_fills": self.total_fills,
            "matched": self.matched,
            "corrected": self.corrected,
            "missing_in_exchange": list(self.missing_in_exchange),
            "missing_in_oms": [item.to_dict() for item in self.missing_in_oms],
            "discrepancies": [item.to_dict() for item in self.discrepancies],
            "total_notional_delta": self.total_notional_delta,
        }

    @property
    def requires_attention(self) -> bool:
        """Return ``True`` when manual follow-up is required."""

        return bool(self.missing_in_exchange or self.missing_in_oms or self.discrepancies)

    @property
    def severity(self) -> OrderStatus:
        """Classify the reconciliation outcome for dashboards."""

        if self.missing_in_exchange or any(
            not item.corrected for item in self.discrepancies
        ):
            return OrderStatus.REJECTED
        if self.discrepancies:
            return OrderStatus.PARTIALLY_FILLED
        return OrderStatus.FILLED

    def summary(self) -> str:
        """Return a human-readable summary string."""

        return (
            "reconciliation matched="
            f"{self.matched}/{self.total_orders} corrected={self.corrected} "
            f"missing_exchange={len(self.missing_in_exchange)} "
            f"missing_oms={len(self.missing_in_oms)}"
        )
