from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class OrderSide(str, Enum):
    """Supported trading directions."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Supported order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Lifecycle states for an order."""

    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(slots=True)
class Order:
    """Trading order aggregate with lifecycle helpers and validation."""

    symbol: str
    side: OrderSide | str
    quantity: float
    price: float | None = None
    order_type: OrderType | str = OrderType.MARKET
    stop_price: float | None = None
    order_id: str | None = None
    status: OrderStatus | str = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float | None = None
    rejection_reason: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(init=False)

    def __post_init__(self) -> None:
        self.side = OrderSide(self.side)
        self.order_type = OrderType(self.order_type)
        self.status = OrderStatus(self.status)
        self.updated_at = self.created_at
        self._validate()

    def _validate(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must be provided")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.price is not None and self.price <= 0:
            raise ValueError("price must be positive when provided")
        if self.stop_price is not None and self.stop_price <= 0:
            raise ValueError("stop_price must be positive when provided")
        if self.average_price is not None and self.average_price <= 0:
            raise ValueError("average_price must be positive when provided")
        if self.filled_quantity < 0:
            raise ValueError("filled_quantity cannot be negative")
        if self.filled_quantity > self.quantity:
            raise ValueError("filled_quantity cannot exceed order quantity")

    def mark_submitted(self, order_id: str) -> None:
        """Assign an identifier after handing the order to an execution venue."""

        if not order_id:
            raise ValueError("order_id must be provided")
        self.order_id = order_id
        self.status = OrderStatus.OPEN
        self.updated_at = datetime.now(timezone.utc)

    def record_fill(self, quantity: float, price: float) -> None:
        """Register a fill, updating average price and status."""

        if quantity <= 0:
            raise ValueError("fill quantity must be positive")
        if price <= 0:
            raise ValueError("fill price must be positive")
        if self.filled_quantity + quantity > self.quantity + 1e-9:
            raise ValueError("fill quantity exceeds remaining order quantity")

        weight_existing = self.filled_quantity
        weight_new = quantity
        if weight_existing + weight_new == 0:
            raise ValueError("cannot compute average price with zero total quantity")

        if self.average_price is None:
            blended_price = price
        else:
            blended_price = (
                self.average_price * weight_existing + price * weight_new
            ) / (weight_existing + weight_new)

        self.average_price = blended_price
        self.filled_quantity += quantity
        self.updated_at = datetime.now(timezone.utc)

        if self.filled_quantity >= self.quantity - 1e-9:
            self.status = OrderStatus.FILLED
            self.filled_quantity = self.quantity
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

    def cancel(self) -> None:
        """Cancel the order."""

        if self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            return
        self.status = OrderStatus.CANCELLED
        self.updated_at = datetime.now(timezone.utc)

    def reject(self, reason: str | None = None) -> None:
        """Mark the order as rejected."""

        self.status = OrderStatus.REJECTED
        if reason:
            self.rejection_reason = reason
        self.updated_at = datetime.now(timezone.utc)

    def synchronize_execution(
        self,
        *,
        filled_quantity: float,
        average_price: float | None,
        status: OrderStatus | str | None = None,
    ) -> None:
        """Synchronize execution state with an external source.

        Args:
            filled_quantity: Quantity acknowledged by the venue.
            average_price: Average execution price, if available.
            status: Optional explicit status override.

        Raises:
            ValueError: If inputs violate order invariants.
        """

        if filled_quantity < 0:
            raise ValueError("filled_quantity cannot be negative")
        if filled_quantity > self.quantity + 1e-9:
            raise ValueError("filled_quantity cannot exceed order quantity")
        if average_price is not None and average_price <= 0:
            raise ValueError("average_price must be positive when provided")

        if status is not None:
            resolved_status = OrderStatus(status)
        else:
            if filled_quantity >= self.quantity - 1e-9:
                resolved_status = OrderStatus.FILLED
                filled_quantity = self.quantity
            elif filled_quantity > 0:
                resolved_status = OrderStatus.PARTIALLY_FILLED
            elif self.status in {OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED}:
                resolved_status = OrderStatus.OPEN
            else:
                resolved_status = self.status

        self.filled_quantity = filled_quantity
        self.average_price = average_price
        self.status = resolved_status
        self.updated_at = datetime.now(timezone.utc)

    @property
    def remaining_quantity(self) -> float:
        """Quantity still outstanding."""

        return max(self.quantity - self.filled_quantity, 0.0)

    @property
    def is_active(self) -> bool:
        """Return whether the order can still receive fills."""

        return self.status in {
            OrderStatus.PENDING,
            OrderStatus.OPEN,
            OrderStatus.PARTIALLY_FILLED,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the order into a transport-friendly representation."""

        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "order_type": self.order_type.value,
            "stop_price": self.stop_price,
            "order_id": self.order_id,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "average_price": self.average_price,
            "rejection_reason": self.rejection_reason,
            "remaining_quantity": self.remaining_quantity,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


__all__ = ["Order", "OrderSide", "OrderStatus", "OrderType"]
