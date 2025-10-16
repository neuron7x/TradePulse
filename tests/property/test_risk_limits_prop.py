from __future__ import annotations

import pytest

try:
    from hypothesis import given, settings, strategies as st
except ImportError:  # pragma: no cover
    pytest.skip("hypothesis not installed", allow_module_level=True)

from core.data.catalog import normalize_symbol
from execution.risk import LimitViolation, RiskLimits, RiskManager


@pytest.mark.property
@settings(max_examples=200, deadline=None)
@given(
    current=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    qty=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    max_position=st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    side=st.sampled_from(["buy", "sell"]),
)
def test_validate_order_enforces_position_caps(
    current: float, qty: float, max_position: float, side: str
) -> None:
    limits = RiskLimits(
        max_position=max_position,
        max_notional=1e12,
        max_orders_per_interval=0,
    )
    manager = RiskManager(limits)

    symbol = "BTC-USD"
    canonical = normalize_symbol(symbol)
    manager._positions[canonical] = current  # type: ignore[attr-defined]

    price = 1.0
    direction = 1.0 if side == "buy" else -1.0
    projected = current + direction * qty

    if abs(projected) > max_position + 1e-9:
        with pytest.raises(LimitViolation):
            manager.validate_order(symbol, side, qty, price)
    else:
        manager.validate_order(symbol, side, qty, price)
