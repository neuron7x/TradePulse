from __future__ import annotations

import math

import pytest

from execution.order import RiskAwarePositionSizer, position_sizing


def test_size_rejects_non_positive_price() -> None:
    sizer = RiskAwarePositionSizer()
    with pytest.raises(ValueError):
        sizer.size(balance=1_000.0, risk=0.1, price=0.0)
    with pytest.raises(ValueError):
        sizer.size(balance=1_000.0, risk=0.1, price=-1.0)


def test_size_clamps_risk_and_handles_zero_notional() -> None:
    sizer = RiskAwarePositionSizer()

    zero_qty = sizer.size(balance=5_000.0, risk=-0.5, price=100.0)
    assert zero_qty == 0.0

    capped_qty = sizer.size(balance=5_000.0, risk=5.0, price=125.0)
    assert capped_qty == pytest.approx((5_000.0 * 1.0) / 125.0)

    assert position_sizing(2_500.0, 0.0, 200.0) == 0.0


@pytest.mark.parametrize(
    "balance,risk,price",
    [
        (0.01, 0.25, 0.29),
    ],
)
def test_size_biases_down_when_rounding_overshoots(
    monkeypatch: pytest.MonkeyPatch, balance: float, risk: float, price: float
) -> None:
    sizer = RiskAwarePositionSizer()
    original_nextafter = math.nextafter
    calls: list[tuple[float, float]] = []

    def tracked_nextafter(x: float, y: float) -> float:
        calls.append((x, y))
        return original_nextafter(x, y)

    monkeypatch.setattr("execution.order.math.nextafter", tracked_nextafter)

    qty = sizer.size(balance=balance, risk=risk, price=price)

    assert calls, "nextafter should be consulted when the initial quantity exceeds the budget"
    assert qty >= 0.0
    assert qty * price <= balance * min(max(risk, 0.0), 1.0) + 1e-12
