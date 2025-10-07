# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from core.indicators.entropy import entropy
from core.indicators.kuramoto import kuramoto_order

hypothesis = pytest.importorskip("hypothesis")
st = pytest.importorskip("hypothesis.strategies", reason="Hypothesis strategies unavailable")

from hypothesis import given

KURAMOTO_ORDER_TOLERANCE = 1e-8


@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=10, max_size=200))
def test_entropy_non_negative(data):
    arr = np.array(data, dtype=float)
    result = entropy(arr)
    assert result >= 0.0, f"Entropy must be non-negative, got {result}"


@given(st.lists(st.floats(min_value=-np.pi, max_value=np.pi), min_size=5, max_size=300))
def test_kuramoto_order_bounded(data):
    phases = np.array(data, dtype=float)
    value = kuramoto_order(phases)
    # Floating-point tolerance for Kuramoto order parameter
    assert -KURAMOTO_ORDER_TOLERANCE <= value <= 1.0 + KURAMOTO_ORDER_TOLERANCE, (
        "Kuramoto order parameter "
        f"{value} out of bounds [-{KURAMOTO_ORDER_TOLERANCE}, {1.0 + KURAMOTO_ORDER_TOLERANCE}]"
    )
