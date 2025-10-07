# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from core.indicators.entropy import entropy
from core.indicators.kuramoto import kuramoto_order

hypothesis = pytest.importorskip("hypothesis")
strategies = pytest.importorskip(
    "hypothesis.strategies", reason="Hypothesis strategies unavailable"
)

from hypothesis import given


@given(strategies.lists(strategies.floats(allow_nan=False, allow_infinity=False), min_size=10, max_size=200))
def test_entropy_non_negative(data):
    arr = np.array(data, dtype=float)
    result = entropy(arr)
    assert result >= 0.0, f"Entropy must be non-negative, got {result}"


@given(strategies.lists(strategies.floats(min_value=-np.pi, max_value=np.pi), min_size=5, max_size=300))
def test_kuramoto_order_bounded(data):
    phases = np.array(data, dtype=float)
    value = kuramoto_order(phases)
    tolerance = 1e-8
    # Floating-point tolerance for Kuramoto order parameter
    assert -tolerance <= value <= 1.0 + tolerance, (
        f"Kuramoto order parameter {value} out of bounds [-{tolerance}, {1.0 + tolerance}]"
    )
