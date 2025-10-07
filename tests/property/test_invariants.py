# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from core.indicators.entropy import entropy
from core.indicators.kuramoto import kuramoto_order

try:  # Optional dependency for property-based coverage
    from hypothesis import given
    from hypothesis import strategies as st
except ImportError:  # pragma: no cover - exercised only without Hypothesis
    given = None
    st = None

TOLERANCE = 1e-8


def _assert_entropy_non_negative(samples: list[float] | np.ndarray) -> None:
    arr = np.asarray(samples, dtype=float)
    result = entropy(arr)
    assert result >= 0.0, f"Entropy must be non-negative, got {result}"


def _assert_kuramoto_order_bounded(phases: list[float] | np.ndarray) -> None:
    angles = np.asarray(phases, dtype=float)
    value = kuramoto_order(angles)
    # Floating-point tolerance for Kuramoto order parameter
    assert -TOLERANCE <= value <= 1.0 + TOLERANCE, (
        f"Kuramoto order parameter {value} out of bounds [-{TOLERANCE}, {1.0 + TOLERANCE}]"
    )


@pytest.mark.parametrize(
    "samples",
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        np.linspace(-5.0, 5.0, 128).tolist(),
        np.sin(np.linspace(0.0, 4 * np.pi, 256)).tolist(),
    ],
    ids=["all-zeros", "linear-spread", "sine-wave"],
)
def test_entropy_non_negative_samples(samples):
    _assert_entropy_non_negative(samples)


@pytest.mark.parametrize(
    "phases",
    [
        [0.0] * 16,
        np.linspace(-np.pi, np.pi, 64).tolist(),
        (np.pi / 3 * np.ones(48)).tolist(),
        np.linspace(-np.pi / 2, np.pi / 2, 33).tolist(),
    ],
    ids=["synchronised", "full-span", "aligned", "semi-span"],
)
def test_kuramoto_order_bounded_samples(phases):
    _assert_kuramoto_order_bounded(phases)


if st is not None and given is not None:

    @given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=10, max_size=200))
    def test_entropy_non_negative_property(data):
        _assert_entropy_non_negative(data)

    @given(
        st.lists(
            st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False),
            min_size=5,
            max_size=300,
        )
    )
    def test_kuramoto_order_bounded_property(data):
        _assert_kuramoto_order_bounded(data)
else:

    @pytest.mark.skip(reason="Hypothesis not installed; property-based variants skipped")
    def test_entropy_non_negative_property():  # pragma: no cover - skip-only wrapper
        pass

    @pytest.mark.skip(reason="Hypothesis not installed; property-based variants skipped")
    def test_kuramoto_order_bounded_property():  # pragma: no cover - skip-only wrapper
        pass
