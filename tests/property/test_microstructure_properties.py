# SPDX-License-Identifier: MIT
"""Transformation invariants for microstructure metrics."""
from __future__ import annotations

import pytest

try:
    from hypothesis import assume, given, settings, strategies as st
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("hypothesis not installed", allow_module_level=True)

from core.metrics.microstructure import hasbrouck_information_impulse, kyles_lambda, queue_imbalance


finite_floats = st.floats(
    min_value=-1_000.0,
    max_value=1_000.0,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def paired_series(draw, *, min_size: int = 5, max_size: int = 60) -> tuple[list[float], list[float]]:
    length = draw(st.integers(min_value=min_size, max_value=max_size))
    pairs = draw(
        st.lists(
            st.tuples(finite_floats, finite_floats),
            min_size=length,
            max_size=length,
        )
    )
    left, right = zip(*pairs)
    return list(left), list(right)


@settings(max_examples=150, deadline=None)
@given(
    bids=st.lists(finite_floats, min_size=2, max_size=32),
    asks=st.lists(finite_floats, min_size=2, max_size=32),
    scale=st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False),
)
def test_queue_imbalance_scaling_invariant(
    bids: list[float], asks: list[float], scale: float
) -> None:
    """Queue imbalance should remain unchanged under positive scaling."""

    base = queue_imbalance(bids, asks)
    scaled = queue_imbalance([scale * b for b in bids], [scale * a for a in asks])
    assert scaled == pytest.approx(base, rel=1e-9, abs=1e-9)


@settings(max_examples=150, deadline=None)
@given(
    paired=paired_series(),
    shift_r=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    shift_q=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    scale=st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False),
)
def test_kyles_lambda_transformation_invariants(
    paired: tuple[list[float], list[float]],
    shift_r: float,
    shift_q: float,
    scale: float,
) -> None:
    """Kyle's lambda is invariant to joint scaling, shifts and sign flips."""

    returns, volume = paired
    assume(any(v != volume[0] for v in volume[1:]))
    assume(any(r != returns[0] for r in returns[1:]))

    base = kyles_lambda(returns, volume)
    shifted = kyles_lambda([r + shift_r for r in returns], [q + shift_q for q in volume])
    scaled = kyles_lambda([scale * r for r in returns], [scale * q for q in volume])
    flipped = kyles_lambda([-r for r in returns], [-q for q in volume])

    assert shifted == pytest.approx(base, rel=1e-9, abs=1e-9)
    assert scaled == pytest.approx(base, rel=1e-9, abs=1e-9)
    assert flipped == pytest.approx(base, rel=1e-9, abs=1e-9)


@settings(max_examples=150, deadline=None)
@given(
    paired=paired_series(),
    shift_r=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    shift_q=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    scale=st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False),
)
def test_hasbrouck_transformation_invariants(
    paired: tuple[list[float], list[float]],
    shift_r: float,
    shift_q: float,
    scale: float,
) -> None:
    """Hasbrouck impulse metric shares the same invariants as Kyle's lambda."""

    returns, volume = paired
    assume(any(v != volume[0] for v in volume[1:]))
    assume(any(r != returns[0] for r in returns[1:]))

    base = hasbrouck_information_impulse(returns, volume)
    shifted = hasbrouck_information_impulse(
        [r + shift_r for r in returns],
        [q + shift_q for q in volume],
    )
    scaled = hasbrouck_information_impulse(
        [scale * r for r in returns],
        [scale * q for q in volume],
    )
    flipped = hasbrouck_information_impulse([-r for r in returns], [-q for q in volume])

    assert shifted == pytest.approx(base, rel=1e-9, abs=1e-9)
    assert scaled == pytest.approx(base, rel=1e-9, abs=1e-9)
    assert flipped == pytest.approx(base, rel=1e-9, abs=1e-9)
