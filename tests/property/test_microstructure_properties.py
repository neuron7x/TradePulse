# SPDX-License-Identifier: MIT
"""Transformation invariants for microstructure metrics."""
from __future__ import annotations

import numpy as np
import pytest

try:
    from hypothesis import assume, given, settings, strategies as st
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("hypothesis not installed", allow_module_level=True)

from core.metrics.microstructure import hasbrouck_information_impulse, kyles_lambda, queue_imbalance


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hasbrouck_transformation_invariants(seed: int) -> None:
    """Regression example showing Hasbrouck invariance for random draws."""

    rng = np.random.default_rng(seed)
    returns = rng.normal(size=128)
    signed_volume = rng.normal(size=128)

    base = hasbrouck_information_impulse(returns, signed_volume)

    shifted = hasbrouck_information_impulse(returns + 5.0, signed_volume - 3.0)
    assert shifted == pytest.approx(base, rel=1e-9, abs=1e-9)

    scaled = hasbrouck_information_impulse(returns * 2.5, signed_volume * 4.0)
    assert scaled == pytest.approx(base, rel=1e-9, abs=1e-9)


finite_floats = st.floats(
    min_value=-1_000.0,
    max_value=1_000.0,
    allow_nan=False,
    allow_infinity=False,
)

_EPS = 1e-12


def _centered(values: list[float] | tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr - np.mean(arr)


def _centered_signed_sqrt(values: list[float] | tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    transformed = np.sign(arr) * np.sqrt(np.abs(arr))
    return transformed - np.mean(transformed)


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
    assume(np.linalg.norm(_centered(volume)) > _EPS)

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
    scale_r=st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False),
    scale_q=st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False),
)
def test_hasbrouck_expected_transform_behaviour(
    paired: tuple[list[float], list[float]],
    shift_r: float,
    scale_r: float,
    scale_q: float,
) -> None:
    """Hasbrouck impulse is invariant to affine rescaling of inputs."""

    returns, volume = paired
    assume(any(v != volume[0] for v in volume[1:]))
    assume(any(r != returns[0] for r in returns[1:]))
    assume(np.linalg.norm(_centered(returns)) > _EPS)
    assume(np.linalg.norm(_centered_signed_sqrt(volume)) > _EPS)

    base = hasbrouck_information_impulse(returns, volume)

    shifted_returns = hasbrouck_information_impulse(
        [r + shift_r for r in returns],
        volume,
    )
    assume(np.linalg.norm(_centered([r + shift_r for r in returns])) > _EPS)

    scaled_returns = hasbrouck_information_impulse(
        [scale_r * r for r in returns],
        volume,
    )
    assume(np.linalg.norm(_centered([scale_r * r for r in returns])) > _EPS)

    scaled_volume = hasbrouck_information_impulse(
        returns,
        [scale_q * q for q in volume],
    )
    assume(
        np.linalg.norm(_centered_signed_sqrt([scale_q * q for q in volume])) > _EPS
    )
    flipped = hasbrouck_information_impulse([-r for r in returns], [-q for q in volume])

    assert shifted_returns == pytest.approx(base, rel=1e-9, abs=1e-9)
    assert scaled_returns == pytest.approx(base, rel=1e-9, abs=1e-9)
    assert scaled_volume == pytest.approx(base, rel=1e-9, abs=1e-9)
    assert flipped == pytest.approx(base, rel=1e-9, abs=1e-9)
