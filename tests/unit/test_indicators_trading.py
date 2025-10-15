"""Tests for lightweight trading indicators."""

from __future__ import annotations

import numpy as np

from core.indicators.kuramoto import compute_phase, kuramoto_order
from core.indicators.trading import KuramotoIndicator


def test_kuramoto_indicator_matches_order_parameter() -> None:
    """KuramotoIndicator should match direct Kuramoto order evaluations."""

    rng = np.random.default_rng(1234)
    prices = np.cumsum(rng.normal(scale=0.5, size=64)) + 100.0
    indicator = KuramotoIndicator(window=16, coupling=1.0)

    result = indicator.compute(prices)

    phases = compute_phase(np.asarray(prices, dtype=float))
    expected = np.zeros_like(phases, dtype=float)
    min_samples = min(indicator.window, 10)
    for idx in range(phases.size):
        start = max(0, idx - indicator.window + 1)
        count = idx - start + 1
        if count < min_samples:
            continue
        expected[idx] = float(kuramoto_order(phases[start : idx + 1]))

    assert np.allclose(result, expected, rtol=1e-6, atol=1e-6)
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_kuramoto_indicator_clips_with_coupling() -> None:
    """KuramotoIndicator should clamp amplified synchrony into [0, 1]."""

    indicator = KuramotoIndicator(window=8, coupling=5.0)
    prices = np.linspace(100.0, 102.0, num=32)

    result = indicator.compute(prices)

    assert np.all((result >= 0.0) & (result <= 1.0))
    assert np.any(result > 0.0)
