from __future__ import annotations

import math

import numpy as np
import pytest

from core.neuro.quantile import P2Quantile


def test_quantile_stays_within_observed_range():
    q = P2Quantile(0.75)
    data = np.linspace(-1.0, 1.0, 101, dtype=np.float32)
    for value in data:
        q.update(float(value))
        assert -1.0 <= q.quantile <= 1.0


def test_quantile_tracks_constant_stream():
    q = P2Quantile(0.3)
    for _ in range(32):
        q.update(0.42)
    assert math.isclose(q.quantile, 0.42, rel_tol=1e-6, abs_tol=1e-6)


def test_quantile_returns_nan_when_empty():
    q = P2Quantile(0.25)
    assert math.isnan(q.quantile)
