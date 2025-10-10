from __future__ import annotations

import math

import numpy as np

from core.neuro.features import (
    EWEntropy,
    EWEntropyConfig,
    ema_update,
    ewvar_update,
)


def test_ema_update_matches_manual_formula():
    prev = 1.25
    x = -0.5
    span = 5
    alpha = 2.0 / (1.0 + span)
    expected = (1.0 - alpha) * prev + alpha * x
    assert math.isclose(ema_update(prev, x, span), expected, rel_tol=1e-6, abs_tol=1e-6)


def test_ewvar_update_is_positive_and_reactive():
    base = ewvar_update(0.002, 0.001, 0.95, 1e-9)
    shocked = ewvar_update(base, 0.05, 0.95, 1e-9)
    assert base > 0.0
    assert shocked > base


def test_entropy_updates_with_repeated_values():
    cfg = EWEntropyConfig(bins=16, xmin=-1.0, xmax=1.0, decay=0.9)
    ent = EWEntropy(cfg)
    initial = ent.value
    seq = np.zeros(64, dtype=np.float32)
    for v in seq:
        ent.update(float(v))
    later = ent.update(0.1)
    assert later <= initial
    assert math.isclose(ent.value, later, rel_tol=1e-6, abs_tol=1e-6)
