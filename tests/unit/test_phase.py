# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np

from core.phase.detector import composite_transition, phase_flags


def test_phase_flags_detects_proto_state() -> None:
    state = phase_flags(R=0.2, dH=0.1, kappa_mean=0.05, H=1.0)
    assert state == "proto"


def test_phase_flags_emergent_requires_negative_curvature() -> None:
    state = phase_flags(R=0.8, dH=-0.2, kappa_mean=-0.1, H=0.5)
    assert state == "emergent"


def test_composite_transition_weighted_sum_behavior() -> None:
    R = 0.6
    dH = -0.1
    kappa = -0.2
    H = 1.0
    result = composite_transition(R, dH, kappa, H)
    expected = 0.4 * R + 0.3 * (-dH) + 0.3 * (-kappa)
    assert np.isclose(result, expected)
