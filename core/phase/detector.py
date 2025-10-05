# SPDX-License-Identifier: MIT
from __future__ import annotations
import numpy as np
from core.indicators.kuramoto import kuramoto_order
from core.indicators.entropy import delta_entropy
from core.indicators.ricci import mean_ricci

def phase_flags(R: float, dH: float, kappa_mean: float, H: float):
    if R < 0.4 and dH > 0:
        return "proto"
    if 0.4 < R < 0.7 and dH <= 0:
        return "precognitive"
    if R > 0.75 and dH < 0 and kappa_mean < 0:
        return "emergent"
    if R < 0.7 and dH > 0:
        return "post-emergent"
    return "neutral"

def composite_transition(R: float, dH: float, kappa_mean: float, H: float) -> float:
    # simple normalized combination; tune via backtest
    return float(0.4*R + 0.3*(-dH) + 0.3*(-kappa_mean))
