# SPDX-License-Identifier: MIT
from __future__ import annotations
import numpy as np

def portfolio_heat(positions) -> float:
    return float(np.sum([abs(p["qty"]*p.get("price",0)) for p in positions]))
