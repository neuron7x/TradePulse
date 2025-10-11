"""Shared numeric tolerances to keep tests stable across platforms."""
from __future__ import annotations

from decimal import Decimal

# Relative tolerance for floating point comparisons across the suite.
FLOAT_REL_TOL = 1e-6

# Absolute tolerance for floating point comparisons.
FLOAT_ABS_TOL = 1e-6

# Timestamp comparisons are performed in seconds at float precision.
TIMESTAMP_ABS_TOL = 1e-6

# Decimal tolerance used when comparing high precision quantities.
DECIMAL_ABS_TOL = Decimal("1e-9")
