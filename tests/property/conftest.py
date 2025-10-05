# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--hypothesis-show-statistics", action="store_true", default=False, help="Compatibility flag when Hypothesis is unavailable")
