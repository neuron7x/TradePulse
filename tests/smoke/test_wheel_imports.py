"""Smoke tests executed against built wheels during cibuildwheel runs."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name, attr",
    [
        ("core.indicators.cache", "FileSystemIndicatorCache"),
        ("core.indicators.multiscale_kuramoto", "KuramotoResult"),
        ("backtest.engine", "LatencyConfig"),
        ("execution.oms", "OrderManagementSystem"),
        ("execution.risk", "RiskLimits"),
    ],
)
def test_public_api_imports(module_name: str, attr: str) -> None:
    module = importlib.import_module(module_name)
    assert hasattr(module, attr), f"{module_name} is missing {attr}"
