from __future__ import annotations

import numpy as np
import pytest

from backtest.monte_carlo import (
    MonteCarloConfig,
    evaluate_scenarios,
    generate_monte_carlo_scenarios,
)


def test_generate_monte_carlo_scenarios_basic() -> None:
    prices = np.linspace(100.0, 110.0, num=20)
    config = MonteCarloConfig(
        n_scenarios=5,
        volatility_scale=(0.5, 0.5),
        lag_range=(0, 2),
        dropout_probability=0.1,
        random_seed=123,
    )

    scenarios = generate_monte_carlo_scenarios(prices, config=config)
    assert len(scenarios) == 5
    for scenario in scenarios:
        assert scenario.prices.shape == prices.shape
        assert scenario.returns.size == prices.size - 1
        assert 0.0 <= scenario.dropout_ratio <= 1.0
        assert scenario.volatility_scale == pytest.approx(0.5)
        assert 0 <= scenario.lag <= 2


def test_evaluate_scenarios_produces_reports() -> None:
    prices = np.linspace(50.0, 55.0, num=10)
    scenarios = generate_monte_carlo_scenarios(
        prices,
        config=MonteCarloConfig(n_scenarios=3, random_seed=1, lag_range=(0, 0)),
    )
    benchmark = np.zeros(prices.size - 1, dtype=float)

    reports = evaluate_scenarios(scenarios, benchmark_returns=benchmark)
    assert len(reports) == 3
    for report in reports:
        payload = report.as_dict()
        assert "alpha" in payload and "information_ratio" in payload
        assert payload["turnover"] is None
        assert payload["beta"] is None or isinstance(payload["beta"], float)
