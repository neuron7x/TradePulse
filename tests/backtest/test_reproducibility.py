from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from cli.tradepulse_cli import _run_backtest
from core.config.cli_models import BacktestConfig, DataSourceConfig, ExecutionConfig, StrategyConfig


@pytest.mark.integration
def test_backtest_run_is_reproducible(tmp_path) -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    prices = [100.0]
    for idx in range(1, 64):
        prices.append(100.0 + np.sin(idx / 5.0))
    timestamps = [(base + timedelta(minutes=idx)).isoformat() for idx in range(len(prices))]
    frame = pd.DataFrame({"timestamp": timestamps, "price": prices})
    csv_path = tmp_path / "prices.csv"
    frame.to_csv(csv_path, index=False)

    cfg = BacktestConfig(
        name="repro-backtest",
        data=DataSourceConfig(path=csv_path),
        strategy=StrategyConfig(
            entrypoint="tests.backtest.strategies:deterministic_signal",
            parameters={"threshold": 0.0},
        ),
        execution=ExecutionConfig(starting_cash=1_000_000.0, fee_bps=0.0),
        results_path=tmp_path / "result.json",
    )

    result_a = _run_backtest(cfg)
    result_b = _run_backtest(cfg)

    assert result_a == result_b
    assert result_a["stats"] == result_b["stats"]
    assert result_a["signals"] == result_b["signals"]
    assert result_a["returns"] == result_b["returns"]
