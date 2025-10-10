from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from backtest.performance import (
    PerformanceReport,
    compute_performance_metrics,
    export_performance_report,
)


def test_compute_performance_metrics_basic() -> None:
    equity_curve = np.array([100.0, 102.0, 101.0, 105.0], dtype=float)
    pnl = equity_curve - np.concatenate(([100.0], equity_curve[:-1]))
    position_changes = np.array([0.5, -0.2, 0.3], dtype=float)

    report = compute_performance_metrics(
        equity_curve=equity_curve,
        pnl=pnl,
        position_changes=position_changes,
        initial_capital=100.0,
        max_drawdown=-4.0,
        periods_per_year=4,
    )

    assert math.isfinite(report.sharpe_ratio)
    assert report.max_drawdown == -4.0
    assert pytest.approx(report.turnover, rel=1e-9) == float(np.sum(np.abs(position_changes)))
    assert pytest.approx(report.hit_ratio, rel=1e-9) == 2.0 / 3.0
    assert report.expected_shortfall is not None


def test_export_performance_report(tmp_path: Path) -> None:
    report = PerformanceReport(
        sharpe_ratio=1.25,
        sortino_ratio=None,
        cagr=0.12,
        max_drawdown=-0.08,
        expected_shortfall=-0.04,
        turnover=2.5,
        hit_ratio=0.55,
    )

    path = export_performance_report("My Strategy!", report, directory=tmp_path)
    assert path.parent == tmp_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["strategy"] == "My Strategy!"
    assert pytest.approx(payload["performance"]["sharpe_ratio"], rel=1e-9) == 1.25
    assert payload["performance"]["sortino_ratio"] is None
