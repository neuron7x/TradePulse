"""Unit tests for analytics.signal pipelines and model selection."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from analytics.signals import (
    FeaturePipelineConfig,
    LeakageGate,
    SignalFeaturePipeline,
    SignalModelSelector,
    build_supervised_learning_frame,
    make_default_candidates,
)
from backtest.time_splits import WalkForwardSplitter


def _sample_market_frame(rows: int = 300) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=rows, freq="1h")
    rng = np.random.default_rng(42)
    price = 100 + np.cumsum(rng.normal(0, 0.5, size=rows))
    high = price + rng.normal(0.2, 0.1, size=rows)
    low = price - rng.normal(0.2, 0.1, size=rows)
    volume = rng.integers(1000, 5000, size=rows).astype(float)
    bid_volume = volume * rng.uniform(0.4, 0.6, size=rows)
    ask_volume = volume - bid_volume
    signed_volume = rng.normal(0, 1.0, size=rows) * volume * 0.01
    frame = pd.DataFrame(
        {
            "close": price,
            "high": high,
            "low": low,
            "volume": volume,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "signed_volume": signed_volume,
        },
        index=index,
    )
    return frame


def test_feature_pipeline_generates_expected_columns() -> None:
    frame = _sample_market_frame(120)
    cfg = FeaturePipelineConfig(technical_windows=(5, 10), microstructure_window=20)
    pipeline = SignalFeaturePipeline(cfg)
    features = pipeline.transform(frame)
    expected = {
        "return_1",
        "sma_5",
        "sma_10",
        "ema_5",
        "ema_10",
        "volatility_5",
        "volatility_10",
        "rsi",
        "macd",
        "price_range",
        "log_volume",
        "volume_z",
        "queue_imbalance",
        "kyles_lambda_20",
        "hasbrouck_20",
        "signed_volume_ema",
    }
    assert expected.issubset(features.columns)
    # ensure leakage control ready: there should be NaNs at start because of rolling windows
    assert features.isna().sum().max() > 0


def test_leakage_gate_alignment() -> None:
    frame = _sample_market_frame(60)
    pipeline = SignalFeaturePipeline(FeaturePipelineConfig(technical_windows=(3,)))
    features = pipeline.transform(frame)
    target = frame["close"].pct_change().shift(-1)
    gate = LeakageGate(lag=1, dropna=True)
    gated_features, gated_target = gate.apply(features, target)
    assert not gated_features.isna().any().any()
    assert len(gated_features) == len(gated_target)
    assert len(gated_features) > 0
    # the gating operation should discard at least the first `lag` observations
    assert all(idx >= features.index[gate.lag] for idx in gated_features.index)


def test_model_selector_walk_forward_runs() -> None:
    frame = _sample_market_frame(220)
    cfg = replace(FeaturePipelineConfig(), technical_windows=(5, 10))
    features, target = build_supervised_learning_frame(frame, config=cfg, gate=LeakageGate(lag=0))
    splitter = WalkForwardSplitter(train_window=100, test_window=40, freq="h")
    candidates = [c for c in make_default_candidates() if c.name == "ols"]
    selector = SignalModelSelector(splitter, candidates=candidates)
    evaluations = selector.evaluate(features, target)
    assert evaluations, "At least one evaluation should be produced"
    report = evaluations[0]
    assert "hit_rate" in report.aggregate_metrics
    assert "sharpe_ratio" in report.aggregate_metrics
    assert "total_pnl" in report.aggregate_metrics
    assert not report.regression_report.empty
    # Ensure regression report has the same number of rows as evaluated splits
    assert len(report.regression_report) == len(report.split_details)
