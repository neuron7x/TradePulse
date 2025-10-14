"""Unit tests for analytics.signal pipelines and model selection."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

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
        "macd_signal",
        "macd_histogram",
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


def test_feature_pipeline_float_precision_consistency() -> None:
    frame_64 = _sample_market_frame(160)
    frame_32 = frame_64.astype(np.float32)
    cfg = FeaturePipelineConfig(technical_windows=(5, 12), microstructure_window=30)
    pipeline = SignalFeaturePipeline(cfg)

    features_64 = pipeline.transform(frame_64)
    features_32 = pipeline.transform(frame_32)

    common_columns = [col for col in features_64.columns if col in features_32.columns]
    features_64 = features_64[common_columns]
    features_32 = features_32[common_columns]

    mask = features_64.notna() & features_32.notna()
    assert mask.to_numpy().any(), "There should be overlapping finite observations"

    stacked_64 = features_64.where(mask).stack()
    stacked_32 = features_32.where(mask).stack()
    np.testing.assert_allclose(stacked_64.values, stacked_32.values, rtol=5e-4, atol=1e-6)
    assert not np.isinf(features_32.to_numpy(dtype=float)).any(), "No overflow should occur in float32 path"


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


def test_leakage_gate_special_value_handling() -> None:
    index = pd.RangeIndex(start=0, stop=3)
    features = pd.DataFrame(
        {
            "a": [1.0, np.inf, -np.inf],
            "b": [0.5, np.nan, 2.0],
        },
        index=index,
    )
    target = pd.Series([0.1, np.inf, -0.3], index=index)
    gate = LeakageGate(lag=0, dropna=False)

    cleaned_features, cleaned_target = gate.apply(features, target)

    assert np.isnan(cleaned_features.loc[1, "a"])
    assert np.isnan(cleaned_features.loc[2, "a"])
    assert np.isnan(cleaned_target.loc[1])
    assert cleaned_target.loc[2] == target.loc[2]


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


@pytest.mark.parametrize(
    "window, expected_non_nan",
    [
        (1, 0),
        (5, 6),
        (10, 1),
        (50, 0),
    ],
)
def test_microstructure_window_edge_cases(window: int, expected_non_nan: int) -> None:
    frame = _sample_market_frame(10)
    cfg = FeaturePipelineConfig(technical_windows=(3,), microstructure_window=window)
    pipeline = SignalFeaturePipeline(cfg)
    features = pipeline.transform(frame)

    kyles_col = f"kyles_lambda_{window}"
    hasbrouck_col = f"hasbrouck_{window}"

    assert kyles_col in features
    assert hasbrouck_col in features

    assert features[kyles_col].notna().sum() == expected_non_nan
    assert features[hasbrouck_col].notna().sum() == expected_non_nan


def test_feature_pipeline_handles_empty_frame() -> None:
    frame = pd.DataFrame(
        {
            "close": pd.Series(dtype=float),
            "high": pd.Series(dtype=float),
            "low": pd.Series(dtype=float),
            "volume": pd.Series(dtype=float),
            "bid_volume": pd.Series(dtype=float),
            "ask_volume": pd.Series(dtype=float),
            "signed_volume": pd.Series(dtype=float),
        }
    )
    pipeline = SignalFeaturePipeline(FeaturePipelineConfig(technical_windows=(3,), microstructure_window=4))
    features = pipeline.transform(frame)

    assert features.empty


def test_macd_features_match_golden_baseline() -> None:
    df = pd.read_csv("data/golden/indicator_macd_baseline.csv", parse_dates=["ts"])
    frame = df.set_index("ts")[["close"]]
    pipeline = SignalFeaturePipeline(FeaturePipelineConfig())
    features = pipeline.transform(frame)

    macd_cols = ["macd", "macd_signal", "macd_histogram"]
    result = features[macd_cols]

    expected = df.set_index("ts")[["macd", "signal", "histogram"]].rename(
        columns={"signal": "macd_signal", "histogram": "macd_histogram"}
    )

    pd.testing.assert_frame_equal(result, expected, atol=1e-10, rtol=0.0)
