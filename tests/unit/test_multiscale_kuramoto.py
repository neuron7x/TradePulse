"""Unit tests for the multi-scale Kuramoto indicator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.indicators.multiscale_kuramoto import MultiScaleKuramoto


def make_frame(values: np.ndarray) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=values.size, freq="min")
    return pd.DataFrame({"close": values}, index=index)


def test_analyze_skips_constant_series() -> None:
    analyzer = MultiScaleKuramoto()
    df = make_frame(np.ones(256))

    result = analyzer.analyze(df)

    assert result.timeframe_results == {}
    assert set(result.skipped_timeframes) == set(analyzer.timeframes)


def test_analyze_handles_reasonable_series() -> None:
    analyzer = MultiScaleKuramoto()
    samples = 512
    values = np.sin(np.linspace(0, 8 * np.pi, samples)) + 0.05 * np.random.default_rng(7).standard_normal(samples)
    df = make_frame(values)

    result = analyzer.analyze(df)

    assert 0.0 <= result.consensus_R <= 1.0
    assert set(result.timeframe_results).issubset(set(analyzer.timeframes))
    assert all(tf not in result.timeframe_results for tf in result.skipped_timeframes)
    assert tuple(sorted(result.skipped_timeframes, key=lambda tf: tf.value)) == result.skipped_timeframes


def test_analyze_rejects_insufficient_history() -> None:
    analyzer = MultiScaleKuramoto()
    df = make_frame(np.linspace(1.0, 2.0, 12))

    result = analyzer.analyze(df)

    assert result.timeframe_results == {}
    assert tuple(analyzer.timeframes) == result.skipped_timeframes
