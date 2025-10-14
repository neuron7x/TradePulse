# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.indicators.temporal_ricci import (
    PriceLevelGraphBuilder,
    TemporalRicciAnalyzer,
)


def _synthetic_series(length: int, volatility: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    increments = rng.normal(0.0, volatility, length)
    return np.cumsum(increments) + 100.0


def test_price_level_graph_builder_creates_edges_based_on_threshold() -> None:
    prices = np.linspace(100.0, 110.0, 32)
    volumes = np.ones(prices.size - 1)
    builder = PriceLevelGraphBuilder(n_levels=8, connection_threshold=0.05)
    graph = builder.build(prices, volumes)
    assert graph.number_of_nodes() == 8
    assert graph.number_of_edges() > 0


def test_price_level_graph_builder_validates_volume_length() -> None:
    prices = np.linspace(100.0, 101.0, 10)
    builder = PriceLevelGraphBuilder(n_levels=5)

    with pytest.raises(ValueError):
        builder.build(prices, np.ones(prices.size - 3))


def test_temporal_ricci_analyzer_reports_metrics() -> None:
    prices = np.concatenate(
        [
            _synthetic_series(400, 0.05, seed=1),
            _synthetic_series(400, 0.25, seed=2),
        ]
    )
    volumes = np.abs(np.sin(np.linspace(0, 4 * np.pi, prices.size - 1))) + 0.1
    dates = pd.date_range("2024-01-01", periods=prices.size, freq="1min")
    df = pd.DataFrame(
        {"close": prices, "volume": np.append(volumes, volumes[-1])}, index=dates
    )

    analyzer = TemporalRicciAnalyzer(window_size=128, n_snapshots=6, n_levels=12)
    result = analyzer.analyze(df)

    assert result.graph_snapshots
    assert result.temporal_curvature <= 0.0
    assert 0.0 <= result.topological_transition_score <= 1.0
    assert 0.0 <= result.structural_stability <= 1.0
    assert 0.0 <= result.edge_persistence <= 1.0


def test_temporal_ricci_analyzer_handles_small_dataset() -> None:
    prices = np.linspace(100.0, 101.0, 32)
    dates = pd.date_range("2024-01-01", periods=prices.size, freq="1min")
    df = pd.DataFrame({"close": prices}, index=dates)

    analyzer = TemporalRicciAnalyzer(window_size=128, n_snapshots=5, n_levels=8)
    outcome = analyzer.analyze(df)

    assert outcome.graph_snapshots == []
    assert outcome.temporal_curvature == 0.0
    assert outcome.topological_transition_score == 0.0
    assert outcome.structural_stability == 1.0
    assert outcome.edge_persistence == 1.0


def test_temporal_ricci_analyzer_requires_close_column_and_non_empty_df() -> None:
    analyzer = TemporalRicciAnalyzer()
    dates = pd.date_range("2024-01-01", periods=4, freq="1min")
    df_missing = pd.DataFrame({"price": np.linspace(100.0, 101.0, 4)}, index=dates)

    with pytest.raises(ValueError):
        analyzer.analyze(df_missing)

    empty_df = pd.DataFrame({"close": []}, index=pd.DatetimeIndex([], name="ts"))

    with pytest.raises(ValueError):
        analyzer.analyze(empty_df)


def test_temporal_transition_score_reacts_to_regime_change() -> None:
    steady_prices = _synthetic_series(512, 0.05, seed=11)
    volatile_prices = np.concatenate(
        [
            _synthetic_series(256, 0.05, seed=21),
            _synthetic_series(256, 0.4, seed=22),
        ]
    )

    dates = pd.date_range("2024-01-01", periods=steady_prices.size, freq="1min")
    steady_df = pd.DataFrame({"close": steady_prices}, index=dates)

    dates_vol = pd.date_range("2024-01-01", periods=volatile_prices.size, freq="1min")
    volatile_df = pd.DataFrame({"close": volatile_prices}, index=dates_vol)

    analyzer = TemporalRicciAnalyzer(window_size=128, n_snapshots=5, n_levels=10)

    steady_result = analyzer.analyze(steady_df)
    volatile_result = analyzer.analyze(volatile_df)

    assert (
        volatile_result.topological_transition_score
        >= steady_result.topological_transition_score
    )
