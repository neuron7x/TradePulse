# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    from hypothesis import given, strategies as st
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("hypothesis not installed", allow_module_level=True)

from core.indicators.kuramoto import kuramoto_order, multi_asset_kuramoto
from core.indicators.ricci import build_price_graph, mean_ricci
from core.indicators.temporal_ricci import TemporalRicciAnalyzer


finite_floats = st.floats(
    allow_nan=True,
    allow_infinity=True,
    width=64,
)


@given(st.lists(finite_floats, min_size=3, max_size=50))
def test_kuramoto_order_handles_non_finite(phases: list[float]) -> None:
    arr = np.asarray(phases, dtype=float)
    result = kuramoto_order(arr)
    assert np.isfinite(result)
    assert 0.0 <= result <= 1.0


@given(
    st.integers(min_value=2, max_value=6),
    st.integers(min_value=8, max_value=64),
    st.data(),
)
def test_multi_asset_kuramoto_supports_variable_windows(
    asset_count: int,
    window: int,
    data: st.DataObject,
) -> None:
    series_list = []
    for _ in range(asset_count):
        samples = data.draw(
            st.lists(finite_floats, min_size=window, max_size=window),
            label="series",
        )
        series_list.append(np.asarray(samples, dtype=float))
    result = multi_asset_kuramoto(series_list)
    assert np.isfinite(result)
    assert 0.0 <= result <= 1.0


@given(st.lists(finite_floats, min_size=3, max_size=40))
def test_mean_ricci_accepts_non_finite_inputs(prices: list[float]) -> None:
    arr = np.asarray(prices, dtype=float)
    graph = build_price_graph(arr)
    curvature = mean_ricci(graph)
    assert np.isfinite(curvature)


@given(
    st.integers(min_value=32, max_value=128),
    st.lists(finite_floats, min_size=256, max_size=512),
)
def test_temporal_ricci_resilient_to_non_finite(window: int, raw_prices: list[float]) -> None:
    length = len(raw_prices)
    index = pd.date_range("2023-01-01", periods=length, freq="T")
    prices = np.asarray(raw_prices, dtype=float)
    volumes = np.linspace(1.0, 2.0, length)
    df = pd.DataFrame({"close": prices, "volume": volumes}, index=index)
    window = min(window, length)
    analyzer = TemporalRicciAnalyzer(window_size=window, n_snapshots=4, retain_history=False)
    result = analyzer.analyze(df)
    assert np.isfinite(result.temporal_curvature)
    assert np.isfinite(result.structural_stability)
