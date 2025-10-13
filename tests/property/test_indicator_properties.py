# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    from hypothesis import given, settings, strategies as st
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("hypothesis not installed", allow_module_level=True)

from core.indicators.kuramoto import compute_phase, kuramoto_order, multi_asset_kuramoto
from core.indicators.ricci import build_price_graph, mean_ricci
from core.indicators.temporal_ricci import TemporalRicciAnalyzer


finite_floats = st.floats(
    allow_nan=True,
    allow_infinity=True,
    width=64,
)


@st.composite
def multi_regime_assets(draw) -> list[np.ndarray]:
    segments = draw(st.integers(min_value=2, max_value=5))
    segment_lengths = draw(
        st.lists(st.integers(min_value=16, max_value=96), min_size=segments, max_size=segments)
    )
    start_price = draw(st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False))
    base_series: list[float] = []
    current = float(start_price)
    for length in segment_lengths:
        volatility = draw(st.floats(min_value=0.05, max_value=5.0, allow_nan=False, allow_infinity=False))
        increments = draw(
            st.lists(
                st.floats(
                    min_value=-volatility,
                    max_value=volatility,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=length,
                max_size=length,
            )
        )
        for delta in increments:
            current = max(1e-6, current + float(delta))
            base_series.append(current)

    base_array = np.asarray(base_series, dtype=float)
    asset_count = draw(st.integers(min_value=2, max_value=5))
    assets: list[np.ndarray] = []
    for _ in range(asset_count):
        noise_scale = draw(st.floats(min_value=0.0, max_value=0.3, allow_nan=False, allow_infinity=False))
        noise = np.asarray(
            draw(
                st.lists(
                    st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                    min_size=len(base_array),
                    max_size=len(base_array),
                )
            ),
            dtype=float,
        )
        perturbed = np.maximum(1e-6, base_array + noise_scale * noise)
        assets.append(perturbed)
    return assets


@given(st.lists(finite_floats, min_size=3, max_size=50))
def test_kuramoto_order_handles_non_finite(phases: list[float]) -> None:
    arr = np.asarray(phases, dtype=float)
    result = kuramoto_order(arr)
    assert np.isfinite(result)
    assert 0.0 <= result
    assert result <= 1.0 or np.isclose(result, 1.0, rtol=1e-9, atol=1e-12)


@settings(max_examples=75, deadline=None)
@given(series_list=multi_regime_assets())
def test_multi_asset_kuramoto_stable_across_regimes(series_list: list[np.ndarray]) -> None:
    """Mixtures of calm and volatile regimes should yield stable Kuramoto outputs."""

    result = multi_asset_kuramoto(series_list)
    assert np.isfinite(result)
    assert 0.0 <= result <= 1.0 or np.isclose(result, 1.0, rtol=1e-9, atol=1e-12)

    for asset in series_list:
        phases = compute_phase(asset)
        assert np.isfinite(phases).all()


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
    assert 0.0 <= result
    assert result <= 1.0 or np.isclose(result, 1.0, rtol=1e-9, atol=1e-12)


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
