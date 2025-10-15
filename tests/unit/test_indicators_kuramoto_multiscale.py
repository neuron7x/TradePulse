# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from tests.tolerances import FLOAT_ABS_TOL, FLOAT_REL_TOL

import core.indicators.multiscale_kuramoto as kuramoto_module
from core.indicators.multiscale_kuramoto import (
    KuramotoResult,
    MultiScaleKuramoto,
    MultiScaleKuramotoFeature,
    MultiScaleResult,
    TimeFrame,
    WaveletWindowSelector,
    _hilbert_phase,
    _kuramoto,
)


def _synth_dataframe(periods: int = 4096) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=periods, freq="1min")
    t = np.arange(periods)
    price = (
        100
        + 5 * np.sin(2 * np.pi * t / 240)
        + 2 * np.sin(2 * np.pi * t / 1024)
        + 0.25 * np.random.default_rng(0).normal(size=periods)
    )
    return pd.DataFrame({"close": price}, index=idx)


def test_timeframe_properties_expose_human_friendly_metadata() -> None:
    assert TimeFrame.M1.pandas_freq == "60s"
    assert TimeFrame.H1.seconds == 3600
    assert str(TimeFrame.M5) == "M5"


def test_wavelet_selector_validates_window_bounds() -> None:
    with pytest.raises(ValueError):
        WaveletWindowSelector(min_window=0, max_window=10)
    with pytest.raises(ValueError):
        WaveletWindowSelector(min_window=128, max_window=64)


def test_wavelet_selector_rejects_excessive_resource_requests() -> None:
    selector = WaveletWindowSelector(min_window=64, max_window=2_000_000)
    with pytest.raises(ValueError):
        selector.select_window([1.0, 2.0, 3.0])

    selector = WaveletWindowSelector(levels=10_000)
    with pytest.raises(ValueError):
        selector.select_window([1.0, 2.0, 3.0])


def test_multiscale_analyzer_requires_price_column() -> None:
    df = _synth_dataframe().rename(columns={"close": "price"})
    analyzer = MultiScaleKuramoto(use_adaptive_window=False)
    with pytest.raises(KeyError):
        analyzer.analyze(df)


def test_multiscale_analyzer_requires_datetime_index() -> None:
    df = _synth_dataframe().reset_index(drop=True)
    analyzer = MultiScaleKuramoto(use_adaptive_window=False)
    with pytest.raises(TypeError):
        analyzer.analyze(df)


def test_multiscale_analyzer_marks_skipped_timeframes_when_insufficient_samples() -> None:
    df = _synth_dataframe(periods=90)
    analyzer = MultiScaleKuramoto(
        timeframes=(TimeFrame.M1, TimeFrame.M15),
        use_adaptive_window=False,
        base_window=64,
        min_samples_per_scale=20,
    )
    result = analyzer.analyze(df)
    assert TimeFrame.M1 in result.timeframe_results
    assert TimeFrame.M15 in result.skipped_timeframes
    assert TimeFrame.M15 not in result.timeframe_results


def test_multiscale_analyzer_uses_selector_for_adaptive_window() -> None:
    class TrackingSelector:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def select_window(self, prices: Sequence[float]) -> int:
            self.calls.append(len(prices))
            return 200

    selector = TrackingSelector()
    analyzer = MultiScaleKuramoto(
        timeframes=(TimeFrame.M1,),
        use_adaptive_window=True,
        base_window=64,
        selector=selector,
        min_samples_per_scale=50,
    )
    df = _synth_dataframe(periods=512)
    result = analyzer.analyze(df)
    assert selector.calls  # ensure selector was invoked
    assert result.adaptive_window == 200


def test_multiscale_feature_reports_metadata_and_custom_price_column() -> None:
    class StubAnalyzer:
        def __init__(self) -> None:
            self.price_cols: list[str] = []

        def analyze(self, _: pd.DataFrame, *, price_col: str = "close") -> MultiScaleResult:
            self.price_cols.append(price_col)
            return MultiScaleResult(
                consensus_R=0.55,
                cross_scale_coherence=0.82,
                dominant_scale=TimeFrame.M5,
                adaptive_window=144,
                timeframe_results={
                    TimeFrame.M1: KuramotoResult(order_parameter=0.42, mean_phase=0.1, window=128),
                    TimeFrame.M5: KuramotoResult(order_parameter=0.68, mean_phase=0.3, window=144),
                },
                skipped_timeframes=(TimeFrame.M15,),
            )

    analyzer = StubAnalyzer()
    feature = MultiScaleKuramotoFeature(analyzer=analyzer, name="calibrated")
    df = _synth_dataframe()
    outcome = feature.transform(df, price_col="custom_price")

    assert analyzer.price_cols == ["custom_price"]
    assert outcome.name == "calibrated"
    assert outcome.value == pytest.approx(0.55, rel=FLOAT_REL_TOL, abs=FLOAT_ABS_TOL)
    assert outcome.metadata["dominant_timeframe"] == "M5"
    assert outcome.metadata["skipped_timeframes"] == ["M15"]
    assert outcome.metadata["cross_scale_coherence"] == pytest.approx(
        0.82, rel=FLOAT_REL_TOL, abs=FLOAT_ABS_TOL
    )
    assert outcome.metadata["R_M1"] == pytest.approx(0.42, rel=FLOAT_REL_TOL, abs=FLOAT_ABS_TOL)
    assert outcome.metadata["window_M5"] == 144


def test_hilbert_phase_fft_fallback_matches_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    scipy_signal = pytest.importorskip("scipy.signal")

    rng = np.random.default_rng(2024)
    t = np.linspace(0.0, 12.0 * np.pi, 4096)
    trend = 0.0005 * np.arange(t.size)
    waveform = np.sin(t) + 0.5 * np.sin(t / 5.0)
    noise = 0.05 * rng.standard_normal(t.shape)
    series = trend + waveform + noise

    original_signal = kuramoto_module._signal
    try:
        kuramoto_module._signal = scipy_signal  # ensure SciPy branch is active
        phases_scipy = _hilbert_phase(series)

        kuramoto_module._signal = None
        phases_fft = _hilbert_phase(series)
    finally:
        kuramoto_module._signal = original_signal

    # Compare on the unit circle to avoid wrap-around issues
    assert np.allclose(
        np.exp(1j * phases_fft),
        np.exp(1j * phases_scipy),
        rtol=1e-6,
        atol=1e-6,
    )

    R_scipy, _ = _kuramoto(phases_scipy)
    R_fft, _ = _kuramoto(phases_fft)
    assert R_fft == pytest.approx(R_scipy, rel=FLOAT_REL_TOL, abs=FLOAT_ABS_TOL)
