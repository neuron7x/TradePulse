from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, cast

import numpy as np
import pandas as pd

try:  # SciPy is optional in lightweight environments
    from scipy import signal as _signal
except Exception:  # pragma: no cover - executed when SciPy unavailable
    _signal = None

from core.utils.cache import (
    BackfillPlan,
    IndicatorCache,
    IndicatorCacheEntry,
    IndicatorCacheKey,
)

from .base import BaseFeature, FeatureResult


class TimeFrame(Enum):
    """Discrete trading horizons expressed in seconds."""

    M1 = 60
    M5 = 300
    M15 = 900
    H1 = 3600

    @property
    def pandas_freq(self) -> str:
        """Return the pandas frequency string for resampling."""

        return f"{int(self.value)}s"

    @property
    def seconds(self) -> int:
        """Expose the time frame in seconds for downstream consumers."""

        return int(self.value)

    def __str__(self) -> str:  # pragma: no cover - tiny helper
        return self.name


@dataclass(slots=True)
class KuramotoResult:
    """Per-timeframe Kuramoto order parameter and associated metadata."""

    order_parameter: float
    mean_phase: float
    window: int


@dataclass(slots=True)
class MultiScaleResult:
    """Aggregate multi-scale consensus metrics produced by the analyzer."""

    consensus_R: float
    cross_scale_coherence: float
    dominant_scale: Optional[TimeFrame]
    adaptive_window: int
    timeframe_results: Mapping[TimeFrame, KuramotoResult]
    skipped_timeframes: Sequence[TimeFrame]


def _hilbert_phase(series: np.ndarray) -> np.ndarray:
    """Return the instantaneous phase of the provided series."""

    x = np.asarray(series, dtype=float)
    if x.size == 0:
        raise ValueError("phase extraction requires at least one sample")
    if not np.all(np.isfinite(x)):
        finite = x[np.isfinite(x)]
        if finite.size == 0:
            x = np.zeros_like(x)
        else:
            fill_value = float(np.mean(finite))
            x = np.where(np.isfinite(x), x, fill_value)
    if _signal is None:
        # fallback: leverage FFT-based analytic signal
        n = x.size
        X = np.fft.fft(x)
        h = np.zeros(n)
        if n % 2 == 0:
            h[0] = h[n // 2] = 1
            h[1 : n // 2] = 2
        else:
            h[0] = 1
            h[1 : (n + 1) // 2] = 2
        analytic = np.fft.ifft(X * h)
    else:
        detrended = _signal.detrend(x)
        analytic = _signal.hilbert(detrended)
    return np.angle(analytic)


def _kuramoto(phases: np.ndarray) -> tuple[float, float]:
    """Compute Kuramoto order parameter and mean phase."""

    complex_mean = np.mean(np.exp(1j * phases))
    return float(np.abs(complex_mean)), float(np.angle(complex_mean))


class WaveletWindowSelector:
    """Selects an analysis window via wavelet energy concentration."""

    def __init__(self, min_window: int = 64, max_window: int = 512, *, wavelet: str = "ricker", levels: int = 16) -> None:
        if min_window <= 0 or max_window <= 0:
            raise ValueError("window bounds must be positive")
        if min_window > max_window:
            raise ValueError("min_window must be <= max_window")

        min_window = int(min_window)
        max_window = int(max_window)

        levels = int(levels)
        if levels <= 0:
            raise ValueError("levels must be positive")

        self.min_window = min_window
        self.max_window = max_window
        self.wavelet = wavelet
        self.levels = max(2, levels)

    def select_window(self, prices: Sequence[float]) -> int:
        if self.max_window > 1_048_576:
            raise ValueError("max_window is excessively large for efficient wavelet analysis")
        if self.levels > 8192:
            raise ValueError("levels is excessively large and could exhaust memory during wavelet selection")
        values = np.asarray(prices, dtype=float)
        if values.size == 0:
            raise ValueError("cannot select window from empty price series")
        if _signal is None:
            # geometric mean ensures deterministic, scale-sensitive fallback
            return int(np.sqrt(self.min_window * self.max_window))

        widths = np.linspace(self.min_window, self.max_window, self.levels)
        widths = np.clip(widths, self.min_window, self.max_window)
        widths = np.unique(widths.astype(int))
        widths = widths[widths > 0]
        if widths.size == 0:
            return self.min_window

        try:
            transform = _signal.cwt(values, _signal.ricker, widths)
        except Exception:  # pragma: no cover - SciPy edge cases
            return int(np.sqrt(self.min_window * self.max_window))

        energy = np.sum(transform**2, axis=1)
        best_idx = int(np.argmax(energy))
        best_width = widths[best_idx]
        return int(np.clip(best_width, self.min_window, self.max_window))


class MultiScaleKuramoto:
    """Compute Kuramoto synchronization metrics across multiple horizons."""

    def __init__(
        self,
        *,
        timeframes: Sequence[TimeFrame] | None = None,
        base_window: int = 256,
        use_adaptive_window: bool = True,
        min_samples_per_scale: int = 64,
        selector: WaveletWindowSelector | None = None,
        cache: IndicatorCache | None = None,
    ) -> None:
        if base_window <= 0:
            raise ValueError("base_window must be positive")
        if min_samples_per_scale <= 0:
            raise ValueError("min_samples_per_scale must be positive")

        self.timeframes: tuple[TimeFrame, ...] = tuple(timeframes or (
            TimeFrame.M1,
            TimeFrame.M5,
            TimeFrame.M15,
            TimeFrame.H1,
        ))
        self.base_window = int(base_window)
        self.use_adaptive_window = use_adaptive_window
        self.min_samples_per_scale = int(min_samples_per_scale)
        self.selector = selector or WaveletWindowSelector(
            min_window=max(32, self.base_window // 2),
            max_window=self.base_window * 2,
        )
        self.cache = cache or IndicatorCache(Path(".cache") / "indicators")
        self._cache_namespace = "MultiScaleKuramoto"

    # -- exposed for unit tests -------------------------------------------------
    def _kuramoto_order_parameter(self, phases: np.ndarray) -> tuple[float, float]:
        return _kuramoto(np.asarray(phases, dtype=float))

    # ---------------------------------------------------------------------------
    def _resample_prices(self, series: pd.Series, timeframe: TimeFrame) -> pd.Series:
        resampled = series.resample(timeframe.pandas_freq).last()
        return resampled.ffill().dropna()

    def _window_for_series(self, values: np.ndarray) -> int:
        if self.use_adaptive_window:
            values_list = values.tolist()
            return int(self.selector.select_window(values_list))
        return self.base_window

    def _context_window(self) -> int:
        base = self.base_window
        if self.use_adaptive_window:
            adaptive_ceiling = max(base, getattr(self.selector, "max_window", base))
        else:
            adaptive_ceiling = base
        context = max(adaptive_ceiling, self.min_samples_per_scale)
        return max(64, int(context) * 4)

    def _cache_params(self, timeframe: TimeFrame, price_col: str) -> Mapping[str, Any]:
        return {
            "timeframe": timeframe.name,
            "seconds": timeframe.seconds,
            "base_window": self.base_window,
            "use_adaptive_window": self.use_adaptive_window,
            "min_samples_per_scale": self.min_samples_per_scale,
            "price_col": price_col,
        }

    def analyze(self, df: pd.DataFrame, *, price_col: str = "close") -> MultiScaleResult:
        if price_col not in df.columns:
            raise KeyError(f"column '{price_col}' not found in dataframe")
        series = df[price_col]
        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError("MultiScaleKuramoto requires a DatetimeIndex")
        series = series.sort_index().astype(float)

        timeframe_results: MutableMapping[TimeFrame, KuramotoResult] = {}
        skipped: list[TimeFrame] = []
        windows: list[int] = []

        for timeframe in self.timeframes:
            try:
                sampled = self._resample_prices(series, timeframe)
            except ValueError:
                skipped.append(timeframe)
                continue
            if sampled.empty:
                skipped.append(timeframe)
                cache_key = IndicatorCacheKey(self._cache_namespace, timeframe.name)
                data_hash_empty = self.cache.hash_series(sampled)
                fingerprint, params_hash = self.cache.make_fingerprint(
                    indicator=self._cache_namespace,
                    params=self._cache_params(timeframe, price_col),
                    data_hash=data_hash_empty,
                    code_version=self.cache.code_version,
                    timeframe=timeframe.name,
                )
                self.cache.store_entry(
                    cache_key,
                    fingerprint=fingerprint,
                    data_hash=data_hash_empty,
                    params_hash=params_hash,
                    latest_timestamp=None,
                    row_count=0,
                    payload=TimeFrameCachePayload(result=None, skipped=True, context_tail=()),
                )
                continue

            cache_key = IndicatorCacheKey(self._cache_namespace, timeframe.name)
            data_hash = self.cache.hash_series(sampled)
            fingerprint, params_hash = self.cache.make_fingerprint(
                indicator=self._cache_namespace,
                params=self._cache_params(timeframe, price_col),
                data_hash=data_hash,
                code_version=self.cache.code_version,
                timeframe=timeframe.name,
            )
            entry_generic = self.cache.load_entry(cache_key)
            entry = (
                IndicatorCacheEntry(
                    metadata=entry_generic.metadata,
                    payload=cast(TimeFrameCachePayload, entry_generic.payload),
                )
                if entry_generic is not None
                else None
            )
            plan: BackfillPlan = self.cache.plan_backfill(
                entry,
                fingerprint=fingerprint,
                latest_timestamp=sampled.index[-1] if not sampled.empty else None,
            )
            if plan.cache_hit and not plan.needs_update and entry is not None:
                payload = entry.payload
                if payload.skipped or payload.result is None:
                    skipped.append(timeframe)
                    continue
                timeframe_results[timeframe] = payload.result
                windows.append(payload.result.window)
                continue

            values = np.asarray(sampled.values, dtype=float)
            if plan.incremental and entry is not None:
                payload = entry.payload
                previous_ts = entry.metadata.latest_timestamp_pd()
                if previous_ts is not None:
                    new_segment = sampled[sampled.index > previous_ts]
                else:
                    new_segment = sampled
                if not new_segment.empty:
                    context = np.asarray(payload.context_tail, dtype=float)
                    new_values = np.asarray(new_segment.values, dtype=float)
                    if context.size:
                        values = np.concatenate([context, new_values])
                    else:
                        values = new_values

            phases = _hilbert_phase(values)
            window = min(self._window_for_series(values), phases.size)
            if window < self.min_samples_per_scale:
                skipped.append(timeframe)
                self.cache.store_entry(
                    cache_key,
                    fingerprint=fingerprint,
                    data_hash=data_hash,
                    params_hash=params_hash,
                    latest_timestamp=sampled.index[-1],
                    row_count=int(sampled.size),
                    payload=TimeFrameCachePayload(
                        result=None,
                        skipped=True,
                        context_tail=tuple(np.asarray(sampled.values[-self._context_window():], dtype=float)),
                    ),
                )
                continue

            R, psi = self._kuramoto_order_parameter(phases[-window:])
            result = KuramotoResult(order_parameter=R, mean_phase=psi, window=window)
            timeframe_results[timeframe] = result
            windows.append(window)
            context_tail = tuple(np.asarray(sampled.values[-self._context_window():], dtype=float))
            self.cache.store_entry(
                cache_key,
                fingerprint=fingerprint,
                data_hash=data_hash,
                params_hash=params_hash,
                latest_timestamp=sampled.index[-1],
                row_count=int(sampled.size),
                payload=TimeFrameCachePayload(result=result, skipped=False, context_tail=context_tail),
            )

        if timeframe_results:
            R_values = np.array([res.order_parameter for res in timeframe_results.values()], dtype=float)
            consensus_R = float(np.mean(R_values))
            if R_values.size > 1:
                dispersion = float(np.std(R_values))
                cross_scale_coherence = float(np.clip(1.0 - dispersion, 0.0, 1.0))
            else:
                cross_scale_coherence = 1.0
            dominant_scale = max(
                timeframe_results.items(),
                key=lambda item: item[1].order_parameter,
            )[0]
        else:
            consensus_R = 0.0
            cross_scale_coherence = 0.0
            dominant_scale = None

        adaptive_window = int(np.median(windows)) if windows and self.use_adaptive_window else self.base_window

        return MultiScaleResult(
            consensus_R=consensus_R,
            cross_scale_coherence=cross_scale_coherence,
            dominant_scale=dominant_scale,
            adaptive_window=adaptive_window,
            timeframe_results=dict(timeframe_results),
            skipped_timeframes=tuple(skipped),
        )


class MultiScaleKuramotoFeature(BaseFeature):
    """Feature wrapper exposing multi-scale Kuramoto consensus as a metric."""

    def __init__(self, analyzer: MultiScaleKuramoto | None = None, *, name: str | None = None) -> None:
        super().__init__(name or "multi_scale_kuramoto")
        self.analyzer = analyzer or MultiScaleKuramoto()

    def transform(self, data: pd.DataFrame, **kwargs: object) -> FeatureResult:
        # Extract price_col if passed
        price_col = kwargs.get("price_col", "close")
        if not isinstance(price_col, str):
            price_col = "close"
        result = self.analyzer.analyze(data, price_col=price_col)
        metadata: Dict[str, object] = {
            "adaptive_window": result.adaptive_window,
            "timeframes": [tf.name for tf in result.timeframe_results.keys()],
            "skipped_timeframes": [tf.name for tf in result.skipped_timeframes],
        }
        for tf, res in result.timeframe_results.items():
            metadata[f"R_{tf.name}"] = res.order_parameter
            metadata[f"phase_{tf.name}"] = res.mean_phase
            metadata[f"window_{tf.name}"] = res.window
        if result.dominant_scale is not None:
            metadata["dominant_timeframe"] = result.dominant_scale.name
        metadata["cross_scale_coherence"] = result.cross_scale_coherence
        return FeatureResult(name=self.name, value=result.consensus_R, metadata=metadata)


__all__ = [
    "MultiScaleKuramoto",
    "MultiScaleKuramotoFeature",
    "MultiScaleResult",
    "KuramotoResult",
    "TimeFrame",
    "WaveletWindowSelector",
]


def analyze_simple(
    df: pd.DataFrame,
    *,
    price_col: str = "close",
    window: int = 128,
) -> MultiScaleResult:
    """Legacy helper retained for backwards compatibility in smoke tests."""

    analyzer = MultiScaleKuramoto(use_adaptive_window=False, base_window=window, min_samples_per_scale=min(window, 64))
    return analyzer.analyze(df, price_col=price_col)

@dataclass(slots=True)
class TimeFrameCachePayload:
    """Payload stored for each timeframe in the indicator cache."""

    result: KuramotoResult | None
    skipped: bool
    context_tail: tuple[float, ...]

