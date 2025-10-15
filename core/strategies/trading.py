"""Algorithmic trading strategy implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from core.indicators import HurstIndicator, KuramotoIndicator, VPINIndicator


@dataclass(slots=True)
class TradingStrategy:
    """Base strategy contract for signal generation."""

    symbol: str
    params: Dict[str, float]

    def __post_init__(self) -> None:
        self._initialise()

    def _initialise(self) -> None:
        """Hook for subclasses to build indicators."""

        return None

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return timestamped trading signals."""

        raise NotImplementedError


class KuramotoStrategy(TradingStrategy):
    """Synchronisation-based momentum detection."""

    def _initialise(self) -> None:
        self._indicator = KuramotoIndicator(
            window=int(self.params.get("window", 200)),
            coupling=float(self.params.get("coupling", 1.0)),
        )
        self._threshold = float(self.params.get("sync_threshold", 0.7))

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if "close" not in data:
            raise ValueError("DataFrame must contain 'close' column")
        closes = data["close"].to_numpy(dtype=float, copy=False)
        sync_values = self._indicator.compute(closes)
        if sync_values.size == 0:
            return pd.DataFrame(columns=["timestamp", "symbol", "signal", "confidence"])
        threshold = abs(self._threshold)
        signals = np.full(sync_values.size, "Hold", dtype=object)
        buy_mask = sync_values > threshold
        sell_mask = sync_values < -threshold
        signals[buy_mask] = "Buy"
        signals[sell_mask] = "Sell"
        max_abs = np.max(np.abs(sync_values)) if sync_values.size else 0.0
        if max_abs > 0.0:
            confidence = np.abs(sync_values) / max_abs
        else:
            confidence = np.zeros_like(sync_values)
        confidence = np.clip(confidence, 0.0, 1.0)
        return pd.DataFrame(
            {
                "timestamp": data.index,
                "symbol": self.symbol,
                "signal": signals,
                "confidence": confidence,
            }
        )


class HurstVPINStrategy(TradingStrategy):
    """Blend trend persistence and order flow toxicity signals."""

    def _initialise(self) -> None:
        self._hurst = HurstIndicator(window=int(self.params.get("hurst_window", 100)))
        self._vpin = VPINIndicator(
            bucket_size=int(self.params.get("vpin_bucket_size", 50)),
            threshold=float(self.params.get("vpin_threshold", 0.8)),
        )
        self._hurst_trend_threshold = float(self.params.get("hurst_trend_threshold", 0.6))
        self._vpin_safe_threshold = float(self.params.get("vpin_safe_threshold", 0.5))

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        required = {"close", "volume", "buy_volume", "sell_volume"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")
        closes = data["close"].to_numpy(dtype=float, copy=False)
        volume_fields = data[["volume", "buy_volume", "sell_volume"]].to_numpy(dtype=float, copy=False)
        hurst_values = self._hurst.compute(closes)
        vpin_values = self._vpin.compute(volume_fields)
        size = len(data)
        if size == 0:
            return pd.DataFrame(columns=["timestamp", "symbol", "signal", "confidence"])
        signals = np.full(size, "Hold", dtype=object)
        confidence = np.full(size, 0.5, dtype=float)
        for idx in range(size):
            hurst_value = float(np.clip(hurst_values[idx], 0.0, 1.0))
            vpin_value = float(np.clip(vpin_values[idx], 0.0, 1.0))
            if hurst_value > self._hurst_trend_threshold and vpin_value < self._vpin_safe_threshold:
                signals[idx] = "Buy"
                confidence[idx] = max(min(hurst_value, 1.0 - vpin_value), 0.0)
            elif vpin_value > self._vpin_safe_threshold:
                signals[idx] = "Sell"
                confidence[idx] = vpin_value
            else:
                confidence[idx] = 0.5
        confidence = np.clip(confidence, 0.0, 1.0)
        return pd.DataFrame(
            {
                "timestamp": data.index,
                "symbol": self.symbol,
                "signal": signals,
                "confidence": confidence,
            }
        )


def register_strategies() -> dict[str, type[TradingStrategy]]:
    """Expose strategy constructors for CLI bindings."""

    return {
        "kuramoto": KuramotoStrategy,
        "hurst_vpin": HurstVPINStrategy,
    }
