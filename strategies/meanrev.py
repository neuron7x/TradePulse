"""Reference mean reversion strategy using Bollinger style z-scores."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, pstdev
from typing import Deque, Dict, Optional

from strategies.base import ExitSignal, Strategy, StrategySignal


@dataclass(slots=True)
class MeanReversionConfig:
    lookback: int = 40
    z_entry: float = 1.5
    z_exit: float = 0.25
    max_position: float = 3.0
    trend_filter_window: int = 100
    trend_filter_threshold: float = 0.0
    timeout_bars: int = 20


@dataclass(slots=True)
class _MeanReversionBuffers:
    closes: Deque[float] = field(default_factory=lambda: deque(maxlen=1024))
    last_timestamp: Optional[datetime] = None
    bars_in_position: int = 0


class MeanReversionStrategy(Strategy):
    def __init__(self, config: MeanReversionConfig | None = None) -> None:
        self.config = config or MeanReversionConfig()
        self._buffers = _MeanReversionBuffers()

    def _zscore(self) -> Optional[float]:
        if len(self._buffers.closes) < self.config.lookback:
            return None
        window = list(self._buffers.closes)[-self.config.lookback :]
        mu = mean(window)
        sigma = pstdev(window) or 1e-12
        return (window[-1] - mu) / sigma

    def _trend_filter(self) -> Optional[float]:
        if len(self._buffers.closes) < self.config.trend_filter_window:
            return None
        window = list(self._buffers.closes)[-self.config.trend_filter_window :]
        return window[-1] - window[0]

    def generate_signals(self, bar: Dict[str, float]) -> StrategySignal:
        timestamp: datetime = bar["timestamp"]
        self._buffers.closes.append(bar["close"])
        self._buffers.last_timestamp = timestamp

        zscore = self._zscore()
        trend_bias = self._trend_filter()

        if zscore is None or trend_bias is None:
            return StrategySignal(side="flat", strength=0.0, ts_utc=timestamp, extra={})

        if trend_bias > self.config.trend_filter_threshold and zscore > 0:
            side = "flat"
            strength = 0.0
        elif trend_bias < -self.config.trend_filter_threshold and zscore < 0:
            side = "flat"
            strength = 0.0
        elif zscore >= self.config.z_entry:
            side = "sell"
            strength = min(zscore / 3.0, 1.0)
        elif zscore <= -self.config.z_entry:
            side = "buy"
            strength = min(abs(zscore) / 3.0, 1.0)
        else:
            side = "flat"
            strength = 0.0

        if side != "flat":
            self._buffers.bars_in_position = 0

        extra = {
            "zscore": zscore,
            "trend_bias": trend_bias,
            "lookback": self.config.lookback,
        }
        return StrategySignal(side=side, strength=strength, ts_utc=timestamp, extra=extra)

    def size_positions(self, signal: StrategySignal, risk_state: Dict[str, float]) -> float:
        sigma = risk_state.get("sigma", 0.0) or 1e-8
        target_vol = risk_state.get("target_vol", 0.10)
        desired = -(signal.strength * target_vol) / sigma
        max_position = self.config.max_position
        return max(min(desired, max_position), -max_position)

    def exits(self, bar: Dict[str, float], position: float) -> ExitSignal | None:
        if position == 0.0:
            self._buffers.bars_in_position = 0
            return None

        self._buffers.bars_in_position += 1
        timestamp = self._buffers.last_timestamp

        zscore = self._zscore()
        if zscore is not None:
            if position > 0 and zscore >= -self.config.z_exit:
                return ExitSignal("mean_revert", bar["close"], timestamp)
            if position < 0 and zscore <= self.config.z_exit:
                return ExitSignal("mean_revert", bar["close"], timestamp)

        if self._buffers.bars_in_position >= self.config.timeout_bars:
            return ExitSignal("timeout", bar["close"], timestamp)

        return None

    def metadata(self) -> Dict[str, object]:
        return {
            "name": "MeanReversion",
            "version": "2025.1",
            "config": self.config.__dict__,
            "depends_on": ["Bollinger", "ZScore"],
        }

