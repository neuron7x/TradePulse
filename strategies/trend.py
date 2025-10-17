"""Reference trend following strategy (MA crossover + ATR filter)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, Optional

from strategies.base import ExitSignal, Strategy, StrategySignal


@dataclass(slots=True)
class TrendStrategyConfig:
    """Hydra compatible configuration for the trend strategy."""

    ma_fast: int = 20
    ma_slow: int = 100
    vol_window: int = 20
    atr_window: int = 20
    chandelier_multiple: float = 3.0
    entry_on_close: bool = True
    min_atr: float = 1e-6
    timeout_bars: Optional[int] = 250


@dataclass(slots=True)
class _TrendBuffers:
    closes: Deque[float] = field(default_factory=lambda: deque(maxlen=512))
    highs: Deque[float] = field(default_factory=lambda: deque(maxlen=512))
    lows: Deque[float] = field(default_factory=lambda: deque(maxlen=512))
    tr_values: Deque[float] = field(default_factory=lambda: deque(maxlen=512))
    last_signal_side: str = "flat"
    last_entry_price: Optional[float] = None
    last_timestamp: Optional[datetime] = None
    bars_in_position: int = 0


class TrendStrategy(Strategy):
    """Moving average crossover with ATR based exit logic."""

    def __init__(self, config: TrendStrategyConfig | None = None) -> None:
        self.config = config or TrendStrategyConfig()
        self._buffers = _TrendBuffers()

    def _update_buffers(self, bar: Dict[str, float]) -> None:
        self._buffers.closes.append(bar["close"])
        self._buffers.highs.append(bar["high"])
        self._buffers.lows.append(bar["low"])

        if len(self._buffers.closes) < 2:
            self._buffers.tr_values.append(0.0)
            return

        prev_close = self._buffers.closes[-2]
        true_range = max(
            self._buffers.highs[-1] - self._buffers.lows[-1],
            abs(self._buffers.highs[-1] - prev_close),
            abs(self._buffers.lows[-1] - prev_close),
        )
        self._buffers.tr_values.append(true_range)

    def _sma(self, values: Deque[float], window: int) -> Optional[float]:
        if len(values) < window:
            return None
        return sum(list(values)[-window:]) / window

    def _atr(self) -> Optional[float]:
        window = self.config.atr_window
        if len(self._buffers.tr_values) < window:
            return None
        return sum(list(self._buffers.tr_values)[-window:]) / window

    def generate_signals(self, bar: Dict[str, float]) -> StrategySignal:
        timestamp: datetime = bar["timestamp"]
        self._update_buffers(bar)
        self._buffers.last_timestamp = timestamp

        ma_fast = self._sma(self._buffers.closes, self.config.ma_fast)
        ma_slow = self._sma(self._buffers.closes, self.config.ma_slow)
        atr = self._atr()

        if ma_fast is None or ma_slow is None or atr is None or atr < self.config.min_atr:
            return StrategySignal(side="flat", strength=0.0, ts_utc=timestamp, extra={})

        strength_raw = (ma_fast - ma_slow) / max(abs(ma_slow), 1e-12)
        strength = max(min(strength_raw, 1.0), -1.0)
        side = "buy" if strength > 0 else "sell" if strength < 0 else "flat"

        if side != "flat":
            self._buffers.last_signal_side = side
            self._buffers.last_entry_price = bar["close"]
            self._buffers.bars_in_position = 0

        extra = {
            "ma_fast": ma_fast,
            "ma_slow": ma_slow,
            "atr": atr,
            "vol_window": self.config.vol_window,
        }
        return StrategySignal(side=side, strength=strength, ts_utc=timestamp, extra=extra)

    def size_positions(self, signal: StrategySignal, risk_state: Dict[str, float]) -> float:
        sigma = risk_state.get("sigma", 0.0) or 1e-8
        target_vol = risk_state.get("target_vol", 0.10)
        max_leverage = risk_state.get("max_leverage", 2.0)

        base_position = (target_vol / sigma) * signal.strength
        return max(min(base_position, max_leverage), -max_leverage)

    def exits(self, bar: Dict[str, float], position: float) -> ExitSignal | None:
        if position == 0:
            self._buffers.bars_in_position = 0
            return None

        atr = self._atr()
        if atr is None:
            return None

        self._buffers.bars_in_position += 1
        chandelier_multiplier = self.config.chandelier_multiple * atr

        entry_price = self._buffers.last_entry_price or bar["close"]
        if position > 0:
            exit_level = max(entry_price - chandelier_multiplier, bar["close"] - chandelier_multiplier)
        else:
            exit_level = min(entry_price + chandelier_multiplier, bar["close"] + chandelier_multiplier)

        timeout_bars = self.config.timeout_bars
        if timeout_bars is not None and self._buffers.bars_in_position >= timeout_bars:
            return ExitSignal(reason="timeout", price_level=bar["close"], ts_utc=self._buffers.last_timestamp)

        if (position > 0 and bar["low"] <= exit_level) or (position < 0 and bar["high"] >= exit_level):
            return ExitSignal(reason="chandelier", price_level=exit_level, ts_utc=self._buffers.last_timestamp)

        ma_fast = self._sma(self._buffers.closes, self.config.ma_fast)
        ma_slow = self._sma(self._buffers.closes, self.config.ma_slow)
        if ma_fast is not None and ma_slow is not None:
            if position > 0 and ma_fast < ma_slow:
                return ExitSignal(reason="ma_reversal", price_level=bar["close"], ts_utc=self._buffers.last_timestamp)
            if position < 0 and ma_fast > ma_slow:
                return ExitSignal(reason="ma_reversal", price_level=bar["close"], ts_utc=self._buffers.last_timestamp)
        return None

    def metadata(self) -> Dict[str, object]:
        return {
            "name": "Trend",
            "version": "2025.1",
            "config": self.config.__dict__,
            "depends_on": ["ATR", "MA"],
        }

