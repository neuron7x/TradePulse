"""Simplified Uniswap v3/v4 AMM reference strategy."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from math import sqrt
from statistics import pstdev
from typing import Deque, Dict, Optional

from strategies.base import ExitSignal, Strategy, StrategySignal


@dataclass(slots=True)
class AMMConfig:
    dex: str = "uniswap_v3"
    pair: str = "ETH/USDC"
    sigma_window: int = 24
    range_width_sigma: float = 1.5
    rebalance_threshold: str = "out_of_range"
    fee_tier: str = "auto_dynamic"
    base_liquidity: float = 1_000_000.0


@dataclass(slots=True)
class _AMMState:
    prices: Deque[float] = field(default_factory=lambda: deque(maxlen=2048))
    last_timestamp: Optional[datetime] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    liquidity_tokens: float = 0.0
    fee_accrued: float = 0.0


class AMMStrategy(Strategy):
    """Adaptive liquidity provision strategy with volatility scaled ranges."""

    def __init__(self, config: AMMConfig | None = None) -> None:
        self.config = config or AMMConfig()
        self.state = _AMMState()

    def _local_sigma(self) -> Optional[float]:
        if len(self.state.prices) < self.config.sigma_window:
            return None
        window = list(self.state.prices)[-self.config.sigma_window :]
        return pstdev(window) or 1e-8

    def _ensure_range(self, price: float, sigma: float) -> None:
        half_width = sigma * self.config.range_width_sigma
        self.state.lower = max(price - half_width, 1e-8)
        self.state.upper = price + half_width

    def generate_signals(self, bar: Dict[str, float]) -> StrategySignal:
        price = bar["close"]
        timestamp = bar["timestamp"]
        self.state.prices.append(price)
        self.state.last_timestamp = timestamp

        sigma = self._local_sigma()
        if sigma is None:
            return StrategySignal("flat", 0.0, timestamp, extra={})

        if self.state.lower is None or self.state.upper is None:
            self._ensure_range(price, sigma)
            return StrategySignal("flat", 0.0, timestamp, extra={})

        if price < self.state.lower:
            direction = "buy"
            strength = -1.0
        elif price > self.state.upper:
            direction = "sell"
            strength = 1.0
        else:
            direction = "flat"
            strength = 0.0

        if direction != "flat":
            self._ensure_range(price, sigma)

        il = self._impermanent_loss(price)
        extra = {
            "lower": self.state.lower,
            "upper": self.state.upper,
            "sigma": sigma,
            "il": il,
            "fee_apr": self._fee_apr(),
        }
        return StrategySignal(direction, strength, timestamp, extra)

    def _impermanent_loss(self, price: float) -> float:
        if not self.state.prices:
            return 0.0
        entry_price = self.state.prices[0]
        d = price / entry_price if entry_price else 1.0
        return 2 * sqrt(d) / (1 + d) - 1

    def _fee_apr(self) -> float:
        if not self.state.prices:
            return 0.0
        elapsed_hours = max(len(self.state.prices), 1) / 60
        if elapsed_hours <= 0:
            return 0.0
        return (self.state.fee_accrued / max(self.config.base_liquidity, 1.0)) * (24 * 365 / elapsed_hours)

    def size_positions(self, signal: StrategySignal, risk_state: Dict[str, float]) -> float:
        target_vol = risk_state.get("target_vol", 0.10)
        sigma = risk_state.get("sigma", 0.0) or 1e-8
        desired_liquidity = self.config.base_liquidity * min(target_vol / sigma, 5.0)
        if signal.side == "flat":
            return self.state.liquidity_tokens
        self.state.liquidity_tokens = desired_liquidity
        return desired_liquidity

    def exits(self, bar: Dict[str, float], position: float) -> ExitSignal | None:
        timestamp = self.state.last_timestamp or bar["timestamp"]
        price = bar["close"]
        if self.state.lower is None or self.state.upper is None:
            return None
        if price < self.state.lower or price > self.state.upper:
            return ExitSignal("rebalance", price, timestamp)
        return None

    def metadata(self) -> Dict[str, object]:
        return {
            "name": "AMM",
            "version": "2025.1",
            "config": self.config.__dict__,
            "depends_on": [self.config.dex, "Uniswap"],
        }

