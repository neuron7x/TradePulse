# SPDX-License-Identifier: MIT
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Sequence

from numpy.typing import NDArray

from core.utils.metrics import get_metrics_collector


@dataclass(slots=True)
class LatencyConfig:
    """Configuration of discrete delays in the execution pipeline."""

    signal_to_order: int = 0
    order_to_execution: int = 0
    execution_to_fill: int = 0

    @property
    def total_delay(self) -> int:
        delay = int(self.signal_to_order + self.order_to_execution + self.execution_to_fill)
        return max(0, delay)


@dataclass(slots=True)
class OrderBookConfig:
    """Synthetic limit order book configuration."""

    spread_bps: float = 0.0
    depth_profile: Sequence[float] = (1.0, 0.75, 0.5)


@dataclass(slots=True)
class SlippageConfig:
    """Model slippage incurred at execution time."""

    per_unit_bps: float = 0.0
    depth_impact_bps: float = 0.0


@dataclass(slots=True)
class Result:
    pnl: float
    max_dd: float
    trades: int
    equity_curve: NDArray[np.float64] | None = None
    latency_steps: int = 0
    slippage_cost: float = 0.0


class _SimpleOrderBook:
    """A lightweight LOB simulator that exposes best levels and depth."""

    def __init__(self, prices: NDArray[np.float64], config: OrderBookConfig) -> None:
        self._prices = prices
        self._config = config

    def _best_quotes(self, idx: int) -> tuple[float, float]:
        mid = float(self._prices[min(idx, self._prices.size - 1)])
        spread = mid * self._config.spread_bps * 1e-4
        best_bid = mid - spread / 2.0
        best_ask = mid + spread / 2.0
        return best_bid, best_ask

    def fill_price(self, side: str, quantity: float, idx: int, slippage: SlippageConfig) -> float:
        quantity = float(abs(quantity))
        if quantity == 0.0 or not np.isfinite(quantity):
            bid, ask = self._best_quotes(idx)
            return ask if side == "buy" else bid

        bid, ask = self._best_quotes(idx)
        remaining = quantity
        total_cost = 0.0
        filled = 0.0
        depth = tuple(float(max(level, 0.0)) for level in self._config.depth_profile)

        for level_idx, capacity in enumerate(depth, start=1):
            if remaining <= 0:
                break
            take = min(remaining, capacity)
            if take <= 0:
                continue
            depth_penalty = slippage.depth_impact_bps * (level_idx - 1) * 1e-4
            if side == "buy":
                level_price = ask * (1.0 + depth_penalty)
            else:
                level_price = bid * (1.0 - depth_penalty)
            total_cost += level_price * take
            filled += take
            remaining -= take

        if remaining > 0:
            depth_penalty = slippage.depth_impact_bps * max(len(depth), 1) * 1e-4
            if side == "buy":
                level_price = ask * (1.0 + depth_penalty)
            else:
                level_price = bid * (1.0 - depth_penalty)
            total_cost += level_price * remaining
            filled += remaining

        avg_price = total_cost / filled if filled else (ask if side == "buy" else bid)
        directional_adjustment = avg_price * slippage.per_unit_bps * 1e-4
        if side == "buy":
            avg_price += directional_adjustment
        else:
            avg_price -= directional_adjustment
        return float(avg_price)


def _compute_positions(signals: NDArray[np.float64], latency: LatencyConfig) -> NDArray[np.float64]:
    executed = np.zeros_like(signals, dtype=float)
    schedule: dict[int, float] = {}
    delay = latency.total_delay
    current = 0.0

    for idx, target in enumerate(signals):
        effective_idx = idx + delay
        if effective_idx >= signals.size:
            continue
        schedule[effective_idx] = float(target)

    for idx in range(signals.size):
        if idx in schedule:
            current = schedule[idx]
        executed[idx] = current
    return executed


def walk_forward(
    prices: np.ndarray,
    signal_fn: Callable[[np.ndarray], np.ndarray],
    fee: float = 0.0005,
    initial_capital: float = 0.0,
    strategy_name: str = "default",
    *,
    latency: LatencyConfig | None = None,
    order_book: OrderBookConfig | None = None,
    slippage: SlippageConfig | None = None,
) -> Result:
    """Vectorised walk-forward backtest with configurable execution realism."""

    metrics = get_metrics_collector()

    with metrics.measure_backtest(strategy_name) as ctx:
        price_array = np.asarray(prices, dtype=float)
        if price_array.ndim != 1 or price_array.size < 2:
            raise ValueError("prices must be a 1-D array with at least two observations")

        latency_cfg = latency or LatencyConfig()
        order_book_cfg = order_book or OrderBookConfig()
        slippage_cfg = slippage or SlippageConfig()

        with metrics.measure_signal_generation(strategy_name) as signal_ctx:
            raw_signals = np.asarray(signal_fn(price_array), dtype=float)
            signal_ctx["status"] = "success"

        if raw_signals.shape != price_array.shape:
            raise ValueError("signal_fn must return an array with the same length as prices")

        signals = np.clip(raw_signals, -1.0, 1.0)
        executed_positions = _compute_positions(signals, latency_cfg)
        price_moves = np.diff(price_array)

        positions = executed_positions[1:]
        prev_positions = executed_positions[:-1]
        position_changes = positions - prev_positions

        book = _SimpleOrderBook(price_array, order_book_cfg)
        slippage_costs = np.zeros_like(position_changes)

        for idx, change in enumerate(position_changes):
            qty = float(abs(change))
            if qty == 0.0:
                continue
            side = "buy" if change > 0 else "sell"
            price_index = min(idx + 1, price_array.size - 1)
            fill_price = book.fill_price(side, qty, price_index, slippage_cfg)
            mid_price = float(price_array[price_index])
            if side == "buy":
                slippage_costs[idx] = max(0.0, (fill_price - mid_price) * qty)
            else:
                slippage_costs[idx] = max(0.0, (mid_price - fill_price) * qty)

        trade_costs = np.abs(position_changes) * fee
        pnl = positions * price_moves - trade_costs - slippage_costs

        equity_curve = np.cumsum(pnl) + initial_capital
        peaks = np.maximum.accumulate(equity_curve)
        drawdowns = equity_curve - peaks
        pnl_total = float(pnl.sum())
        max_dd = float(drawdowns.min()) if drawdowns.size else 0.0
        trades = int(np.count_nonzero(position_changes))
        total_slippage = float(slippage_costs.sum())

        if metrics.enabled:
            for step, value in enumerate(equity_curve):
                metrics.record_equity_point(strategy_name, step, float(value))

        ctx["pnl"] = pnl_total
        ctx["max_dd"] = max_dd
        ctx["trades"] = trades
        ctx["equity"] = float(equity_curve[-1]) if equity_curve.size else initial_capital

        return Result(
            pnl=pnl_total,
            max_dd=max_dd,
            trades=trades,
            equity_curve=equity_curve,
            latency_steps=latency_cfg.total_delay,
            slippage_cost=total_slippage,
        )

