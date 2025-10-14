# SPDX-License-Identifier: MIT
"""Walk-forward backtesting engine with execution realism controls.

The engine models signal latency, slippage, spreads, and financing to produce a
traceable equity curve, aligning with the methodology described in
``docs/performance.md`` and the operational checklist in
``docs/runbook_live_trading.md``. It serves as the reference implementation for
portfolio walk-forward evaluation across TradePulse components and feeds
observability metrics expected by ``docs/quality_gates.md``.

Dependencies include NumPy for vectorised PnL computation, transaction cost
models from :mod:`backtest.transaction_costs`, and metrics collectors for
governance-compliant telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from backtest.transaction_costs import (
    PerUnitCommission,
    TransactionCostModel,
    load_market_costs,
)
from core.utils.metrics import get_metrics_collector
from interfaces.backtest import BacktestEngine

from .performance import (
    PerformanceReport,
    compute_performance_metrics,
    export_performance_report,
)


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
    infinite_depth: bool = True


@dataclass(slots=True)
class SlippageConfig:
    """Model slippage incurred at execution time."""

    per_unit_bps: float = 0.0
    depth_impact_bps: float = 0.0
    stochastic_bps: float = 0.0


@dataclass(slots=True)
class PortfolioConstraints:
    """Risk controls applied to target positions before execution."""

    max_gross_exposure: float | None = None
    max_net_exposure: float | None = None
    target_volatility: float | None = None
    volatility_lookback: int = 20


@dataclass(slots=True)
class Result:
    pnl: float
    max_dd: float
    trades: int
    equity_curve: NDArray[np.float64] | None = None
    latency_steps: int = 0
    slippage_cost: float = 0.0
    commission_cost: float = 0.0
    spread_cost: float = 0.0
    financing_cost: float = 0.0
    performance: PerformanceReport | None = None
    report_path: Path | None = None


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


def _apply_portfolio_constraints(
    signals: NDArray[np.float64],
    prices: NDArray[np.float64],
    constraints: PortfolioConstraints,
) -> NDArray[np.float64]:
    adjusted = np.asarray(signals, dtype=float).copy()
    max_gross = constraints.max_gross_exposure
    if max_gross is not None:
        limit = float(abs(max_gross))
        if limit > 0.0:
            adjusted = np.clip(adjusted, -limit, limit)

    max_net = constraints.max_net_exposure
    if max_net is not None:
        limit = float(abs(max_net))
        if limit > 0.0:
            adjusted = np.clip(adjusted, -limit, limit)

    target_vol = constraints.target_volatility
    if target_vol is not None and target_vol > 0.0:
        returns = np.diff(prices) / prices[:-1]
        if returns.size:
            lookback = min(len(returns), max(int(constraints.volatility_lookback), 1))
            window = returns[-lookback:] if lookback else returns
            with np.errstate(invalid="ignore", divide="ignore"):
                if window.size > 1:
                    realized_vol = float(np.std(window, ddof=1))
                else:
                    realized_vol = float(np.std(window))
            if realized_vol > 1e-12:
                scale = float(target_vol) / realized_vol
                adjusted = adjusted * scale
                if max_gross is not None or max_net is not None:
                    candidate_limits = [
                        float(abs(value))
                        for value in (max_gross, max_net)
                        if value is not None
                    ]
                    limit = min(candidate_limits or [1.0])
                    adjusted = np.clip(adjusted, -limit, limit)
    return np.clip(adjusted, -1.0, 1.0)


class WalkForwardEngine(BacktestEngine[Result]):
    """Concrete implementation of :class:`interfaces.backtest.BacktestEngine`."""

    def run(
        self,
        prices: np.ndarray,
        signal_fn: Callable[[np.ndarray], np.ndarray],
        *,
        fee: float = 0.0005,
        initial_capital: float = 0.0,
        strategy_name: str = "default",
        latency: LatencyConfig | None = None,
        order_book: OrderBookConfig | None = None,
        slippage: SlippageConfig | None = None,
        market: str | None = None,
        cost_model: TransactionCostModel | None = None,
        cost_config: str | Path | Mapping[str, Any] | None = None,
        constraints: PortfolioConstraints | None = None,
    ) -> Result:
        """Execute a vectorised walk-forward backtest.

        Args:
            prices: One-dimensional array of prices. The series must be strictly
                positive and length two or greater.
            signal_fn: Callable that maps the price series to normalised target
                positions. The returned array must align with ``prices`` in
                shape.
            fee: Fallback per-unit commission when a market-specific model is not
                supplied.
            initial_capital: Starting equity for the cumulative PnL series.
            strategy_name: Identifier used for metrics, reporting, and file
                exports.
            latency: Optional :class:`LatencyConfig` describing signal, order, and
                fill delays.
            order_book: Optional :class:`OrderBookConfig` modelling spread and
                depth.
            slippage: Optional :class:`SlippageConfig` capturing execution impact.
            market: Market code used to load transaction cost templates via
                :func:`backtest.transaction_costs.load_market_costs`.
            cost_model: Explicit transaction cost model overriding autodetection.
            cost_config: Path-like or mapping configuration source passed to
                :func:`load_market_costs` when ``market`` is provided.
            constraints: Optional :class:`PortfolioConstraints` limiting the target
                positions, as suggested in ``docs/execution.md``.

        Returns:
            :class:`Result` dataclass containing realised PnL, drawdown, trade
            count, equity curve, and cost breakdowns.

        Raises:
            ValueError: If ``prices`` is not a 1-D vector of sufficient length, or
                if ``signal_fn`` produces an array of incompatible shape.

        Examples:
            >>> prices = np.linspace(100, 101, 16)
            >>> def mean_revert(series):
            ...     return np.sign(np.mean(series) - series)
            >>> engine = WalkForwardEngine()
            >>> result = engine.run(prices, mean_revert)
            >>> round(result.pnl, 4)
            0.0

        Notes:
            - PnL is computed as ``position[t] * Δprice[t]`` minus the configured
              cost legs.
            - Latency shifts are applied before cost evaluation to emulate a live
              order pipeline, matching the governance requirements of
              ``docs/runbook_live_trading.md``.
            - The generated :class:`PerformanceReport` is exported to disk for
              audit trails mandated in ``docs/documentation_governance.md``.
        """

        metrics = get_metrics_collector()

        with metrics.measure_backtest(strategy_name) as ctx:
            price_array = np.asarray(prices, dtype=float)
            if price_array.ndim != 1 or price_array.size < 2:
                raise ValueError("prices must be a 1-D array with at least two observations")

            latency_cfg = latency or LatencyConfig()
            order_book_cfg = order_book or OrderBookConfig()
            slippage_cfg = slippage or SlippageConfig()

            transaction_cost_model = cost_model
            if transaction_cost_model is None and market:
                config_source = cost_config
                if config_source is None:
                    default_config = Path("configs/markets.yaml")
                    if default_config.exists():
                        config_source = default_config
                if config_source is not None:
                    transaction_cost_model = load_market_costs(config_source, market)

            if transaction_cost_model is None:
                transaction_cost_model = PerUnitCommission(fee)

            with metrics.measure_signal_generation(strategy_name) as signal_ctx:
                raw_signals = np.asarray(signal_fn(price_array), dtype=float)
                signal_ctx["status"] = "success"

            if raw_signals.shape != price_array.shape:
                raise ValueError("signal_fn must return an array with the same length as prices")

            signals = np.clip(raw_signals, -1.0, 1.0)
            if constraints is not None:
                signals = _apply_portfolio_constraints(signals, price_array, constraints)
            executed_positions = _compute_positions(signals, latency_cfg)
            price_moves = np.diff(price_array)

            positions = executed_positions[1:]
            prev_positions = executed_positions[:-1]
            position_changes = positions - prev_positions

            book = _SimpleOrderBook(price_array, order_book_cfg)
            commission_costs = np.zeros_like(position_changes)
            spread_costs = np.zeros_like(position_changes)
            slippage_costs = np.zeros_like(position_changes)
            financing_costs = np.zeros_like(position_changes)

            for idx, change in enumerate(position_changes):
                prev_position = float(prev_positions[idx])
                price_index = min(idx, price_array.size - 1)
                ref_price = float(price_array[price_index])
                financing_cost = transaction_cost_model.get_financing(
                    prev_position,
                    ref_price,
                )
                financing_costs[idx] = float(financing_cost)

                qty = float(abs(change))
                if qty == 0.0:
                    continue
                side = "buy" if change > 0 else "sell"
                trade_price_index = min(idx + 1, price_array.size - 1)
                mid_price = float(price_array[trade_price_index])
                fill_price = float(mid_price)

                book_fill_price = book.fill_price(side, qty, trade_price_index, slippage_cfg)
                if side == "buy":
                    slippage_costs[idx] += max(0.0, (book_fill_price - mid_price) * qty)
                else:
                    slippage_costs[idx] += max(0.0, (mid_price - book_fill_price) * qty)
                fill_price = float(book_fill_price)

                spread_adj = float(max(transaction_cost_model.get_spread(mid_price, side), 0.0))
                if spread_adj > 0.0:
                    spread_costs[idx] = spread_adj * qty
                    if side == "buy":
                        fill_price += spread_adj
                    else:
                        fill_price -= spread_adj

                model_slippage = transaction_cost_model.get_slippage(qty, mid_price, side)
                slippage_adj = float(max(model_slippage, 0.0))
                if slippage_adj > 0.0:
                    slippage_costs[idx] += slippage_adj * qty
                    if side == "buy":
                        fill_price += slippage_adj
                    else:
                        fill_price -= slippage_adj

                commission = transaction_cost_model.get_commission(qty, fill_price)
                commission_costs[idx] = max(0.0, float(commission))

            pnl = (
                positions * price_moves
                - commission_costs
                - spread_costs
                - slippage_costs
                - financing_costs
            )

            equity_curve = np.cumsum(pnl) + initial_capital
            peaks = np.maximum.accumulate(equity_curve)
            drawdowns = equity_curve - peaks
            pnl_total = float(pnl.sum())
            max_dd = float(drawdowns.min()) if drawdowns.size else 0.0
            trades = int(np.count_nonzero(position_changes))
            total_commission = float(commission_costs.sum())
            total_spread = float(spread_costs.sum())
            total_slippage = float(slippage_costs.sum())
            total_financing = float(financing_costs.sum())

            performance = compute_performance_metrics(
                equity_curve=equity_curve,
                pnl=pnl,
                position_changes=position_changes,
                initial_capital=initial_capital,
                max_drawdown=max_dd,
            )
            report_path = export_performance_report(strategy_name, performance)

            if metrics.enabled:
                for step, value in enumerate(equity_curve):
                    metrics.record_equity_point(strategy_name, step, float(value))

            ctx["pnl"] = pnl_total
            ctx["performance"] = performance.as_dict()
            ctx["report_path"] = str(report_path)
            ctx["max_dd"] = max_dd
            ctx["trades"] = trades
            ctx["equity"] = float(equity_curve[-1]) if equity_curve.size else initial_capital
            ctx["commission_cost"] = total_commission
            ctx["spread_cost"] = total_spread
            ctx["slippage_cost"] = total_slippage
            ctx["financing_cost"] = total_financing

        return Result(
            pnl=pnl_total,
            max_dd=max_dd,
            trades=trades,
            equity_curve=equity_curve,
            latency_steps=int(latency_cfg.total_delay),
            slippage_cost=total_slippage,
            commission_cost=total_commission,
            spread_cost=total_spread,
            financing_cost=total_financing,
            performance=performance,
            report_path=report_path,
        )


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
    market: str | None = None,
    cost_model: TransactionCostModel | None = None,
    cost_config: str | Path | Mapping[str, Any] | None = None,
    constraints: PortfolioConstraints | None = None,
) -> Result:
    """Compatibility wrapper that delegates to :class:`WalkForwardEngine`."""

    engine = WalkForwardEngine()
    return engine.run(
        prices,
        signal_fn,
        fee=fee,
        initial_capital=initial_capital,
        strategy_name=strategy_name,
        latency=latency,
        order_book=order_book,
        slippage=slippage,
        market=market,
        cost_model=cost_model,
        cost_config=cost_config,
        constraints=constraints,
    )

