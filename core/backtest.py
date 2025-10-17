"""Backtesting utilities including walk-forward analysis and purged CV."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from math import sqrt
from statistics import mean, pstdev
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import mlflow

from core.execution import ExecutionModel, Order
from core.tca import TCA
from risk.manager import RiskManager
from strategies.base import ExitSignal, Strategy, StrategySignal


@dataclass(slots=True)
class BarData:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    atr: float | None = None

    def as_dict(self) -> Dict[str, float]:
        payload = {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }
        if self.vwap is not None:
            payload["vwap"] = self.vwap
        if self.atr is not None:
            payload["atr"] = self.atr
        return payload


@dataclass(slots=True)
class BacktestResult:
    pnl: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    hit_rate: float
    equity_curve: List[float]
    orders: List[Order]


class PurgedKFold:
    def __init__(self, n_splits: int, embargo: int) -> None:
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, data: Sequence[BarData]) -> Iterator[Tuple[List[int], List[int]]]:
        n = len(data)
        fold_size = n // self.n_splits
        for fold in range(self.n_splits):
            start = fold * fold_size
            stop = start + fold_size
            test_idx = list(range(start, min(stop, n)))
            embargo_start = max(start - self.embargo, 0)
            embargo_end = min(stop + self.embargo, n)
            train_idx = [i for i in range(n) if i < embargo_start or i >= embargo_end]
            yield train_idx, test_idx


class Backtester:
    def __init__(self, execution_model: ExecutionModel, risk_manager: RiskManager, tca: TCA | None = None) -> None:
        self.execution_model = execution_model
        self.risk_manager = risk_manager
        self.tca = tca or TCA()

    def run(self, strategy: Strategy, data: Sequence[BarData]) -> BacktestResult:
        position = 0.0
        equity = 0.0
        pnl_series: List[float] = []
        orders: List[Order] = []
        closes: List[float] = []
        wins = 0
        total_trades = 0
        prev_price = None

        for bar in data:
            bar_dict = bar.as_dict()
            closes.append(bar.close)
            signal = strategy.generate_signals(bar_dict)

            sigma = self._realised_volatility(closes)
            risk_state = {
                "sigma": sigma,
                "target_vol": self.risk_manager.cfg.target_vol,
                "max_leverage": self.risk_manager.cfg.max_leverage,
            }
            desired_position = strategy.size_positions(signal, risk_state)
            if signal.side == "flat":
                desired_position = 0.0
            desired_position = self.risk_manager.clip_position(desired_position, bar.close)

            exit_signal = strategy.exits(bar_dict, position)
            exit_payload = exit_signal.as_dict() if isinstance(exit_signal, ExitSignal) else None

            order = self.execution_model.route(signal.as_dict(), desired_position, exit_payload, bar_dict)
            orders.append(order)
            self.tca.consume(asdict(order), bar_dict)

            if prev_price is not None:
                pnl = (order.price - prev_price) * position
            else:
                pnl = 0.0
            equity += pnl
            pnl_series.append(pnl)
            position = order.quantity
            prev_price = bar.close
            self.risk_manager.update_state(position, bar.close, equity, bar.timestamp)
            state_snapshot = {
                "notional": abs(position * bar.close),
                "pnl": equity,
                "drawdown": self._max_drawdown(pnl_series),
            }
            self.risk_manager.check_limits(state_snapshot)

            if pnl != 0:
                total_trades += 1
                if pnl > 0:
                    wins += 1

            self._log_step(signal, order, pnl, equity)

        metrics = self._metrics(pnl_series, closes)
        return BacktestResult(
            pnl=equity,
            cagr=metrics["cagr"],
            sharpe=metrics["sharpe"],
            sortino=metrics["sortino"],
            calmar=metrics["calmar"],
            hit_rate=wins / total_trades if total_trades else 0.0,
            equity_curve=self._equity_curve(pnl_series),
            orders=orders,
        )

    def _realised_volatility(self, closes: List[float]) -> float:
        if len(closes) < 2:
            return 1e-3
        returns = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
        vol = pstdev(returns) * sqrt(252) if len(returns) > 1 else abs(returns[0])
        return max(vol, 1e-4)

    def _max_drawdown(self, pnl_series: List[float]) -> float:
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in pnl_series:
            equity += pnl
            peak = max(peak, equity)
            if peak > 0:
                max_dd = max(max_dd, (peak - equity) / peak)
        return max_dd

    def _equity_curve(self, pnl_series: List[float]) -> List[float]:
        equity = 0.0
        curve: List[float] = []
        for pnl in pnl_series:
            equity += pnl
            curve.append(equity)
        return curve

    def _metrics(self, pnl_series: List[float], closes: List[float]) -> Dict[str, float]:
        if not pnl_series:
            return {"cagr": 0.0, "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0}
        total_return = sum(pnl_series)
        years = max(len(pnl_series) / 252, 1 / 252)
        cagr = (1 + total_return) ** (1 / years) - 1
        returns = pnl_series
        mean_return = mean(returns)
        vol = pstdev(returns) if len(returns) > 1 else abs(mean_return)
        downside = pstdev([min(r, 0) for r in returns]) if len(returns) > 1 else 0.0
        sharpe = mean_return / vol if vol else 0.0
        sortino = mean_return / downside if downside else 0.0
        calmar = cagr / max(self._max_drawdown(pnl_series), 1e-4)
        return {"cagr": cagr, "sharpe": sharpe, "sortino": sortino, "calmar": calmar}

    def _log_step(self, signal: StrategySignal, order: Order, pnl: float, equity: float) -> None:
        if not mlflow.active_run():
            return
        mlflow.log_metrics({"pnl": pnl, "equity": equity}, step=int(order.ts_utc.timestamp()))


class WalkForwardAnalyzer:
    def __init__(self, backtester: Backtester, window: int, step: int) -> None:
        self.backtester = backtester
        self.window = window
        self.step = step

    def run(self, strategy: Strategy, data: Sequence[BarData]) -> List[BacktestResult]:
        results: List[BacktestResult] = []
        for start in range(0, len(data) - self.window, self.step):
            segment = data[start : start + self.window]
            result = self.backtester.run(strategy, segment)
            results.append(result)
        return results

