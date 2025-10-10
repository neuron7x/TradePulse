"""Event-driven backtest engine with chunked data ingestion."""
from __future__ import annotations

import heapq
import logging
import queue
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from core.utils.metrics import get_metrics_collector

from .engine import LatencyConfig, OrderBookConfig, Result, SlippageConfig
from .performance import compute_performance_metrics, export_performance_report
from .events import FillEvent, MarketEvent, OrderEvent, SignalEvent
from interfaces.backtest import BacktestEngine

LOGGER = logging.getLogger(__name__)


class MarketDataStream(Iterator[MarketEvent]):
    """Iterator interface for market data events."""

    def __next__(self) -> MarketEvent:  # pragma: no cover - protocol definition
        raise NotImplementedError


class MarketDataHandler:
    """Base class for chunked market data handlers."""

    def stream(self) -> Iterator[Iterable[MarketEvent]]:
        raise NotImplementedError


@dataclass(slots=True)
class ArrayDataHandler(MarketDataHandler):
    """Create market events from an in-memory array with optional chunking."""

    prices: Sequence[float]
    symbol: str = "asset"
    chunk_size: Optional[int] = None

    def stream(self) -> Iterator[Iterable[MarketEvent]]:
        total = len(self.prices)
        if total == 0:
            return

        chunk = int(self.chunk_size or total)
        if chunk <= 0:
            chunk = total

        step = 0
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            events = [
                MarketEvent(symbol=self.symbol, price=float(price), step=step + idx)
                for idx, price in enumerate(self.prices[start:end])
            ]
            step += len(events)
            LOGGER.debug("array chunk %s-%s -> %s events", start, end, len(events))
            yield events


@dataclass(slots=True)
class CSVChunkDataHandler(MarketDataHandler):
    """Stream market data from a CSV file using pandas chunking."""

    path: str
    price_column: str = "close"
    symbol: str = "asset"
    chunk_size: int = 50_000
    parse_dates: bool = False
    date_column: Optional[str] = None
    dtype: Optional[dict[str, str]] = None

    def stream(self) -> Iterator[Iterable[MarketEvent]]:
        reader = pd.read_csv(
            self.path,
            usecols=[self.price_column] if self.date_column is None else [self.date_column, self.price_column],
            parse_dates=[self.date_column] if self.parse_dates and self.date_column else None,
            dtype=self.dtype,
            chunksize=self.chunk_size,
        )

        step = 0
        for chunk in reader:
            prices = chunk[self.price_column].to_numpy(dtype=float, copy=False)
            timestamps: List[pd.Timestamp | None]
            if self.date_column is not None:
                timestamps = list(chunk[self.date_column])
            else:
                timestamps = [None] * len(prices)

            events = [
                MarketEvent(
                    symbol=self.symbol,
                    price=float(price),
                    step=step + idx,
                    timestamp=None if ts is None else ts.to_pydatetime(),
                )
                for idx, (price, ts) in enumerate(zip(prices, timestamps, strict=True))
            ]
            step += len(events)
            LOGGER.debug("csv chunk produced %s events", len(events))
            yield events


class Strategy:
    """Base strategy interface for event-driven backtests."""

    def on_market_event(self, event: MarketEvent) -> Iterable[SignalEvent]:  # pragma: no cover - interface
        raise NotImplementedError


class VectorisedStrategy(Strategy):
    """Adapter turning pre-computed vectorised signals into events."""

    def __init__(self, signals: NDArray[np.float64], *, symbol: str = "asset") -> None:
        self._signals = np.asarray(signals, dtype=float)
        self._symbol = symbol

    @classmethod
    def from_signal_function(
        cls,
        prices: NDArray[np.float64],
        signal_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        *,
        symbol: str = "asset",
    ) -> VectorisedStrategy:
        price_array = np.asarray(prices, dtype=float)
        signals = np.asarray(signal_fn(price_array), dtype=float)
        if signals.shape != price_array.shape:
            raise ValueError("signal_fn must return an array with the same length as prices")
        signals = np.clip(signals, -1.0, 1.0)
        return cls(signals, symbol=symbol)

    def on_market_event(self, event: MarketEvent) -> Iterable[SignalEvent]:
        next_index = event.step + 1
        if next_index >= self._signals.size:
            return ()
        signal_value = float(self._signals[next_index])
        LOGGER.debug(
            "strategy emitted precomputed signal %.4f for future step %s", signal_value, next_index
        )
        return (SignalEvent(symbol=self._symbol, target_position=signal_value, step=event.step),)


@dataclass(slots=True)
class Portfolio:
    """Single-asset portfolio that reacts to fills and market updates."""

    symbol: str
    initial_capital: float
    fee_per_unit: float

    cash: float = 0.0
    position: float = 0.0
    equity_curve: List[float] | None = None
    position_history: List[float] | None = None
    trades: int = 0
    _last_price: float | None = field(init=False, default=None, repr=False)
    _pending_target: float = field(init=False, default=0.0, repr=False)

    def __post_init__(self) -> None:
        self.cash = float(self.initial_capital)
        self.equity_curve = []
        self.position_history = []
        self._pending_target = self.position

    def on_market_event(self, event: MarketEvent) -> None:
        if self._last_price is not None:
            delta = event.price - self._last_price
            self.cash += self.position * delta
        self._last_price = event.price
        self.equity_curve.append(self.cash)
        if self.position_history is not None:
            self.position_history.append(self.position)
        LOGGER.debug("portfolio equity updated to %.4f", self.cash)

    def create_order(self, signal: SignalEvent) -> Optional[OrderEvent]:
        target = float(signal.target_position)
        delta = target - self._pending_target
        if abs(delta) < 1e-12:
            return None
        self._pending_target += delta
        LOGGER.debug(
            "portfolio creating order for %.4f units (target=%.4f)",
            delta,
            self._pending_target,
        )
        return OrderEvent(symbol=self.symbol, quantity=delta, step=signal.step)

    def on_fill(self, event: FillEvent) -> None:
        self.position += event.quantity
        self.cash -= event.fee + event.slippage
        self.trades += 1
        self._pending_target = self.position
        LOGGER.debug(
            "fill processed: qty=%.4f price=%.4f cash=%.4f position=%.4f",
            event.quantity,
            event.price,
            self.cash,
            self.position,
        )
        if self.equity_curve:
            self.equity_curve[-1] = self.cash

    @property
    def last_price(self) -> float:
        if self._last_price is None:
            raise RuntimeError("No market data has been processed yet")
        return self._last_price


class SimulatedExecutionHandler:
    """Simple execution handler that uses a synthetic order book."""

    def __init__(
        self,
        order_book: OrderBookConfig,
        slippage: SlippageConfig,
        fee_per_unit: float,
    ) -> None:
        self._order_book = order_book
        self._slippage = slippage
        self._fee_per_unit = fee_per_unit
        self._price_history: List[float] = []

    def on_market_event(self, event: MarketEvent) -> None:
        self._price_history.append(event.price)

    def execute(self, order: OrderEvent, current_step: int) -> tuple[FillEvent, float]:
        if not self._price_history:
            raise RuntimeError("Cannot execute order before receiving market data")

        prices = np.asarray(self._price_history, dtype=float)
        current_idx = min(current_step, prices.size - 1)
        side = "buy" if order.quantity > 0 else "sell"
        quantity = abs(order.quantity)
        mid_price = float(prices[current_idx])
        best_bid, best_ask = self._best_quotes(mid_price)
        fill_price = self._fill_price(side, quantity, best_bid, best_ask)
        if side == "buy":
            slippage_cost = max(0.0, (fill_price - mid_price) * quantity)
        else:
            slippage_cost = max(0.0, (mid_price - fill_price) * quantity)
        fee = quantity * self._fee_per_unit
        fill = FillEvent(
            symbol=order.symbol,
            quantity=order.quantity,
            price=fill_price,
            fee=fee,
            slippage=slippage_cost,
            step=current_step,
        )
        LOGGER.debug(
            "executed order qty=%.4f fill_price=%.4f slippage=%.6f fee=%.6f",
            order.quantity,
            fill_price,
            slippage_cost,
            fee,
        )
        return fill, slippage_cost

    def _best_quotes(self, mid_price: float) -> tuple[float, float]:
        spread = mid_price * self._order_book.spread_bps * 1e-4
        best_bid = mid_price - spread / 2.0
        best_ask = mid_price + spread / 2.0
        return best_bid, best_ask

    def _fill_price(self, side: str, quantity: float, best_bid: float, best_ask: float) -> float:
        remaining = quantity
        total_cost = 0.0
        filled = 0.0
        depth = tuple(float(max(level, 0.0)) for level in self._order_book.depth_profile)

        for level_idx, capacity in enumerate(depth, start=1):
            if remaining <= 0:
                break
            take = min(remaining, capacity)
            if take <= 0:
                continue
            depth_penalty = self._slippage.depth_impact_bps * (level_idx - 1) * 1e-4
            if side == "buy":
                level_price = best_ask * (1.0 + depth_penalty)
            else:
                level_price = best_bid * (1.0 - depth_penalty)
            total_cost += level_price * take
            filled += take
            remaining -= take

        if remaining > 0:
            depth_penalty = self._slippage.depth_impact_bps * max(len(depth), 1) * 1e-4
            if side == "buy":
                level_price = best_ask * (1.0 + depth_penalty)
            else:
                level_price = best_bid * (1.0 - depth_penalty)
            total_cost += level_price * remaining
            filled += remaining

        avg_price = total_cost / filled if filled else (best_ask if side == "buy" else best_bid)
        directional_adjustment = avg_price * self._slippage.per_unit_bps * 1e-4
        if side == "buy":
            avg_price += directional_adjustment
        else:
            avg_price -= directional_adjustment
        return float(avg_price)


class EventDrivenBacktestEngine(BacktestEngine[Result]):
    """Event-driven backtest engine with memory-aware chunked ingestion."""

    def run(
        self,
        prices: NDArray[np.float64],
        signal_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        *,
        fee: float = 0.0005,
        initial_capital: float = 0.0,
        strategy_name: str = "default",
        latency: LatencyConfig | None = None,
        order_book: OrderBookConfig | None = None,
        slippage: SlippageConfig | None = None,
        data_handler: MarketDataHandler | None = None,
        strategy: Strategy | None = None,
        chunk_size: Optional[int] = None,
    ) -> Result:
        latency_cfg = latency or LatencyConfig()
        order_book_cfg = order_book or OrderBookConfig()
        slippage_cfg = slippage or SlippageConfig()

        price_array = np.asarray(prices, dtype=float)
        if price_array.ndim > 1:
            raise ValueError("prices must be a 1-D array")

        if data_handler is None:
            data_handler = ArrayDataHandler(price_array, chunk_size=chunk_size)

        symbol = getattr(data_handler, "symbol", "asset")

        if strategy is None:
            if price_array.size == 0:
                raise ValueError("prices must be provided when using the default strategy")
            strategy_impl = VectorisedStrategy.from_signal_function(price_array, signal_fn, symbol=symbol)
        else:
            strategy_impl = strategy

        metrics = get_metrics_collector()
        with metrics.measure_backtest(strategy_name) as ctx:
            event_queue: queue.Queue[FillEvent | MarketEvent | OrderEvent | SignalEvent] = queue.Queue()
            delayed: list[tuple[int, int, SignalEvent | OrderEvent | FillEvent]] = []
            counter = 0
            current_step = -1

            portfolio = Portfolio(symbol=symbol, initial_capital=initial_capital, fee_per_unit=fee)
            execution_handler = SimulatedExecutionHandler(order_book_cfg, slippage_cfg, fee)

            total_slippage = 0.0

            def schedule(event: SignalEvent | OrderEvent | FillEvent, delay: int) -> None:
                nonlocal counter
                release = current_step + max(0, delay)
                event.step = release
                heapq.heappush(delayed, (release, counter, event))
                counter += 1
                LOGGER.debug("scheduled %s with delay %s", event.type, delay)

            def release_ready() -> None:
                while delayed and delayed[0][0] <= current_step:
                    _, _, evt = heapq.heappop(delayed)
                    event_queue.put(evt)

            for chunk in data_handler.stream():
                for market_event in chunk:
                    current_step = market_event.step
                    event_queue.put(market_event)
                    release_ready()

                    while True:
                        try:
                            event = event_queue.get_nowait()
                        except queue.Empty:
                            break

                        if isinstance(event, MarketEvent):
                            execution_handler.on_market_event(event)
                            portfolio.on_market_event(event)
                            for signal in strategy_impl.on_market_event(event):
                                schedule(signal, latency_cfg.signal_to_order)
                        elif isinstance(event, SignalEvent):
                            order = portfolio.create_order(event)
                            if order is not None:
                                schedule(order, latency_cfg.order_to_execution)
                        elif isinstance(event, OrderEvent):
                            fill, slippage_cost = execution_handler.execute(event, current_step)
                            total_slippage += slippage_cost
                            schedule(fill, latency_cfg.execution_to_fill)
                        elif isinstance(event, FillEvent):
                            portfolio.on_fill(event)
                        else:  # pragma: no cover - safety net
                            LOGGER.warning("Unhandled event type: %s", type(event))

                        release_ready()

            while delayed:
                next_step, _, evt = heapq.heappop(delayed)
                if next_step > current_step:
                    current_step = next_step
                event_queue.put(evt)
                release_ready()

                while True:
                    try:
                        pending = event_queue.get_nowait()
                    except queue.Empty:
                        break

                    if isinstance(pending, SignalEvent):
                        order = portfolio.create_order(pending)
                        if order is not None:
                            schedule(order, latency_cfg.order_to_execution)
                    elif isinstance(pending, OrderEvent):
                        fill, slippage_cost = execution_handler.execute(pending, current_step)
                        total_slippage += slippage_cost
                        schedule(fill, latency_cfg.execution_to_fill)
                    elif isinstance(pending, FillEvent):
                        portfolio.on_fill(pending)
                    else:  # pragma: no cover - safety net
                        LOGGER.warning("Unhandled delayed event type: %s", type(pending))

                    release_ready()

            equity_curve = np.asarray(portfolio.equity_curve, dtype=float)
            positions = (
                np.asarray(portfolio.position_history, dtype=float)
                if portfolio.position_history is not None
                else np.array([], dtype=float)
            )
            pnl_total = float(equity_curve[-1] - initial_capital) if equity_curve.size else 0.0
            peaks = np.maximum.accumulate(equity_curve) if equity_curve.size else np.array([], dtype=float)
            drawdowns = equity_curve - peaks if peaks.size else np.array([], dtype=float)
            max_dd = float(drawdowns.min()) if drawdowns.size else 0.0

            trades = portfolio.trades

            ctx["pnl"] = pnl_total
            ctx["max_dd"] = max_dd
            ctx["trades"] = trades
            ctx["equity"] = float(equity_curve[-1]) if equity_curve.size else initial_capital
            ctx["status"] = "success"

            if metrics.enabled:
                for step, value in enumerate(equity_curve):
                    metrics.record_equity_point(strategy_name, step, float(value))

            pnl_series = (
                equity_curve
                - np.concatenate(([float(initial_capital)], equity_curve[:-1]))
                if equity_curve.size
                else np.array([], dtype=float)
            )
            position_changes = np.diff(positions) if positions.size else np.array([], dtype=float)
            performance = compute_performance_metrics(
                equity_curve=equity_curve,
                pnl=pnl_series,
                position_changes=position_changes,
                initial_capital=initial_capital,
                max_drawdown=max_dd,
            )
            report_path = export_performance_report(strategy_name, performance)
            ctx["performance"] = performance.as_dict()
            ctx["report_path"] = str(report_path)

            return Result(
                pnl=pnl_total,
                max_dd=max_dd,
                trades=trades,
                equity_curve=equity_curve,
                latency_steps=int(latency_cfg.total_delay),
                slippage_cost=float(total_slippage),
                performance=performance,
                report_path=report_path,
            )


__all__ = [
    "ArrayDataHandler",
    "CSVChunkDataHandler",
    "EventDrivenBacktestEngine",
    "Portfolio",
    "SimulatedExecutionHandler",
    "Strategy",
    "VectorisedStrategy",
]
