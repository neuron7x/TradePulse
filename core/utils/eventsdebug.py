from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from backtest.event_driven import Portfolio
from backtest.events import Event, FillEvent, MarketEvent, OrderEvent, SignalEvent


@dataclass(slots=True)
class ReplaySummary:
    """Summary statistics produced from replaying an event log."""

    final_cash: float
    position: float
    equity_curve: tuple[float, ...]
    trades: int
    total_slippage: float


def replay_event_log(
    events: Sequence[Event] | Iterable[Event],
    *,
    initial_capital: float,
    fee_per_unit: float,
    symbol: str | None = None,
) -> ReplaySummary:
    """Replay an ordered stream of backtest events and derive aggregate metrics.

    The function mirrors the accounting rules applied inside
    :class:`backtest.event_driven.EventDrivenBacktestEngine`. It is intended for
    debugging sessions where a recorded event journal needs to be validated for
    reproducibility.

    Args:
        events: Ordered iterable of events captured from the event-driven engine.
        initial_capital: Starting cash balance used by the original run.
        fee_per_unit: Fee applied per filled unit in the original run.
        symbol: Optional explicit symbol if the log does not contain market events.

    Returns:
        A :class:`ReplaySummary` describing the reconstructed portfolio state.

    Raises:
        TypeError: If the log contains an unsupported event type.
    """

    materialised = list(events)
    if not materialised:
        return ReplaySummary(
            final_cash=float(initial_capital),
            position=0.0,
            equity_curve=(),
            trades=0,
            total_slippage=0.0,
        )

    derived_symbol = symbol
    if derived_symbol is None:
        for event in materialised:
            if isinstance(event, MarketEvent):
                derived_symbol = event.symbol
                break
        if derived_symbol is None:
            derived_symbol = "asset"

    portfolio = Portfolio(symbol=derived_symbol, initial_capital=initial_capital, fee_per_unit=fee_per_unit)
    total_slippage = 0.0

    for event in materialised:
        if isinstance(event, MarketEvent):
            portfolio.on_market_event(event)
        elif isinstance(event, FillEvent):
            portfolio.on_fill(event)
            total_slippage += float(event.slippage)
        elif isinstance(event, (SignalEvent, OrderEvent)):
            # Signals and orders do not directly mutate portfolio state during replay.
            continue
        else:
            raise TypeError(f"Unsupported event type in replay: {type(event)!r}")

    equity_curve = tuple(float(value) for value in portfolio.equity_curve or [])
    final_cash = equity_curve[-1] if equity_curve else float(initial_capital)

    return ReplaySummary(
        final_cash=final_cash,
        position=float(portfolio.position),
        equity_curve=equity_curve,
        trades=portfolio.trades,
        total_slippage=total_slippage,
    )


__all__ = ["ReplaySummary", "replay_event_log"]
