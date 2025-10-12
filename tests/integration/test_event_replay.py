from __future__ import annotations

import numpy as np
import pytest

from backtest.event_driven import EventDrivenBacktestEngine
from backtest.events import Event
from core.utils.eventsdebug import ReplaySummary, replay_event_log

pytestmark = pytest.mark.eventsdebug


def _momentum_signal(prices: np.ndarray) -> np.ndarray:
    signal = np.zeros_like(prices)
    signal[1:] = np.sign(prices[1:] - prices[:-1])
    return signal


def test_event_replay_matches_engine_output() -> None:
    prices = np.array([100.0, 101.0, 100.5, 102.0, 103.5, 103.0], dtype=float)
    engine = EventDrivenBacktestEngine()
    captured: list[Event] = []

    initial_capital = 10_000.0
    fee = 0.0005

    result = engine.run(
        prices,
        _momentum_signal,
        fee=fee,
        initial_capital=initial_capital,
        event_recorder=captured.append,
    )

    assert captured, "event recorder should capture market/signals/orders/fills"

    replay = replay_event_log(captured, initial_capital=initial_capital, fee_per_unit=fee)

    assert isinstance(replay, ReplaySummary)
    assert replay.trades == result.trades
    assert replay.final_cash == pytest.approx(float(result.equity_curve[-1]))
    assert replay.total_slippage == pytest.approx(result.slippage_cost)
    assert replay.final_cash - initial_capital == pytest.approx(result.pnl)


class _UnknownEvent(Event):
    def __init__(self) -> None:
        Event.__init__(self, type="UNKNOWN", step=0)


def test_event_replay_rejects_unknown_event_type() -> None:
    with pytest.raises(TypeError):
        replay_event_log([_UnknownEvent()], initial_capital=0.0, fee_per_unit=0.0)
