from datetime import datetime, timezone

from core.backtest import Backtester, BarData
from core.execution import ExecutionModel
from core.tca import TCA
from risk.limits import LimitConfig
from risk.manager import RiskManager
from strategies.trend import TrendStrategy, TrendStrategyConfig


def _bars() -> list[BarData]:
    bars = []
    for i in range(30):
        price = 100 + i
        bars.append(
            BarData(
                timestamp=datetime.fromtimestamp(i, tz=timezone.utc),
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=1_000,
            )
        )
    return bars


def test_backtester_runs_and_returns_metrics():
    strategy = TrendStrategy(TrendStrategyConfig(ma_fast=2, ma_slow=5, atr_window=2))
    execution_model = ExecutionModel(venue="unit-test")
    risk_manager = RiskManager(LimitConfig())
    backtester = Backtester(execution_model, risk_manager, TCA())
    result = backtester.run(strategy, _bars())
    assert result.pnl is not None
    assert isinstance(result.orders, list)

