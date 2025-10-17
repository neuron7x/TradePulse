from datetime import datetime, timezone

import pytest

from risk.limits import LimitConfig
from risk.manager import RiskManager


def test_risk_manager_enforces_notional_limit():
    cfg = LimitConfig(max_position_notional=1000.0)
    rm = RiskManager(cfg)
    rm.update_state(position=0.0, price=100.0, pnl=0.0, timestamp=datetime.now(tz=timezone.utc))
    with pytest.raises(ValueError):
        rm.check_limits({"notional": 2000.0, "pnl": 0.0, "drawdown": 0.0})

