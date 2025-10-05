# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from backtest.engine import walk_forward


def trend_following_signal(prices: np.ndarray) -> np.ndarray:
    signal = np.zeros_like(prices)
    signal[1:] = np.sign(prices[1:] - prices[:-1])
    return signal


def test_walk_forward_trend_following_strategy() -> None:
    prices = np.array([100.0, 101.5, 101.0, 102.5, 103.0])
    result = walk_forward(prices, trend_following_signal, fee=0.0)
    expected_pnl = 4.0
    assert result.pnl == pytest.approx(expected_pnl, rel=1e-12)
    assert result.trades == 3
    assert result.max_dd <= 0
