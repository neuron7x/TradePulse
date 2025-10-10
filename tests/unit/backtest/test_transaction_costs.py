import math
from pathlib import Path

import numpy as np
import pytest

from backtest.engine import WalkForwardEngine, walk_forward
from backtest.transaction_costs import (
    BpsSpread,
    CompositeTransactionCostModel,
    FixedBpsCommission,
    FixedSlippage,
    FixedSpread,
    PerUnitCommission,
    PercentVolumeCommission,
    SquareRootSlippage,
    TransactionCostModel,
    VolumeProportionalSlippage,
    load_market_costs,
)


class DummyModel(TransactionCostModel):
    def __init__(self) -> None:
        self.commission_calls: list[tuple[float, float]] = []
        self.spread_calls: list[tuple[float, str | None]] = []
        self.slippage_calls: list[tuple[float, float, str | None]] = []

    def get_commission(self, volume: float, price: float) -> float:
        self.commission_calls.append((volume, price))
        return volume * 0.5

    def get_spread(self, price: float, side: str | None = None) -> float:
        self.spread_calls.append((price, side))
        return price * 0.01

    def get_slippage(self, volume: float, price: float, side: str | None = None) -> float:
        self.slippage_calls.append((volume, price, side))
        return volume * 0.1


def test_component_models_behaviour() -> None:
    assert FixedBpsCommission(10).get_commission(5, 100) == pytest.approx(5 * 100 * 10 * 1e-4)
    assert PercentVolumeCommission(0.5).get_commission(2, 50) == pytest.approx(2 * 50 * 0.5 * 0.01)
    assert PerUnitCommission(1.2).get_commission(3, 10) == pytest.approx(3.6)

    assert FixedSpread(0.25).get_spread(100, "buy") == pytest.approx(0.25)
    assert BpsSpread(5).get_spread(200, "sell") == pytest.approx(200 * 5 * 1e-4)

    assert FixedSlippage(0.05).get_slippage(10, 100) == pytest.approx(0.05)
    assert VolumeProportionalSlippage(0.01).get_slippage(4, 100) == pytest.approx(0.04)
    assert SquareRootSlippage(a=0.1, b=0.5).get_slippage(9, 100) == pytest.approx(100 * (0.1 + 0.5 * math.sqrt(9)))


def test_composite_model_delegates() -> None:
    dummy = DummyModel()
    composite = CompositeTransactionCostModel(
        commission_model=dummy,
        spread_model=dummy,
        slippage_model=dummy,
    )

    assert composite.get_commission(2, 100) == pytest.approx(1.0)
    assert composite.get_spread(50, "buy") == pytest.approx(0.5)
    assert composite.get_slippage(3, 75, "sell") == pytest.approx(0.3)
    assert dummy.commission_calls == [(2, 100)]
    assert dummy.spread_calls == [(50, "buy")]
    assert dummy.slippage_calls == [(3, 75, "sell")]


def test_load_market_costs_from_mapping(tmp_path: Path) -> None:
    config = {
        "X-test": {
            "commission_model": "fixed_bps",
            "commission_params": {"bps": 12},
            "spread_model": "bps",
            "spread_params": {"bps": 4},
            "slippage_model": "volume",
            "slippage_params": {"coefficient": 0.02},
        }
    }
    model = load_market_costs(config, "X-test")
    assert isinstance(model, CompositeTransactionCostModel)
    assert model.get_commission(1, 100) == pytest.approx(100 * 12 * 1e-4)
    assert model.get_spread(100) == pytest.approx(100 * 4 * 1e-4)
    assert model.get_slippage(5, 100) == pytest.approx(5 * 0.02)

    file_config = tmp_path / "markets.yaml"
    file_config.write_text("X-test:\n  commission_bps: 10\n", encoding="utf8")
    file_model = load_market_costs(file_config, "X-test")
    assert file_model.get_commission(2, 50) == pytest.approx(2 * 50 * 10 * 1e-4)


def test_walk_forward_respects_default_fee() -> None:
    prices = np.array([100.0, 101.0, 102.0], dtype=float)

    def signals(_: np.ndarray) -> np.ndarray:
        return np.array([0.0, 1.0, 1.0])

    result = walk_forward(prices, signals, fee=0.01)
    assert result.trades == 1
    assert result.commission_cost == pytest.approx(0.01)
    assert result.spread_cost == pytest.approx(0.0)
    assert result.slippage_cost == pytest.approx(0.0)


def test_walk_forward_market_configuration(tmp_path: Path) -> None:
    config = tmp_path / "markets.yaml"
    config.write_text(
        "Test:\n"
        "  commission_per_unit: 1.25\n"
        "  spread: 0.5\n"
        "  slippage_model: fixed\n"
        "  slippage_params:\n"
        "    value: 0.25\n",
        encoding="utf8",
    )

    prices = np.array([100.0, 100.0, 100.0], dtype=float)

    def signals(_: np.ndarray) -> np.ndarray:
        return np.array([0.0, 1.0, 1.0])

    engine = WalkForwardEngine()
    result = engine.run(
        prices,
        signals,
        fee=0.0,
        cost_config=config,
        market="Test",
    )

    assert result.trades == 1
    assert result.commission_cost == pytest.approx(1.25)
    assert result.spread_cost == pytest.approx(0.5)
    assert result.slippage_cost == pytest.approx(0.25)
    assert result.pnl == pytest.approx(-(1.25 + 0.5 + 0.25))
