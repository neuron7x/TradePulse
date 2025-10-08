# SPDX-License-Identifier: MIT
from __future__ import annotations

import math
import random
import time

import numpy as np
import pandas as pd

import pytest

from core.agent.bandits import EpsilonGreedy, UCB1
from core.agent.memory import StrategyMemory, StrategyRecord
from core.agent.strategy import PiAgent, Strategy


def test_epsilon_greedy_prefers_best_arm_when_exploit() -> None:
    agent = EpsilonGreedy(["a", "b"], epsilon=0.0)
    agent.update("a", 0.1)
    agent.update("b", 0.5)
    assert agent.select() == "b"


def test_ucb1_selects_unseen_arm_first() -> None:
    agent = UCB1(["x", "y"])
    choice1 = agent.select()
    agent.update(choice1, 0.1)
    choice2 = agent.select()
    assert {choice1, choice2} == {"x", "y"}


def test_epsilon_greedy_explores(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = EpsilonGreedy(["a", "b"], epsilon=1.0)
    monkeypatch.setattr(random, "choice", lambda seq: seq[1])
    assert agent.select() == "b"


def test_strategy_memory_topk_orders_by_freshness(monkeypatch: pytest.MonkeyPatch) -> None:
    memory = StrategyMemory(decay_lambda=0.0)
    now = time.time()
    rec_old = StrategyRecord("old", (0, 0, 0, 0, 0), score=0.2, ts=now - 10)
    rec_new = StrategyRecord("new", (0, 0, 0, 0, 0), score=0.1, ts=now)
    memory.records = [rec_old, rec_new]
    top = memory.topk(1)
    assert top[0].name == "old"


def test_strategy_memory_cleanup_removes_low_scores() -> None:
    memory = StrategyMemory(decay_lambda=0.0)
    memory.add("keep", (0, 0, 0, 0, 0), 1.0)
    memory.add("drop", (0, 0, 0, 0, 0), -1.0)
    memory.cleanup(min_score=0.5)
    names = {r.name for r in memory.records}
    assert names == {"keep"}


def test_strategy_mutation_changes_numeric_parameters() -> None:
    random.seed(42)
    strategy = Strategy(name="base", params={"alpha": 1.0, "beta": 2})
    mutant = strategy.generate_mutation()
    assert mutant.name.startswith("base_mut")
    assert mutant.params["alpha"] != strategy.params["alpha"]


def test_strategy_simulate_performance_within_expected_range() -> None:
    random.seed(1)
    strategy = Strategy(name="strat", params={})
    score = strategy.simulate_performance(data=None)
    assert -1.0 <= score <= 2.0


def test_strategy_simulate_performance_populates_diagnostics() -> None:
    strategy = Strategy(name="rich", params={"lookback": 12, "threshold": 0.4})
    prices = pd.Series(np.linspace(100.0, 120.0, 256))

    score = strategy.simulate_performance(prices)

    diagnostics = strategy.diagnostics
    assert diagnostics is not None
    assert isinstance(diagnostics.trades, int)
    assert strategy.params["last_equity_curve"] == diagnostics.equity_curve
    assert strategy.params["max_drawdown"] == pytest.approx(diagnostics.max_drawdown)
    assert strategy.params["sharpe"] == pytest.approx(diagnostics.sharpe)
    assert strategy.params["sortino"] == pytest.approx(diagnostics.sortino)
    assert strategy.params["volatility"] == pytest.approx(diagnostics.volatility)
    assert strategy.params["hit_rate"] == pytest.approx(diagnostics.hit_rate)
    assert strategy.params["turnover"] == pytest.approx(diagnostics.turnover)
    assert strategy.params["profit_factor"] == pytest.approx(diagnostics.profit_factor)
    assert strategy.params["value_at_risk"] == pytest.approx(diagnostics.value_at_risk)
    assert strategy.params["conditional_value_at_risk"] == pytest.approx(
        diagnostics.conditional_value_at_risk
    )
    assert strategy.params["terminal_value"] == pytest.approx(diagnostics.terminal_value)
    assert strategy.params["sample_size"] == diagnostics.sample_size
    assert diagnostics.equity_curve[0] == pytest.approx(1.0)
    assert diagnostics.sample_size > 0
    assert -1.0 <= score <= 2.0


def test_strategy_risk_penalty_reduces_score() -> None:
    strategy = Strategy(name="risk", params={"lookback": 8, "threshold": 0.2})
    cliff = pd.Series(np.linspace(100.0, 40.0, 256))

    raw = strategy.simulate_performance(cliff)
    diagnostics = strategy.diagnostics
    assert diagnostics is not None
    baseline = (diagnostics.terminal_value - 1.0) + 0.5 * diagnostics.sharpe
    penalty = 0.1 * min(diagnostics.value_at_risk, diagnostics.conditional_value_at_risk)
    expected = np.clip(baseline - penalty, -1.0, 2.0)
    assert raw == pytest.approx(expected)
    assert diagnostics.value_at_risk >= 0.0
    assert diagnostics.conditional_value_at_risk >= diagnostics.value_at_risk


def test_pi_agent_detects_instability_and_repair() -> None:
    agent = PiAgent(strategy=Strategy(name="s", params={"alpha": 1.0, "beta": math.nan}))
    state = {"R": 0.8, "delta_H": -0.1, "kappa_mean": -0.05}
    assert agent.evaluate_and_adapt(state) == "enter"
    agent.repair()
    assert agent.strategy.params["beta"] == 0.0


def test_pi_agent_mutation_creates_new_strategy() -> None:
    random.seed(7)
    agent = PiAgent(strategy=Strategy(name="s", params={"alpha": 1.0}))
    mutant = agent.mutate()
    assert mutant.strategy.name != agent.strategy.name
    assert mutant.strategy.params["alpha"] != agent.strategy.params["alpha"]


def test_pi_agent_exit_and_hold_paths() -> None:
    agent = PiAgent(strategy=Strategy(name="s", params={}))
    exit_state = {"R": 0.3, "delta_H": 0.0, "kappa_mean": 0.1, "phase_reversal": True}
    assert agent.evaluate_and_adapt(exit_state) == "exit"
    hold_state = {"R": 0.2, "delta_H": 0.0, "kappa_mean": 0.1}
    assert agent.evaluate_and_adapt(hold_state) == "hold"
