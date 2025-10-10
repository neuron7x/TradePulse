# Backtesting

TradePulse ships with a deterministic, vectorised walk-forward engine designed
for rapid iteration over strategy parameter sets. This guide explains the data
contract, scoring outputs, and extension hooks for more advanced simulations.

---

## Walk-Forward Engine

`backtest.engine.walk_forward(prices, signal_fn, fee=0.0005, initial_capital=0.0, market=None, cost_model=None)`
performs a rolling evaluation of a strategy by:

1. Validating that `prices` is a 1-D array with at least two observations.
2. Requesting a synchronised signal array from `signal_fn(prices)` and clipping
   it to the `[-1, 1]` range to enforce long/short bounds.
3. Calculating P&L from position changes and price moves while deducting
   configurable commissions, spreads, and slippage.
4. Returning a `Result` dataclass with total P&L, maximum drawdown, and the
   number of trades executed. 【F:backtest/engine.py†L1-L32】

```python
from backtest.engine import walk_forward
import numpy as np

def momentum_signal(prices: np.ndarray) -> np.ndarray:
    returns = np.diff(prices, prepend=prices[0])
    signal = np.sign(returns)
    return signal

prices = np.linspace(100, 110, 500) + np.random.randn(500)
result = walk_forward(
    prices,
    momentum_signal,
    fee=0.0002,
    initial_capital=10_000,
    market="BTC-USD",
)
print(result.pnl, result.max_dd, result.trades)
```

---

## Data Requirements

- **Alignment** – `signal_fn` must return an array with the same length as the
  price series; otherwise a `ValueError` is raised. 【F:backtest/engine.py†L15-L22】
- **Leverage & bounds** – signals outside `[-1, 1]` are clamped automatically.
- **Execution costs** – specify `market` to pull per-instrument settings from
  `configs/markets.yaml`, or pass a custom `cost_model` implementing
  `TransactionCostModel` for bespoke handling. Falling back to the scalar `fee`
  reproduces the legacy proportional cost behaviour.

When you need richer book-keeping (cash balances, borrowing costs, event-driven
fills), wrap the existing engine so legacy tests remain deterministic.

---

## Diagnostics & Extensions

- Use `Result.commission_cost`, `Result.spread_cost`, and
  `Result.slippage_cost` to reconcile the breakdown of execution frictions.
- Construct reusable commission/spread/slippage policies via
  `backtest.transaction_costs.CompositeTransactionCostModel` and point the
  engine at bespoke YAML configuration files with `cost_config="my_markets.yaml"`.

- Record intermediate arrays (`positions`, `equity_curve`, `drawdowns`) to
  inspect trade-by-trade performance.
- Compose with the agent layer: `Strategy.simulate_performance` in the agent
  module already uses this walk-forward logic to ensure consistent scoring.
- For multi-asset or order-book simulations, fork the engine and maintain the
  same `Result` signature so downstream tooling keeps working.

---

## Execution Simulation

When strategies require microstructure-aware analysis (latency, partial fills,
order queueing, halts, and time-in-force semantics), layer the
[`backtest.execution_simulation`](../backtest/execution_simulation.py) module on
top of the walk-forward loop. The protocol described in
[`docs/backtest_execution_simulation.md`](backtest_execution_simulation.md)
covers latency modelling, halt policies, order types, and integration steps for
deterministic research environments.
