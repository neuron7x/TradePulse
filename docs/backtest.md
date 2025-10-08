# Backtesting

TradePulse ships with a deterministic, vectorised walk-forward engine designed
for rapid iteration over strategy parameter sets. This guide explains the data
contract, scoring outputs, and extension hooks for more advanced simulations.

---

## Walk-Forward Engine

`backtest.engine.walk_forward(prices, signal_fn, fee=0.0005, initial_capital=0.0)`
performs a rolling evaluation of a strategy by:

1. Validating that `prices` is a 1-D array with at least two observations.
2. Requesting a synchronised signal array from `signal_fn(prices)` and clipping
   it to the `[-1, 1]` range to enforce long/short bounds.
3. Calculating P&L from position changes and price moves while deducting
   proportional transaction costs.
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
result = walk_forward(prices, momentum_signal, fee=0.0002, initial_capital=10_000)
print(result.pnl, result.max_dd, result.trades)
```

---

## Data Requirements

- **Alignment** – `signal_fn` must return an array with the same length as the
  price series; otherwise a `ValueError` is raised. 【F:backtest/engine.py†L15-L22】
- **Leverage & bounds** – signals outside `[-1, 1]` are clamped automatically.
- **Fees** – the `fee` parameter models proportional costs per unit of position
  change; set it to zero for idealised runs or scale it by expected slippage.

When you need richer book-keeping (cash balances, borrowing costs, event-driven
fills), wrap the existing engine so legacy tests remain deterministic.

---

## Diagnostics & Extensions

- Record intermediate arrays (`positions`, `equity_curve`, `drawdowns`) to
  inspect trade-by-trade performance.
- Compose with the agent layer: `Strategy.simulate_performance` in the agent
  module already uses this walk-forward logic to ensure consistent scoring.
- For multi-asset or order-book simulations, fork the engine and maintain the
  same `Result` signature so downstream tooling keeps working.
