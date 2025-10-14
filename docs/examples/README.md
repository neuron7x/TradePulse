# Usage Examples

This directory contains practical examples for using TradePulse.

---

## Quick Examples

### 1. Basic Analysis

```python
import numpy as np
import pandas as pd
from core.indicators.kuramoto import compute_phase, kuramoto_order
from core.indicators.entropy import entropy
from core.indicators.ricci import build_price_graph, mean_ricci

# Load data
df = pd.read_csv('sample.csv')
prices = df['close'].to_numpy()

# Compute Kuramoto order
phases = compute_phase(prices)
R = kuramoto_order(phases[-200:])
print(f"Kuramoto Order: {R:.3f}")

# Compute entropy
H = entropy(prices[-200:])
print(f"Entropy: {H:.3f}")

# Compute Ricci curvature
G = build_price_graph(prices[-200:], delta=0.005)
kappa = mean_ricci(G)
print(f"Mean Ricci Curvature: {kappa:.3f}")
```

### 2. Simple Backtest

```python
from backtest.engine import walk_forward
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('sample.csv')
prices = df['close'].to_numpy()

# Define signal function
def moving_average_crossover(prices: np.ndarray, window: int = 50) -> np.ndarray:
    """Generate signals based on MA crossover."""
    signals = np.zeros(len(prices))
    
    fast = pd.Series(prices).rolling(window).mean()
    slow = pd.Series(prices).rolling(window*2).mean()
    
    signals[fast > slow] = 1  # Buy
    signals[fast < slow] = -1  # Sell
    
    return signals

# Run backtest
results = walk_forward(
    prices=prices,
    signal_func=moving_average_crossover,
    train_window=500,
    test_window=100,
    initial_capital=10000.0
)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Number of Trades: {results['num_trades']}")
```

### 3. Live Data Stream

```python
from core.data.ingestion import DataIngestor, Ticker
from core.indicators.kuramoto import compute_phase, kuramoto_order
import numpy as np

# Collect ticks
ticks = []

def on_tick(tick: Ticker):
    ticks.append(tick.price)
    
    # Compute indicator every 100 ticks
    if len(ticks) >= 200:
        prices = np.array(ticks[-200:])
        phases = compute_phase(prices)
        R = kuramoto_order(phases)
        print(f"{tick.symbol} @ {tick.ts}: R = {R:.3f}")

# Ingest from CSV (demo)
ingestor = DataIngestor()
ingestor.historical_csv('sample.csv', on_tick)
```

### 4. Custom Indicator

```python
from core.indicators.base import BaseFeature, FeatureResult
import numpy as np

class SimpleMovingAverage(BaseFeature):
    """Simple moving average indicator."""
    
    def __init__(self, name: str = "sma", period: int = 20):
        super().__init__(name, period=period)
        self.period = period
    
    def transform(self, data: np.ndarray) -> FeatureResult:
        """Compute SMA."""
        self.validate_input(data)
        
        if len(data) < self.period:
            raise ValueError(f"Need at least {self.period} data points")
        
        sma = np.mean(data[-self.period:])
        
        return FeatureResult(
            value=float(sma),
            metadata={"period": self.period, "n_samples": len(data)},
            name=self.name
        )

# Use it
import pandas as pd
df = pd.read_csv('sample.csv')
prices = df['close'].to_numpy()

sma = SimpleMovingAverage(period=50)
result = sma.transform(prices)
print(f"SMA(50): {result.value:.2f}")
```

### 5. Risk Management

```python
from execution.risk import position_sizing, calculate_stop_loss

# Account settings
balance = 10000.0
risk_per_trade = 0.01  # 1% risk

# Current trade
entry_price = 50000.0
stop_loss_pct = 0.02  # 2% stop loss

# Calculate position size
size = position_sizing(
    balance=balance,
    risk=risk_per_trade,
    price=entry_price,
    stop_loss_pct=stop_loss_pct
)

# Calculate stop loss price
stop_loss_price = calculate_stop_loss(entry_price, stop_loss_pct, 'long')

print(f"Position Size: {size:.6f}")
print(f"Stop Loss: ${stop_loss_price:.2f}")
print(f"Risk Amount: ${balance * risk_per_trade:.2f}")
```

### 6. Strategy Selection with Bandits

TradePulse provides classic multi-armed bandit algorithms to manage simple
strategy selection loops. The example below shows how to use the
`EpsilonGreedy` policy to explore available strategies and reinforce the one
that yields the highest reward.

```python
from core.agent.bandits import EpsilonGreedy
import numpy as np
import pandas as pd

df = pd.read_csv('sample.csv')
prices = df['close'].to_numpy()

# Candidate strategies that output a position (+1 long, -1 short, 0 flat)
STRATEGIES = {
    "trend_following": lambda window, prices: np.sign(
        np.mean(prices[-window:]) - np.mean(prices[-2 * window:])
    ),
    "mean_reversion": lambda window, prices: -np.sign(
        prices[-1] - np.mean(prices[-window:])
    ),
}

def backtest_once(strategy_name: str, prices: np.ndarray) -> float:
    """Return a mock reward from executing the strategy on recent data."""
    window = 20
    position = STRATEGIES[strategy_name](window, prices)
    returns = np.diff(prices)[-window:]
    return float(position * np.mean(returns))

# Initialize bandit with available strategies (arms)
bandit = EpsilonGreedy(list(STRATEGIES.keys()), epsilon=0.1)

for _ in range(100):
    arm = bandit.select()
    reward = backtest_once(arm, prices)
    bandit.update(arm, reward)

print("Estimated action values:", bandit.Q)
```

### 7. Metrics Calculation

```python
import numpy as np
from core.metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    win_rate,
    profit_factor
)

# Sample returns
returns = np.array([0.02, -0.01, 0.03, -0.02, 0.04, 0.01, -0.01])

# Calculate metrics
sharpe = sharpe_ratio(returns, risk_free_rate=0.02)
sortino = sortino_ratio(returns, risk_free_rate=0.02)
max_dd = max_drawdown(returns)
calmar = calmar_ratio(returns)
win_pct = win_rate(returns)
pf = profit_factor(returns)

print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Sortino Ratio: {sortino:.2f}")
print(f"Max Drawdown: {max_dd:.2%}")
print(f"Calmar Ratio: {calmar:.2f}")
print(f"Win Rate: {win_pct:.2%}")
print(f"Profit Factor: {pf:.2f}")
```

### 8. Multi-Indicator Analysis

```python
from core.indicators.kuramoto import KuramotoOrder
from core.indicators.entropy import Entropy
from core.indicators.ricci import RicciCurvature
from core.indicators.hurst import HurstExponent
from core.indicators.base import FeatureBlock

# Create indicator block
block = FeatureBlock("market_regime")
block.add_feature(KuramotoOrder(window=200))
block.add_feature(Entropy(bins=50))
block.add_feature(RicciCurvature(delta=0.005))
block.add_feature(HurstExponent())

# Process data
import pandas as pd
df = pd.read_csv('sample.csv')
prices = df['close'].to_numpy()

results = block.transform_all(prices)

for name, result in results.items():
    print(f"{name}: {result.value:.3f}")
```

### 9. Phase Detection

```python
from core.phase.detector import phase_flags, composite_transition
from core.indicators.kuramoto import compute_phase, kuramoto_order
from core.indicators.entropy import entropy, delta_entropy
from core.indicators.ricci import build_price_graph, mean_ricci
import numpy as np
import pandas as pd

# Load data
df = pd.read_csv('sample.csv')
prices = df['close'].to_numpy()

# Compute indicators
phases = compute_phase(prices)
R = kuramoto_order(phases[-200:])
H = entropy(prices[-200:])
dH = delta_entropy(prices, window=200)
G = build_price_graph(prices[-200:], delta=0.005)
kappa = mean_ricci(G)

# Detect phase
phase = phase_flags(R, dH, kappa, H)
transition = composite_transition(R, dH, kappa)

print(f"Market Phase: {phase}")
print(f"Transition Signal: {transition}")
```

### 10. Complete Trading System

```python
# complete_system.py
import pandas as pd
import numpy as np
from core.indicators.kuramoto import compute_phase, kuramoto_order
from core.indicators.entropy import entropy
from backtest.engine import walk_forward
from execution.risk import position_sizing

class TradingSystem:
    """Complete trading system."""
    
    def __init__(self, balance: float = 10000.0):
        self.balance = balance
        self.positions = []
    
    def analyze(self, prices: np.ndarray) -> dict:
        """Analyze market conditions."""
        phases = compute_phase(prices)
        R = kuramoto_order(phases[-200:])
        H = entropy(prices[-200:])
        
        return {"R": R, "H": H}
    
    def generate_signal(self, analysis: dict) -> str:
        """Generate trading signal."""
        if analysis["R"] > 0.7 and analysis["H"] < 2.0:
            return "buy"
        elif analysis["R"] < 0.3 or analysis["H"] > 3.0:
            return "sell"
        return "hold"
    
    def execute_trade(self, signal: str, price: float):
        """Execute trade with risk management."""
        if signal == "buy":
            size = position_sizing(
                self.balance,
                risk=0.01,
                price=price,
                stop_loss_pct=0.02
            )
            self.positions.append({"side": "long", "size": size, "price": price})
        elif signal == "sell" and self.positions:
            # Close positions
            self.positions = []
    
    def run(self, data_file: str):
        """Run complete system."""
        df = pd.read_csv(data_file)
        prices = df['close'].to_numpy()
        
        for i in range(200, len(prices)):
            window = prices[i-200:i]
            analysis = self.analyze(window)
            signal = self.generate_signal(analysis)
            self.execute_trade(signal, prices[i])

# Run system
system = TradingSystem(balance=10000.0)
system.run('sample.csv')
```

---

## More Examples

For more detailed examples, see:
- [Extending TradePulse](../extending.md)
- [Integration API](../integration-api.md)
- [Developer Scenarios](../scenarios.md)

---

## Running Examples

```bash
# Save example to file
cat > example.py << 'EOF'
# ... example code ...
EOF

# Run example
python example.py
```

---

**Last Updated**: 2025-01-01
