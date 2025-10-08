# Execution

The execution layer converts strategy intentions into orders while enforcing
risk controls. TradePulse currently provides thin utility functions for sizing
positions and estimating aggregate risk; these utilities are intended to be
wrapped by exchange adapters or execution daemons.

---

## Order Model

```python
from execution.order import Order

order = Order(side="buy", qty=0.5, price=25_000.0, type="limit")
```

- `side` – direction (`"buy"` or `"sell"`).
- `qty` – base-asset quantity.
- `price` – optional limit price; leave `None` for market orders.
- `type` – order type label; defaults to `"market"`. 【F:execution/order.py†L1-L11】

The dataclass can be extended with venue-specific metadata (time-in-force,
client IDs) when integrating with real exchanges.

---

## Position Sizing

`execution.order.position_sizing(balance, risk, price, max_leverage=5.0)` returns
risk-aware quantity in base units:

1. Validates that `price` is positive.
2. Caps `risk` between 0 and 1 (fraction of account equity to deploy).
3. Limits exposure by both the risk budget (`balance * risk / price`) and the
   leverage ceiling (`balance * max_leverage / price`).
4. Returns the minimum of those limits, floored at zero. 【F:execution/order.py†L13-L23】

Example:

```python
from execution.order import position_sizing

size = position_sizing(balance=25_000, risk=0.02, price=20_000, max_leverage=3.0)
# -> 0.75 BTC equivalent
```

---

## Portfolio Heat

`execution.risk.portfolio_heat(positions)` computes aggregate notional exposure
with optional directionality and risk weights. It expects each position mapping
(`dict` or similar) to contain:

- `qty` – signed or absolute quantity
- `price` – entry price
- `risk_weight` – multiplier for instrument-specific scaling (defaults to 1.0)
- `side` – `"long"` or `"short"` to inject direction into the heat total

The helper iterates through positions, multiplies `qty * price * risk_weight`,
adjusts sign based on `side`, and accumulates the absolute contribution. 【F:execution/risk.py†L1-L18】

```python
from execution.risk import portfolio_heat

positions = [
    {"side": "long", "qty": 1.2, "price": 30_000, "risk_weight": 0.8},
    {"side": "short", "qty": 0.5, "price": 25_000, "risk_weight": 1.2},
]
heat = portfolio_heat(positions)
```

Use the resulting scalar to enforce account-level limits or trigger staged
liquidations when exposure exceeds policy thresholds.

---

## Implementation Notes

- Layer venue integrations on top of these primitives so that deterministic
  behaviour in tests is preserved.
- Validate upstream strategy outputs before forwarding to exchanges—e.g., check
  that `PiAgent` actions map cleanly to order intents.
- Extend this guide whenever new adapters (FIX, REST, websockets) or risk checks
  (e.g., pre-trade credit, margin requirements) are added.
