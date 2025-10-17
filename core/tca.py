"""Transaction cost analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(slots=True)
class TCARecord:
    order_id: int
    arrival_price: float
    execution_price: float
    vwap: float
    twap: float
    quantity: float

    @property
    def slippage_bps(self) -> float:
        return (self.execution_price / self.arrival_price - 1) * 10_000


class TCA:
    def __init__(self) -> None:
        self._records: List[TCARecord] = []

    def consume(self, order: Dict[str, float], bar: Dict[str, float]) -> None:
        record = TCARecord(
            order_id=len(self._records) + 1,
            arrival_price=bar["open"],
            execution_price=order["price"],
            vwap=bar.get("vwap", bar["close"]),
            twap=(bar["open"] + bar["close"]) / 2,
            quantity=order["quantity"],
        )
        self._records.append(record)

    def summary(self) -> Dict[str, float]:
        if not self._records:
            return {"avg_slippage_bps": 0.0, "vwap_diff": 0.0, "twap_diff": 0.0}
        avg_slip = sum(r.slippage_bps for r in self._records) / len(self._records)
        vwap_diff = sum(r.execution_price - r.vwap for r in self._records) / len(self._records)
        twap_diff = sum(r.execution_price - r.twap for r in self._records) / len(self._records)
        return {
            "avg_slippage_bps": avg_slip,
            "vwap_diff": vwap_diff,
            "twap_diff": twap_diff,
        }

