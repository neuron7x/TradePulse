"""Canonical strategy interface used by the 2025 trading core."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass(slots=True)
class StrategySignal:
    """Normalised signal container."""

    side: str
    strength: float
    ts_utc: datetime
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "side": self.side,
            "strength": self.strength,
            "ts_utc": self.ts_utc.isoformat(),
        }
        if self.extra:
            payload["extra"] = self.extra
        return payload


@dataclass(slots=True)
class ExitSignal:
    """Representation of a strategy exit/stop instruction."""

    reason: str
    price_level: Optional[float]
    ts_utc: datetime
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "reason": self.reason,
            "price_level": self.price_level,
            "ts_utc": self.ts_utc.isoformat(),
        }
        if self.extra:
            payload["extra"] = self.extra
        return payload


class Strategy(ABC):
    """Abstract base class shared by all reference strategies."""

    @abstractmethod
    def generate_signals(self, bar: Dict[str, Any]) -> StrategySignal:
        """Return the raw signal before risk controls are applied."""

    @abstractmethod
    def size_positions(self, signal: StrategySignal, risk_state: Dict[str, Any]) -> float:
        """Compute the desired position size incorporating risk constraints."""

    @abstractmethod
    def exits(self, bar: Dict[str, Any], position: float) -> ExitSignal | None:
        """Evaluate exit conditions for the currently open position."""

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Describe the strategy for logging and reproducibility purposes."""

