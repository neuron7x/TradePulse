"""Utilities for assembling and running end-to-end TradePulse pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from analytics.signals.pipeline import FeaturePipelineConfig
from application.system import (
    ExchangeAdapterConfig,
    LiveLoopSettings,
    TradePulseSystem,
    TradePulseSystemConfig,
)
from core.data.models import InstrumentType
from domain import Order, Signal
from execution.connectors import BinanceConnector, CoinbaseConnector
from execution.risk import RiskLimits


@dataclass(slots=True)
class MarketDataSource:
    """Declarative description of a market data CSV source."""

    path: Path
    symbol: str
    venue: str
    instrument_type: InstrumentType = InstrumentType.SPOT
    market: str | None = None


@dataclass(slots=True)
class StrategyRun:
    """Result of executing a strategy over ingested market data."""

    market_frame: pd.DataFrame
    feature_frame: pd.DataFrame
    signals: list[Signal]
    payloads: list[dict[str, object]]


@dataclass(slots=True)
class ExecutionRequest:
    """Parameters required to hand a signal over to execution."""

    signal: Signal
    venue: str
    quantity: float
    price: float | None = None
    order_type: str | None = None
    correlation_id: str | None = None


def build_tradepulse_system(
    venues: Sequence[ExchangeAdapterConfig] | None = None,
    *,
    feature_pipeline: FeaturePipelineConfig | None = None,
    risk_limits: RiskLimits | None = None,
    live_settings: LiveLoopSettings | None = None,
    allowed_data_roots: Iterable[str | Path] | None = None,
    max_csv_bytes: int | None = None,
) -> TradePulseSystem:
    """Return a ready-to-use :class:`TradePulseSystem` instance.

    The helper provides sensible defaults so tests and prototypes can stand up a
    full pipeline with a couple of lines of code while still allowing callers to
    supply bespoke connectors, feature pipelines, or risk limits when required.
    """

    if venues is None:
        venues = (
            ExchangeAdapterConfig(name="binance", connector=BinanceConnector()),
            ExchangeAdapterConfig(name="coinbase", connector=CoinbaseConnector()),
        )

    pipeline_config = feature_pipeline or FeaturePipelineConfig()
    risk = risk_limits or RiskLimits()
    live = live_settings or LiveLoopSettings()

    config = TradePulseSystemConfig(
        venues=tuple(venues),
        feature_pipeline=pipeline_config,
        risk_limits=risk,
        live_settings=live,
        allowed_data_roots=allowed_data_roots,
        max_csv_bytes=max_csv_bytes,
    )
    return TradePulseSystem(config)


class TradePulseOrchestrator:
    """High-level façade that wires ingestion, analytics, and execution."""

    def __init__(self, system: TradePulseSystem) -> None:
        self._system = system

    @property
    def system(self) -> TradePulseSystem:
        """Expose the underlying :class:`TradePulseSystem`."""

        return self._system

    def ingest_market_data(self, source: MarketDataSource) -> pd.DataFrame:
        """Load a CSV data source into a normalised OHLCV frame."""

        return self._system.ingest_csv(
            source.path,
            symbol=source.symbol,
            venue=source.venue,
            instrument_type=source.instrument_type,
            market=source.market,
        )

    def build_features(self, market_frame: pd.DataFrame) -> pd.DataFrame:
        """Return a feature-enriched frame derived from *market_frame*."""

        return self._system.build_feature_frame(market_frame)

    def run_strategy(
        self,
        source: MarketDataSource,
        strategy: Callable[[np.ndarray], np.ndarray],
    ) -> StrategyRun:
        """Execute the canonical ingestion → features → strategy pipeline."""

        market = self.ingest_market_data(source)
        features = self.build_features(market)
        signals = self._system.generate_signals(features, strategy=strategy, symbol=source.symbol)
        payloads = self._system.signals_to_dtos(signals)
        return StrategyRun(market, features, signals, payloads)

    def submit_signal(self, request: ExecutionRequest) -> Order:
        """Forward a signal to execution and return the resulting order."""

        return self._system.submit_signal(
            request.signal,
            venue=request.venue,
            quantity=request.quantity,
            price=request.price,
            order_type=request.order_type,
            correlation_id=request.correlation_id,
        )

    def ensure_live_loop(self) -> None:
        """Ensure the live loop has been instantiated."""

        self._system.ensure_live_loop()


__all__ = [
    "ExecutionRequest",
    "MarketDataSource",
    "StrategyRun",
    "TradePulseOrchestrator",
    "build_tradepulse_system",
]
