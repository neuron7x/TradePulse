"""Market data connectors wrapping ingestion adapters with schema-aware payloads."""

from .market import (
    BaseMarketDataConnector,
    BinanceMarketDataConnector,
    CoinbaseMarketDataConnector,
    DeadLetterItem,
    DeadLetterQueue,
    PolygonMarketDataConnector,
)

__all__ = [
    "BaseMarketDataConnector",
    "BinanceMarketDataConnector",
    "CoinbaseMarketDataConnector",
    "DeadLetterItem",
    "DeadLetterQueue",
    "PolygonMarketDataConnector",
]
