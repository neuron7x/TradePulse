"""Exchange-specific data ingestion connectors with schema validation and resilience."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from threading import Lock
from typing import Any, AsyncIterator, Deque, Optional
from uuid import uuid4

from core.data.adapters.base import (
    IngestionAdapter,
    RateLimitConfig,
    RetryConfig,
    TimeoutConfig,
)
from core.data.adapters.ccxt import CCXTIngestionAdapter
from core.data.adapters.polygon import PolygonIngestionAdapter
from core.data.models import InstrumentType
from core.data.models import PriceTick as Ticker
from core.events import TickEvent
from core.messaging.schema_registry import EventSchemaRegistry, SchemaFormat
from core.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SCHEMA_ROOT = Path(__file__).resolve().parents[3] / "schemas" / "events"


@dataclass(frozen=True)
class DeadLetterItem:
    """Record captured when payload processing fails."""

    payload: Any
    error: str
    context: str
    timestamp: float

    def asdict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the item."""

        return {
            "payload": self.payload,
            "error": self.error,
            "context": self.context,
            "timestamp": self.timestamp,
        }


class DeadLetterQueue:
    """In-memory bounded dead-letter queue retaining the latest failures."""

    def __init__(
        self,
        max_items: int = 1024,
        *,
        persistent_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        self._items: Deque[DeadLetterItem] = deque(maxlen=max_items)
        self._max_items = max_items
        self._persistent_path = (
            Path(persistent_path) if persistent_path is not None else None
        )
        self._lock = Lock()
        if self._persistent_path is not None:
            self._persistent_path.parent.mkdir(parents=True, exist_ok=True)

    def push(self, payload: Any, error: Exception | str, *, context: str) -> None:
        message = str(error)
        if hasattr(payload, "model_dump"):
            try:
                payload = payload.model_dump()
            except Exception:  # pragma: no cover - defensive guard
                payload = repr(payload)
        elif isinstance(payload, (dict, list, tuple, str, int, float, type(None))):
            pass
        else:
            payload = repr(payload)
        item = DeadLetterItem(
            payload=payload, error=message, context=context, timestamp=time.time()
        )
        self._items.append(item)
        logger.debug("dead_letter_enqueued", size=len(self._items), context=context)
        self._persist_item(item)

    def drain(self) -> list[DeadLetterItem]:
        items = list(self._items)
        self._items.clear()
        return items

    def peek(self) -> list[DeadLetterItem]:
        return list(self._items)

    def __len__(self) -> int:  # pragma: no cover - trivial accessor
        return len(self._items)

    @property
    def max_items(self) -> int:  # pragma: no cover - trivial accessor
        return self._max_items

    def persist(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        drain: bool = False,
    ) -> Path:
        """Persist the queue content to ``path`` for operational triage.

        Args:
            path: Destination file. When omitted, the queue must have been
                initialised with ``persistent_path``.
            drain: When ``True`` the queue is emptied after persisting.

        Returns:
            The resolved :class:`pathlib.Path` to the persisted payload.
        """

        target = Path(path) if path is not None else self._persistent_path
        if target is None:
            raise ValueError("No persistence path configured for dead-letter queue")

        with self._lock:
            if drain:
                items = list(self._items)
                self._items.clear()
            else:
                items = list(self._items)
            payload = [item.asdict() for item in items]

            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")

        return target

    def _persist_item(self, item: DeadLetterItem) -> None:
        if self._persistent_path is None:
            return
        record = json.dumps(item.asdict(), sort_keys=True)
        with self._lock:
            with self._persistent_path.open("a", encoding="utf-8") as handle:
                handle.write(record)
                handle.write("\n")


class BaseMarketDataConnector:
    """Shared functionality for schema-aware market data connectors."""

    def __init__(
        self,
        adapter: IngestionAdapter,
        *,
        schema_registry: Optional[EventSchemaRegistry] = None,
        event_type: str = "ticks",
        dead_letter_queue: Optional[DeadLetterQueue] = None,
    ) -> None:
        self._adapter = adapter
        self._dead_letters = dead_letter_queue or DeadLetterQueue()
        self._registry = schema_registry or EventSchemaRegistry.from_directory(
            DEFAULT_SCHEMA_ROOT
        )
        schema_info = self._registry.latest(event_type, SchemaFormat.AVRO)
        self._schema_version = schema_info.version_str
        schema_doc = schema_info.load()
        self._required_fields = {
            field["name"]
            for field in schema_doc.get("fields", [])
            if not _is_nullable(field.get("type")) and "default" not in field
        }
        self._event_type = event_type
        logger.debug(
            "market_connector_initialised",
            adapter=adapter.__class__.__name__,
            schema_version=self._schema_version,
            event_type=event_type,
        )

    async def fetch_snapshot(self, **kwargs: Any) -> list[TickEvent]:
        """Fetch a bounded snapshot of ticks and return schema-compliant events."""

        raw_ticks = await self._adapter.fetch(**kwargs)
        events: list[TickEvent] = []
        for tick in raw_ticks:
            try:
                event = self._convert_tick(tick)
                self._validate_event(event)
            except (
                Exception
            ) as exc:  # pragma: no cover - resilience path exercised in tests
                logger.warning(
                    "tick_conversion_failed",
                    error=str(exc),
                    context="fetch",
                    adapter=self._adapter.__class__.__name__,
                )
                self._dead_letters.push(tick, exc, context="fetch")
            else:
                events.append(event)
        return events

    async def stream_ticks(self, **kwargs: Any) -> AsyncIterator[TickEvent]:
        """Stream live ticks, yielding validated events with resilience features."""

        attempt = 0
        while True:
            try:
                async for tick in self._adapter.stream(**kwargs):
                    attempt = 0
                    try:
                        event = self._convert_tick(tick)
                        self._validate_event(event)
                    except Exception as exc:
                        logger.warning(
                            "tick_conversion_failed",
                            error=str(exc),
                            context="stream",
                            adapter=self._adapter.__class__.__name__,
                        )
                        self._dead_letters.push(tick, exc, context="stream")
                        continue
                    yield event
            except (
                Exception
            ) as exc:  # pragma: no cover - exercised in tests via dummy adapters
                attempt += 1
                backoff = min(2**attempt, 60)
                logger.warning(
                    "stream_restart",
                    attempt=attempt,
                    delay=backoff,
                    error=str(exc),
                    adapter=self._adapter.__class__.__name__,
                )
                await asyncio.sleep(backoff)
            else:  # pragma: no cover - loop exit path
                break

    def _convert_tick(self, tick: Ticker) -> TickEvent:
        price = _decimal_to_float(tick.price)
        timestamp_us = _timestamp_to_micros(tick.timestamp)
        volume = _decimal_to_int(tick.volume)
        event = TickEvent(
            event_id=str(uuid4()),
            schema_version=self._schema_version,
            symbol=tick.symbol,
            timestamp=timestamp_us,
            bid_price=price,
            ask_price=price,
            last_price=price,
            volume=volume,
        )
        return event

    def _validate_event(self, event: TickEvent) -> None:
        for field_name in self._required_fields:
            if getattr(event, field_name) is None:
                raise ValueError(f"Field '{field_name}' must not be None")

    @property
    def dead_letter_queue(self) -> DeadLetterQueue:
        return self._dead_letters

    async def aclose(self) -> None:
        await self._adapter.aclose()

    async def __aenter__(
        self,
    ) -> "BaseMarketDataConnector":  # pragma: no cover - trivial
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        await self.aclose()


class BinanceMarketDataConnector(BaseMarketDataConnector):
    """Connector configured for Binance market data via CCXT."""

    def __init__(
        self,
        *,
        retry: Optional[RetryConfig] = None,
        rate_limit: Optional[RateLimitConfig] = None,
        client_params: Optional[dict[str, Any]] = None,
        schema_registry: Optional[EventSchemaRegistry] = None,
        dead_letter_queue: Optional[DeadLetterQueue] = None,
        adapter: Optional[IngestionAdapter] = None,
    ) -> None:
        retry = retry or RetryConfig()
        rate_limit = rate_limit or RateLimitConfig(rate=1200, period_seconds=60.0)
        ccxt_adapter = adapter or CCXTIngestionAdapter(
            exchange_id="binance",
            retry=retry,
            rate_limit=rate_limit,
            client_params=client_params,
        )
        super().__init__(
            ccxt_adapter,
            schema_registry=schema_registry,
            dead_letter_queue=dead_letter_queue,
        )


class CoinbaseMarketDataConnector(BaseMarketDataConnector):
    """Connector configured for Coinbase market data via CCXT."""

    def __init__(
        self,
        *,
        retry: Optional[RetryConfig] = None,
        rate_limit: Optional[RateLimitConfig] = None,
        client_params: Optional[dict[str, Any]] = None,
        schema_registry: Optional[EventSchemaRegistry] = None,
        dead_letter_queue: Optional[DeadLetterQueue] = None,
        adapter: Optional[IngestionAdapter] = None,
    ) -> None:
        retry = retry or RetryConfig(attempts=6, multiplier=0.4, max_backoff=20.0)
        rate_limit = rate_limit or RateLimitConfig(rate=10, period_seconds=1.0)
        params = client_params or {"enableRateLimit": True}
        ccxt_adapter = adapter or CCXTIngestionAdapter(
            exchange_id="coinbasepro",
            retry=retry,
            rate_limit=rate_limit,
            client_params=params,
        )
        super().__init__(
            ccxt_adapter,
            schema_registry=schema_registry,
            dead_letter_queue=dead_letter_queue,
        )


class PolygonMarketDataConnector(BaseMarketDataConnector):
    """Connector configured for Polygon.io REST and WebSocket APIs."""

    def __init__(
        self,
        *,
        api_key: str,
        retry: Optional[RetryConfig] = None,
        rate_limit: Optional[RateLimitConfig] = None,
        timeout: Optional[TimeoutConfig] = None,
        base_url: str = "https://api.polygon.io",
        websocket_url: str = "wss://socket.polygon.io",
        schema_registry: Optional[EventSchemaRegistry] = None,
        dead_letter_queue: Optional[DeadLetterQueue] = None,
        adapter: Optional[IngestionAdapter] = None,
    ) -> None:
        retry = retry or RetryConfig(attempts=6, multiplier=0.5, max_backoff=30.0)
        rate_limit = rate_limit or RateLimitConfig(rate=5, period_seconds=1.0)
        timeout = timeout or TimeoutConfig(total_seconds=10.0)
        polygon_adapter = adapter or PolygonIngestionAdapter(
            api_key=api_key,
            base_url=base_url,
            websocket_url=websocket_url,
            retry=retry,
            rate_limit=rate_limit,
            timeout=timeout,
        )
        super().__init__(
            polygon_adapter,
            schema_registry=schema_registry,
            dead_letter_queue=dead_letter_queue,
        )

    async def fetch_aggregates(
        self,
        *,
        symbol: str,
        start: str,
        end: str,
        multiplier: int = 1,
        timespan: str = "minute",
        limit: int = 5000,
        adjusted: bool = True,
        instrument_type: InstrumentType = InstrumentType.SPOT,
    ) -> list[TickEvent]:
        """Fetch Polygon aggregates and adapt them into tick events."""

        ticks = await self._adapter.fetch(
            symbol=symbol,
            start=start,
            end=end,
            multiplier=multiplier,
            timespan=timespan,
            limit=limit,
            adjusted=adjusted,
            instrument_type=instrument_type,
        )
        events: list[TickEvent] = []
        for tick in ticks:
            try:
                event = self._convert_tick(tick)
                self._validate_event(event)
            except Exception as exc:
                logger.warning(
                    "tick_conversion_failed",
                    error=str(exc),
                    context="fetch",
                    adapter=self._adapter.__class__.__name__,
                )
                self._dead_letters.push(tick, exc, context="fetch")
            else:
                events.append(event)
        return events


def _decimal_to_float(value: Decimal | float | int) -> float:
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _decimal_to_int(value: Optional[Decimal]) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _timestamp_to_micros(timestamp: datetime) -> int:
    return int(timestamp.timestamp() * 1_000_000)


def _is_nullable(avro_type: Any) -> bool:
    if isinstance(avro_type, list):
        return any(_is_nullable(member) for member in avro_type)
    if isinstance(avro_type, dict):
        return avro_type.get("type") == "null"
    return avro_type == "null"
