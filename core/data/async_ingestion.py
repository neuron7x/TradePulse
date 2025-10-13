# SPDX-License-Identifier: MIT
"""Async data ingestion APIs for TradePulse with strict path validation."""
from __future__ import annotations

import asyncio
import csv
from datetime import datetime, timezone
from decimal import InvalidOperation
from pathlib import Path
from typing import AsyncIterator, Callable, Iterable, Optional

from core.utils.logging import get_logger
from core.utils.metrics import get_metrics_collector

from core.data.models import InstrumentType, PriceTick as Ticker
from core.data.path_guard import DataPathGuard
from core.data.timeutils import normalize_timestamp
from interfaces.ingestion import AsyncDataIngestionService

__all__ = ["AsyncDataIngestor", "AsyncWebSocketStream", "Ticker", "merge_streams"]

logger = get_logger(__name__)
metrics = get_metrics_collector()


class AsyncDataIngestor(AsyncDataIngestionService):
    """Async data ingestion with support for CSV and streaming sources."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        *,
        allowed_roots: Iterable[str | Path] | None = None,
        max_csv_bytes: Optional[int] = None,
        follow_symlinks: bool = False,
    ):
        """Initialize async data ingestor.

        Args:
            api_key: Optional API key for authenticated sources
            api_secret: Optional API secret for authenticated sources
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self._path_guard = DataPathGuard(
            allowed_roots=allowed_roots,
            max_bytes=max_csv_bytes,
            follow_symlinks=follow_symlinks,
        )
        
    async def read_csv(
        self,
        path: str,
        *,
        symbol: str = "UNKNOWN",
        venue: str = "CSV",
        instrument_type: InstrumentType = InstrumentType.SPOT,
        market: Optional[str] = None,
        chunk_size: int = 1000,
        delay_ms: int = 0,
    ) -> AsyncIterator[Ticker]:
        """Async CSV reader that yields ticks.
        
        Args:
            path: Path to CSV file
            symbol: Trading symbol for the data
            venue: Name of the data venue (used for metadata)
            instrument_type: Instrument classification (spot or futures)
            market: Optional market calendar identifier for timezone normalization
            chunk_size: Number of rows to read at a time
            delay_ms: Optional delay between chunks (for simulation)
            
        Yields:
            Ticker objects from CSV
            
        Raises:
            ValueError: If CSV is missing required columns
        """
        resolved_path = self._path_guard.resolve(path, description="CSV data file")

        with logger.operation("async_csv_read", path=str(resolved_path), symbol=symbol):
            try:
                with resolved_path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    
                    if reader.fieldnames is None:
                        raise ValueError("CSV file must include a header row")
                        
                    required = {"ts", "price"}
                    missing = required - set(reader.fieldnames)
                    if missing:
                        raise ValueError(f"CSV missing columns: {', '.join(missing)}")
                        
                    chunk: List[Ticker] = []
                    for row_number, row in enumerate(reader, start=2):
                        try:
                            ts_raw = float(row["ts"])
                            price = row["price"]
                            volume = row.get("volume", 0.0) or 0.0

                            tick = Ticker.create(
                                symbol=symbol,
                                venue=venue,
                                price=price,
                                timestamp=normalize_timestamp(ts_raw, market=market),
                                volume=volume,
                                instrument_type=instrument_type,
                            )
                            chunk.append(tick)
                            
                            if len(chunk) >= chunk_size:
                                for tick in chunk:
                                    yield tick
                                    metrics.record_tick_processed("csv", symbol)
                                chunk = []
                                
                                if delay_ms > 0:
                                    await asyncio.sleep(delay_ms / 1000.0)
                                    
                        except (TypeError, ValueError, InvalidOperation) as exc:
                            logger.warning(
                                f"Skipping malformed row {row_number}",
                                path=path,
                                error=str(exc)
                            )
                            continue
                            
                    # Yield remaining ticks
                    for tick in chunk:
                        yield tick
                        metrics.record_tick_processed("csv", symbol)
                        
            except Exception as exc:
                logger.error("CSV ingestion failed", path=path, error=str(exc))
                raise
                
    async def stream_ticks(
        self,
        source: str,
        symbol: str,
        *,
        instrument_type: InstrumentType = InstrumentType.SPOT,
        interval_ms: int = 1000,
        max_ticks: Optional[int] = None,
    ) -> AsyncIterator[Ticker]:
        """Stream ticks from a source (placeholder for real implementations).
        
        Args:
            source: Data source name
            symbol: Trading symbol
            instrument_type: Instrument classification (spot or futures)
            interval_ms: Polling interval in milliseconds
            max_ticks: Optional maximum number of ticks to yield
            
        Yields:
            Ticker objects from the stream
        """
        with logger.operation("async_stream_ticks", source=source, symbol=symbol):
            count = 0
            
            while max_ticks is None or count < max_ticks:
                # This is a placeholder - real implementation would connect to
                # actual data source (WebSocket, API, etc.)
                await asyncio.sleep(interval_ms / 1000.0)
                
                # Placeholder tick generation
                tick = Ticker.create(
                    symbol=symbol,
                    venue=source.upper(),
                    price=100.0 + (count % 10),
                    timestamp=normalize_timestamp(datetime.now(timezone.utc)),
                    volume=1000.0,
                    instrument_type=instrument_type,
                )
                
                yield tick
                metrics.record_tick_processed(source, symbol)
                count += 1
                
    async def batch_process(
        self,
        ticks: AsyncIterator[Ticker],
        callback: Callable[[list[Ticker]], None],
        batch_size: int = 100,
        *,
        max_retries: int = 3,
        retry_backoff_ms: int = 0,
        retry_exceptions: tuple[type[Exception], ...] = (ConnectionError, TimeoutError, OSError),
    ) -> int:
        """Process ticks in batches with async callback.
        
        Args:
            ticks: Async iterator of ticks
            callback: Function to call with each batch
            batch_size: Number of ticks per batch
            
        Returns:
            Total number of ticks processed
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if retry_backoff_ms < 0:
            raise ValueError("retry_backoff_ms cannot be negative")

        batch: list[Ticker] = []
        total = 0
        
        async for tick in ticks:
            batch.append(tick)
            total += 1
            
            if len(batch) >= batch_size:
                await self._dispatch_batch(
                    callback,
                    batch,
                    max_retries=max_retries,
                    retry_backoff_ms=retry_backoff_ms,
                    retry_exceptions=retry_exceptions,
                )
                batch = []
                
        # Process remaining ticks
        if batch:
            await self._dispatch_batch(
                callback,
                batch,
                max_retries=max_retries,
                retry_backoff_ms=retry_backoff_ms,
                retry_exceptions=retry_exceptions,
            )

        return total

    async def _dispatch_batch(
        self,
        callback: Callable[[list[Ticker]], None],
        batch: list[Ticker],
        *,
        max_retries: int,
        retry_backoff_ms: int,
        retry_exceptions: tuple[type[Exception], ...],
    ) -> None:
        attempts = 0
        while True:
            try:
                callback(batch)
                return
            except retry_exceptions:
                if attempts >= max_retries:
                    raise
                attempts += 1
                if retry_backoff_ms > 0:
                    await asyncio.sleep(retry_backoff_ms / 1000 * attempts)


class AsyncWebSocketStream:
    """Async WebSocket stream handler (base class for exchange-specific implementations)."""
    
    def __init__(self, url: str, symbol: str):
        """Initialize WebSocket stream.
        
        Args:
            url: WebSocket URL
            symbol: Trading symbol to subscribe to
        """
        self.url = url
        self.symbol = symbol
        self._running = False
        
    async def connect(self) -> None:
        """Connect to WebSocket (to be implemented by subclasses)."""
        raise NotImplementedError
        
    async def disconnect(self) -> None:
        """Disconnect from WebSocket (to be implemented by subclasses)."""
        raise NotImplementedError
        
    async def subscribe(self) -> AsyncIterator[Ticker]:
        """Subscribe to tick updates (to be implemented by subclasses).
        
        Yields:
            Ticker objects from WebSocket
        """
        raise NotImplementedError
        

async def merge_streams(*streams: AsyncIterator[Ticker]) -> AsyncIterator[Ticker]:
    """Merge multiple async tick streams into one.
    
    Args:
        *streams: Variable number of async iterators
        
    Yields:
        Ticks from all streams in arrival order
    """
    from typing import Any
    pending_tasks: Dict[Any, AsyncIterator[Ticker]] = {
        asyncio.create_task(anext(stream)): stream  # type: ignore[arg-type]
        for stream in streams
    }
    
    while pending_tasks:
        done, pending = await asyncio.wait(
            pending_tasks.keys(),
            return_when=asyncio.FIRST_COMPLETED
        )
        
        for task in done:
            stream = pending_tasks.pop(task)
            try:
                tick = task.result()
                yield tick
                
                # Schedule next read from this stream
                pending_tasks[asyncio.create_task(anext(stream))] = stream  # type: ignore[arg-type]
            except StopAsyncIteration:
                # Stream exhausted, don't reschedule
                pass


__all__ = [
    "Ticker",
    "AsyncDataIngestor",
    "AsyncWebSocketStream",
    "merge_streams",
]
