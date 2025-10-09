# SPDX-License-Identifier: MIT
"""Tests for async data ingestion."""

import asyncio
import csv
import tempfile
from pathlib import Path

import pytest

from core.data.async_ingestion import AsyncDataIngestor, Ticker, merge_streams


class TestAsyncDataIngestor:
    """Test async data ingestion functionality."""
    
    @pytest.mark.asyncio
    async def test_read_csv_basic(self, tmp_path: Path) -> None:
        """Test basic CSV reading."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("ts,price,volume\n1.0,100.0,1000\n2.0,101.0,2000\n")
        
        ingestor = AsyncDataIngestor()
        ticks = []
        
        async for tick in ingestor.read_csv(str(csv_file), symbol="TEST"):
            ticks.append(tick)
            
        assert len(ticks) == 2
        assert ticks[0].price == 100.0
        assert ticks[1].price == 101.0
        assert ticks[0].symbol == "TEST"
        
    @pytest.mark.asyncio
    async def test_read_csv_chunked(self, tmp_path: Path) -> None:
        """Test CSV reading with chunks."""
        csv_file = tmp_path / "test.csv"
        
        # Create CSV with 10 rows
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ts", "price", "volume"])
            for i in range(10):
                writer.writerow([float(i), 100.0 + i, 1000])
                
        ingestor = AsyncDataIngestor()
        ticks = []
        
        async for tick in ingestor.read_csv(str(csv_file), chunk_size=3):
            ticks.append(tick)
            
        assert len(ticks) == 10
        assert ticks[5].price == 105.0
        
    @pytest.mark.asyncio
    async def test_read_csv_missing_columns(self, tmp_path: Path) -> None:
        """Test CSV with missing required columns."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("timestamp,value\n1.0,100.0\n")
        
        ingestor = AsyncDataIngestor()
        
        with pytest.raises(ValueError, match="missing columns"):
            async for _ in ingestor.read_csv(str(csv_file)):
                pass
                
    @pytest.mark.asyncio
    async def test_stream_ticks_basic(self) -> None:
        """Test basic tick streaming."""
        ingestor = AsyncDataIngestor()
        ticks = []
        
        async for tick in ingestor.stream_ticks(
            "test_source",
            "BTC",
            interval_ms=10,
            max_ticks=5
        ):
            ticks.append(tick)
            
        assert len(ticks) == 5
        assert all(tick.symbol == "BTC" for tick in ticks)
        
    @pytest.mark.asyncio
    async def test_batch_process(self, tmp_path: Path) -> None:
        """Test batch processing of ticks."""
        csv_file = tmp_path / "test.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ts", "price", "volume"])
            for i in range(25):
                writer.writerow([float(i), 100.0 + i, 1000])
                
        ingestor = AsyncDataIngestor()
        batches = []
        
        def collect_batch(batch):
            batches.append(len(batch))
            
        ticks_iter = ingestor.read_csv(str(csv_file))
        total = await ingestor.batch_process(ticks_iter, collect_batch, batch_size=10)
        
        assert total == 25
        assert len(batches) == 3  # 10 + 10 + 5
        assert batches[0] == 10
        assert batches[1] == 10
        assert batches[2] == 5
        

class TestMergeStreams:
    """Test stream merging functionality."""
    
    async def generate_ticks(self, symbol: str, count: int, delay_ms: int = 5) -> Ticker:
        """Helper to generate ticks."""
        for i in range(count):
            await asyncio.sleep(delay_ms / 1000.0)
            yield Ticker(ts=float(i), price=100.0 + i, volume=1000, symbol=symbol)
            
    @pytest.mark.asyncio
    async def test_merge_two_streams(self) -> None:
        """Test merging two async streams."""
        stream1 = self.generate_ticks("BTC", 3)
        stream2 = self.generate_ticks("ETH", 3)
        
        ticks = []
        async for tick in merge_streams(stream1, stream2):
            ticks.append(tick)
            
        assert len(ticks) == 6
        symbols = [tick.symbol for tick in ticks]
        assert "BTC" in symbols
        assert "ETH" in symbols
        
    @pytest.mark.asyncio
    async def test_merge_empty_stream(self) -> None:
        """Test merging with empty stream."""
        async def empty_stream():
            return
            yield  # Make it a generator
            
        stream1 = self.generate_ticks("BTC", 2)
        stream2 = empty_stream()
        
        ticks = []
        async for tick in merge_streams(stream1, stream2):
            ticks.append(tick)
            
        assert len(ticks) == 2
        assert all(tick.symbol == "BTC" for tick in ticks)


class TestAsyncWebSocketStream:
    """Test WebSocket stream base class."""
    
    @pytest.mark.asyncio
    async def test_websocket_not_implemented(self) -> None:
        """Test that base WebSocket methods raise NotImplementedError."""
        from core.data.async_ingestion import AsyncWebSocketStream
        
        stream = AsyncWebSocketStream("ws://test", "BTC")
        
        with pytest.raises(NotImplementedError):
            await stream.connect()
            
        with pytest.raises(NotImplementedError):
            await stream.disconnect()
            
        # subscribe() should raise NotImplementedError when awaited
        with pytest.raises(NotImplementedError):
            await stream.subscribe()
