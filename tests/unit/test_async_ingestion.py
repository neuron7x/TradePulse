# SPDX-License-Identifier: MIT
"""Tests for async data ingestion."""

import asyncio
import csv
from datetime import datetime, timezone
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

        async for tick in ingestor.read_csv(str(csv_file), symbol="TEST", venue="TEST"):
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
            "test_source", "BTC", interval_ms=10, max_ticks=5
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

    @pytest.mark.asyncio
    async def test_read_csv_respects_allowed_roots(self, tmp_path: Path) -> None:
        """The async ingestor should reject paths outside the configured roots."""

        csv_file = tmp_path / "outside.csv"
        csv_file.write_text("ts,price\n1,1\n", encoding="utf-8")
        allowed_root = tmp_path / "allowed"
        allowed_root.mkdir()

        ingestor = AsyncDataIngestor(allowed_roots=[allowed_root])

        with pytest.raises(PermissionError):
            async for _ in ingestor.read_csv(str(csv_file)):
                pass

    @pytest.mark.asyncio
    async def test_read_csv_respects_size_limit(self, tmp_path: Path) -> None:
        """The async ingestor should enforce configured file size limits."""

        csv_file = tmp_path / "big.csv"
        csv_file.write_text(
            "ts,price\n" + "\n".join("1,1" for _ in range(40)), encoding="utf-8"
        )

        ingestor = AsyncDataIngestor(allowed_roots=[tmp_path], max_csv_bytes=32)

        with pytest.raises(ValueError, match="exceeds"):
            async for _ in ingestor.read_csv(str(csv_file)):
                pass


class TestMergeStreams:
    """Test stream merging functionality."""

    async def generate_ticks(
        self, symbol: str, count: int, delay_ms: int = 5
    ) -> Ticker:
        """Helper to generate ticks."""
        for i in range(count):
            await asyncio.sleep(delay_ms / 1000.0)
            yield Ticker.create(
                symbol=symbol,
                venue="TEST",
                price=100.0 + i,
                timestamp=datetime.fromtimestamp(float(i), tz=timezone.utc),
                volume=1000,
            )

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

    @pytest.mark.asyncio
    async def test_merge_streams_handles_failures(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Failed streams should be logged and skipped while others continue."""

        async def flaky_stream():
            yield Ticker.create(
                symbol="FLAKY",
                venue="TEST",
                price=101.0,
                timestamp=datetime.fromtimestamp(0, tz=timezone.utc),
                volume=1_000,
            )
            raise ConnectionError("network down")

        stream_ok = self.generate_ticks("BTC", 3, delay_ms=1)

        caplog.set_level("WARNING")
        collected: list[Ticker] = []

        async for tick in merge_streams(flaky_stream(), stream_ok):
            collected.append(tick)

        symbols = [tick.symbol for tick in collected]
        assert symbols.count("BTC") == 3
        assert "FLAKY" in symbols
        assert any(
            getattr(record, "extra_fields", {}).get("error") == "network down"
            for record in caplog.records
        )


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
