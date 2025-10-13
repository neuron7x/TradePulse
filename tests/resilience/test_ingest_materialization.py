from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from core.data.async_ingestion import AsyncDataIngestor
from core.data.materialization import InMemoryCheckpointStore, StreamMaterializer
from core.data.models import PriceTick as Ticker
from core.data.timeutils import normalize_timestamp


@dataclass
class WriterProbe:
    calls: int = 0
    successes: int = 0
    failures_before_success: int = 0

    def __call__(self, feature_view: str, frame: pd.DataFrame) -> None:
        self.calls += 1
        if self.failures_before_success > 0:
            self.failures_before_success -= 1
            raise ConnectionError("transient network drop")
        self.successes += 1
        # Simulate persisted side effect by validating the frame is not empty
        assert not frame.empty


@pytest.mark.asyncio
async def test_async_batch_process_recovers_from_transient_failure() -> None:
    ingestor = AsyncDataIngestor()
    ticks = [
        Ticker.create(
            symbol="BTC-USD",
            venue="TEST",
            price=100 + idx,
            timestamp=normalize_timestamp(float(idx)),
            volume=1.0,
        )
        for idx in range(6)
    ]

    async def tick_stream():
        for tick in ticks:
            yield tick

    batches: list[list[AsyncDataIngestor.Ticker]] = []
    probe: dict[str, int] = {"failures": 0}

    def callback(batch: list[AsyncDataIngestor.Ticker]) -> None:
        if probe["failures"] < 2:
            probe["failures"] += 1
            raise ConnectionError("downstream sink unavailable")
        batches.append(batch)

    total = await ingestor.batch_process(
        tick_stream(),
        callback,
        batch_size=3,
        max_retries=3,
        retry_backoff_ms=0,
    )
    assert total == len(ticks)
    assert len(batches) == 2
    assert probe["failures"] == 2


@pytest.mark.asyncio
async def test_async_batch_process_raises_when_retries_exhausted() -> None:
    ingestor = AsyncDataIngestor()

    async def tick_stream():
        for idx in range(3):
            yield Ticker.create(
                symbol="ETH-USD",
                venue="TEST",
                price=200 + idx,
                timestamp=normalize_timestamp(float(idx)),
                volume=1.0,
            )

    def callback(batch: list[AsyncDataIngestor.Ticker]) -> None:
        raise TimeoutError("db locked")

    with pytest.raises(TimeoutError):
        await ingestor.batch_process(
            tick_stream(),
            callback,
            batch_size=2,
            max_retries=1,
            retry_backoff_ms=0,
        )


def test_stream_materializer_retries_and_checkpoint_once() -> None:
    frame = pd.DataFrame({"entity_id": np.arange(5), "ts": np.arange(5), "value": np.random.rand(5)})
    writer = WriterProbe(failures_before_success=2)
    checkpoints = InMemoryCheckpointStore()
    materializer = StreamMaterializer(
        writer,
        checkpoints,
        microbatch_size=10,
        max_retries=3,
        retry_backoff_seconds=0.0,
    )

    materializer.materialize("features.orders", frame)
    assert writer.calls == 3
    checkpoint = checkpoints.load("features.orders")
    assert checkpoint is not None
    assert len(checkpoint.checkpoint_ids) == 1


def test_stream_materializer_does_not_checkpoint_on_failure() -> None:
    frame = pd.DataFrame({"entity_id": np.arange(3), "ts": np.arange(3), "value": np.random.rand(3)})
    writer = WriterProbe(failures_before_success=5)
    checkpoints = InMemoryCheckpointStore()
    materializer = StreamMaterializer(
        writer,
        checkpoints,
        microbatch_size=5,
        max_retries=1,
        retry_backoff_seconds=0.0,
    )

    with pytest.raises(ConnectionError):
        materializer.materialize("features.signals", frame)
    assert checkpoints.load("features.signals") is None

    writer.failures_before_success = 0
    materializer.materialize("features.signals", frame)
    assert checkpoints.load("features.signals") is not None
