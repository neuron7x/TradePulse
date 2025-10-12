from __future__ import annotations

from datetime import UTC

import pandas as pd
import pytest

from core.data.materialization import (
    Checkpoint,
    InMemoryCheckpointStore,
    StreamMaterializer,
)


@pytest.fixture
def stream_payload() -> pd.DataFrame:
    ts = pd.Timestamp("2024-01-01 00:00:00", tz=UTC)
    return pd.DataFrame(
        {
            "entity_id": ["a", "a", "b", "b"],
            "ts": [
                ts,
                ts,
                ts - pd.Timedelta(minutes=1),
                ts + pd.Timedelta(minutes=1),
            ],
            "value": [1.0, 1.0, 2.0, 3.0],
        }
    )


def test_microbatch_checkpointing(stream_payload: pd.DataFrame) -> None:
    checkpoint_store = InMemoryCheckpointStore()
    written: list[pd.DataFrame] = []

    def writer(_name: str, frame: pd.DataFrame) -> None:
        written.append(frame.copy())

    materializer = StreamMaterializer(
        writer,
        checkpoint_store,
        microbatch_size=2,
        dedup_keys=("entity_id", "ts"),
        backfill_loader=lambda _: pd.DataFrame(
            {
                "entity_id": ["c"],
                "ts": [pd.Timestamp("2023-12-31 23:59:00", tz=UTC)],
                "value": [9.0],
            }
        ),
    )

    materializer.materialize("features.demo", stream_payload)

    assert len(written) == 2
    # Backfill row is merged with each batch; ensure dedup keeps latest.
    assert all("c" not in batch["entity_id"].values for batch in written)
    assert all(batch["entity_id"].isin(["a", "b"]).all() for batch in written)

    # Replaying the same payload must be idempotent.
    materializer.materialize("features.demo", stream_payload)
    assert len(written) == 2
    checkpoint = checkpoint_store.load("features.demo")
    assert isinstance(checkpoint, Checkpoint)
    assert len(checkpoint.checkpoint_ids) == 2


def test_deduplicate_requires_keys(stream_payload: pd.DataFrame) -> None:
    checkpoint_store = InMemoryCheckpointStore()

    materializer = StreamMaterializer(
        lambda *_: None,
        checkpoint_store,
        microbatch_size=2,
        dedup_keys=("entity_id", "ts"),
    )

    with pytest.raises(KeyError):
        bad_payload = stream_payload.drop(columns=["ts"])
        materializer.materialize("features.demo", bad_payload)


def test_materializer_configuration_guards() -> None:
    checkpoint_store = InMemoryCheckpointStore()

    with pytest.raises(ValueError):
        StreamMaterializer(lambda *_: None, checkpoint_store, microbatch_size=0)

    with pytest.raises(ValueError):
        StreamMaterializer(lambda *_: None, checkpoint_store, dedup_keys=())


def test_materialize_accepts_iterable_batches(stream_payload: pd.DataFrame) -> None:
    checkpoint_store = InMemoryCheckpointStore()
    writes: list[pd.DataFrame] = []

    materializer = StreamMaterializer(
        lambda _name, frame: writes.append(frame.copy()),
        checkpoint_store,
        microbatch_size=3,
    )

    materializer.materialize("features.demo", [stream_payload])
    assert writes


def test_backfill_only_updates_checkpoint(stream_payload: pd.DataFrame) -> None:
    checkpoint_store = InMemoryCheckpointStore()
    writes: list[pd.DataFrame] = []

    deduped = stream_payload.sort_values(by=["entity_id", "ts"], kind="mergesort").drop_duplicates(
        ["entity_id", "ts"], keep="last"
    )

    materializer = StreamMaterializer(
        lambda _name, frame: writes.append(frame.copy()),
        checkpoint_store,
        backfill_loader=lambda _name: deduped,
    )

    materializer.materialize("features.demo", stream_payload)
    assert writes == []
    checkpoint = checkpoint_store.load("features.demo")
    assert checkpoint is not None
    assert len(checkpoint.checkpoint_ids) == 1


def test_checkpoint_add_does_not_mutate_original() -> None:
    checkpoint = Checkpoint("demo", frozenset({"one"}))
    updated = checkpoint.add("two")

    assert checkpoint is not updated
    assert checkpoint.checkpoint_ids == frozenset({"one"})
    assert updated.checkpoint_ids == frozenset({"one", "two"})
