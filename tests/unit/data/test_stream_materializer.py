from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from core.data import OnlineFeatureStore, RetentionPolicy
from core.data.materialization import InMemoryCheckpointStore, StreamMaterializer
from core.messaging.idempotency import InMemoryEventIdempotencyStore


def _records(values: list[tuple[str, datetime, float]]) -> list[dict[str, object]]:
    return [
        {
            "entity_id": entity,
            "timestamp": ts,
            "value": value,
        }
        for entity, ts, value in values
    ]


def test_stream_materializer_deduplicates_and_is_idempotent(tmp_path) -> None:
    store = OnlineFeatureStore(
        tmp_path,
        retention=RetentionPolicy(max_rows=10),
        dedup_keys=["entity_id", "timestamp"],
    )
    idempotency = InMemoryEventIdempotencyStore()
    checkpoints = InMemoryCheckpointStore()
    materializer = StreamMaterializer(
        store,
        "live.features",
        idempotency_store=idempotency,
        checkpoint_store=checkpoints,
        dedup_keys=("entity_id", "timestamp"),
    )

    ts = datetime(2024, 6, 1, tzinfo=UTC)
    batch = _records([("A", ts, 1.0), ("A", ts, 2.0), ("B", ts, 3.0)])
    report = materializer.process(batch, checkpoint_id="cp-1")
    assert report is not None
    stored = store.load("live.features")
    assert stored.shape[0] == 2
    assert stored.loc[stored["entity_id"] == "A", "value"].iloc[0] == 2.0

    # Reprocessing the same checkpoint should have no effect
    skipped = materializer.process(batch, checkpoint_id="cp-1")
    assert skipped is None
    stored_again = store.load("live.features")
    pd.testing.assert_frame_equal(stored_again, stored)

    # Backfill older data still dedups while preserving latest values
    backfill_ts = datetime(2024, 5, 20, tzinfo=UTC)
    backfill = pd.DataFrame(
        {
            "entity_id": ["A", "C"],
            "timestamp": [backfill_ts, backfill_ts],
            "value": [0.5, 4.0],
        }
    )
    materializer.backfill(backfill, checkpoint_id="cp-2")
    stored_final = store.load("live.features")
    assert stored_final.shape[0] == 4
    assert checkpoints.last_checkpoint("live.features") == "cp-2"
    assert stored_final.loc[stored_final["entity_id"] == "A", "value"].iloc[0] == 2.0
    assert stored_final.loc[stored_final["entity_id"] == "C", "value"].iloc[0] == 4.0
