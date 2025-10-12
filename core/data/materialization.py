"""Streaming materialisation helpers for the online feature store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Protocol, Sequence

import pandas as pd

from core.data.feature_store import IntegrityReport, OnlineFeatureStore
from core.messaging.idempotency import EventIdempotencyStore


class CheckpointStore(Protocol):
    """Minimal interface for persisting stream checkpoints."""

    def last_checkpoint(self, feature_view: str) -> str | None:
        ...

    def store_checkpoint(self, feature_view: str, checkpoint_id: str) -> None:
        ...


class InMemoryCheckpointStore:
    """Thread-unsafe in-memory checkpoint store for testing and small jobs."""

    def __init__(self) -> None:
        self._state: MutableMapping[str, str] = {}

    def last_checkpoint(self, feature_view: str) -> str | None:  # pragma: no cover - trivial
        return self._state.get(feature_view)

    def store_checkpoint(self, feature_view: str, checkpoint_id: str) -> None:
        self._state[feature_view] = checkpoint_id


@dataclass
class MicroBatch:
    """Micro-batch payload produced by stream ingestion."""

    records: pd.DataFrame
    checkpoint_id: str
    backfill: bool = False


class StreamMaterializer:
    """Perform idempotent micro-batch writes into an :class:`OnlineFeatureStore`."""

    def __init__(
        self,
        store: OnlineFeatureStore,
        feature_view: str,
        *,
        idempotency_store: EventIdempotencyStore,
        checkpoint_store: CheckpointStore | None = None,
        dedup_keys: Sequence[str] = ("entity_id", "timestamp"),
    ) -> None:
        self._store = store
        self._feature_view = feature_view
        self._idempotency_store = idempotency_store
        self._checkpoint_store = checkpoint_store or InMemoryCheckpointStore()
        self._dedup_keys = list(dedup_keys)

    def process(
        self,
        records: Iterable[Mapping[str, object]] | pd.DataFrame,
        *,
        checkpoint_id: str,
        backfill: bool = False,
        validate: bool = True,
    ) -> IntegrityReport | None:
        """Process a micro-batch and persist it into the online store."""

        if self._idempotency_store.was_processed(checkpoint_id):
            return None

        frame = self._to_frame(records)
        if not frame.empty and self._dedup_keys:
            missing = [key for key in self._dedup_keys if key not in frame.columns]
            if missing:
                raise ValueError(
                    "Deduplication keys missing from payload: "
                    f"{', '.join(sorted(missing))}"
                )
            frame = frame.drop_duplicates(subset=self._dedup_keys, keep="last")

        if frame.empty:
            self._idempotency_store.mark_processed(checkpoint_id)
            self._checkpoint_store.store_checkpoint(self._feature_view, checkpoint_id)
            return None

        report = self._store.sync(
            self._feature_view,
            frame,
            mode="append" if not backfill else "append",
            validate=validate,
        )

        self._idempotency_store.mark_processed(checkpoint_id)
        self._checkpoint_store.store_checkpoint(self._feature_view, checkpoint_id)
        return report

    def backfill(
        self,
        frame: pd.DataFrame,
        *,
        checkpoint_id: str,
        validate: bool = True,
    ) -> IntegrityReport | None:
        """Convenience wrapper around :meth:`process` for backfill payloads."""

        return self.process(frame, checkpoint_id=checkpoint_id, backfill=True, validate=validate)

    @staticmethod
    def _to_frame(records: Iterable[Mapping[str, object]] | pd.DataFrame) -> pd.DataFrame:
        if isinstance(records, pd.DataFrame):
            return records.copy(deep=True)
        return pd.DataFrame(list(records))


__all__ = [
    "CheckpointStore",
    "InMemoryCheckpointStore",
    "MicroBatch",
    "StreamMaterializer",
]
