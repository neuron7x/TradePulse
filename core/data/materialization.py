"""Streaming materialisation utilities for online feature stores."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable, Iterable, MutableMapping, Protocol, Sequence

import pandas as pd

from core.data.feature_store import OnlineFeatureStore


@dataclass(frozen=True)
class Checkpoint:
    """Collection of processed checkpoint identifiers for a feature view."""

    feature_view: str
    checkpoint_ids: frozenset[str]

    def contains(self, checkpoint_id: str) -> bool:
        return checkpoint_id in self.checkpoint_ids

    def add(self, checkpoint_id: str) -> "Checkpoint":
        ids = set(self.checkpoint_ids)
        ids.add(checkpoint_id)
        return Checkpoint(self.feature_view, frozenset(ids))


class CheckpointStore(Protocol):
    """Minimal persistence abstraction for checkpoint metadata."""

    def load(self, feature_view: str) -> Checkpoint | None:
        ...

    def save(self, checkpoint: Checkpoint) -> None:
        ...


class InMemoryCheckpointStore:
    """In-memory checkpoint store used in tests and lightweight deployments."""

    def __init__(self) -> None:
        self._state: MutableMapping[str, Checkpoint] = {}

    def load(self, feature_view: str) -> Checkpoint | None:  # pragma: no cover - trivial
        return self._state.get(feature_view)

    def save(self, checkpoint: Checkpoint) -> None:
        current = self._state.get(checkpoint.feature_view)
        if current is None:
            self._state[checkpoint.feature_view] = checkpoint
            return
        ids = set(current.checkpoint_ids)
        ids.update(checkpoint.checkpoint_ids)
        self._state[checkpoint.feature_view] = Checkpoint(checkpoint.feature_view, frozenset(ids))


class StreamMaterializer:
    """Materialise streaming payloads into micro-batches with idempotent writes."""

    def __init__(
        self,
        writer: Callable[[str, pd.DataFrame], None],
        checkpoint_store: CheckpointStore,
        *,
        microbatch_size: int = 500,
        dedup_keys: Sequence[str] = ("entity_id", "ts"),
        backfill_loader: Callable[[str], pd.DataFrame] | None = None,
    ) -> None:
        if microbatch_size <= 0:
            raise ValueError("microbatch_size must be positive")
        if not dedup_keys:
            raise ValueError("dedup_keys cannot be empty")
        self._writer = writer
        self._checkpoint_store = checkpoint_store
        self._microbatch_size = microbatch_size
        self._dedup_keys = tuple(dedup_keys)
        self._backfill_loader = backfill_loader

    def materialize(
        self,
        feature_view: str,
        payload: pd.DataFrame | Iterable[pd.DataFrame],
    ) -> None:
        """Materialise ``payload`` into the online store respecting checkpoints."""

        batches = self._iter_batches(payload)
        for batch in batches:
            if batch.empty:
                continue
            self._process_batch(feature_view, batch)

    def _process_batch(self, feature_view: str, batch: pd.DataFrame) -> None:
        deduped = self._deduplicate(batch)
        if deduped.empty:
            return
        checkpoint_id = self._checkpoint_id(feature_view, deduped)
        last = self._checkpoint_store.load(feature_view)
        if last and last.contains(checkpoint_id):
            return

        new_rows = deduped
        if self._backfill_loader is not None:
            history = self._backfill_loader(feature_view)
            if not history.empty:
                combined = self._deduplicate(
                    pd.concat([history, deduped], ignore_index=True)
                )
                key_columns = list(self._dedup_keys)
                history_keys = history[key_columns].drop_duplicates()
                merged = combined.merge(
                    history_keys,
                    on=key_columns,
                    how="left",
                    indicator=True,
                )
                new_rows = merged[merged["_merge"] == "left_only"].drop(columns="_merge")
        if new_rows.empty:
            self._checkpoint_store.save(Checkpoint(feature_view, frozenset({checkpoint_id})))
            return

        self._writer(feature_view, new_rows.reset_index(drop=True))
        self._checkpoint_store.save(Checkpoint(feature_view, frozenset({checkpoint_id})))

    def _deduplicate(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(self._dedup_keys) - set(frame.columns)
        if missing:
            raise KeyError(
                f"Unable to deduplicate batch missing columns: {sorted(missing)}"
            )
        ordered = frame.sort_values(by=list(self._dedup_keys), kind="mergesort")
        deduped = ordered.drop_duplicates(list(self._dedup_keys), keep="last")
        return deduped.reset_index(drop=True)

    def _checkpoint_id(self, feature_view: str, frame: pd.DataFrame) -> str:
        canonical = OnlineFeatureStore._canonicalize(frame)
        payload = canonical.to_json(
            orient="split",
            index=False,
            date_format="iso",
            date_unit="ns",
            double_precision=15,
        )
        digest = hashlib.sha256()
        digest.update(feature_view.encode("utf-8"))
        digest.update(payload.encode("utf-8"))
        return digest.hexdigest()

    def _iter_batches(
        self,
        payload: pd.DataFrame | Iterable[pd.DataFrame],
    ) -> Iterable[pd.DataFrame]:
        if isinstance(payload, pd.DataFrame):
            yield from self._chunk_frame(payload)
            return
        for frame in payload:
            yield from self._chunk_frame(frame)

    def _chunk_frame(self, frame: pd.DataFrame) -> Iterable[pd.DataFrame]:
        if frame.empty:
            return []
        for start in range(0, len(frame), self._microbatch_size):
            stop = start + self._microbatch_size
            yield frame.iloc[start:stop].reset_index(drop=True)


__all__ = [
    "Checkpoint",
    "CheckpointStore",
    "InMemoryCheckpointStore",
    "StreamMaterializer",
]
