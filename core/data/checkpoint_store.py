"""Persistent checkpoint store implementations for streaming materialisation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .materialization import Checkpoint, CheckpointStore

__all__ = ["JsonCheckpointStore"]


class JsonCheckpointStore(CheckpointStore):
    """File-backed checkpoint store storing payloads as JSON."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._state: Dict[str, Checkpoint] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Checkpoint store payload must be a mapping")
        checkpoints = raw.get("checkpoints")
        if checkpoints is None:
            raise ValueError("Checkpoint store payload missing 'checkpoints' key")
        if not isinstance(checkpoints, dict):
            raise ValueError("'checkpoints' must be a mapping")
        parsed: Dict[str, Checkpoint] = {}
        for feature_view, ids in checkpoints.items():
            if not isinstance(feature_view, str):
                raise ValueError("Feature view keys must be strings")
            if not isinstance(ids, list):
                raise ValueError("Checkpoint identifiers must be stored as a list")
            if not all(isinstance(item, str) for item in ids):
                raise ValueError("Checkpoint identifiers must be strings")
            parsed[feature_view] = Checkpoint(feature_view, frozenset(ids))
        self._state = parsed

    def _flush(self) -> None:
        payload = {
            "checkpoints": {
                feature_view: sorted(checkpoint.checkpoint_ids)
                for feature_view, checkpoint in self._state.items()
            }
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def load(self, feature_view: str) -> Checkpoint | None:  # pragma: no cover - trivial
        return self._state.get(feature_view)

    def save(self, checkpoint: Checkpoint) -> None:
        current = self._state.get(checkpoint.feature_view)
        if current is None:
            merged = checkpoint
        else:
            ids = set(current.checkpoint_ids)
            ids.update(checkpoint.checkpoint_ids)
            merged = Checkpoint(checkpoint.feature_view, frozenset(ids))
        self._state[checkpoint.feature_view] = merged
        self._flush()
