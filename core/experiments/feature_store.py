"""Feature store integrations for TradePulse experiments."""

from __future__ import annotations

import importlib
import importlib.util
from datetime import datetime
from typing import Iterable, Mapping


class FeastFeatureStoreClient:
    """Thin wrapper around Feast's `FeatureStore` with lazy import semantics."""

    def __init__(self, repo_path: str, *, project: str | None = None) -> None:
        spec = importlib.util.find_spec("feast")
        if spec is None:
            raise RuntimeError("feast must be installed to use FeastFeatureStoreClient")
        feast = importlib.import_module("feast")
        self._store = feast.FeatureStore(repo_path=repo_path, project=project)

    def apply(self, objects: Iterable[object]) -> None:
        self._store.apply(objects)

    def materialize(self, start_date: datetime, end_date: datetime) -> None:
        self._store.materialize(start_date, end_date)

    def materialize_incremental(self, end_date: datetime) -> None:
        self._store.materialize_incremental(end_date)

    def get_online_features(
        self,
        feature_refs: Iterable[str],
        entity_rows: Iterable[Mapping[str, object]],
    ) -> Mapping[str, list]:
        result = self._store.get_online_features(features=list(feature_refs), entity_rows=list(entity_rows))
        return result.to_dict()
