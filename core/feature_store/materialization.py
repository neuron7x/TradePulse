"""Materialization workflows for offline and online feature stores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import FeatureStoreConfig, OnlineStoreConfig
from .models import FeatureSet
from .offline import OfflineStore, OfflineStoreFactory, WriteMode
from .online import OnlineStore, RedisOnlineStore, SQLiteOnlineStore


@dataclass(slots=True)
class MaterializationResult:
    """Result information returned by materialization operations."""

    feature_set: str
    offline_path: Optional[Path]
    rows_written_offline: int
    rows_written_online: int
    latest_timestamp: Optional[pd.Timestamp]


class FeatureMaterializer:
    """Coordinates feature materialization across offline and online stores."""

    def __init__(
        self,
        config: FeatureStoreConfig,
        offline_store: Optional[OfflineStore] = None,
        online_store: Optional[OnlineStore] = None,
    ) -> None:
        self._config = config
        self._offline_store = offline_store or OfflineStoreFactory(config.offline).create()
        self._online_store = online_store or self._create_online_store(config.online)

    @staticmethod
    def _create_online_store(config: OnlineStoreConfig) -> OnlineStore:
        if config.backend == "sqlite":
            return SQLiteOnlineStore(config)
        if config.backend == "redis":
            return RedisOnlineStore(config)
        raise ValueError(f"Unsupported online store backend: {config.backend}")

    def materialize(self, dataframe: pd.DataFrame, feature_set_name: str, mode: WriteMode = "append") -> MaterializationResult:
        """Persist new feature values to both offline and online stores."""

        if dataframe.empty:
            return MaterializationResult(
                feature_set=feature_set_name,
                offline_path=None,
                rows_written_offline=0,
                rows_written_online=0,
                latest_timestamp=self._offline_store.latest_timestamp(
                    feature_set_name, self._config.materialization.timestamp_column
                ),
            )
        feature_set = FeatureSet(
            name=feature_set_name,
            dataframe=dataframe,
            entity_columns=self._config.entity_columns(),
            timestamp_column=self._config.timestamp_column(),
        )
        offline_path = self._offline_store.write(feature_set, mode=mode)
        rows_online = 0
        if self._online_store is not None:
            # Validate freshness before mutating the online store so we never delete
            # the authoritative snapshot when attempting to load stale data.
            self._ensure_not_regressing(feature_set)
            if mode == "overwrite":
                # Align the online store with the offline overwrite by clearing out
                # rows that are no longer present in the incoming dataset.
                self._online_store.purge(feature_set_name)
            self._online_store.write(feature_set)
            rows_online = len(feature_set.dataframe)
        latest = self._offline_store.latest_timestamp(
            feature_set_name, self._config.materialization.timestamp_column
        )
        return MaterializationResult(
            feature_set=feature_set_name,
            offline_path=offline_path,
            rows_written_offline=len(feature_set.dataframe),
            rows_written_online=rows_online,
            latest_timestamp=latest,
        )

    def _ensure_not_regressing(self, feature_set: FeatureSet) -> None:
        if self._config.materialization.allow_overwrite:
            return
        latest_online = self._online_store.latest_timestamp(
            feature_set.name, feature_set.timestamp_column
        )
        newest_candidate = feature_set.dataframe[feature_set.timestamp_column].max()
        if latest_online is not None and newest_candidate <= latest_online:
            raise ValueError(
                "Incoming feature set contains data that is not fresher than the existing online snapshot."
            )

    def sync_online_from_offline(
        self, feature_set_name: str, since: Optional[pd.Timestamp] = None
    ) -> MaterializationResult:
        """Materialize offline changes to the online store starting from an optional timestamp."""

        frame = self._offline_store.read(feature_set_name)
        if since is not None:
            frame = frame[frame[self._config.timestamp_column()] > since]
        if frame.empty:
            latest = self._offline_store.latest_timestamp(
                feature_set_name, self._config.materialization.timestamp_column
            )
            return MaterializationResult(
                feature_set=feature_set_name,
                offline_path=self._offline_store.config.base_path / feature_set_name,
                rows_written_offline=0,
                rows_written_online=0,
                latest_timestamp=latest,
            )
        feature_set = FeatureSet(
            name=feature_set_name,
            dataframe=frame,
            entity_columns=self._config.entity_columns(),
            timestamp_column=self._config.timestamp_column(),
        )
        self._online_store.write(feature_set)
        latest = frame[self._config.timestamp_column()].max()
        return MaterializationResult(
            feature_set=feature_set_name,
            offline_path=self._offline_store.config.base_path / feature_set_name,
            rows_written_offline=0,
            rows_written_online=len(frame),
            latest_timestamp=latest,
        )

    def validate_consistency(self, feature_set_name: str) -> bool:
        """Check that offline and online stores are aligned up to the latest timestamp."""

        offline_latest = self._offline_store.latest_timestamp(
            feature_set_name, self._config.materialization.timestamp_column
        )
        if self._online_store is None:
            return True
        online_latest = self._online_store.latest_timestamp(
            feature_set_name, self._config.materialization.timestamp_column
        )
        if offline_latest is None:
            return online_latest is None
        if online_latest is None:
            return False
        return online_latest >= offline_latest


__all__ = ["FeatureMaterializer", "MaterializationResult"]
