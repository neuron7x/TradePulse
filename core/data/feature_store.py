"""Online feature store helpers with integrity guards."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
from pathlib import Path
from typing import Literal

import pandas as pd

from core.utils.dataframe_io import (
    purge_dataframe_artifacts,
    read_dataframe,
    write_dataframe,
)


class FeatureStoreIntegrityError(RuntimeError):
    """Raised when integrity invariants fail for the online feature store."""


@dataclass(frozen=True)
class IntegritySnapshot:
    """Compact representation of a dataset used for integrity comparisons."""

    row_count: int
    data_hash: str


@dataclass(frozen=True)
class IntegrityReport:
    """Integrity comparison between the offline payload and persisted store."""

    feature_view: str
    offline_rows: int
    online_rows: int
    row_count_diff: int
    offline_hash: str
    online_hash: str
    hash_differs: bool

    def ensure_valid(self) -> None:
        """Raise :class:`FeatureStoreIntegrityError` when invariants are violated."""

        if self.row_count_diff != 0:
            raise FeatureStoreIntegrityError(
                f"Row count mismatch for {self.feature_view!r}: "
                f"offline={self.offline_rows}, online={self.online_rows}"
            )
        if self.hash_differs:
            raise FeatureStoreIntegrityError(
                f"Hash mismatch for {self.feature_view!r}: "
                f"offline={self.offline_hash}, online={self.online_hash}"
            )


class OnlineFeatureStore:
    """Simple parquet-backed store providing overwrite/append semantics."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, feature_view: str) -> Path:
        safe_name = feature_view.replace("/", "__").replace(".", "__")
        return self._root / safe_name

    def purge(self, feature_view: str) -> None:
        """Remove persisted artefacts for ``feature_view`` if they exist."""

        path = self._resolve_path(feature_view)
        purge_dataframe_artifacts(path)

    def load(self, feature_view: str) -> pd.DataFrame:
        """Load the persisted dataframe for ``feature_view``."""

        path = self._resolve_path(feature_view)
        return read_dataframe(path, allow_json_fallback=True)

    def sync(
        self,
        feature_view: str,
        frame: pd.DataFrame,
        *,
        mode: Literal["append", "overwrite"] = "append",
        validate: bool = True,
    ) -> IntegrityReport:
        """Persist ``frame`` and return an integrity report.

        When ``mode`` is ``"overwrite"`` the existing payload is purged before
        writing.  ``"append"`` concatenates to the existing dataset while still
        returning an integrity report for the delta written in this operation.
        """

        if mode not in {"append", "overwrite"}:
            raise ValueError("mode must be either 'append' or 'overwrite'")

        offline_frame = frame.copy(deep=True)
        path = self._resolve_path(feature_view)

        if mode == "overwrite":
            self.purge(feature_view)
            stored = self._write_frame(path, offline_frame)
            report = self._build_report(feature_view, offline_frame, stored)
        else:
            existing = self.load(feature_view)
            if not existing.empty:
                missing = set(existing.columns) ^ set(offline_frame.columns)
                if missing:
                    raise ValueError(
                        "Cannot append payload with mismatched columns: "
                        f"{sorted(missing)}"
                    )
                offline_frame = offline_frame[existing.columns]
            stored = self._append_frames(existing, offline_frame)
            self._write_frame(path, stored)
            delta_rows = offline_frame.shape[0]
            if delta_rows:
                online_delta = stored.tail(delta_rows).reset_index(drop=True)
            else:
                online_delta = stored.iloc[0:0]
            report = self._build_report(feature_view, offline_frame, online_delta)

        if validate:
            report.ensure_valid()
        return report

    def compute_integrity(self, feature_view: str, frame: pd.DataFrame) -> IntegrityReport:
        """Compare ``frame`` against the currently persisted dataset."""

        online = self.load(feature_view)
        return self._build_report(feature_view, frame.copy(deep=True), online)

    @staticmethod
    def _append_frames(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
        if existing.empty:
            return incoming.reset_index(drop=True)
        combined = pd.concat([existing.reset_index(drop=True), incoming.reset_index(drop=True)], ignore_index=True)
        return combined

    def _write_frame(self, path: Path, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.reset_index(drop=True)
        write_dataframe(prepared, path, index=False, allow_json_fallback=True)
        return prepared

    def _build_report(
        self,
        feature_view: str,
        offline_frame: pd.DataFrame,
        online_frame: pd.DataFrame,
    ) -> IntegrityReport:
        offline_snapshot = self._snapshot(offline_frame)
        online_snapshot = self._snapshot(online_frame)
        hash_differs = not hmac.compare_digest(offline_snapshot.data_hash, online_snapshot.data_hash)
        return IntegrityReport(
            feature_view=feature_view,
            offline_rows=offline_snapshot.row_count,
            online_rows=online_snapshot.row_count,
            row_count_diff=online_snapshot.row_count - offline_snapshot.row_count,
            offline_hash=offline_snapshot.data_hash,
            online_hash=online_snapshot.data_hash,
            hash_differs=hash_differs,
        )

    @staticmethod
    def _snapshot(frame: pd.DataFrame) -> IntegritySnapshot:
        canonical = OnlineFeatureStore._canonicalize(frame)
        payload = canonical.to_json(
            orient="split",
            index=False,
            date_format="iso",
            date_unit="ns",
            double_precision=15,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return IntegritySnapshot(row_count=int(canonical.shape[0]), data_hash=digest)

    @staticmethod
    def _canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        columns = sorted(frame.columns)
        canonical = frame.loc[:, columns].copy()
        if columns:
            canonical = canonical.sort_values(by=columns, kind="mergesort")
        return canonical.reset_index(drop=True)


__all__ = [
    "FeatureStoreIntegrityError",
    "IntegrityReport",
    "OnlineFeatureStore",
]
