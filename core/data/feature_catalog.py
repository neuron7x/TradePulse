"""Feature and signal catalog management for TradePulse."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

from pydantic import BaseModel

from core.config.template_manager import model_to_hash

__all__ = ["CatalogEntry", "FeatureCatalog"]


@dataclass(slots=True)
class CatalogEntry:
    """Metadata persisted for a registered artifact."""

    name: str
    path: Path
    checksum: Optional[str]
    config_hash: str
    lineage: List[str]
    metadata: Dict[str, Any]
    timestamp: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "checksum": self.checksum,
            "config_hash": self.config_hash,
            "lineage": self.lineage,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class FeatureCatalog:
    """Persist metadata about generated features, signals, and reports."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps({"artifacts": []}, indent=2), encoding="utf-8")

    def _load(self) -> MutableMapping[str, Any]:
        with self.path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save(self, payload: Mapping[str, Any]) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    @staticmethod
    def _checksum(path: Path) -> Optional[str]:
        if not path.exists():
            return None
        if path.is_dir():
            digest = __import__("hashlib").sha256()
            for root, _dirs, files in os.walk(path):
                for file_name in sorted(files):
                    file_path = Path(root) / file_name
                    digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
                    digest.update(file_path.read_bytes())
            return digest.hexdigest()
        return __import__("hashlib").sha256(path.read_bytes()).hexdigest()

    def register(
        self,
        name: str,
        artifact_path: Path,
        *,
        config: BaseModel,
        lineage: Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CatalogEntry:
        """Register an artifact and persist its metadata."""

        payload = self._load()
        artifacts: List[Dict[str, Any]] = list(payload.get("artifacts", []))
        checksum = self._checksum(artifact_path)
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        entry = CatalogEntry(
            name=name,
            path=artifact_path.resolve(),
            checksum=checksum,
            config_hash=model_to_hash(config),
            lineage=list(lineage or []),
            metadata=dict(metadata or {}),
            timestamp=timestamp,
        )
        artifacts = [item for item in artifacts if item.get("name") != name]
        artifacts.append(entry.as_dict())
        payload["artifacts"] = sorted(artifacts, key=lambda item: item["name"])
        self._save(payload)
        return entry

    def find(self, name: str) -> Optional[CatalogEntry]:
        """Return catalog entry for *name* if registered."""

        data = self._load()
        for item in data.get("artifacts", []):
            if item.get("name") == name:
                return CatalogEntry(
                    name=item["name"],
                    path=Path(item["path"]),
                    checksum=item.get("checksum"),
                    config_hash=item.get("config_hash", ""),
                    lineage=list(item.get("lineage", [])),
                    metadata=dict(item.get("metadata", {})),
                    timestamp=item.get("timestamp", ""),
                )
        return None

