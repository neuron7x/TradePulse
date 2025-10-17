"""Minimal artifact versioning helpers for the CLI."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from core.config.cli_models import VersioningConfig


class VersioningError(RuntimeError):
    """Raised when versioning operations cannot be completed."""

    pass


class DataVersionManager:
    """Persist lightweight metadata about produced artifacts."""

    def __init__(self, config: VersioningConfig) -> None:
        self.config = config

    def snapshot(
        self, artifact_path: Path, metadata: Optional[Dict[str, object]] = None
    ) -> Dict[str, object]:
        artifact_path = Path(artifact_path)
        info = {
            "backend": self.config.backend,
            "artifact": str(artifact_path.resolve()),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        if self.config.backend != "none":
            info["repo_path"] = str(self.config.repo_path)
        version_path = artifact_path.with_suffix(f"{artifact_path.suffix}.version.json")
        version_path.write_text(
            json.dumps(info, indent=2, sort_keys=True), encoding="utf-8"
        )
        return info
