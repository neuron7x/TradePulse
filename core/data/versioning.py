"""Abstractions for dataset and artifact versioning using DVC or LakeFS."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from core.config.cli_models import VersioningConfig

__all__ = ["VersioningError", "DataVersionManager"]


class VersioningError(RuntimeError):
    """Raised when version control operations fail."""


@dataclass(slots=True)
class DataVersionManager:
    """Manage dataset snapshots across optional backends."""

    config: VersioningConfig

    def snapshot(self, artifact_path: Path, *, push: bool = False) -> Dict[str, Any]:
        """Attempt to version control *artifact_path* according to config."""

        artifact_path = Path(artifact_path)
        result: Dict[str, Any] = {
            "backend": self.config.backend,
            "artifact": str(artifact_path.resolve()),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        if self.config.backend == "none":
            result["status"] = "skipped"
            return self._write_metadata(artifact_path, result)

        if self.config.backend == "dvc":
            return self._handle_dvc(artifact_path, result, push=push)
        if self.config.backend == "lakefs":
            return self._handle_lakefs(artifact_path, result)
        result["status"] = "unknown-backend"
        return self._write_metadata(artifact_path, result)

    def _handle_dvc(self, artifact_path: Path, result: Dict[str, Any], *, push: bool) -> Dict[str, Any]:
        dvc_path = shutil.which("dvc")
        if dvc_path is None:
            result["status"] = "dvc-missing"
            return self._write_metadata(artifact_path, result)

        commands = [[dvc_path, "add", str(artifact_path)]]
        if push:
            commands.append([dvc_path, "push"])
        try:
            for command in commands:
                self._run(command, cwd=self.config.repo_path)
        except VersioningError as exc:
            result["status"] = "error"
            result["error"] = str(exc)
            return self._write_metadata(artifact_path, result)

        result["status"] = "tracked"
        if push:
            result["pushed"] = True
        return self._write_metadata(artifact_path, result)

    def _handle_lakefs(self, artifact_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
        lakectl = shutil.which("lakectl")
        if lakectl is None:
            result["status"] = "lakefs-missing"
            return self._write_metadata(artifact_path, result)

        repo = self.config.remote or "tradepulse"
        branch = self.config.branch or "main"
        try:
            self._run(
                [
                    lakectl,
                    "fs",
                    "upload",
                    repo,
                    branch,
                    str(artifact_path),
                    str(artifact_path.name),
                ],
                cwd=self.config.repo_path,
            )
        except VersioningError as exc:
            result["status"] = "error"
            result["error"] = str(exc)
            return self._write_metadata(artifact_path, result)

        result["status"] = "uploaded"
        result["branch"] = branch
        result["repository"] = repo
        return self._write_metadata(artifact_path, result)

    def _run(self, command: list[str], *, cwd: Optional[Path]) -> None:
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                cwd=None if cwd is None else str(cwd),
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive guard
            raise VersioningError(
                f"Command {' '.join(command)} failed: {exc.stderr or exc.stdout}"
            ) from exc
        if completed.stdout:
            return

    def _write_metadata(self, artifact_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
        metadata_path = self._metadata_path(artifact_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result

    @staticmethod
    def _metadata_path(artifact_path: Path) -> Path:
        if artifact_path.is_dir():
            return artifact_path / ".version.json"
        suffix = artifact_path.suffix or ""
        return artifact_path.with_suffix(f"{suffix}.version.json")

