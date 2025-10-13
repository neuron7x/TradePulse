"""Runtime utilities shared across TradePulse maintenance scripts."""
from __future__ import annotations

# SPDX-License-Identifier: MIT

import locale
import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .artifacts import ArtifactManager, create_artifact_manager
from .checksum import ChecksumMismatchError, compute_checksum, verify_checksum
from .exit_codes import EXIT_CODES
from .pathfinder import find_resources
from .progress import ProgressBar
from .retry import create_resilient_session
from .task_queue import TaskQueue, task_queue
from .transfer import TransferError, transfer_with_resume

DEFAULT_SEED = 1337
DEFAULT_LOCALE = "C"
_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


class UTCFormatter(logging.Formatter):
    """Format timestamps using ISO-8601 in UTC regardless of host settings."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:  # noqa: N802
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat(timespec="seconds")


@dataclass(frozen=True)
class LoadedEnvironment:
    """Representation of key/value pairs sourced from ``.env`` style files."""

    variables: Mapping[str, str]
    source: Path


def configure_deterministic_runtime(
    *, seed: int | None = None, locale_name: str | None = None
) -> None:
    """Apply deterministic defaults for random seed and locale."""

    resolved_seed = seed if seed is not None else int(os.getenv("SCRIPTS_RANDOM_SEED", DEFAULT_SEED))
    resolved_locale = locale_name or os.getenv("SCRIPTS_LOCALE", DEFAULT_LOCALE)

    os.environ["PYTHONHASHSEED"] = str(resolved_seed)
    random.seed(resolved_seed)

    try:  # pragma: no cover - numpy is optional in many environments
        import numpy as np  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - import guard is trivial
        pass
    else:  # pragma: no branch - simple deterministic seeding
        np.random.seed(resolved_seed)

    try:
        locale.setlocale(locale.LC_ALL, resolved_locale)
    except locale.Error:
        locale.setlocale(locale.LC_ALL, "")


def configure_logging(level: int) -> None:
    """Initialise the logging stack with UTC ISO-8601 timestamps."""

    handler = logging.StreamHandler()
    handler.setFormatter(UTCFormatter(_LOG_FORMAT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def parse_env_file(path: Path) -> LoadedEnvironment | None:
    """Parse a dotenv style file without leaking secret values."""

    if not path.exists():
        return None

    variables: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        variables[key] = value

    return LoadedEnvironment(variables=variables, source=path)


def apply_environment(overrides: Mapping[str, str]) -> None:
    """Update :data:`os.environ` without exposing secrets in the logs."""

    for key, value in overrides.items():
        os.environ[key] = value


__all__ = [
    "ArtifactManager",
    "ChecksumMismatchError",
    "EXIT_CODES",
    "LoadedEnvironment",
    "ProgressBar",
    "TaskQueue",
    "TransferError",
    "UTCFormatter",
    "apply_environment",
    "compute_checksum",
    "configure_deterministic_runtime",
    "configure_logging",
    "create_artifact_manager",
    "create_resilient_session",
    "find_resources",
    "parse_env_file",
    "task_queue",
    "transfer_with_resume",
    "verify_checksum",
]
