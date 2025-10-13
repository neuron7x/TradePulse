"""Runtime utilities shared across TradePulse maintenance scripts."""
from __future__ import annotations

# SPDX-License-Identifier: MIT

from importlib import util as importlib_util
from pathlib import Path
import sys
from types import ModuleType

from .artifacts import ArtifactManager, create_artifact_manager
from .checksum import ChecksumMismatchError, compute_checksum, verify_checksum
from .exit_codes import EXIT_CODES
from .pathfinder import find_resources
from .progress import ProgressBar
from .retry import create_resilient_session
from .task_queue import TaskQueue, task_queue
from .transfer import TransferError, transfer_with_resume


def _load_runtime_module() -> ModuleType:
    module_name = "scripts._runtime_impl"
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    runtime_path = Path(__file__).resolve().parent.parent / "runtime.py"
    spec = importlib_util.spec_from_file_location(module_name, runtime_path)
    if spec is None or spec.loader is None:
        msg = f"Unable to load runtime helpers from {runtime_path!s}"
        raise ImportError(msg)

    module = importlib_util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_runtime_impl = _load_runtime_module()
apply_environment = _runtime_impl.apply_environment
configure_deterministic_runtime = _runtime_impl.configure_deterministic_runtime
configure_logging = _runtime_impl.configure_logging
parse_env_file = _runtime_impl.parse_env_file


__all__ = [
    "ArtifactManager",
    "ChecksumMismatchError",
    "EXIT_CODES",
    "ProgressBar",
    "TaskQueue",
    "TransferError",
    "apply_environment",
    "configure_deterministic_runtime",
    "configure_logging",
    "compute_checksum",
    "create_artifact_manager",
    "create_resilient_session",
    "parse_env_file",
    "find_resources",
    "task_queue",
    "transfer_with_resume",
    "verify_checksum",
]
