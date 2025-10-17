"""Utilities for integrating with MLflow while tolerating optional installs."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

try:  # pragma: no cover - exercised indirectly when MLflow is available.
    import mlflow as _mlflow  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - stub path covered in tests.
    _mlflow = None


class _MlflowStub:
    """Minimal MLflow facade that degrades gracefully when the package is absent."""

    def active_run(self) -> bool:
        return False

    @contextmanager
    def start_run(self, *_args: Any, **_kwargs: Any):
        yield None

    def set_tracking_uri(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def set_experiment(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def log_params(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def log_dict(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def log_metrics(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def log_artifact(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def log_text(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def set_tags(self, *_args: Any, **_kwargs: Any) -> None:
        return None


mlflow = _mlflow if _mlflow is not None else _MlflowStub()


def mlflow_available() -> bool:
    """Return whether the real MLflow client is available in the runtime."""

    return _mlflow is not None

