"""Experiment tracking integrations for TradePulse."""

from __future__ import annotations

import importlib
import importlib.util
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Mapping


class ExperimentTracker:
    """Abstract base class for experiment trackers."""

    def start_run(self, name: str | None = None, **metadata: Any) -> None:
        raise NotImplementedError

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        raise NotImplementedError

    def log_params(self, params: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def log_artifact(self, path: str | Path) -> None:
        raise NotImplementedError

    def end_run(self) -> None:
        raise NotImplementedError


class MLflowTracker(ExperimentTracker):
    """Wrapper around MLflow's tracking API with lazy import semantics."""

    def __init__(
        self,
        *,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
        experiment: str | None = None,
    ) -> None:
        spec = importlib.util.find_spec("mlflow")
        if spec is None:
            raise RuntimeError("mlflow must be installed to use MLflowTracker")
        self._mlflow = importlib.import_module("mlflow")
        if tracking_uri:
            self._mlflow.set_tracking_uri(tracking_uri)
        if registry_uri:
            self._mlflow.set_registry_uri(registry_uri)
        if experiment:
            self._mlflow.set_experiment(experiment)
        self._active = None

    def start_run(self, name: str | None = None, **metadata: Any) -> None:
        self._active = self._mlflow.start_run(run_name=name, tags=metadata or None)

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._mlflow.log_params(params)

    def log_artifact(self, path: str | Path) -> None:
        self._mlflow.log_artifact(str(path))

    def end_run(self) -> None:
        self._mlflow.end_run()
        self._active = None

    @contextmanager
    def run(self, name: str | None = None, **metadata: Any):
        self.start_run(name, **metadata)
        try:
            yield self
        finally:
            self.end_run()


class WeightsAndBiasesTracker(ExperimentTracker):
    """Integration with Weights & Biases (wandb)."""

    def __init__(self, *, project: str, entity: str | None = None, config: Mapping[str, Any] | None = None) -> None:
        spec = importlib.util.find_spec("wandb")
        if spec is None:
            raise RuntimeError("wandb must be installed to use WeightsAndBiasesTracker")
        self._wandb = importlib.import_module("wandb")
        self._project = project
        self._entity = entity
        self._config = dict(config or {})
        self._run = None

    def start_run(self, name: str | None = None, **metadata: Any) -> None:
        kwargs: Dict[str, Any] = {"project": self._project, "config": self._config}
        if self._entity:
            kwargs["entity"] = self._entity
        if name:
            kwargs["name"] = name
        if metadata:
            kwargs.setdefault("tags", []).extend(f"{key}:{value}" for key, value in metadata.items())
        self._run = self._wandb.init(**kwargs)

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        if self._run is None:
            raise RuntimeError("wandb run has not been started")
        self._run.log(dict(metrics), step=step)

    def log_params(self, params: Mapping[str, Any]) -> None:
        if self._run is None:
            raise RuntimeError("wandb run has not been started")
        self._run.config.update(dict(params), allow_val_change=True)

    def log_artifact(self, path: str | Path) -> None:
        if self._run is None:
            raise RuntimeError("wandb run has not been started")
        artifact = self._wandb.Artifact(Path(path).stem, type="dataset")
        artifact.add_file(str(path))
        self._run.log_artifact(artifact)

    def end_run(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None

    @contextmanager
    def run(self, name: str | None = None, **metadata: Any):
        self.start_run(name, **metadata)
        try:
            yield self
        finally:
            self.end_run()
