from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.experiments import feature_store as feature_store_module
from core.experiments import tracking as tracking_module
from core.experiments.feature_store import FeastFeatureStoreClient
from core.experiments.tracking import MLflowTracker, WeightsAndBiasesTracker


def test_mlflow_tracker_logs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    class FakeRun:
        def __init__(self) -> None:
            self.metrics: dict[str, float] = {}

    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda uri: calls.append(f"uri:{uri}"),
        set_registry_uri=lambda uri: calls.append(f"registry:{uri}"),
        set_experiment=lambda name: calls.append(f"experiment:{name}"),
        start_run=lambda run_name=None, tags=None: FakeRun(),
        log_metrics=lambda metrics, step=None: calls.append(f"metrics:{sorted(metrics.items())}:{step}"),
        log_params=lambda params: calls.append(f"params:{sorted(params.items())}"),
        log_artifact=lambda path: calls.append(f"artifact:{path}"),
        end_run=lambda: calls.append("end"),
    )

    original_find_spec = tracking_module.importlib.util.find_spec
    original_import_module = tracking_module.importlib.import_module
    monkeypatch.setattr(
        tracking_module.importlib.util,
        "find_spec",
        lambda name: object() if name == "mlflow" else original_find_spec(name),
    )
    monkeypatch.setattr(
        tracking_module.importlib,
        "import_module",
        lambda name: fake_mlflow if name == "mlflow" else original_import_module(name),
    )

    tracker = MLflowTracker(tracking_uri="file://mlruns", registry_uri="sqlite:///registry.db", experiment="test")
    tracker.start_run("example", team="core")
    tracker.log_metrics({"loss": 0.1}, step=1)
    tracker.log_params({"lr": 1e-3})
    tracker.log_artifact(tmp_path / "model.pkl")
    tracker.end_run()

    assert "metrics:[('loss', 0.1)]:1" in calls
    assert "params:[('lr', 0.001)]" in calls


def test_wandb_tracker(monkeypatch: pytest.MonkeyPatch) -> None:
    logged: list[tuple] = []

    class FakeRun:
        def __init__(self) -> None:
            self.config = SimpleNamespace(update=lambda data, allow_val_change=True: logged.append(("config", dict(data))))

        def log(self, metrics, step=None):
            logged.append(("log", dict(metrics), step))

        def log_artifact(self, artifact):
            logged.append(("artifact", artifact.name))

        def finish(self):
            logged.append(("finish",))

    class FakeArtifact:
        def __init__(self, name, type):
            self.name = name

        def add_file(self, path):
            logged.append(("add_file", path))

    fake_wandb = SimpleNamespace(init=lambda **kwargs: FakeRun(), Artifact=FakeArtifact)

    original_find_spec = tracking_module.importlib.util.find_spec
    original_import_module = tracking_module.importlib.import_module
    monkeypatch.setattr(
        tracking_module.importlib.util,
        "find_spec",
        lambda name: object() if name == "wandb" else original_find_spec(name),
    )
    monkeypatch.setattr(
        tracking_module.importlib,
        "import_module",
        lambda name: fake_wandb if name == "wandb" else original_import_module(name),
    )

    tracker = WeightsAndBiasesTracker(project="tradepulse", config={"lr": 0.01})
    tracker.start_run("run", stage="dev")
    tracker.log_metrics({"accuracy": 0.99}, step=10)
    tracker.log_params({"epochs": 5})
    tracker.log_artifact("/tmp/model.bin")
    tracker.end_run()

    assert any(entry[0] == "log" for entry in logged)
    assert any(entry[0] == "finish" for entry in logged)


def test_feast_feature_store(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeFeatureStore:
        def __init__(self, repo_path, project=None):
            self.repo_path = repo_path
            self.project = project
            self.applied = []

        def apply(self, objects):
            self.applied.extend(list(objects))

        def materialize(self, start, end):
            self.materialized = (start, end)

        def materialize_incremental(self, end):
            self.incremental = end

        def get_online_features(self, features, entity_rows):
            return SimpleNamespace(to_dict=lambda: {"features": features, "entities": entity_rows})

    fake_feast_module = SimpleNamespace(FeatureStore=FakeFeatureStore)

    original_find_spec = feature_store_module.importlib.util.find_spec
    original_import_module = feature_store_module.importlib.import_module
    monkeypatch.setattr(
        feature_store_module.importlib.util,
        "find_spec",
        lambda name: object() if name == "feast" else original_find_spec(name),
    )
    monkeypatch.setattr(
        feature_store_module.importlib,
        "import_module",
        lambda name: fake_feast_module if name == "feast" else original_import_module(name),
    )

    client = FeastFeatureStoreClient(repo_path="/repo")
    client.apply(["entity"])
    features = client.get_online_features(["feature:a"], [{"entity_id": 1}])
    assert "features" in features
