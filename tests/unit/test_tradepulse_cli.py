from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from cli.tradepulse_cli import cli
from core.config.cli_models import IngestConfig, VersioningConfig
from core.config.template_manager import ConfigTemplateManager
from core.data.feature_catalog import FeatureCatalog
from core.data.versioning import DataVersionManager


@pytest.fixture()
def sample_prices(tmp_path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "price": [100.0, 101.0, 99.0, 102.0, 103.0],
        }
    )
    path = tmp_path / "prices.csv"
    frame.to_csv(path, index=False)
    return path


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_cli_generates_templates(tmp_path: Path) -> None:
    runner = CliRunner()
    for command in (
        "ingest",
        "materialize",
        "backtest",
        "train",
        "optimize",
        "exec",
        "serve",
        "report",
    ):
        destination = tmp_path / f"{command}.yaml"
        result = runner.invoke(cli, [command, "--generate-config", "--template-output", str(destination)])
        assert result.exit_code == 0, result.output
        assert destination.exists()


def test_full_cli_flow(tmp_path: Path, sample_prices: Path) -> None:
    manager = ConfigTemplateManager(Path("configs/templates"))

    catalog_path = tmp_path / "catalog.json"
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    runner = CliRunner()

    # Ingest
    ingest_cfg_path = tmp_path / "ingest.yaml"
    manager.render("ingest", ingest_cfg_path)
    ingest_cfg = _load_yaml(ingest_cfg_path)
    ingest_cfg["source"]["path"] = str(sample_prices)
    ingest_destination = tmp_path / "ingested.csv"
    ingest_cfg["destination"] = str(ingest_destination)
    ingest_cfg["catalog"] = {"path": str(catalog_path)}
    ingest_cfg["versioning"] = {"backend": "dvc", "repo_path": str(repo_path)}
    _write_yaml(ingest_cfg_path, ingest_cfg)

    result = runner.invoke(cli, ["ingest", "--config", str(ingest_cfg_path)])
    assert result.exit_code == 0, result.output
    assert ingest_destination.exists()
    assert (ingest_destination.with_suffix(".csv.version.json")).exists()

    # Materialize
    materialize_cfg_path = tmp_path / "materialize.yaml"
    manager.render("materialize", materialize_cfg_path)
    materialize_cfg = _load_yaml(materialize_cfg_path)
    materialize_payload = pd.DataFrame(
        {
            "entity_id": ["asset-1"] * 6,
            "ts": pd.date_range("2024-01-01", periods=6, freq="h"),
            "feature": np.linspace(0.0, 1.0, 6),
        }
    )
    materialize_source = tmp_path / "materialize.csv"
    materialize_payload.to_csv(materialize_source, index=False)
    materialize_cfg["source"]["path"] = str(materialize_source)
    store_root = tmp_path / "online_store"
    materialize_cfg["store_root"] = str(store_root)
    materialize_cfg["checkpoint_path"] = str(tmp_path / "materialize_checkpoints.json")
    materialize_cfg["feature_view"] = "features.demo"
    materialize_cfg["catalog"] = {"path": str(catalog_path)}
    materialize_cfg["versioning"] = {"backend": "dvc", "repo_path": str(repo_path)}
    _write_yaml(materialize_cfg_path, materialize_cfg)

    result = runner.invoke(cli, ["materialize", "--config", str(materialize_cfg_path)])
    assert result.exit_code == 0, result.output
    safe_name = materialize_cfg["feature_view"].replace("/", "__").replace(".", "__")
    candidates = list(Path(materialize_cfg["store_root"]).glob(f"{safe_name}.*"))
    assert candidates, "materialize command did not create a persisted artifact"
    materialized_path = candidates[0]
    assert materialized_path.exists()

    # Backtest
    backtest_cfg_path = tmp_path / "backtest.yaml"
    manager.render("backtest", backtest_cfg_path)
    backtest_cfg = _load_yaml(backtest_cfg_path)
    backtest_cfg["data"]["path"] = str(sample_prices)
    backtest_results = tmp_path / "backtest.json"
    backtest_cfg["results_path"] = str(backtest_results)
    backtest_cfg["catalog"] = {"path": str(catalog_path)}
    backtest_cfg["versioning"] = {"backend": "dvc", "repo_path": str(repo_path)}
    _write_yaml(backtest_cfg_path, backtest_cfg)

    result = runner.invoke(cli, ["backtest", "--config", str(backtest_cfg_path)])
    assert result.exit_code == 0, result.output
    backtest_payload = json.loads(backtest_results.read_text())
    assert backtest_payload["stats"]["trades"] >= 0

    # Exec
    exec_cfg_path = tmp_path / "exec.yaml"
    manager.render("exec", exec_cfg_path)
    exec_cfg = _load_yaml(exec_cfg_path)
    exec_cfg["data"]["path"] = str(sample_prices)
    exec_results = tmp_path / "exec.json"
    exec_cfg["results_path"] = str(exec_results)
    exec_cfg["catalog"] = {"path": str(catalog_path)}
    exec_cfg["versioning"] = {"backend": "dvc", "repo_path": str(repo_path)}
    _write_yaml(exec_cfg_path, exec_cfg)

    result = runner.invoke(cli, ["exec", "--config", str(exec_cfg_path)])
    assert result.exit_code == 0, result.output
    exec_payload = json.loads(exec_results.read_text())
    assert "latest_signal" in exec_payload

    # Optimize
    optimize_cfg_path = tmp_path / "optimize.yaml"
    manager.render("optimize", optimize_cfg_path)
    optimize_cfg = _load_yaml(optimize_cfg_path)
    optimize_cfg["metadata"]["backtest"]["data"]["path"] = str(sample_prices)
    optimize_cfg["metadata"]["backtest"]["results_path"] = str(tmp_path / "opt_backtest.json")
    optimize_cfg["results_path"] = str(tmp_path / "optimize.json")
    optimize_cfg["versioning"] = {"backend": "dvc", "repo_path": str(repo_path)}
    _write_yaml(optimize_cfg_path, optimize_cfg)

    result = runner.invoke(cli, ["optimize", "--config", str(optimize_cfg_path)])
    assert result.exit_code == 0, result.output
    optimize_payload = json.loads((tmp_path / "optimize.json").read_text())
    assert optimize_payload["best_params"] is not None
    assert optimize_payload["trials"]

    # Train
    train_cfg_path = tmp_path / "train.yaml"
    manager.render("train", train_cfg_path)
    train_cfg = _load_yaml(train_cfg_path)
    train_source = tmp_path / "train.csv"
    price_frame = pd.read_csv(sample_prices)
    train_frame = pd.DataFrame(
        {
            "signal": price_frame["price"].pct_change().fillna(0.0),
            "reward": price_frame["price"].diff().fillna(0.0),
            "kappa": np.linspace(0.0, 1.0, len(price_frame)),
        }
    )
    train_frame.to_csv(train_source, index=False)
    train_cfg["data"]["path"] = str(train_source)
    train_results = tmp_path / "train.json"
    train_cfg["results_path"] = str(train_results)
    train_cfg["catalog"] = {"path": str(catalog_path)}
    train_cfg["versioning"] = {"backend": "dvc", "repo_path": str(repo_path)}
    _write_yaml(train_cfg_path, train_cfg)

    result = runner.invoke(cli, ["train", "--config", str(train_cfg_path)])
    assert result.exit_code == 0, result.output
    train_payload = json.loads(train_results.read_text())
    assert train_payload["best_params"]
    assert train_payload["records"] == len(train_frame)

    # Serve
    serve_cfg_path = tmp_path / "serve.yaml"
    manager.render("serve", serve_cfg_path)
    serve_cfg = _load_yaml(serve_cfg_path)
    serve_cfg["data"]["path"] = str(sample_prices)
    serve_results = tmp_path / "serve.json"
    serve_cfg["results_path"] = str(serve_results)
    serve_cfg["catalog"] = {"path": str(catalog_path)}
    serve_cfg["versioning"] = {"backend": "dvc", "repo_path": str(repo_path)}
    _write_yaml(serve_cfg_path, serve_cfg)

    result = runner.invoke(cli, ["serve", "--config", str(serve_cfg_path)])
    assert result.exit_code == 0, result.output
    serve_payload = json.loads(serve_results.read_text())
    assert "latest_signal" in serve_payload

    # Report
    report_cfg_path = tmp_path / "report.yaml"
    manager.render("report", report_cfg_path)
    report_cfg = _load_yaml(report_cfg_path)
    report_output = tmp_path / "report.md"
    report_cfg["inputs"] = [
        str(backtest_results),
        str(exec_results),
        str(train_results),
        str(serve_results),
    ]
    report_cfg["output_path"] = str(report_output)
    report_cfg["versioning"] = {"backend": "dvc", "repo_path": str(repo_path)}
    _write_yaml(report_cfg_path, report_cfg)

    result = runner.invoke(cli, ["report", "--config", str(report_cfg_path)])
    assert result.exit_code == 0, result.output
    text = report_output.read_text()
    lower_text = text.lower()
    assert "backtest" in lower_text
    assert "exec" in lower_text

    catalog = json.loads(catalog_path.read_text())
    assert len(catalog["artifacts"]) >= 6


def test_backtest_outputs_jsonl(tmp_path: Path, sample_prices: Path) -> None:
    manager = ConfigTemplateManager(Path("configs/templates"))
    backtest_cfg_path = tmp_path / "backtest.yaml"
    manager.render("backtest", backtest_cfg_path)
    cfg = _load_yaml(backtest_cfg_path)
    cfg["data"]["path"] = str(sample_prices)
    cfg["results_path"] = str(tmp_path / "results.json")
    _write_yaml(backtest_cfg_path, cfg)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "backtest",
            "--config",
            str(backtest_cfg_path),
            "--output",
            "jsonl",
        ],
    )
    assert result.exit_code == 0, result.output
    lines = [line for line in result.output.splitlines() if line.startswith("{")]
    assert any("\"metric\": \"total_return\"" in line for line in lines)


def test_versioning_manager_writes_metadata(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("hello", encoding="utf-8")

    cfg = VersioningConfig(backend="dvc", repo_path=tmp_path)
    manager = DataVersionManager(cfg)
    result = manager.snapshot(artifact, metadata={"size": 5})
    metadata_path = artifact.with_suffix(".txt.version.json")
    assert metadata_path.exists()
    assert result["backend"] == "dvc"
    assert result["metadata"]["size"] == 5


def test_feature_catalog_register(tmp_path: Path, sample_prices: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog = FeatureCatalog(catalog_path)
    config = IngestConfig(
        name="test",
        source={"kind": "csv", "path": sample_prices},
        destination=tmp_path / "dest.csv",
    )
    config.destination.write_text("data", encoding="utf-8")
    entry = catalog.register("artifact", config.destination, config=config, lineage=["input"], metadata={"owner": "qa"})
    assert entry.name == "artifact"
    stored = catalog.find("artifact")
    assert stored is not None and stored.checksum == entry.checksum
