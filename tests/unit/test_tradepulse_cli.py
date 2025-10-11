from __future__ import annotations

import json
from pathlib import Path

import yaml
from click.testing import CliRunner

from cli.tradepulse_cli import cli
from core.config.cli_models import IngestConfig, VersioningConfig
from core.config.template_manager import ConfigTemplateManager
from core.data.feature_catalog import FeatureCatalog
from core.data.versioning import DataVersionManager


def test_generate_ingest_template(tmp_path: Path) -> None:
    runner = CliRunner()
    destination = tmp_path / "ingest.yaml"
    result = runner.invoke(cli, ["ingest", "--generate-config", "--output", str(destination)])
    assert result.exit_code == 0, result.output
    assert destination.exists()
    data = yaml.safe_load(destination.read_text())
    assert data["name"] == "sample_ingest_job"


def test_ingest_flow_registers_catalog(tmp_path: Path) -> None:
    manager = ConfigTemplateManager(Path("configs/templates"))
    config_path = tmp_path / "ingest.yaml"
    manager.render("ingest", config_path)

    config_data = yaml.safe_load(config_path.read_text())
    config_data["source"]["path"] = str(Path("data/sample.csv").resolve())
    destination = tmp_path / "out.csv"
    config_data["destination"] = str(destination)
    catalog_path = tmp_path / "catalog.json"
    config_data["catalog"] = {"path": str(catalog_path)}
    config_data["versioning"] = {"backend": "none"}
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["ingest", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    assert destination.exists()

    catalog = json.loads(catalog_path.read_text())
    entries = catalog.get("artifacts", [])
    assert entries and entries[0]["name"] == config_data["name"]


def test_versioning_manager_writes_metadata(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("hello", encoding="utf-8")

    cfg = VersioningConfig(backend="dvc", repo_path=tmp_path)
    manager = DataVersionManager(cfg)
    result = manager.snapshot(artifact)
    metadata_path = artifact.with_suffix(".txt.version.json")
    assert metadata_path.exists()
    assert result["backend"] == "dvc"


def test_feature_catalog_register(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog = FeatureCatalog(catalog_path)
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")

    config = IngestConfig.model_validate(
        {
            "name": "test",
            "source": {"kind": "csv", "path": str(artifact)},
            "destination": str(artifact),
            "versioning": {"backend": "none"},
            "catalog": {"path": str(catalog_path)},
        }
    )
    entry = catalog.register("artifact", artifact, config=config, lineage=["input"], metadata={"owner": "qa"})
    assert entry.name == "artifact"
    stored = catalog.find("artifact")
    assert stored is not None and stored.checksum == entry.checksum
