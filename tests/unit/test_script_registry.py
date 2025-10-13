from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.runtime import (
    ConfirmationRequiredError,
    MissingDependencyError,
    SchemaValidationError,
    ScriptRegistry,
    ScriptRegistryError,
    ScriptRunner,
)


def test_registry_loads_builtin_file() -> None:
    registry = ScriptRegistry.from_path()
    spec = registry.get("data_sanity")
    assert spec.name == "data_sanity"
    assert "pandas" in spec.python_dependencies


def test_unknown_script_raises() -> None:
    registry = ScriptRegistry.from_path()
    with pytest.raises(ScriptRegistryError):
        registry.get("does-not-exist")


def test_missing_dependency_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ScriptRegistry.from_path()
    spec = registry.get("data_sanity")

    monkeypatch.setattr(spec, "python_dependencies", ["surely_missing_package"], raising=False)
    with pytest.raises(MissingDependencyError):
        spec.ensure_dependencies(auto_install=False)


def test_markdown_index_contains_expected_rows() -> None:
    registry = ScriptRegistry.from_path()
    markdown = registry.generate_markdown_index()
    assert "data_sanity" in markdown
    assert "gen_synth_amm_data" in markdown
    assert "TradePulse Script Index" in markdown


def test_schema_validation_json(tmp_path: Path) -> None:
    registry = ScriptRegistry.from_path()
    spec = registry.get("gen_synth_amm_data")
    schema = spec.output_artifacts[0].schema
    assert schema is not None

    payload = {"columns": ["x", "R", "kappa", "H"], "row_count": 100, "path": "/tmp/out.csv"}
    schema.validate(payload)

    with pytest.raises(SchemaValidationError):
        schema.validate({"columns": ["x"], "row_count": "bad"})


def test_schema_validation_pandera() -> None:
    registry = ScriptRegistry.from_path()
    spec = registry.get("data_sanity")
    schema = spec.input_schemas[0]

    df = pd.DataFrame({"ts": ["2023-01-01"], "price": [1.0], "volume": [2.0]})
    schema.validate(df)


def test_runner_dry_run_outputs(tmp_path: Path) -> None:
    registry = ScriptRegistry.from_path()
    runner = ScriptRunner(registry)

    metrics = tmp_path / "metrics.prom"
    manifest = tmp_path / "manifest.json"

    result = runner.run(
        "data_sanity",
        dry_run=True,
        metrics_path=metrics,
        manifest_path=manifest,
    )

    assert result.success is True
    assert metrics.exists()
    assert manifest.exists()


def test_runner_requires_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ScriptRegistry.from_path()
    runner = ScriptRunner(registry)

    with pytest.raises(ConfirmationRequiredError):
        runner.run("resilient_data_sync", dry_run=True)

    monkeypatch.setenv("TRADEPULSE_ALLOW_LIVE", "1")
    runner.run("resilient_data_sync", dry_run=True)
