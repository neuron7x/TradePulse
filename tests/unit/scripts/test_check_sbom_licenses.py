from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import check_sbom_licenses


def _write_sbom(tmp_path: Path, components: list[dict[str, object]]) -> Path:
    payload = {"bomFormat": "CycloneDX", "components": components}
    path = tmp_path / "sbom.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_license_validation_allows_permitted_components(tmp_path: Path) -> None:
    sbom = _write_sbom(
        tmp_path,
        [
            {"name": "numpy", "licenses": [{"license": {"id": "BSD-3-Clause"}}]},
            {"name": "pydantic", "licenses": [{"license": {"id": "MIT"}}]},
        ],
    )
    components = check_sbom_licenses.load_components(sbom)
    assert len(components) == 2
    assert check_sbom_licenses.extract_license(components[0]) == {"BSD-3-Clause"}
    exit_code, missing, violations = check_sbom_licenses.validate_sbom(sbom)
    assert exit_code == 0
    assert missing == []
    assert violations == []


def test_license_validation_rejects_disallowed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sbom = _write_sbom(
        tmp_path,
        [
            {"name": "bad-lib", "licenses": [{"license": {"id": "AGPL-3.0"}}]},
        ],
    )
    exit_code = check_sbom_licenses.main([str(sbom)])
    assert exit_code == 1
    output = capsys.readouterr().out
    assert "AGPL-3.0" in output
