from __future__ import annotations

import json
from pathlib import Path

import pytest

from reports import ci_html_reports


def _read_html(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    return text


@pytest.mark.parametrize("target", ["backtest", "train", "publish"])
def test_generate_reports_creates_html(tmp_path: Path, target: str) -> None:
    artifacts = ci_html_reports.generate_reports(target, tmp_path)
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.path.exists()
    html = _read_html(artifact.path)
    # JSON metrics payload should be embedded in the summary block for traceability
    for key in artifact.metrics:
        assert key in html
    parsed_metrics = json.loads(artifact.path.read_text(encoding="utf-8").split("<p class=\"summary\">")[1].split("</p>")[0])
    assert set(parsed_metrics) == set(artifact.metrics)


def test_generate_reports_all_targets(tmp_path: Path) -> None:
    artifacts = ci_html_reports.generate_reports("all", tmp_path)
    names = {artifact.name for artifact in artifacts}
    assert {"Backtest Job Report", "Training Job Report", "Publish Job Report"} <= names
    for artifact in artifacts:
        _read_html(artifact.path)
