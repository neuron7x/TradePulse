# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from core.utils.security import SecretDetector, check_for_hardcoded_secrets


def test_secret_detector_identifies_api_keys(tmp_path_factory: pytest.TempPathFactory) -> None:
    workspace = tmp_path_factory.mktemp("security")
    target = workspace / "config.py"
    target.write_text("API_KEY = 'abcdef1234567890'\n", encoding="utf-8")

    detector = SecretDetector()
    findings = detector.scan_file(target)

    assert findings
    secret_type, line_num, masked = findings[0]
    assert secret_type == "api_key"
    assert line_num == 1
    assert "********" in masked


def test_secret_detector_respects_ignore_patterns(tmp_path_factory: pytest.TempPathFactory) -> None:
    workspace = tmp_path_factory.mktemp("security-ignore")
    node_modules = workspace / "node_modules"
    node_modules.mkdir()
    ignored = node_modules / "config.js"
    ignored.write_text("const password = 'hunter2';\n", encoding="utf-8")

    detector = SecretDetector()
    assert detector.scan_file(ignored) == []


def test_scan_directory_filters_extensions(tmp_path_factory: pytest.TempPathFactory) -> None:
    workspace = tmp_path_factory.mktemp("security-scan")
    allowed = workspace / "settings.py"
    allowed.write_text("password='supersecret'\n", encoding="utf-8")
    skipped = workspace / "README.md"
    skipped.write_text("api_secret='value'\n", encoding="utf-8")

    detector = SecretDetector()
    results = detector.scan_directory(workspace, extensions=[".py"])

    assert "settings.py" in results
    assert "README.md" not in results


def test_check_for_hardcoded_secrets_reports_findings(tmp_path_factory: pytest.TempPathFactory, capsys: pytest.CaptureFixture[str]) -> None:
    project = tmp_path_factory.mktemp("security-project")
    secret_file = project / "secrets.py"
    secret_file.write_text("github_token = 'ghp_" + "a" * 36 + "'\n", encoding="utf-8")

    found = check_for_hardcoded_secrets(str(project))
    captured = capsys.readouterr()

    assert found is True
    assert "Potential secrets detected" in captured.out
    assert "********" in captured.out


def test_check_for_hardcoded_secrets_returns_false_when_clean(tmp_path_factory: pytest.TempPathFactory, capsys: pytest.CaptureFixture[str]) -> None:
    clean_project = tmp_path_factory.mktemp("security-clean")

    found = check_for_hardcoded_secrets(str(clean_project))
    captured = capsys.readouterr()

    assert found is False
    assert "No hardcoded secrets detected" in captured.out
