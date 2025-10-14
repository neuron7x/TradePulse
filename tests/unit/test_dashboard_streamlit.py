"""Smoke tests for the Streamlit dashboard UI."""

from __future__ import annotations

from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest


@pytest.fixture()
def dashboard_app() -> AppTest:
    app_path = Path(__file__).resolve().parents[2] / "interfaces" / "dashboard_streamlit.py"
    return AppTest.from_file(str(app_path))


def test_dashboard_renders_core_metrics(
    dashboard_app: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "dashboard_sample.csv"
    monkeypatch.setenv("TRADEPULSE_DASHBOARD_TEST_UPLOAD", str(fixture))

    app = dashboard_app.run(timeout=10)

    labels = {metric.label for metric in app.metric}
    assert {"Kuramoto R", "Entropy H", "ΔH"}.issubset(labels)
    assert not app.error
