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


def test_dashboard_renders_core_metrics(dashboard_app: AppTest) -> None:
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "dashboard_sample.csv"
    data = fixture.read_bytes()

    app = dashboard_app.run()
    app = app.file_uploader[0].upload(
        data=data,
        file_name="dashboard_sample.csv",
        mime_type="text/csv",
    ).run()

    labels = {metric.label for metric in app.metric}
    assert {"Kuramoto R", "Entropy H", "ΔH"}.issubset(labels)
    assert not app.error
