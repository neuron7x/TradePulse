from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("TRADEPULSE_ADMIN_TOKEN", "import-admin-token")
os.environ.setdefault("TRADEPULSE_AUDIT_SECRET", "import-audit-secret")

from application.api.service import create_app  # noqa: E402  - env vars must be set before import


BASELINE = Path("interfaces/http/openapi/1.0.0.json")


@pytest.fixture()
def fastapi_app():
    return create_app(admin_token="import-admin-token", audit_secret="import-audit-secret")


def test_openapi_contract_matches_baseline(fastapi_app) -> None:
    generated = fastapi_app.openapi()
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert generated == baseline


def test_openapi_defines_expected_routes(fastapi_app) -> None:
    schema = fastapi_app.openapi()
    paths = schema.get("paths", {})
    assert "/features" in paths
    assert "post" in paths["/features"]
    assert "/predictions" in paths
    assert "post" in paths["/predictions"]
    feature_response = schema["components"]["schemas"]["FeatureResponse"]
    prediction_response = schema["components"]["schemas"]["PredictionResponse"]
    assert {"symbol", "features", "generated_at"} <= set(feature_response["properties"].keys())
    assert {"symbol", "signal", "horizon_seconds"} <= set(prediction_response["properties"].keys())
