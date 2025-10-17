from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("TRADEPULSE_OAUTH2_ISSUER", "https://issuer.tradepulse.test")
os.environ.setdefault("TRADEPULSE_OAUTH2_AUDIENCE", "tradepulse-api")
os.environ.setdefault(
    "TRADEPULSE_OAUTH2_JWKS_URI", "https://issuer.tradepulse.test/jwks"
)
os.environ.setdefault("TRADEPULSE_AUDIT_SECRET", "import-audit-secret")

from application.api.service import (
    create_app,
)  # noqa: E402  - env vars must be set before import
from application.settings import AdminApiSettings

BASELINE = Path("interfaces/http/openapi/0.2.0.json")


@pytest.fixture()
def fastapi_app():
    settings = AdminApiSettings(
        audit_secret="import-audit-secret",
    )
    return create_app(settings=settings)


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
    assert {
        "symbol",
        "features",
        "items",
        "pagination",
        "filters",
    } <= set(feature_response["properties"].keys())
    assert {
        "symbol",
        "signal",
        "items",
        "pagination",
        "filters",
    } <= set(prediction_response["properties"].keys())
