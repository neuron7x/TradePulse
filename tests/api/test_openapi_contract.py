from __future__ import annotations

import pytest


@pytest.mark.contract
def test_openapi_exposes_core_endpoints(api_app):
    schema = api_app.openapi()
    paths = schema.get("paths", {})
    for endpoint in ("/health", "/features", "/predictions"):
        assert endpoint in paths


@pytest.mark.contract
def test_openapi_schemas_are_consistent(api_app):
    schema = api_app.openapi()
    components = schema.get("components", {}).get("schemas", {})
    assert "FeatureResponse" in components
    assert "PredictionResponse" in components
    feature_props = components["FeatureResponse"]["properties"]
    prediction_props = components["PredictionResponse"]["properties"]
    assert {"symbol", "features", "generated_at"} <= set(feature_props)
    assert {"symbol", "signal", "horizon_seconds", "score"} <= set(prediction_props)
