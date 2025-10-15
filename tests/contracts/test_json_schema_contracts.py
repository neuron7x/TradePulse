from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("TRADEPULSE_ADMIN_TOKEN", "import-admin-token")
os.environ.setdefault("TRADEPULSE_ADMIN_TOKEN_ID", "TRADEPULSE_ADMIN_TOKEN")
os.environ.setdefault("TRADEPULSE_AUDIT_SECRET", "import-audit-secret")
os.environ.setdefault("TRADEPULSE_AUDIT_SECRET_ID", "TRADEPULSE_AUDIT_SECRET")

from application.api.service import (  # noqa: E402  - environment variables must be set before import
    FeatureRequest,
    FeatureResponse,
    PredictionRequest,
    PredictionResponse,
)


BASELINE_DIR = Path("schemas/http/json/1.0.0")


@pytest.mark.parametrize(
    ("model", "filename"),
    [
        (FeatureRequest, "feature_request.schema.json"),
        (FeatureResponse, "feature_response.schema.json"),
        (PredictionRequest, "prediction_request.schema.json"),
        (PredictionResponse, "prediction_response.schema.json"),
    ],
)
def test_dto_json_schema_matches_baseline(model: type, filename: str) -> None:
    baseline = json.loads((BASELINE_DIR / filename).read_text(encoding="utf-8"))
    current = model.model_json_schema()
    assert current == baseline


@pytest.mark.parametrize(
    ("model", "filename"),
    [
        (FeatureRequest, "feature_request.schema.json"),
        (PredictionRequest, "prediction_request.schema.json"),
    ],
)
def test_dto_required_fields_are_stable(model: type, filename: str) -> None:
    baseline = json.loads((BASELINE_DIR / filename).read_text(encoding="utf-8"))
    current = model.model_json_schema()
    assert set(baseline.get("required", ())) <= set(current.get("required", ()))
