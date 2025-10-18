import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("TRADEPULSE_AUDIT_SECRET", "contract-test-secret")
os.environ.setdefault("TRADEPULSE_OAUTH2_ISSUER", "https://openapi.test")
os.environ.setdefault("TRADEPULSE_OAUTH2_AUDIENCE", "tradepulse-api")
os.environ.setdefault("TRADEPULSE_OAUTH2_JWKS_URI", "https://openapi.test/jwks")

from application.api.service import create_app
from tests.api.test_service import security_context  # noqa: F401


@pytest.mark.usefixtures("security_context")
def test_openapi_contract_is_stable() -> None:
    app = create_app()
    runtime_schema = app.openapi()
    spec_path = Path(__file__).resolve().parents[2] / "schemas" / "openapi" / (
        "tradepulse-online-inference-v1.json"
    )
    expected_schema = json.loads(spec_path.read_text(encoding="utf-8"))
    assert runtime_schema == expected_schema
