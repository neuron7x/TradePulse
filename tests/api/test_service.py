from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Callable

import jwt
from jwt.algorithms import RSAAlgorithm
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ.setdefault("TRADEPULSE_AUDIT_SECRET", "test-audit-secret")
os.environ.setdefault("TRADEPULSE_OAUTH2_ISSUER", "https://issuer.tradepulse.test")
os.environ.setdefault("TRADEPULSE_OAUTH2_AUDIENCE", "tradepulse-api")
os.environ.setdefault("TRADEPULSE_OAUTH2_JWKS_URI", "https://issuer.tradepulse.test/jwks")

from application.api import security as security_module
from application.api.rate_limit import InMemorySlidingWindowBackend, SlidingWindowRateLimiter
from application.api.service import DependencyProbeResult, create_app
from application.settings import (
    AdminApiSettings,
    ApiRateLimitSettings,
    ApiSecuritySettings,
    RateLimitPolicy,
)


@pytest.fixture()
def security_context(monkeypatch: pytest.MonkeyPatch) -> Callable[..., str]:
    if hasattr(security_module.get_api_security_settings, "_instance"):
        delattr(security_module.get_api_security_settings, "_instance")

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    jwk_dict = RSAAlgorithm.to_jwk(public_key, as_dict=True)
    kid = "unit-test-key"
    jwk_dict.update({"kid": kid, "alg": "RS256", "use": "sig"})

    settings = ApiSecuritySettings(
        oauth2_issuer="https://issuer.tradepulse.test",
        oauth2_audience="tradepulse-api",
        oauth2_jwks_uri="https://issuer.tradepulse.test/jwks",
        trusted_hosts=["testserver", "localhost"],
    )

    monkeypatch.setattr(security_module, "_default_settings_loader", lambda: settings)

    async def fake_get_key(uri: str, request_kid: str) -> dict[str, str] | None:
        assert uri == str(settings.oauth2_jwks_uri)
        if request_kid == kid:
            return jwk_dict
        return None

    monkeypatch.setattr(security_module._jwks_resolver, "get_key", fake_get_key)

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    def mint_token(
        *,
        subject: str = "unit-user",
        audience: str | None = None,
        issuer: str | None = None,
        lifetime: timedelta = timedelta(minutes=5),
    ) -> str:
        now = datetime.now(timezone.utc)
        payload = {
            "iss": issuer or str(settings.oauth2_issuer),
            "aud": audience or settings.oauth2_audience,
            "sub": subject,
            "iat": int(now.timestamp()),
            "exp": int((now + lifetime).timestamp()),
        }
        return jwt.encode(payload, private_pem, algorithm="RS256", headers={"kid": kid})

    return mint_token


@pytest.fixture()
def configured_app(
    monkeypatch: pytest.MonkeyPatch,
    security_context: Callable[..., str],
) -> FastAPI:
    monkeypatch.delenv("TRADEPULSE_AUDIT_SECRET", raising=False)
    settings = AdminApiSettings(
        audit_secret="unit-audit-secret",
    )
    return create_app(settings=settings)


def test_create_app_requires_secrets(monkeypatch: pytest.MonkeyPatch, security_context: Callable[..., str]) -> None:
    monkeypatch.delenv("TRADEPULSE_AUDIT_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="TRADEPULSE_AUDIT_SECRET"):
        create_app()


def _build_payload() -> dict[str, object]:
    base = datetime(2024, 12, 1, 12, 0, tzinfo=timezone.utc)
    bars = []
    price = 100.0
    for idx in range(60):
        ts = base + timedelta(minutes=idx)
        price += 0.1 if idx % 2 == 0 else -0.05
        bars.append(
            {
                "timestamp": ts.isoformat(),
                "open": price - 0.2,
                "high": price + 0.3,
                "low": price - 0.4,
                "close": price,
                "volume": 1000 + idx * 2,
                "bidVolume": 500 + idx,
                "askVolume": 480 + idx,
                "signedVolume": (-1) ** idx * 20.0,
            }
        )
    return {"symbol": "TEST-USD", "bars": bars}


def _auth_headers(token: str, *, client_cert: bool = False) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {token}"}
    if client_cert:
        headers["X-Client-Cert"] = "test-cert"
    return headers


def test_feature_endpoint_rejects_missing_token(configured_app: FastAPI) -> None:
    client = TestClient(configured_app)
    payload = _build_payload()
    response = client.post("/features", json=payload)
    assert response.status_code == 401


def test_feature_endpoint_computes_latest_vector(
    configured_app: FastAPI, security_context: Callable[..., str]
) -> None:
    app = configured_app
    client = TestClient(app)

    payload = _build_payload()
    token = security_context(subject="feature-user")
    headers = _auth_headers(token)

    response = client.post("/features", json=payload, headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "TEST-USD"
    assert "features" in body
    features = body["features"]
    for column in [
        "macd",
        "macd_signal",
        "macd_histogram",
        "macd_ema_fast",
        "macd_ema_slow",
    ]:
        assert column in features, f"Expected {column} in feature payload"
        assert features[column] is not None
    assert response.headers["X-Cache-Status"] == "miss"
    assert response.headers["Cache-Control"] == "private, max-age=30"
    assert "Accept" in response.headers.get("Vary", "")

    cached_response = client.post("/features", json=payload, headers=headers)
    assert cached_response.headers["X-Cache-Status"] == "hit"
    assert cached_response.json() == body


def test_prediction_endpoint_returns_signal(
    configured_app: FastAPI, security_context: Callable[..., str]
) -> None:
    app = configured_app
    client = TestClient(app)

    payload = _build_payload()
    payload["horizon_seconds"] = 900
    token = security_context(subject="prediction-user")
    headers = _auth_headers(token)

    response = client.post("/predictions", json=payload, headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "TEST-USD"
    assert body["horizon_seconds"] == 900
    signal = body["signal"]
    assert signal["symbol"] == "TEST-USD"
    assert 0.0 <= signal["confidence"] <= 1.0
    assert "score" in signal["metadata"]
    assert response.headers["X-Cache-Status"] == "miss"
    assert response.headers["Cache-Control"] == "private, max-age=30"
    assert "Accept" in response.headers.get("Vary", "")

    cached = client.post("/predictions", json=payload, headers=headers)
    assert cached.headers["X-Cache-Status"] == "hit"
    assert cached.json() == body


def test_invalid_token_is_rejected(
    configured_app: FastAPI, security_context: Callable[..., str]
) -> None:
    client = TestClient(configured_app)
    payload = _build_payload()
    invalid_headers = _auth_headers("malformed-token")
    response = client.post("/features", json=payload, headers=invalid_headers)
    assert response.status_code == 401


def test_admin_endpoints_require_client_certificate(
    configured_app: FastAPI, security_context: Callable[..., str]
) -> None:
    client = TestClient(configured_app)
    token = security_context(subject="admin-user")
    headers = _auth_headers(token)
    response = client.get("/admin/kill-switch", headers=headers)
    assert response.status_code == 401


def test_admin_endpoints_accept_jwt_and_certificate(
    configured_app: FastAPI, security_context: Callable[..., str]
) -> None:
    client = TestClient(configured_app)
    token = security_context(subject="admin-user")
    headers = _auth_headers(token, client_cert=True)

    response = client.get("/admin/kill-switch", headers=headers)
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["Pragma"] == "no-cache"
    vary_header = response.headers.get("Vary", "")
    assert "Authorization" in vary_header
    assert "Accept" in vary_header

    payload = {"reason": "manual intervention"}
    response = client.post("/admin/kill-switch", headers=headers, json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["kill_switch_engaged"] is True
    assert body["already_engaged"] is False


def test_admin_endpoint_rejects_wrong_audience(
    configured_app: FastAPI, security_context: Callable[..., str]
) -> None:
    client = TestClient(configured_app)
    bad_token = security_context(audience="different-audience")
    headers = _auth_headers(bad_token, client_cert=True)
    response = client.get("/admin/kill-switch", headers=headers)
    assert response.status_code == 401


def test_client_rate_limit_is_enforced(security_context: Callable[..., str]) -> None:
    rate_settings = ApiRateLimitSettings(
        default_policy=RateLimitPolicy(max_requests=5, window_seconds=60),
        client_policies={"feature-user": RateLimitPolicy(max_requests=1, window_seconds=60)},
    )
    limiter = SlidingWindowRateLimiter(InMemorySlidingWindowBackend(), rate_settings)
    app = create_app(
        settings=AdminApiSettings(audit_secret="unit-audit-secret"),
        rate_limiter=limiter,
        rate_limit_settings=rate_settings,
    )
    client = TestClient(app)

    payload = _build_payload()
    token = security_context(subject="feature-user")
    response_ok = client.post("/features", json=payload, headers=_auth_headers(token))
    assert response_ok.status_code == 200

    response_limited = client.post("/features", json=payload, headers=_auth_headers(token))
    assert response_limited.status_code == 429

    other_token = security_context(subject="different-user")
    recovery = client.post("/features", json=payload, headers=_auth_headers(other_token))
    assert recovery.status_code == 200


def test_health_probe_reports_ready_state(configured_app: FastAPI) -> None:
    client = TestClient(configured_app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    risk_component = body["components"]["risk_manager"]
    assert risk_component["healthy"] is True
    assert risk_component["status"] == "operational"
    assert "inference_cache" in body["components"]
    assert "client_rate_limiter" in body["components"]


def test_health_probe_reflects_kill_switch(configured_app: FastAPI) -> None:
    app = configured_app
    app.state.risk_manager.kill_switch.trigger("scheduled maintenance")
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "failed"
    risk = body["components"]["risk_manager"]
    assert risk["healthy"] is False
    assert risk["status"] == "failed"
    assert risk["detail"] == "scheduled maintenance"


def test_health_probe_flags_dependency_failure() -> None:
    probes = {
        "postgres": lambda: DependencyProbeResult(healthy=False, detail="connection refused"),
    }
    app = create_app(settings=AdminApiSettings(audit_secret="unit-audit-secret"), dependency_probes=probes)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "failed"
    dependency = body["components"]["dependency:postgres"]
    assert dependency["healthy"] is False
    assert dependency["status"] == "failed"
    assert dependency["detail"] == "connection refused"


def test_trusted_host_middleware_blocks_unlisted_hosts(
    monkeypatch: pytest.MonkeyPatch, security_context: Callable[..., str]
) -> None:
    restricted_settings = ApiSecuritySettings(
        oauth2_issuer="https://issuer.tradepulse.test",
        oauth2_audience="tradepulse-api",
        oauth2_jwks_uri="https://issuer.tradepulse.test/jwks",
        trusted_hosts=["api.tradepulse.test"],
    )
    monkeypatch.setattr(security_module, "_default_settings_loader", lambda: restricted_settings)
    if hasattr(security_module.get_api_security_settings, "_instance"):
        delattr(security_module.get_api_security_settings, "_instance")

    app = create_app(settings=AdminApiSettings(audit_secret="unit-audit-secret"))
    client = TestClient(app)
    payload = _build_payload()
    token = security_context(subject="feature-user")

    bad_host_headers = {**_auth_headers(token), "Host": "attacker.example"}
    denied = client.post("/features", json=payload, headers=bad_host_headers)
    assert denied.status_code == 400

    good_host_headers = {**_auth_headers(token), "Host": "api.tradepulse.test"}
    permitted = client.post("/features", json=payload, headers=good_host_headers)
    assert permitted.status_code == 200


def test_payload_guard_rejects_large_and_suspicious_bodies(
    monkeypatch: pytest.MonkeyPatch, security_context: Callable[..., str]
) -> None:
    tuned_settings = ApiSecuritySettings(
        oauth2_issuer="https://issuer.tradepulse.test",
        oauth2_audience="tradepulse-api",
        oauth2_jwks_uri="https://issuer.tradepulse.test/jwks",
        trusted_hosts=["testserver"],
        max_request_bytes=512,
    )
    monkeypatch.setattr(security_module, "_default_settings_loader", lambda: tuned_settings)
    if hasattr(security_module.get_api_security_settings, "_instance"):
        delattr(security_module.get_api_security_settings, "_instance")

    app = create_app(settings=AdminApiSettings(audit_secret="unit-audit-secret"))
    client = TestClient(app)
    token = security_context(subject="feature-user")

    oversized_payload = _build_payload()
    oversized_payload["bars"] *= 20
    response_large = client.post(
        "/features",
        json=oversized_payload,
        headers=_auth_headers(token),
    )
    assert response_large.status_code == 413

    suspicious_payload = _build_payload()
    suspicious_payload["symbol"] = "<script>alert(1)</script>"
    suspicious_payload["bars"] = suspicious_payload["bars"][:1]
    response_suspicious = client.post(
        "/features",
        json=suspicious_payload,
        headers=_auth_headers(token),
    )
    assert response_suspicious.status_code == 400
