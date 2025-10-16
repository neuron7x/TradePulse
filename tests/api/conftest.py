from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import jwt
import pytest
import pytest_asyncio
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import httpx
from jwt.algorithms import RSAAlgorithm

os.environ.setdefault("TRADEPULSE_AUDIT_SECRET", "bootstrap-test-secret-1234")

from application.api import security as security_module
from application.api.service import create_app
from application.settings import (
    AdminApiSettings,
    ApiRateLimitSettings,
    ApiSecuritySettings,
    RateLimitPolicy,
)


@pytest.fixture
def admin_token_factory(monkeypatch: pytest.MonkeyPatch) -> Callable[..., str]:
    """Return a callable that mints OAuth2 bearer tokens for the test FastAPI app."""

    if hasattr(security_module.get_api_security_settings, "_instance"):
        delattr(security_module.get_api_security_settings, "_instance")

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    jwk_dict = RSAAlgorithm.to_jwk(public_key, as_dict=True)
    kid = "test-key"
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
        subject: str = "unit-user",
        *,
        audience: str | None = None,
        issuer: str | None = None,
        lifetime: timedelta = timedelta(minutes=5),
        roles: tuple[str, ...] = (),
    ) -> str:
        now = datetime.now(timezone.utc)
        payload: dict[str, object] = {
            "iss": issuer or str(settings.oauth2_issuer),
            "aud": audience or settings.oauth2_audience,
            "sub": subject,
            "iat": int(now.timestamp()),
            "exp": int((now + lifetime).timestamp()),
        }
        if roles:
            payload["roles"] = list(roles)
        return jwt.encode(payload, private_pem, algorithm="RS256", headers={"kid": kid})

    return mint_token


@pytest.fixture
def api_app(tmp_path: Path, admin_token_factory: Callable[..., str]):
    """Instantiate the FastAPI application with deterministic, test-friendly settings."""

    cache_policy = RateLimitPolicy(max_requests=2, window_seconds=1.0)
    rate_settings = ApiRateLimitSettings(
        default_policy=cache_policy,
        unauthenticated_policy=cache_policy,
    )
    settings = AdminApiSettings(
        audit_secret="integration-secret-123456",
        kill_switch_store_path=tmp_path / "kill_switch.sqlite",
    )
    return create_app(settings=settings, rate_limit_settings=rate_settings)


@pytest_asyncio.fixture
async def async_api_client(api_app):
    transport = httpx.ASGITransport(app=api_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.fixture
def feature_payload() -> dict[str, object]:
    base = datetime(2024, 12, 1, 12, 0, tzinfo=timezone.utc)
    bars: list[dict[str, object]] = []
    price = 100.0
    for idx in range(60):
        timestamp = base + timedelta(minutes=idx)
        price += 0.25 if idx % 2 == 0 else -0.1
        bars.append(
            {
                "timestamp": timestamp.isoformat(),
                "open": price - 0.5,
                "high": price + 0.6,
                "low": price - 0.7,
                "close": price,
                "volume": 1_000 + idx * 3,
                "bidVolume": 500 + idx,
                "askVolume": 480 + idx,
                "signedVolume": (-1) ** idx * 25.0,
            }
        )
    return {"symbol": "TEST-USD", "bars": bars}
