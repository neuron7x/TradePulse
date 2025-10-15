from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jwt.algorithms import RSAAlgorithm
from starlette.requests import Request

os.environ.setdefault("TRADEPULSE_AUDIT_SECRET", "test-audit-secret")
os.environ.setdefault("TRADEPULSE_OAUTH2_ISSUER", "https://issuer.tradepulse.test")
os.environ.setdefault("TRADEPULSE_OAUTH2_AUDIENCE", "tradepulse-api")
os.environ.setdefault("TRADEPULSE_OAUTH2_JWKS_URI", "https://issuer.tradepulse.test/jwks")

from application.api.security import get_api_security_settings, verify_request_identity
from application.settings import ApiSecuritySettings


@dataclass(slots=True)
class OAuthContext:
    mint_token: Callable[..., str]
    settings: ApiSecuritySettings
    kid: str
    jwk_dict: dict[str, Any]
    get_key_calls: Callable[[], int]


@pytest.fixture()
def oauth2_context(monkeypatch: pytest.MonkeyPatch) -> OAuthContext:
    if hasattr(get_api_security_settings, "_instance"):
        delattr(get_api_security_settings, "_instance")

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    jwk_dict = RSAAlgorithm.to_jwk(public_key, as_dict=True)
    kid = "unit-test-kid"
    jwk_dict.update({"kid": kid, "alg": "RS256", "use": "sig"})

    settings = ApiSecuritySettings(
        oauth2_issuer="https://issuer.tradepulse.test",
        oauth2_audience="tradepulse-api",
        oauth2_jwks_uri="https://issuer.tradepulse.test/jwks",
    )

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    call_count = 0

    async def fake_get_key(uri: str, request_kid: str) -> dict[str, Any] | None:
        nonlocal call_count
        call_count += 1
        assert uri == str(settings.oauth2_jwks_uri)
        if request_kid == kid:
            return jwk_dict
        return None

    monkeypatch.setattr(
        "application.api.security._jwks_resolver.get_key", fake_get_key
    )

    def mint_token(
        *,
        subject: str = "unit-user",
        audience: str | None = None,
        issuer: str | None = None,
        kid_override: str | None = None,
        include_subject: bool = True,
        lifetime: timedelta = timedelta(minutes=5),
    ) -> str:
        now = datetime.now(timezone.utc)
        payload: dict[str, Any] = {
            "iss": issuer or str(settings.oauth2_issuer),
            "aud": audience or settings.oauth2_audience,
            "iat": int(now.timestamp()),
            "exp": int((now + lifetime).timestamp()),
        }
        if include_subject:
            payload["sub"] = subject
        headers: dict[str, Any] = {"alg": "RS256"}
        headers["kid"] = kid_override or kid
        return jwt.encode(payload, private_pem, algorithm="RS256", headers=headers)

    return OAuthContext(
        mint_token=mint_token,
        settings=settings,
        kid=kid,
        jwk_dict=jwk_dict,
        get_key_calls=lambda: call_count,
    )


def _make_request(
    *, headers: dict[str, str] | None = None, scope_cert: dict[str, Any] | None = None
) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/admin",
        "headers": [],
    }
    if headers:
        scope["headers"] = [
            (key.lower().encode("ascii"), value.encode("utf-8"))
            for key, value in headers.items()
        ]
    if scope_cert is not None:
        scope["client_cert"] = scope_cert

    async def receive() -> dict[str, Any]:  # pragma: no cover - Starlette protocol hook
        return {"type": "http.request"}

    return Request(scope, receive)


@pytest.mark.anyio
async def test_valid_token_populates_identity_context(
    oauth2_context: OAuthContext,
) -> None:
    dependency = verify_request_identity()
    token = oauth2_context.mint_token(subject="feature-user")
    request = _make_request()
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    identity = await dependency(request, credentials, oauth2_context.settings)

    assert identity.subject == "feature-user"
    assert request.state.token_claims["sub"] == "feature-user"
    assert not hasattr(request.state, "client_certificate")
    assert oauth2_context.get_key_calls() == 1


@pytest.mark.anyio
async def test_missing_credentials_are_rejected(
    oauth2_context: OAuthContext,
) -> None:
    dependency = verify_request_identity()
    request = _make_request()

    with pytest.raises(HTTPException) as exc:
        await dependency(request, None, oauth2_context.settings)

    assert exc.value.status_code == 401
    assert exc.value.detail == "Bearer token required for this endpoint."


@pytest.mark.anyio
async def test_unknown_signing_key_is_rejected(oauth2_context: OAuthContext) -> None:
    dependency = verify_request_identity()
    token = oauth2_context.mint_token(kid_override="different-key")
    request = _make_request()
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    with pytest.raises(HTTPException) as exc:
        await dependency(request, credentials, oauth2_context.settings)

    assert exc.value.status_code == 401
    assert exc.value.detail == "Bearer token signed by unknown key."


@pytest.mark.anyio
async def test_missing_subject_claim_is_rejected(
    oauth2_context: OAuthContext,
) -> None:
    dependency = verify_request_identity()
    token = oauth2_context.mint_token(subject="")
    request = _make_request()
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    with pytest.raises(HTTPException) as exc:
        await dependency(request, credentials, oauth2_context.settings)

    assert exc.value.status_code == 401
    assert exc.value.detail == "Bearer token is missing the subject claim."


@pytest.mark.anyio
async def test_certificate_required_but_missing_is_rejected(
    oauth2_context: OAuthContext,
) -> None:
    dependency = verify_request_identity(require_client_certificate=True)
    token = oauth2_context.mint_token()
    request = _make_request()
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    with pytest.raises(HTTPException) as exc:
        await dependency(request, credentials, oauth2_context.settings)

    assert exc.value.status_code == 401
    assert exc.value.detail == "Mutual TLS client certificate required."


@pytest.mark.anyio
async def test_certificate_from_header_is_accepted(
    oauth2_context: OAuthContext,
) -> None:
    dependency = verify_request_identity(require_client_certificate=True)
    token = oauth2_context.mint_token(subject="admin-user")
    request = _make_request(headers={"X-Client-Cert": "client-cert"})
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    identity = await dependency(request, credentials, oauth2_context.settings)

    assert identity.subject == "admin-user"
    assert request.state.client_certificate == {"pem": "client-cert"}


@pytest.mark.anyio
async def test_certificate_from_scope_is_accepted(
    oauth2_context: OAuthContext,
) -> None:
    dependency = verify_request_identity(require_client_certificate=True)
    token = oauth2_context.mint_token(subject="scope-user")
    request = _make_request(scope_cert={"serial": "01"})
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    identity = await dependency(request, credentials, oauth2_context.settings)

    assert identity.subject == "scope-user"
    assert request.state.client_certificate == {"serial": "01"}


@pytest.mark.anyio
async def test_preexisting_state_certificate_is_preserved(
    oauth2_context: OAuthContext,
) -> None:
    dependency = verify_request_identity(require_client_certificate=True)
    token = oauth2_context.mint_token(subject="state-user")
    request = _make_request()
    request.state.client_certificate = {"thumbprint": "abc123"}
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    identity = await dependency(request, credentials, oauth2_context.settings)

    assert identity.subject == "state-user"
    assert request.state.client_certificate == {"thumbprint": "abc123"}

