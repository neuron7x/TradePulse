"""Central configuration for the TradePulse FastAPI service."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, HttpUrl, PositiveFloat, PositiveInt, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AdminApiSettings(BaseSettings):
    """Configuration governing administrative controls and audit logging."""

    audit_secret: SecretStr = Field(
        ...,
        description="Secret used to sign administrative audit records.",
    )
    admin_subject: str = Field(
        "remote-admin",
        min_length=1,
        description="Default subject recorded for administrative actions when no override is provided.",
    )
    admin_rate_limit_max_attempts: PositiveInt = Field(
        5,
        description="Number of administrative requests allowed within the configured interval.",
    )
    admin_rate_limit_interval_seconds: PositiveFloat = Field(
        60.0,
        description="Length of the rolling window used for administrative rate limiting.",
    )
    audit_webhook_url: HttpUrl | None = Field(
        default=None,
        description="Optional HTTP endpoint that receives signed audit records for external storage.",
    )

    model_config = SettingsConfigDict(env_prefix="TRADEPULSE_", extra="ignore")


class ApiSecuritySettings(BaseSettings):
    """Runtime configuration for OAuth2 and mutual TLS enforcement."""

    oauth2_issuer: HttpUrl = Field(
        ...,
        description="Expected issuer claim for incoming OAuth2 JWT bearer tokens.",
    )
    oauth2_audience: str = Field(
        ...,
        min_length=1,
        description="Audience that must be present within validated JWT access tokens.",
    )
    oauth2_jwks_uri: HttpUrl = Field(
        ...,
        description="JWKS endpoint used to discover signing keys for JWT validation.",
    )
    mtls_trusted_ca_path: Path | None = Field(
        default=None,
        description=(
            "Optional path to a PEM bundle containing certificate authorities trusted for "
            "mutual TLS client authentication."
        ),
    )
    mtls_revocation_list_path: Path | None = Field(
        default=None,
        description=(
            "Optional path to a certificate revocation list checked during mTLS handshakes."
        ),
    )

    model_config = SettingsConfigDict(env_prefix="TRADEPULSE_", extra="ignore")


__all__ = ["AdminApiSettings", "ApiSecuritySettings"]
