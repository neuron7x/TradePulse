"""Central configuration for the TradePulse FastAPI service."""

from __future__ import annotations

from pathlib import Path

from pydantic import (
    AnyUrl,
    BaseModel,
    Field,
    HttpUrl,
    PositiveFloat,
    PositiveInt,
    SecretStr,
)
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

    trusted_hosts: list[str] = Field(
        default_factory=lambda: ["testserver", "localhost"],
        description=(
            "Host header values accepted by the API gateway. Requests from other hosts "
            "are rejected before hitting route handlers."
        ),
        min_length=1,
    )
    max_request_bytes: PositiveInt = Field(
        1_000_000,
        description="Maximum request payload size, in bytes, accepted by the gateway.",
    )
    suspicious_json_keys: list[str] = Field(
        default_factory=lambda: ["$where", "__proto__", "$regex"],
        description=(
            "JSON keys that trigger an early rejection when present in request payloads."
        ),
    )
    suspicious_json_substrings: list[str] = Field(
        default_factory=lambda: ["<script", "javascript:"],
        description=(
            "Case-insensitive substrings that mark a JSON value as suspicious and cause "
            "the request to be rejected."
        ),
    )

    model_config = SettingsConfigDict(env_prefix="TRADEPULSE_", extra="ignore")


class RateLimitPolicy(BaseModel):
    """Rate limit definition expressed as a sliding-window quota."""

    max_requests: PositiveInt = Field(
        ...,
        description="Number of requests permitted within the configured window.",
    )
    window_seconds: PositiveFloat = Field(
        ...,
        description="Duration of the sliding window, in seconds, used for quota checks.",
    )


class ApiRateLimitSettings(BaseSettings):
    """Runtime configuration for per-client API rate limiting."""

    default_policy: RateLimitPolicy = Field(
        default_factory=lambda: RateLimitPolicy(max_requests=120, window_seconds=60.0),
        description="Fallback policy applied when a subject specific policy is not defined.",
    )
    unauthenticated_policy: RateLimitPolicy | None = Field(
        default=None,
        description=(
            "Optional policy applied to unauthenticated requests. When unset the "
            "default policy is used."
        ),
    )
    client_policies: dict[str, RateLimitPolicy] = Field(
        default_factory=dict,
        description=(
            "Mapping of authenticated subject identifiers to dedicated rate policies."
        ),
    )
    redis_url: AnyUrl | None = Field(
        default=None,
        description=(
            "Redis connection string used to coordinate rate limits across instances. "
            "When omitted an in-memory limiter is used."
        ),
    )
    redis_key_prefix: str = Field(
        default="tradepulse:rate", description="Prefix applied to Redis keys."
    )

    model_config = SettingsConfigDict(env_prefix="TRADEPULSE_RATE_", extra="ignore")


__all__ = [
    "AdminApiSettings",
    "ApiSecuritySettings",
    "RateLimitPolicy",
    "ApiRateLimitSettings",
]
