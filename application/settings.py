"""Central configuration for the TradePulse FastAPI service."""

from __future__ import annotations

from pydantic import Field, HttpUrl, PositiveFloat, PositiveInt, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AdminApiSettings(BaseSettings):
    """Configuration governing administrative controls and audit logging."""

    admin_token: SecretStr = Field(
        ...,
        description="Static bearer token for administrative access.",
    )
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


__all__ = ["AdminApiSettings"]
