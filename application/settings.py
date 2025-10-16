"""Central configuration for the TradePulse FastAPI service."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from src.audit.audit_logger import AuditLogger

from pydantic import (
    AnyUrl,
    BaseModel,
    Field,
    HttpUrl,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class AdminApiSettings(BaseSettings):
    """Configuration governing administrative controls and audit logging."""

    audit_secret: SecretStr = Field(
        ...,
        min_length=16,
        description="Secret used to sign administrative audit records.",
    )
    audit_secret_path: Path | None = Field(
        default=None,
        description=(
            "Optional filesystem path managed by the platform secret manager that contains "
            "the audit signing secret. When supplied the application refreshes the key "
            "periodically to honour rotations."
        ),
    )
    secret_refresh_interval_seconds: PositiveFloat = Field(
        300.0,
        description="Minimum interval, in seconds, between managed secret refresh attempts.",
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
    kill_switch_store_path: Path = Field(
        Path("state/kill_switch_state.sqlite"),
        description="Filesystem path used to persist the risk kill-switch state.",
    )
    siem_endpoint: HttpUrl | None = Field(
        default=None,
        description="Optional SIEM API endpoint that receives replicated audit records.",
    )
    siem_client_id: str | None = Field(
        default=None,
        min_length=1,
        description="Client identifier used when authenticating against the SIEM ingest API.",
    )
    siem_client_secret: SecretStr | None = Field(
        default=None,
        description=(
            "Client secret used for SIEM authentication. Provide via environment variable "
            "or mounted secrets directory to avoid embedding credentials in configuration files."
        ),
    )
    siem_client_secret_path: Path | None = Field(
        default=None,
        description=(
            "Optional filesystem path monitored for SIEM client secret rotations."
        ),
    )
    siem_scope: str | None = Field(
        default=None,
        description="Optional OAuth2 scope requested when exchanging SIEM credentials for a token.",
    )

    @model_validator(mode="after")
    def _validate_siem_configuration(self) -> "AdminApiSettings":
        if self.siem_endpoint is not None:
            has_secret = self.siem_client_secret is not None or self.siem_client_secret_path is not None
            if not self.siem_client_id or not has_secret:
                raise ValueError(
                    "siem_client_id and siem_client_secret must be configured when siem_endpoint is set"
                )
        return self

    def build_secret_manager(
        self,
        *,
        audit_logger_factory: Callable[["SecretManager"], "AuditLogger"] | None = None,
    ) -> "SecretManager":
        """Return a configured secret manager for administrative components."""

        from application.secrets.manager import ManagedSecret, ManagedSecretConfig, SecretManager

        refresh_interval = float(self.secret_refresh_interval_seconds)
        secrets: dict[str, ManagedSecret] = {
            "audit_secret": ManagedSecret(
                config=ManagedSecretConfig(
                    name="audit_secret",
                    path=self.audit_secret_path,
                    min_length=16,
                ),
                fallback=self.audit_secret.get_secret_value(),
                refresh_interval_seconds=refresh_interval,
            )
        }

        if self.siem_client_secret is not None or self.siem_client_secret_path is not None:
            fallback: str | None = None
            if self.siem_client_secret is not None:
                fallback = self.siem_client_secret.get_secret_value()
            secrets["siem_client_secret"] = ManagedSecret(
                config=ManagedSecretConfig(
                    name="siem_client_secret",
                    path=self.siem_client_secret_path,
                    min_length=12,
                ),
                fallback=fallback,
                refresh_interval_seconds=refresh_interval,
            )

        return SecretManager(secrets, audit_logger_factory=audit_logger_factory)

    model_config = SettingsConfigDict(
        env_prefix="TRADEPULSE_",
        extra="ignore",
        secrets_dir=Path("/run/secrets"),
    )


class ApiSecuritySettings(BaseSettings):
    """Runtime configuration for OAuth2, mutual TLS, and upstream WAF hand-off."""

    oauth2_issuer: HttpUrl = Field(
        "https://auth.tradepulse.invalid/issuer",
        description=(
            "Expected issuer claim for incoming OAuth2 JWT bearer tokens."
            " Defaults ensure unit tests can import the API without extra"
            " environment configuration."
        ),
    )
    oauth2_audience: str = Field(
        "tradepulse-api",
        min_length=1,
        description=(
            "Audience that must be present within validated JWT access tokens."
            " Override in production via environment variables."
        ),
    )
    oauth2_jwks_uri: HttpUrl = Field(
        "https://auth.tradepulse.invalid/jwks",
        description=(
            "JWKS endpoint used to discover signing keys for JWT validation."
            " Defaults are non-routable placeholders suitable for tests."
        ),
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
    upstream_waf_request_id_header: str = Field(
        "X-Request-ID",
        min_length=1,
        description=(
            "Header normalised by the external gateway or cloud WAF that uniquely tags each "
            "request for downstream log correlation."
        ),
    )
    upstream_waf_forwarded_for_header: str = Field(
        "X-Forwarded-For",
        min_length=1,
        description=(
            "Header populated by the upstream WAF containing the client IP chain that the "
            "FastAPI layer trusts for rate-limiting and audit trails."
        ),
    )
    upstream_waf_event_header: str = Field(
        "X-WAF-Event",
        min_length=1,
        description=(
            "Header propagated from the external gateway describing the inspection decision "
            "(allow, challenged, mitigated) to be recorded alongside local security logs."
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


class EmailNotificationSettings(BaseModel):
    """SMTP configuration used for email notifications."""

    host: str = Field(..., min_length=1, description="SMTP server hostname.")
    port: PositiveInt = Field(587, description="SMTP server port.")
    sender: str = Field(..., min_length=3, description="Email address used as the sender.")
    recipients: list[str] = Field(
        default_factory=list,
        description="Email recipients that receive TradePulse notifications.",
    )
    username: str | None = Field(
        default=None, description="Optional username used for SMTP authentication."
    )
    password: SecretStr | None = Field(
        default=None, description="Optional password used for SMTP authentication."
    )
    use_tls: bool = Field(True, description="Enable STARTTLS for SMTP connections.")
    use_ssl: bool = Field(False, description="Use implicit TLS when connecting to SMTP.")
    timeout_seconds: PositiveFloat = Field(10.0, description="SMTP connection timeout.")

    @model_validator(mode="after")
    def _validate_configuration(self) -> "EmailNotificationSettings":
        if not self.recipients:
            raise ValueError("recipients must contain at least one address")
        if self.use_tls and self.use_ssl:
            raise ValueError("use_tls and use_ssl are mutually exclusive")
        return self


class NotificationSettings(BaseSettings):
    """Runtime configuration for out-of-band notifications."""

    email: EmailNotificationSettings | None = Field(
        default=None,
        description="Optional SMTP configuration for email alerts.",
    )
    slack_webhook_url: HttpUrl | None = Field(
        default=None,
        description="Incoming webhook URL used for Slack notifications.",
    )
    slack_channel: str | None = Field(
        default=None,
        description="Override Slack channel routed by the webhook.",
    )
    slack_username: str | None = Field(
        default=None,
        description="Display name used by the Slack notifier.",
    )
    slack_timeout_seconds: PositiveFloat = Field(
        5.0,
        description="HTTP timeout used for Slack webhook requests.",
    )

    model_config = SettingsConfigDict(env_prefix="TRADEPULSE_NOTIFY_", extra="ignore")


__all__ = [
    "AdminApiSettings",
    "ApiSecuritySettings",
    "RateLimitPolicy",
    "ApiRateLimitSettings",
    "EmailNotificationSettings",
    "NotificationSettings",
]
