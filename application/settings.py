"""Central configuration for the TradePulse FastAPI service."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

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
    secret_backend_provider: Literal["vault", "aws_secrets_manager"] | None = Field(
        default=None,
        description=(
            "Optional secret backend provider used instead of file-based secrets. When omitted the application "
            "relies on local files with optional fallbacks."
        ),
    )
    secret_backend_vault_url: AnyUrl | None = Field(
        default=None,
        description="Base URL of the HashiCorp Vault cluster used for managed secrets.",
    )
    secret_backend_vault_namespace: str | None = Field(
        default=None,
        description="Vault namespace that scopes secret requests when using Vault as backend.",
    )
    secret_backend_vault_token: SecretStr | None = Field(
        default=None,
        description="Authentication token used when connecting to the Vault API.",
    )
    secret_backend_vault_mount_point: str = Field(
        "secret",
        description="KV mount point that stores managed secrets when Vault backend is enabled.",
    )
    secret_backend_vault_secret_prefix: str | None = Field(
        default=None,
        description="Optional prefix prepended to secret identifiers when using the Vault backend.",
    )
    secret_backend_aws_region: str | None = Field(
        default=None,
        description="AWS region that hosts the Secrets Manager instance for managed secrets.",
    )
    secret_backend_aws_profile: str | None = Field(
        default=None,
        description="Optional named AWS profile used to authenticate when retrieving secrets.",
    )
    secret_backend_aws_secret_prefix: str | None = Field(
        default=None,
        description="Optional prefix prepended to secret identifiers before resolving them in Secrets Manager.",
    )

    @model_validator(mode="after")
    def _validate_siem_configuration(self) -> "AdminApiSettings":
        if self.siem_endpoint is not None:
            has_secret = self.siem_client_secret is not None or self.siem_client_secret_path is not None
            if not self.siem_client_id or not has_secret:
                raise ValueError(
                    "siem_client_id and siem_client_secret must be configured when siem_endpoint is set"
                )
        if self.secret_backend_provider == "vault":
            if self.secret_backend_vault_url is None:
                raise ValueError("secret_backend_vault_url is required when using the Vault backend")
            if self.secret_backend_vault_token is None:
                raise ValueError("secret_backend_vault_token is required when using the Vault backend")
        if self.secret_backend_provider == "aws_secrets_manager":
            if self.secret_backend_aws_region is None:
                raise ValueError(
                    "secret_backend_aws_region is required when using the AWS Secrets Manager backend"
                )
        return self

    def build_secret_manager(self) -> "SecretManager":
        """Return a configured secret manager for administrative components."""

        from application.secrets.backends import AwsSecretsManagerBackend, VaultSecretBackend
        from application.secrets.manager import (
            ManagedSecret,
            ManagedSecretConfig,
            SecretBackend,
            SecretManager,
        )

        refresh_interval = float(self.secret_refresh_interval_seconds)
        backend: SecretBackend | None = None
        provider = self.secret_backend_provider
        if provider == "vault":
            backend = VaultSecretBackend(
                url=str(self.secret_backend_vault_url),
                namespace=self.secret_backend_vault_namespace,
                auth_token=self.secret_backend_vault_token.get_secret_value()
                if self.secret_backend_vault_token is not None
                else None,
                mount_point=self.secret_backend_vault_mount_point,
            )
        elif provider == "aws_secrets_manager":
            backend = AwsSecretsManagerBackend(
                region_name=self.secret_backend_aws_region or "",
                profile_name=self.secret_backend_aws_profile,
                secret_prefix=self.secret_backend_aws_secret_prefix,
            )

        def backend_identifier(name: str) -> str | None:
            if backend is None:
                return None
            if provider == "vault":
                prefix = self.secret_backend_vault_secret_prefix or ""
                return f"{prefix}{name}" if prefix else name
            return name

        secrets: dict[str, ManagedSecret] = {
            "audit_secret": ManagedSecret(
                config=ManagedSecretConfig(
                    name="audit_secret",
                    path=self.audit_secret_path,
                    min_length=16,
                    backend_identifier=backend_identifier("audit_secret"),
                ),
                fallback=self.audit_secret.get_secret_value(),
                refresh_interval_seconds=refresh_interval,
                backend=backend,
            )
        }

        siem_backend_identifier = backend_identifier("siem_client_secret")
        if (
            self.siem_client_secret is not None
            or self.siem_client_secret_path is not None
            or siem_backend_identifier is not None
        ):
            fallback: str | None = None
            if self.siem_client_secret is not None:
                fallback = self.siem_client_secret.get_secret_value()
            secrets["siem_client_secret"] = ManagedSecret(
                config=ManagedSecretConfig(
                    name="siem_client_secret",
                    path=self.siem_client_secret_path,
                    min_length=12,
                    backend_identifier=siem_backend_identifier,
                ),
                fallback=fallback,
                refresh_interval_seconds=refresh_interval,
                backend=backend,
            )

        return SecretManager(secrets)

    model_config = SettingsConfigDict(
        env_prefix="TRADEPULSE_",
        extra="ignore",
        secrets_dir=Path("/run/secrets"),
    )


class ApiSecuritySettings(BaseSettings):
    """Runtime configuration for OAuth2, mutual TLS, and upstream WAF hand-off."""

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


__all__ = [
    "AdminApiSettings",
    "ApiSecuritySettings",
    "RateLimitPolicy",
    "ApiRateLimitSettings",
]
