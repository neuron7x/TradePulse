"""AWS Secrets Manager backend implementation."""

from __future__ import annotations

import importlib
import importlib.util
import logging
from typing import Mapping

from application.secrets.manager import SecretBackend, SecretManagerError

_LOGGER = logging.getLogger("tradepulse.secrets.aws")


class AwsSecretsManagerBackend(SecretBackend):
    """Interact with AWS Secrets Manager for secret retrieval."""

    def __init__(
        self,
        *,
        region_name: str,
        secret_prefix: str | None = None,
        profile_name: str | None = None,
        client: object | None = None,
    ) -> None:
        spec = importlib.util.find_spec("boto3")
        if spec is None:
            raise SecretManagerError(
                "AwsSecretsManagerBackend requires the 'boto3' package to be installed."
            )
        boto3 = importlib.import_module("boto3")
        session_kwargs: dict[str, str] = {"region_name": region_name}
        if profile_name is not None:
            session_kwargs["profile_name"] = profile_name
        session = boto3.session.Session(**session_kwargs)
        self._client = client or session.client("secretsmanager")
        self._region_name = region_name
        self._secret_prefix = secret_prefix or ""

    def _secret_id(self, identifier: str) -> str:
        if self._secret_prefix:
            return f"{self._secret_prefix}{identifier}"
        return identifier

    def read(self, *, identifier: str) -> str:
        secret_id = self._secret_id(identifier)
        try:
            response = self._client.get_secret_value(SecretId=secret_id)
        except Exception as exc:  # pragma: no cover - boto3 specific
            raise SecretManagerError(f"Unable to read secret '{secret_id}' from AWS Secrets Manager") from exc
        if "SecretString" in response and response["SecretString"]:
            return response["SecretString"]
        if "SecretBinary" in response and response["SecretBinary"]:
            return response["SecretBinary"].decode("utf-8")
        raise SecretManagerError(f"Secret '{secret_id}' in AWS Secrets Manager is empty")

    def write(self, *, identifier: str, value: str) -> None:
        secret_id = self._secret_id(identifier)
        try:
            self._client.put_secret_value(SecretId=secret_id, SecretString=value)
        except Exception as exc:  # pragma: no cover - boto3 specific
            raise SecretManagerError(f"Unable to write secret '{secret_id}' to AWS Secrets Manager") from exc

    def audit_metadata(self, *, identifier: str) -> Mapping[str, str]:
        secret_id = self._secret_id(identifier)
        metadata = {
            "provider": "aws_secrets_manager",
            "region": self._region_name,
            "secret_id": secret_id,
        }
        try:
            description = self._client.describe_secret(SecretId=secret_id)
        except Exception:  # pragma: no cover - metadata fetch best-effort
            _LOGGER.debug("Failed to fetch AWS Secrets Manager metadata", extra={"secret_id": secret_id})
        else:
            rotation_enabled = description.get("RotationEnabled")
            if rotation_enabled is not None:
                metadata["rotation_enabled"] = str(rotation_enabled)
            if description.get("LastChangedDate") is not None:
                metadata["last_changed"] = description["LastChangedDate"].isoformat()
        return metadata
