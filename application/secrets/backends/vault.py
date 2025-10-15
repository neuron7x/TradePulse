"""HashiCorp Vault backend implementation."""

from __future__ import annotations

import importlib
import importlib.util
import logging
from typing import Mapping

from application.secrets.manager import SecretBackend, SecretManagerError

_LOGGER = logging.getLogger("tradepulse.secrets.vault")


class VaultSecretBackend(SecretBackend):
    """Retrieve managed secrets from HashiCorp Vault."""

    def __init__(
        self,
        *,
        url: str,
        namespace: str | None = None,
        auth_token: str | None = None,
        mount_point: str = "secret",
        client: object | None = None,
        verify: bool = True,
    ) -> None:
        spec = importlib.util.find_spec("hvac")
        if spec is None:
            raise SecretManagerError("VaultSecretBackend requires the 'hvac' package to be installed.")
        hvac = importlib.import_module("hvac")
        if client is None:
            client = hvac.Client(url=url, token=auth_token, namespace=namespace, verify=verify)
        self._client = client
        self._mount_point = mount_point
        self._url = url
        self._namespace = namespace
        if getattr(self._client, "is_authenticated", lambda: True)() is False:
            raise SecretManagerError("Failed to authenticate against HashiCorp Vault")

    def read(self, *, identifier: str) -> str:
        try:
            response = self._client.secrets.kv.v2.read_secret_version(
                path=identifier,
                mount_point=self._mount_point,
            )
        except Exception as exc:  # pragma: no cover - hvac specific
            raise SecretManagerError(
                f"Unable to read secret '{identifier}' from Vault mount '{self._mount_point}'"
            ) from exc
        data = response.get("data", {}).get("data", {})
        if not data:
            raise SecretManagerError(
                f"Secret '{identifier}' from Vault mount '{self._mount_point}' returned no data"
            )
        if "value" in data:
            return str(data["value"])
        if len(data) == 1:
            return str(next(iter(data.values())))
        raise SecretManagerError(
            f"Secret '{identifier}' from Vault mount '{self._mount_point}' returned composite data"
        )

    def write(self, *, identifier: str, value: str) -> None:
        try:
            self._client.secrets.kv.v2.create_or_update_secret(
                path=identifier,
                secret={"value": value},
                mount_point=self._mount_point,
            )
        except Exception as exc:  # pragma: no cover - hvac specific
            raise SecretManagerError(
                f"Unable to write secret '{identifier}' to Vault mount '{self._mount_point}'"
            ) from exc

    def audit_metadata(self, *, identifier: str) -> Mapping[str, str]:
        metadata = {
            "provider": "hashicorp_vault",
            "url": self._url,
            "mount_point": self._mount_point,
        }
        if self._namespace:
            metadata["namespace"] = self._namespace
        try:
            secret_metadata = self._client.secrets.kv.v2.read_metadata(
                path=identifier,
                mount_point=self._mount_point,
            )
        except Exception:  # pragma: no cover - metadata fetch best-effort
            _LOGGER.debug(
                "Failed to fetch Vault secret metadata",
                extra={"secret": identifier, "mount_point": self._mount_point},
            )
        else:
            data = secret_metadata.get("data", {})
            if "created_time" in data:
                metadata["created_time"] = data["created_time"]
            if "updated_time" in data:
                metadata["updated_time"] = data["updated_time"]
            if "current_version" in data:
                metadata["version"] = str(data["current_version"])
        return metadata
