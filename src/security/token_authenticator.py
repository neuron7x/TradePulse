"""Static token authenticator backed by the secret manager."""

from __future__ import annotations

import hmac
import logging
from threading import RLock
from typing import Optional

from .secret_manager import SecretManager, SecretValue

__all__ = ["TokenAuthenticator"]


class TokenAuthenticator:
    """Validate shared-secret administrative tokens with rotation support."""

    def __init__(
        self,
        secret_manager: SecretManager,
        secret_id: str,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        if not secret_id:
            raise ValueError("secret_id must be provided")
        self._secret_manager = secret_manager
        self._secret_id = secret_id
        self._logger = logger or logging.getLogger("tradepulse.security.token_auth")
        self._lock = RLock()
        self._token: Optional[str] = None
        self._version: Optional[str] = None
        self._refresh()
        self._secret_manager.subscribe(secret_id, self._on_rotation)

    @property
    def secret_id(self) -> str:
        """Identifier for the secret backing the authenticator."""

        return self._secret_id

    @property
    def version(self) -> Optional[str]:
        """Return the version of the cached token secret."""

        with self._lock:
            return self._version

    def authenticate(self, candidate: str) -> bool:
        """Return ``True`` when *candidate* matches the configured token."""

        if not candidate:
            self._logger.warning("admin.token.missing")
            return False
        expected = self._get_token()
        if expected is None:
            self._logger.error("admin.token.unavailable")
            return False
        valid = hmac.compare_digest(candidate, expected)
        if valid:
            self._logger.debug(
                "admin.token.validated", extra={"token": {"secret_id": self._secret_id, "version": self.version}}
            )
        else:
            self._logger.warning(
                "admin.token.invalid", extra={"token": {"secret_id": self._secret_id, "version": self.version}}
            )
        return valid

    def invalidate(self) -> None:
        """Drop the cached token forcing a refresh on the next authentication."""

        with self._lock:
            self._token = None
            self._version = None
        self._logger.info("admin.token.cache_cleared", extra={"token": {"secret_id": self._secret_id}})

    def _get_token(self) -> Optional[str]:
        with self._lock:
            token = self._token
        if token is not None:
            return token
        self._refresh()
        with self._lock:
            return self._token

    def _refresh(self) -> None:
        secret = self._secret_manager.get_secret(self._secret_id)
        with self._lock:
            self._token = secret.value
            self._version = secret.version
        self._logger.info(
            "admin.token.loaded",
            extra={"token": {"secret_id": self._secret_id, "version": secret.version, "expires_at": secret.expires_at.isoformat()}},
        )

    def _on_rotation(self, secret: SecretValue) -> None:
        with self._lock:
            self._token = secret.value
            self._version = secret.version
        self._logger.info(
            "admin.token.rotated",
            extra={"token": {"secret_id": self._secret_id, "version": secret.version, "expires_at": secret.expires_at.isoformat()}},
        )
