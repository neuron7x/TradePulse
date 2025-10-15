"""Secret management utilities with support for rotation."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Protocol, runtime_checkable

__all__ = (
    "ManagedSecret",
    "ManagedSecretConfig",
    "SecretBackend",
    "SecretManager",
    "SecretManagerError",
)


class SecretManagerError(RuntimeError):
    """Raised when managed secrets cannot be resolved."""


@runtime_checkable
class SecretBackend(Protocol):
    """Protocol implemented by secret storage providers."""

    def read(self, *, identifier: str) -> str:
        """Return the clear-text secret for the given identifier."""

    def write(self, *, identifier: str, value: str) -> None:
        """Persist the secret in the backend."""

    def audit_metadata(self, *, identifier: str) -> Mapping[str, str]:
        """Return metadata suitable for audit logging."""


@dataclass(slots=True)
class ManagedSecretConfig:
    """Configuration describing how a secret should be sourced."""

    name: str
    path: Path | None = None
    min_length: int = 16
    backend_identifier: str | None = None


class ManagedSecret:
    """Represent a secret value that can refresh itself from disk."""

    def __init__(
        self,
        *,
        config: ManagedSecretConfig,
        fallback: str | None,
        refresh_interval_seconds: float,
        logger: logging.Logger | None = None,
        backend: SecretBackend | None = None,
    ) -> None:
        self._config = config
        self._refresh_interval = max(0.0, refresh_interval_seconds)
        self._logger = logger or logging.getLogger("tradepulse.secrets")
        self._lock = threading.Lock()
        self._value: str | None = None
        self._last_refresh = 0.0
        self._backend = backend
        if config.backend_identifier is not None and backend is None:
            raise SecretManagerError(
                f"Secret '{config.name}' declared backend identifier without configured backend"
            )
        if fallback is not None:
            self._ensure_min_length(fallback)
            self._value = fallback
        if (
            config.path is None
            and config.backend_identifier is None
            and self._value is None
        ):
            raise SecretManagerError(
                f"Secret '{config.name}' must provide a fallback value, managed path, or backend identifier."
            )
        if config.path is not None or config.backend_identifier is not None:
            # Attempt an eager refresh so missing files or backend access issues are detected on startup. If the
            # refresh fails but a fallback value is available we continue using the fallback and log the failure so
            # operators can investigate.
            try:
                self._refresh(force=True)
            except SecretManagerError as exc:
                if self._value is None:
                    raise
                extra: dict[str, str | None] = {"secret": config.name}
                if config.path is not None:
                    extra["path"] = str(config.path)
                if config.backend_identifier is not None:
                    extra["backend_identifier"] = config.backend_identifier
                self._logger.warning(
                    "Falling back to static secret after refresh failure",
                    extra=extra,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

    def get_secret(self) -> str:
        """Return the secret value, refreshing it when stale."""

        with self._lock:
            self._refresh()
            if self._value is None:
                raise SecretManagerError(
                    f"Secret '{self._config.name}' is unavailable after refresh."
                )
            return self._value

    def force_refresh(self) -> None:
        """Refresh the secret irrespective of the configured interval."""

        with self._lock:
            self._refresh(force=True)

    def _refresh(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and self._refresh_interval and now - self._last_refresh < self._refresh_interval:
            return
        secret: str | None = None
        source: str | None = None
        backend_identifier = self._config.backend_identifier
        if self._backend is not None and backend_identifier is not None:
            try:
                secret = self._backend.read(identifier=backend_identifier).strip()
                source = "backend"
            except Exception as exc:  # pragma: no cover - backend specifics
                self._logger.warning(
                    "Managed secret backend read failed",
                    extra={
                        "secret": self._config.name,
                        "backend_identifier": backend_identifier,
                        "backend": type(self._backend).__name__,
                    },
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
        if secret is None and self._config.path is not None:
            try:
                secret = self._config.path.read_text(encoding="utf-8").strip()
                source = "file"
            except FileNotFoundError as exc:
                self._logger.warning(
                    "Managed secret file missing",
                    extra={"secret": self._config.name, "path": str(self._config.path)},
                )
                if self._value is not None:
                    self._last_refresh = now
                    return
                raise SecretManagerError(
                    f"Secret '{self._config.name}' missing at {self._config.path}"
                ) from exc
        if not secret:
            self._logger.warning(
                "Managed secret source returned empty value",
                extra={
                    "secret": self._config.name,
                    "source": source,
                    "backend_identifier": backend_identifier,
                    "path": str(self._config.path) if self._config.path is not None else None,
                },
            )
            if self._value is not None:
                self._last_refresh = now
                return
            raise SecretManagerError(
                f"Secret '{self._config.name}' resolved empty value from managed sources"
            )
        try:
            self._ensure_min_length(secret)
        except SecretManagerError:
            self._logger.warning(
                "Managed secret failed minimum length check",
                extra={"secret": self._config.name, "source": source},
            )
            if self._value is not None:
                self._last_refresh = now
                return
            raise
        if secret != self._value:
            self._logger.info(
                "Managed secret rotated",
                extra={
                    "secret": self._config.name,
                    "source": source,
                    "path": str(self._config.path) if self._config.path is not None else None,
                },
            )
        self._value = secret
        self._last_refresh = now

    def _ensure_min_length(self, secret: str) -> None:
        if len(secret) < self._config.min_length:
            raise SecretManagerError(
                f"Secret '{self._config.name}' must be at least {self._config.min_length} characters."
            )


class SecretManager:
    """Coordinate retrieval of managed secrets for the application."""

    def __init__(self, secrets: Mapping[str, ManagedSecret]) -> None:
        if not secrets:
            raise ValueError("At least one secret must be managed")
        self._secrets: Dict[str, ManagedSecret] = dict(secrets)

    def get(self, name: str) -> str:
        secret = self._secrets.get(name)
        if secret is None:
            raise SecretManagerError(f"Unknown secret '{name}'")
        return secret.get_secret()

    def provider(self, name: str) -> Callable[[], str]:
        secret = self._secrets.get(name)
        if secret is None:
            raise SecretManagerError(f"Unknown secret '{name}'")

        def _resolver() -> str:
            return secret.get_secret()

        return _resolver

    def force_refresh(self, name: str) -> None:
        secret = self._secrets.get(name)
        if secret is None:
            raise SecretManagerError(f"Unknown secret '{name}'")
        secret.force_refresh()
