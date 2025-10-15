"""Secret management utilities with refresh and monitoring hooks."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Mapping, MutableMapping, Protocol

__all__ = [
    "SecretNotFoundError",
    "SecretProvider",
    "SecretValue",
    "SecretManager",
    "EnvSecretProvider",
    "get_secret_manager",
    "set_secret_manager",
]


_LOGGER = logging.getLogger("tradepulse.security.secrets")
_DEFAULT_REFRESH_MARGIN = timedelta(minutes=5)


class SecretNotFoundError(KeyError):
    """Raised when a requested secret cannot be resolved by the provider."""


class SecretProvider(Protocol):
    """Protocol implemented by backends that surface secret material."""

    def fetch(self, secret_id: str) -> "SecretValue":
        """Return the latest value for *secret_id* or raise :class:`SecretNotFoundError`."""

    def secret_exists(self, secret_id: str) -> bool:
        """Return ``True`` when *secret_id* is resolvable by the provider."""


@dataclass(frozen=True)
class SecretValue:
    """Container describing a retrieved secret and its metadata."""

    value: str
    version: str
    expires_at: datetime

    def ttl(self, *, now: datetime | None = None) -> timedelta:
        """Return the remaining lifetime for the secret."""

        current = now or datetime.now(timezone.utc)
        return max(self.expires_at - current, timedelta(0))


@dataclass
class _SecretCacheEntry:
    value: SecretValue
    watchers: list[Callable[[SecretValue], None]]


class SecretManager:
    """Coordinate secret retrieval, caching, rotation, and monitoring."""

    def __init__(
        self,
        provider: SecretProvider,
        *,
        refresh_margin: timedelta = _DEFAULT_REFRESH_MARGIN,
        clock: Callable[[], datetime] | None = None,
        metrics_hook: Callable[[str, Mapping[str, str]], None] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._provider = provider
        self._refresh_margin = refresh_margin
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._metrics_hook = metrics_hook
        self._logger = logger or _LOGGER
        self._lock = threading.RLock()
        self._cache: MutableMapping[str, _SecretCacheEntry] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background refresh loop if it is not already running."""

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._refresh_loop, name="secret-refresh", daemon=True)
            self._thread.start()
            self._logger.debug("secret.refresh_loop.started")

    def stop(self) -> None:
        """Stop the background refresh loop and wait for it to exit."""

        thread: threading.Thread | None
        with self._lock:
            thread = self._thread
            self._thread = None
            self._stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
            self._logger.debug("secret.refresh_loop.stopped")

    def register_secret(self, secret_id: str) -> SecretValue:
        """Ensure *secret_id* is tracked and return the current value."""

        value = self._fetch_secret(secret_id)
        with self._lock:
            entry = self._cache.get(secret_id)
            if entry is None:
                entry = _SecretCacheEntry(value=value, watchers=[])
                self._cache[secret_id] = entry
            else:
                entry.value = value
        self._logger.info(
            "secret.registered",
            extra={
                "secret": {
                    "id": secret_id,
                    "version": value.version,
                    "expires_at": value.expires_at.isoformat(),
                }
            },
        )
        return value

    def subscribe(self, secret_id: str, callback: Callable[[SecretValue], None]) -> None:
        """Invoke *callback* whenever *secret_id* rotates."""

        with self._lock:
            entry = self._cache.get(secret_id)
            if entry is None:
                entry = _SecretCacheEntry(value=self._fetch_secret(secret_id), watchers=[])
                self._cache[secret_id] = entry
            entry.watchers.append(callback)

    def get_secret(self, secret_id: str) -> SecretValue:
        """Return the cached secret value, refreshing when necessary."""

        with self._lock:
            entry = self._cache.get(secret_id)
        if entry is None:
            secret_value = self.register_secret(secret_id)
        else:
            secret_value = entry.value
        now = self._clock()
        ttl = secret_value.ttl(now=now)
        if self._metrics_hook is not None:
            self._metrics_hook(
                "secret.access",
                {
                    "secret_id": secret_id,
                    "version": secret_value.version,
                },
            )
        self._logger.debug(
            "secret.accessed",
            extra={
                "secret": {
                    "id": secret_id,
                    "version": secret_value.version,
                    "expires_at": secret_value.expires_at.isoformat(),
                }
            },
        )
        if ttl <= self._refresh_margin:
            self._emit_near_expiry(secret_id, secret_value, ttl)
            if ttl <= timedelta(0):
                secret_value = self._refresh(secret_id)
        return secret_value

    def invalidate(self, secret_id: str) -> None:
        """Force the cached value for *secret_id* to refresh on next access."""

        with self._lock:
            if secret_id in self._cache:
                del self._cache[secret_id]
        self._logger.info("secret.invalidated", extra={"secret": {"id": secret_id}})

    def _emit_near_expiry(self, secret_id: str, secret_value: SecretValue, ttl: timedelta) -> None:
        self._logger.warning(
            "secret.near_expiry",
            extra={
                "secret": {
                    "id": secret_id,
                    "version": secret_value.version,
                    "expires_in_seconds": ttl.total_seconds(),
                }
            },
        )
        if self._metrics_hook is not None:
            self._metrics_hook(
                "secret.near_expiry",
                {
                    "secret_id": secret_id,
                    "version": secret_value.version,
                },
            )

    def _refresh_loop(self) -> None:
        while not self._stop_event.wait(self._next_refresh_interval()):
            for secret_id in self._tracked_secret_ids():
                try:
                    value = self.get_secret(secret_id)
                except SecretNotFoundError:
                    continue
                now = self._clock()
                if value.ttl(now=now) <= self._refresh_margin:
                    self._refresh(secret_id)

    def _tracked_secret_ids(self) -> list[str]:
        with self._lock:
            return list(self._cache.keys())

    def _next_refresh_interval(self) -> float:
        with self._lock:
            if not self._cache:
                return 30.0
            now = self._clock()
            next_deadline: datetime | None = None
            for entry in self._cache.values():
                candidate = entry.value.expires_at - self._refresh_margin
                if candidate <= now:
                    return 0.0
                if next_deadline is None or candidate < next_deadline:
                    next_deadline = candidate
            if next_deadline is None:
                return 30.0
            return max((next_deadline - now).total_seconds(), 1.0)

    def _refresh(self, secret_id: str) -> SecretValue:
        value = self._fetch_secret(secret_id)
        watchers: Iterable[Callable[[SecretValue], None]] = ()
        previous: SecretValue | None = None
        with self._lock:
            entry = self._cache.get(secret_id)
            if entry is None:
                entry = _SecretCacheEntry(value=value, watchers=[])
                self._cache[secret_id] = entry
            else:
                previous = entry.value
                entry.value = value
            watchers = tuple(entry.watchers)
        if previous is None or previous.version != value.version or previous.value != value.value:
            self._logger.info(
                "secret.rotated",
                extra={
                    "secret": {
                        "id": secret_id,
                        "version": value.version,
                        "expires_at": value.expires_at.isoformat(),
                    }
                },
            )
            for callback in watchers:
                try:
                    callback(value)
                except Exception:  # pragma: no cover - defensive logging
                    self._logger.exception("secret.rotation_callback_failed", extra={"secret": {"id": secret_id}})
        return value

    def _fetch_secret(self, secret_id: str) -> SecretValue:
        try:
            return self._provider.fetch(secret_id)
        except SecretNotFoundError:
            raise
        except Exception as exc:  # pragma: no cover - provider failure guard
            self._logger.error("secret.provider_failure", exc_info=exc, extra={"secret": {"id": secret_id}})
            raise


class EnvSecretProvider:
    """Secret provider that sources values from environment variables."""

    def __init__(
        self,
        env: Mapping[str, str] | None = None,
        *,
        default_ttl: timedelta = timedelta(hours=12),
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._env = env or os.environ
        self._default_ttl = default_ttl
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def fetch(self, secret_id: str) -> SecretValue:
        raw = self._env.get(secret_id)
        if raw is None:
            raise SecretNotFoundError(secret_id)
        ttl_seconds = self._env.get(f"{secret_id}_TTL_SECONDS")
        expires_at = self._clock() + self._default_ttl
        if ttl_seconds is not None:
            try:
                expires_at = self._clock() + timedelta(seconds=float(ttl_seconds))
            except ValueError as exc:  # pragma: no cover - invalid TTL metadata
                raise RuntimeError(f"Invalid TTL for secret {secret_id}") from exc
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return SecretValue(value=raw, version=digest, expires_at=expires_at)

    def secret_exists(self, secret_id: str) -> bool:
        return secret_id in self._env


def get_secret_manager() -> SecretManager:
    """Return the process-wide secret manager singleton."""

    if not hasattr(get_secret_manager, "_instance"):
        provider = EnvSecretProvider()
        manager = SecretManager(provider)
        manager.start()
        get_secret_manager._instance = manager  # type: ignore[attr-defined]
    return get_secret_manager._instance  # type: ignore[attr-defined]


def set_secret_manager(manager: SecretManager | None) -> None:
    """Override or clear the process-wide secret manager singleton."""

    if hasattr(get_secret_manager, "_instance"):
        existing = get_secret_manager._instance  # type: ignore[attr-defined]
        if isinstance(existing, SecretManager):
            existing.stop()
        delattr(get_secret_manager, "_instance")
    if manager is not None:
        manager.start()
        get_secret_manager._instance = manager  # type: ignore[attr-defined]
