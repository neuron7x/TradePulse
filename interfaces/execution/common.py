"""Shared utilities for authenticated execution connectors."""

from __future__ import annotations

import hashlib
import hmac
import os
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from queue import Empty, Queue
from typing import Any, Callable, Dict, Iterable, Mapping

import httpx

try:  # pragma: no cover - optional dependency guard
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - library not installed
    def load_dotenv(*_: object, **__: object) -> None:
        """Fallback noop when python-dotenv is not available."""

else:
    load_dotenv()


from execution.connectors import ExecutionConnector, OrderError, TransientOrderError


VaultResolver = Callable[[str], Mapping[str, str]]
RotationHook = Callable[[Mapping[str, str]], None]
TimestampExtractor = Callable[[httpx.Response], float | None]


class CredentialError(RuntimeError):
    """Raised when credentials are missing or invalid."""


@dataclass(slots=True)
class APICredentials:
    """Simple container for API credentials."""

    api_key: str
    api_secret: str
    passphrase: str | None = None
    extra: Mapping[str, str] = field(default_factory=dict)


class CredentialProvider:
    """Load credentials from environment variables or a Vault resolver."""

    def __init__(
        self,
        env_prefix: str,
        *,
        required_keys: Iterable[str] = ("API_KEY", "API_SECRET"),
        optional_keys: Iterable[str] | None = None,
        vault_resolver: VaultResolver | None = None,
        rotation_hook: RotationHook | None = None,
    ) -> None:
        self.env_prefix = env_prefix
        self.required_keys = tuple(required_keys)
        self.optional_keys = tuple(optional_keys or ())
        self.vault_resolver = vault_resolver
        self.rotation_hook = rotation_hook
        self._cache: Mapping[str, str] | None = None
        self._lock = threading.Lock()

    def _load_from_env(self) -> Dict[str, str]:
        values: Dict[str, str] = {}
        prefix = self.env_prefix.upper()
        for key in (*self.required_keys, *self.optional_keys):
            env_key = f"{prefix}_{key.upper()}"
            value = os.getenv(env_key)
            if value:
                values[key.upper()] = value
        return values

    def _load_from_vault(self) -> Dict[str, str]:
        if not self.vault_resolver:
            return {}
        vault_path = os.getenv(f"{self.env_prefix.upper()}_VAULT_PATH")
        if not vault_path:
            return {}
        return {k.upper(): v for k, v in self.vault_resolver(vault_path).items()}

    def load(self, *, force: bool = False) -> Mapping[str, str]:
        with self._lock:
            if self._cache is not None and not force:
                return self._cache
            values = {}
            values.update(self._load_from_vault())
            values.update(self._load_from_env())
            missing = [key for key in self.required_keys if key.upper() not in values]
            if missing:
                raise CredentialError(
                    f"Missing credential values for {', '.join(missing)} (prefix={self.env_prefix})"
                )
            self._cache = values
            return values

    def rotate(self, new_values: Mapping[str, str] | None = None) -> Mapping[str, str]:
        with self._lock:
            if new_values:
                normalized = {k.upper(): v for k, v in new_values.items()}
                missing = [key for key in self.required_keys if key.upper() not in normalized]
                if missing:
                    raise CredentialError(
                        "Cannot rotate credentials because required keys are missing: "
                        + ", ".join(missing)
                    )
                self._cache = normalized
            else:
                self._cache = None
                normalized = self.load(force=True)
            if self.rotation_hook:
                self.rotation_hook(normalized)
            return normalized


class HMACSigner:
    """Helper for computing HMAC signatures for authenticated requests."""

    def __init__(self, secret: str, *, algorithm: str = "sha256") -> None:
        self.secret = secret.encode()
        self.algorithm = algorithm

    def sign(self, payload: str) -> str:
        digest = hmac.new(self.secret, payload.encode(), getattr(hashlib, self.algorithm))
        return digest.hexdigest()


class HTTPBackoffController:
    """Adaptive rate limiter/backoff controller for REST calls."""

    def __init__(self, *, base_delay: float = 0.25, max_delay: float = 8.0) -> None:
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._lock = threading.Lock()
        self._backoff_until: float = 0.0
        self._attempts: int = 0

    def throttle(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._backoff_until:
                time.sleep(self._backoff_until - now)
        self._attempts = 0
        self._backoff_until = 0.0

    def reset(self) -> None:
        with self._lock:
            self._attempts = 0
            self._backoff_until = 0.0

    def backoff(self, response: httpx.Response | None = None) -> None:
        with self._lock:
            self._attempts += 1
            delay = min(self.base_delay * (2 ** (self._attempts - 1)), self.max_delay)
            if response is not None:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
            jitter = random.uniform(0, delay / 2)
            delay += jitter
            self._backoff_until = time.monotonic() + delay
            time.sleep(delay)


class ConnectionHealthMonitor:
    """Track websocket heartbeats to detect unhealthy connections."""

    def __init__(self, *, heartbeat_interval: float = 30.0) -> None:
        self.heartbeat_interval = heartbeat_interval
        self._lock = threading.Lock()
        self._last_heartbeat = time.monotonic()

    def touch(self) -> None:
        with self._lock:
            self._last_heartbeat = time.monotonic()

    def is_stale(self) -> bool:
        with self._lock:
            return time.monotonic() - self._last_heartbeat > self.heartbeat_interval * 2


class IdempotencyStore:
    """In-memory idempotency key registry with reconciliation support."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: Dict[str, Mapping[str, str]] = {}

    def get(self, key: str) -> Mapping[str, str] | None:
        with self._lock:
            return self._records.get(key)

    def put(self, key: str, payload: Mapping[str, str]) -> None:
        with self._lock:
            self._records[key] = dict(payload)

    def reconcile(
        self,
        key: str,
        fetcher: Callable[[Mapping[str, str]], Mapping[str, str] | None],
    ) -> Mapping[str, str] | None:
        """Ensure previously stored state matches remote data."""

        record = self.get(key)
        if not record:
            return None
        remote = fetcher(record)
        if remote is None:
            return None
        if remote.get("order_id") == record.get("order_id"):
            return remote
        # Update cached mapping to remote canonical state
        self.put(key, remote)
        return remote


def parse_server_time(response: httpx.Response) -> float | None:
    """Attempt to parse a timestamp from HTTP response headers."""

    date_header = response.headers.get("Date")
    if date_header:
        try:
            parsed = parsedate_to_datetime(date_header)
        except Exception:
            parsed = None
        if parsed is not None:
            return parsed.timestamp()
    server_time = response.headers.get("X-Server-Time") or response.headers.get("Server-Time")
    if server_time:
        try:
            return float(server_time) / 1000 if len(server_time) > 11 else float(server_time)
        except ValueError:
            return None
    return None


def ensure_timestamp_skew(response: httpx.Response, *, max_skew: float = 5.0) -> None:
    """Validate that the server time is within the allowed skew."""

    server_ts = parse_server_time(response)
    if server_ts is None:
        return
    local_ts = datetime.now(timezone.utc).timestamp()
    if abs(local_ts - server_ts) > max_skew:
        raise RuntimeError(
            f"Timestamp skew exceeds tolerance: local={local_ts}, server={server_ts}"
        )


class AuthenticatedRESTExecutionConnector(ExecutionConnector):
    """Base class bundling authentication, rate limiting and WS plumbing."""

    def __init__(
        self,
        env_prefix: str,
        *,
        sandbox: bool = True,
        base_url: str,
        sandbox_url: str | None = None,
        ws_url: str | None = None,
        sandbox_ws_url: str | None = None,
        credential_provider: CredentialProvider | None = None,
        optional_credential_keys: Iterable[str] | None = None,
        vault_resolver: VaultResolver | None = None,
        http_client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        backoff: HTTPBackoffController | None = None,
        health_monitor: ConnectionHealthMonitor | None = None,
        ws_factory: Callable[[str], Any] | None = None,
        enable_stream: bool = True,
        timeout: float = 10.0,
    ) -> None:
        super().__init__(sandbox=sandbox)
        self.env_prefix = env_prefix
        self._base_url = (sandbox_url if sandbox and sandbox_url else base_url).rstrip("/")
        self._ws_url = (sandbox_ws_url if sandbox and sandbox_ws_url else ws_url)
        self._timeout = timeout
        self._transport = transport
        self._credential_provider = credential_provider or CredentialProvider(
            env_prefix,
            optional_keys=optional_credential_keys,
            vault_resolver=vault_resolver,
        )
        self._http_client = http_client
        self._backoff = backoff or HTTPBackoffController()
        self._health = health_monitor or ConnectionHealthMonitor()
        self._ws_factory = ws_factory or self._default_ws_factory
        self._ws_enabled = enable_stream and self._ws_url is not None
        self._ws_thread: threading.Thread | None = None
        self._ws_stop = threading.Event()
        self._event_queue: "Queue[dict[str, Any]]" = Queue()
        self._credentials: Mapping[str, str] | None = None
        self._signer: HMACSigner | None = None
        self._rotation_attempted = False
        self._idempotency_store = IdempotencyStore()

    @property
    def credentials(self) -> Mapping[str, str]:
        if self._credentials is None:
            raise CredentialError("Connector is not connected")
        return self._credentials

    def connect(self, credentials: Mapping[str, str] | None = None) -> None:  # type: ignore[override]
        if credentials is not None:
            self._credentials = self._credential_provider.rotate(credentials)
        else:
            self._credentials = self._credential_provider.load()
        self._signer = self._create_signer(self.credentials)
        if self._http_client is None:
            self._http_client = httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout,
                transport=self._transport,
                headers=self._default_headers(),
            )
        if self._ws_enabled:
            self._start_streaming()

    def disconnect(self) -> None:  # type: ignore[override]
        self._ws_stop.set()
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=2.0)
        self._ws_thread = None
        if self._http_client is not None:
            self._http_client.close()
        self._http_client = None
        self._credentials = None
        self._signer = None
        self._rotation_attempted = False

    # --- Hook points -------------------------------------------------

    def _default_headers(self) -> dict[str, str]:
        return {}

    def _create_signer(self, credentials: Mapping[str, str]) -> HMACSigner:
        return HMACSigner(credentials["API_SECRET"])

    def _apply_signature(
        self,
        method: str,
        path: str,
        params: dict[str, Any],
        headers: dict[str, str],
        body: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, Any] | None]:
        return params, headers, body

    def _handle_stream_payload(self, payload: dict[str, Any]) -> None:
        self._event_queue.put(payload)

    # --- HTTP helpers -------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        signed: bool = True,
        allow_retry: bool = True,
        **kwargs: Any,
    ) -> httpx.Response:
        if self._http_client is None:
            raise RuntimeError("HTTP client is not initialised")
        params = dict(params or {})
        headers = dict(headers or {})
        attempt = 0
        while True:
            self._backoff.throttle()
            req_params = dict(params)
            req_headers = dict(headers)
            req_body = dict(body) if body is not None else None
            if signed:
                if self._signer is None:
                    raise CredentialError("Cannot sign request without credentials")
                req_params, req_headers, req_body = self._apply_signature(
                    method, path, req_params, req_headers, req_body
                )
            response = self._http_client.request(
                method,
                path,
                params=req_params or None,
                json=req_body if req_body is not None else None,
                headers=req_headers or None,
                **kwargs,
            )
            try:
                ensure_timestamp_skew(response)
            except RuntimeError:
                # Hard failure: treat as transient to allow rotation/backoff logic to decide.
                response.raise_for_status()
            if response.status_code == 429:
                if not allow_retry:
                    break
                self._backoff.backoff(response)
                continue
            if response.status_code in (401, 403) and allow_retry:
                if self._rotation_attempted:
                    break
                self._refresh_credentials()
                attempt += 1
                self._rotation_attempted = True
                continue
            if response.status_code >= 500 and allow_retry:
                self._backoff.backoff(response)
                continue
            self._rotation_attempted = False
            self._backoff.reset()
            if not response.is_success:
                response.raise_for_status()
            return response
        response.raise_for_status()
        return response

    def _refresh_credentials(self) -> None:
        self._credentials = self._credential_provider.rotate()
        self._signer = self._create_signer(self.credentials)

    # --- Websocket handling ------------------------------------------

    def _start_streaming(self) -> None:
        if self._ws_thread and self._ws_thread.is_alive():
            return
        self._ws_stop.clear()
        self._ws_thread = threading.Thread(target=self._ws_loop, daemon=True)
        self._ws_thread.start()

    def _default_ws_factory(self, url: str) -> Any:
        if not url:
            raise RuntimeError("Websocket URL is not configured")
        try:
            from websockets.sync.client import connect  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("websockets library is required for streaming support") from exc
        return connect(url)

    def _ws_loop(self) -> None:
        if not self._ws_url:
            return
        backoff = 1.0
        while not self._ws_stop.is_set():
            try:
                ws_obj = self._ws_factory(self._ws_url)
                context = ws_obj if hasattr(ws_obj, "__enter__") else None
                if context is not None:
                    with context as ws:
                        self._consume_ws(ws)
                else:
                    ws = ws_obj
                    try:
                        self._consume_ws(ws)
                    finally:
                        close = getattr(ws, "close", None)
                        if callable(close):
                            close()
                backoff = 1.0
            except Exception:
                if self._ws_stop.wait(backoff + random.uniform(0, 0.5)):
                    return
                backoff = min(backoff * 2, 30.0)

    def _consume_ws(self, ws: Any) -> None:
        self._health.touch()
        while not self._ws_stop.is_set():
            try:
                message = ws.recv()
            except Exception:
                break
            if message is None:
                continue
            self._health.touch()
            payload = self._normalise_ws_message(message)
            if payload is not None:
                self._handle_stream_payload(payload)

    def _normalise_ws_message(self, message: Any) -> dict[str, Any] | None:
        if isinstance(message, bytes):
            try:
                message = message.decode()
            except UnicodeDecodeError:
                return None
        if isinstance(message, str):
            try:
                import json

                data = json.loads(message)
            except Exception:
                return None
            if isinstance(data, dict):
                return data
            return None
        if isinstance(message, dict):
            return message
        return None

    # --- Event helpers -----------------------------------------------

    def next_event(self, timeout: float | None = None) -> dict[str, Any] | None:
        try:
            return self._event_queue.get(timeout=timeout)
        except Empty:
            return None

    def stream_is_healthy(self) -> bool:
        return not self._health.is_stale()
