# SPDX-License-Identifier: MIT
"""Foundational primitives for authenticated REST/WebSocket connectors."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import deque
from typing import Any, AsyncContextManager, Callable, Deque, Dict, Mapping, MutableMapping

import httpx

from domain import Order

from core.utils.metrics import get_metrics_collector

from execution.connectors import ExecutionConnector, OrderError, TransientOrderError


class SlidingWindowRateLimiter:
    """Simple sliding-window rate limiter with blocking semantics."""

    def __init__(
        self,
        *,
        max_requests: int,
        interval_seconds: float,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if max_requests <= 0:
            raise ValueError("max_requests must be positive")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._max_requests = max_requests
        self._interval = interval_seconds
        self._clock = clock or time.monotonic
        self._events: Deque[tuple[float, int]] = deque()
        self._in_window = 0
        self._lock = threading.Lock()

    def acquire(self, weight: int = 1) -> None:
        if weight <= 0:
            return
        backoff = 0.0
        while True:
            with self._lock:
                now = self._clock()
                while self._events and now - self._events[0][0] >= self._interval:
                    _, event_weight = self._events.popleft()
                    self._in_window = max(0, self._in_window - event_weight)
                if self._in_window + weight <= self._max_requests:
                    self._events.append((now, weight))
                    self._in_window += weight
                    return
                oldest_time, _ = self._events[0]
                backoff = max(0.0, self._interval - (now - oldest_time))
            time.sleep(min(backoff, 0.5) if backoff else 0.01)


class RESTWebSocketConnector(ExecutionConnector):
    """Base class combining REST interactions with WebSocket streaming."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        sandbox: bool,
        http_client: httpx.Client | None = None,
        ws_factory: Callable[[str], AsyncContextManager[Any]] | None = None,
        rate_limit: tuple[int, float] = (1200, 60.0),
        max_backoff: float = 30.0,
    ) -> None:
        super().__init__(sandbox=sandbox)
        self.name = name
        self._base_url = base_url.rstrip("/")
        self._logger = logging.getLogger(f"execution.connector.{name}")
        self._http_client = http_client
        self._owns_client = http_client is None
        self._ws_factory = ws_factory
        self._rate_limiter = SlidingWindowRateLimiter(
            max_requests=max(rate_limit[0], 1), interval_seconds=max(rate_limit[1], 0.1)
        )
        self._max_backoff = max(1.0, float(max_backoff))
        self._ws_stop = threading.Event()
        self._ws_thread: threading.Thread | None = None
        self._orders: MutableMapping[str, Order] = {}
        self._idempotency_cache: Dict[str, Order] = {}
        self._lock = threading.Lock()
        self._connected = False
        self._credentials: Mapping[str, str] = {}
        self._metrics = get_metrics_collector()

    # ------------------------------------------------------------------
    # Abstract hooks for subclasses
    def _resolve_credentials(self, credentials: Mapping[str, str] | None) -> Mapping[str, str]:
        raise NotImplementedError

    def _sign_request(
        self,
        method: str,
        path: str,
        *,
        params: Dict[str, Any],
        json_payload: Dict[str, Any] | None,
        headers: Dict[str, str],
    ) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, str]]:
        raise NotImplementedError

    def _order_endpoint(self) -> str:
        raise NotImplementedError

    def _build_place_payload(self, order: Order, idempotency_key: str | None) -> Dict[str, Any]:
        raise NotImplementedError

    def _parse_order(self, payload: Mapping[str, Any], *, original: Order | None = None) -> Order:
        raise NotImplementedError

    def _cancel_endpoint(self, order_id: str) -> tuple[str, Dict[str, Any]]:
        raise NotImplementedError

    def _fetch_endpoint(self, order_id: str) -> tuple[str, Dict[str, Any]]:
        raise NotImplementedError

    def _open_orders_endpoint(self) -> tuple[str, Dict[str, Any]]:
        raise NotImplementedError

    def _positions_endpoint(self) -> tuple[str, Dict[str, Any]]:
        raise NotImplementedError

    def _parse_positions(self, payload: Mapping[str, Any]) -> list[dict]:
        raise NotImplementedError

    def _stream_url(self) -> str | None:
        return None

    def _handle_stream_message(self, payload: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def _default_headers(self) -> Dict[str, str]:
        return {}

    # ------------------------------------------------------------------
    # Public interface
    def connect(self, credentials: Mapping[str, str] | None = None) -> None:  # type: ignore[override]
        if self._connected:
            return
        resolved = self._resolve_credentials(credentials)
        if self._http_client is None:
            self._http_client = httpx.Client(base_url=self._base_url, timeout=httpx.Timeout(10.0, read=30.0))
        self._credentials = resolved
        self._connected = True
        stream_url = self._stream_url()
        if stream_url:
            self._start_stream(stream_url)

    def disconnect(self) -> None:  # type: ignore[override]
        self._ws_stop.set()
        thread = self._ws_thread
        if thread is not None:
            thread.join(timeout=5.0)
            self._ws_thread = None
        if self._owns_client and self._http_client is not None:
            self._http_client.close()
            self._http_client = None
        self._connected = False

    # ------------------------------------------------------------------
    # REST helpers
    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Dict[str, Any] | None = None,
        json_payload: Dict[str, Any] | None = None,
        signed: bool = False,
        weight: int = 1,
    ) -> Mapping[str, Any] | list:
        if not self._connected or self._http_client is None:
            raise RuntimeError("Connector is not connected")
        self._rate_limiter.acquire(weight)
        request_params = dict(params or {})
        request_json = dict(json_payload) if json_payload is not None else None
        headers = self._default_headers()
        if signed:
            request_params, request_json, headers = self._sign_request(
                method,
                path,
                params=request_params,
                json_payload=request_json,
                headers=headers,
            )
        response = self._http_client.request(
            method,
            path,
            params=request_params,
            json=request_json,
            headers=headers,
        )
        if response.status_code == 429:
            raise TransientOrderError("HTTP 429: rate limited")
        if 500 <= response.status_code < 600:
            raise TransientOrderError(f"HTTP {response.status_code}: transient server error")
        if response.is_error:
            raise OrderError(f"HTTP {response.status_code}: {response.text}")
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise OrderError("Invalid JSON response from exchange") from exc
        if not isinstance(payload, (Mapping, list)):
            raise OrderError("Unexpected response payload type")
        return payload

    # ------------------------------------------------------------------
    # ExecutionConnector API
    def place_order(self, order: Order, *, idempotency_key: str | None = None) -> Order:  # type: ignore[override]
        if idempotency_key and idempotency_key in self._idempotency_cache:
            return self._idempotency_cache[idempotency_key]
        payload = self._build_place_payload(order, idempotency_key)
        started_at = time.perf_counter()
        response = self._send_place_order(
            order,
            payload,
            idempotency_key=idempotency_key,
        )
        finished_at = time.perf_counter()
        submitted = self._parse_order(response, original=order)
        self._record_trade_latency(order=submitted, started_at=started_at, finished_at=finished_at)
        with self._lock:
            self._orders[submitted.order_id or ""] = submitted
            if idempotency_key:
                self._idempotency_cache[idempotency_key] = submitted
        return submitted

    def _send_place_order(
        self,
        order: Order,
        payload: Dict[str, Any],
        *,
        idempotency_key: str | None,
    ) -> Mapping[str, Any] | list:
        """Dispatch the place-order request and return the raw payload."""

        return self._request("POST", self._order_endpoint(), params=payload, signed=True)

    def _record_trade_latency(self, *, order: Order, started_at: float, finished_at: float) -> None:
        metrics = getattr(self, "_metrics", None)
        if metrics is None or not metrics.enabled:
            return

        latency_ms = max(0.0, (finished_at - started_at) * 1000.0)
        try:
            order_type = order.order_type.value  # type: ignore[union-attr]
        except AttributeError:
            order_type = str(order.order_type)

        exchange = getattr(self, "name", self.__class__.__name__.lower())
        metrics.observe_trade_latency_ms(
            exchange=exchange,
            adapter=self.__class__.__name__,
            symbol=order.symbol,
            order_type=order_type,
            latency_ms=latency_ms,
        )

    def cancel_order(self, order_id: str) -> bool:  # type: ignore[override]
        path, payload = self._cancel_endpoint(order_id)
        self._request("DELETE", path, params=payload, signed=True)
        with self._lock:
            if order_id in self._orders:
                self._orders[order_id].cancel()
        return True

    def fetch_order(self, order_id: str) -> Order:  # type: ignore[override]
        path, payload = self._fetch_endpoint(order_id)
        response = self._request("GET", path, params=payload, signed=True)
        order = self._parse_order(response)
        with self._lock:
            self._orders[order.order_id or order_id] = order
        return order

    def open_orders(self) -> list[Order]:  # type: ignore[override]
        path, payload = self._open_orders_endpoint()
        response = self._request("GET", path, params=payload, signed=True)
        orders_payload = response.get("orders") if "orders" in response else response
        if isinstance(orders_payload, Mapping):
            values = list(orders_payload.values())
        else:
            values = list(orders_payload) if isinstance(orders_payload, list) else []
        orders: list[Order] = []
        for item in values:
            if not isinstance(item, Mapping):
                continue
            orders.append(self._parse_order(item))
        with self._lock:
            for order in orders:
                if order.order_id:
                    self._orders[order.order_id] = order
        return orders

    def get_positions(self) -> list[dict]:  # type: ignore[override]
        path, payload = self._positions_endpoint()
        response = self._request("GET", path, params=payload, signed=True)
        return self._parse_positions(response)

    # ------------------------------------------------------------------
    # Streaming helpers
    def _start_stream(self, url: str) -> None:
        if self._ws_factory is None:
            self._logger.debug("Streaming disabled because no WebSocket factory was provided")
            return
        self._ws_stop.clear()
        self._ws_thread = threading.Thread(
            target=self._run_stream_loop,
            args=(url,),
            name=f"{self.name}-ws",
            daemon=True,
        )
        self._ws_thread.start()

    def _run_stream_loop(self, url: str) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._consume_stream(url))
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()

    async def _consume_stream(self, url: str) -> None:
        backoff = 1.0
        attempt = 0
        while not self._ws_stop.is_set():
            try:
                async with self._ws_factory(url) as websocket:  # type: ignore[misc]
                    attempt = 0
                    backoff = 1.0
                    while not self._ws_stop.is_set():
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue
                        except Exception as exc:  # pragma: no cover - defensive
                            self._logger.warning("WebSocket receive failed", extra={"error": str(exc)})
                            break
                        if message is None:
                            continue
                        try:
                            payload = json.loads(message)
                        except json.JSONDecodeError:
                            self._logger.debug("Ignoring non-JSON message from stream")
                            continue
                        if isinstance(payload, Mapping):
                            self._handle_stream_message(payload)
            except Exception as exc:
                attempt += 1
                delay = min(self._max_backoff, backoff * (2 ** max(0, attempt - 1)))
                self._logger.warning(
                    "WebSocket connection failed", extra={"attempt": attempt, "delay": delay, "error": str(exc)}
                )
                try:
                    await asyncio.wait_for(asyncio.sleep(delay), timeout=delay)
                except asyncio.TimeoutError:
                    pass
        self._logger.debug("WebSocket loop exiting")


__all__ = ["RESTWebSocketConnector", "SlidingWindowRateLimiter"]
