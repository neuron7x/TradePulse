# SPDX-License-Identifier: MIT
"""Production-grade Coinbase Advanced Trade REST/WebSocket connector."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any, Dict, Mapping

from domain import Order, OrderSide, OrderStatus, OrderType

from .base import RESTWebSocketConnector


_STATUS_MAP = {
    "OPEN": OrderStatus.OPEN,
    "PENDING": OrderStatus.OPEN,
    "FILLED": OrderStatus.FILLED,
    "CANCELED": OrderStatus.CANCELLED,
    "CANCELLED": OrderStatus.CANCELLED,
    "EXPIRED": OrderStatus.CANCELLED,
    "FAILED": OrderStatus.REJECTED,
    "REJECTED": OrderStatus.REJECTED,
    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
}

_TYPE_ALIASES: dict[str, OrderType] = {
    "market": OrderType.MARKET,
    "market_market_ioc": OrderType.MARKET,
    "limit": OrderType.LIMIT,
    "limit_limit_gtc": OrderType.LIMIT,
    "limit_limit_gtc_post_only": OrderType.LIMIT,
    "stop": OrderType.STOP,
    "stop_limit": OrderType.STOP_LIMIT,
    "stop_limit_stop_limit_gtc": OrderType.STOP_LIMIT,
}


class CoinbaseRESTConnector(RESTWebSocketConnector):
    """Authenticated connector for Coinbase Advanced Trade API."""

    def __init__(
        self,
        *,
        sandbox: bool = True,
        http_client=None,
        ws_factory=None,
    ) -> None:
        base_url = "https://api-public.sandbox.exchange.coinbase.com" if sandbox else "https://api.coinbase.com"
        if not sandbox:
            base_url = "https://api.coinbase.com"
        api_base = f"{base_url.rstrip('/')}/api/v3/brokerage"
        super().__init__(
            name="coinbase",
            base_url=api_base,
            sandbox=sandbox,
            http_client=http_client,
            ws_factory=ws_factory,
            rate_limit=(120, 1.0),
        )
        self._stream_base = "wss://advanced-trade-ws.coinbase.com"
        if sandbox:
            self._stream_base = "wss://advanced-trade-ws-public.sandbox.exchange.coinbase.com"
        self._api_key = ""
        self._api_secret = ""
        self._passphrase = ""

    # ------------------------------------------------------------------
    def _resolve_credentials(self, credentials: Mapping[str, str] | None) -> Mapping[str, str]:
        supplied = {str(k).lower(): str(v) for k, v in (credentials or {}).items()}
        api_key = supplied.get("api_key") or os.getenv("COINBASE_API_KEY")
        api_secret = supplied.get("api_secret") or os.getenv("COINBASE_API_SECRET")
        passphrase = supplied.get("passphrase") or os.getenv("COINBASE_API_PASSPHRASE")
        if not api_key or not api_secret or not passphrase:
            raise ValueError("Coinbase credentials must provide api_key, api_secret, and passphrase")
        self._api_key = api_key
        self._api_secret = api_secret
        self._passphrase = passphrase
        return {"api_key": api_key, "api_secret": api_secret, "passphrase": passphrase}

    def _default_headers(self) -> Dict[str, str]:
        headers = super()._default_headers()
        if self._api_key:
            headers["CB-ACCESS-KEY"] = self._api_key
            headers["CB-ACCESS-PASSPHRASE"] = self._passphrase
        headers.setdefault("Content-Type", "application/json")
        return headers

    def _sign_request(
        self,
        method: str,
        path: str,
        *,
        params: Dict[str, Any],
        json_payload: Dict[str, Any] | None,
        headers: Dict[str, str],
    ) -> tuple[Dict[str, Any], Dict[str, Any] | None, Dict[str, str]]:
        timestamp = str(int(time.time()))
        body = json.dumps(json_payload or {}) if json_payload is not None else ""
        request_path = path if path.startswith("/") else f"/{path}"
        message = f"{timestamp}{method.upper()}{request_path}{body}"
        secret = base64.b64decode(self._api_secret)
        signature = hmac.new(
            secret,
            message.encode("utf-8"),
            hashlib.sha256,  # lgtm[py/weak-sensitive-data-hashing]
        ).digest()
        headers["CB-ACCESS-TIMESTAMP"] = timestamp
        headers["CB-ACCESS-SIGN"] = base64.b64encode(signature).decode("utf-8")
        return params, json_payload, headers

    def _order_endpoint(self) -> str:
        return "/orders"

    def _build_place_payload(self, order: Order, idempotency_key: str | None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "client_order_id": idempotency_key or order.order_id or "",
            "product_id": order.symbol.replace("_", "-"),
            "side": order.side.value.upper(),
            "order_configuration": {
                "market_market_ioc": None,
            },
        }
        if order.order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT}:
            payload["order_configuration"] = {
                "limit_limit_gtc": {
                    "base_size": f"{order.quantity:.10f}",
                    "limit_price": f"{order.price or 0:.10f}",
                }
            }
        else:
            payload["order_configuration"] = {
                "market_market_ioc": {
                    "base_size": f"{order.quantity:.10f}",
                }
            }
        return payload

    def place_order(self, order: Order, *, idempotency_key: str | None = None) -> Order:  # type: ignore[override]
        if idempotency_key and idempotency_key in self._idempotency_cache:
            return self._idempotency_cache[idempotency_key]
        payload = self._build_place_payload(order, idempotency_key)
        response = self._request("POST", self._order_endpoint(), json_payload=payload, signed=True)
        submitted = self._parse_order(response, original=order)
        with self._lock:
            self._orders[submitted.order_id or ""] = submitted
            if idempotency_key:
                self._idempotency_cache[idempotency_key] = submitted
        return submitted

    def _parse_order(self, payload: Mapping[str, Any], *, original: Order | None = None) -> Order:
        if "order" in payload and isinstance(payload["order"], Mapping):
            payload = payload["order"]
        symbol = str(payload.get("product_id") or (original.symbol if original else ""))
        if not symbol:
            raise ValueError("Order payload missing product identifier")
        side = str(payload.get("side") or (original.side.value if original else "buy")).lower()
        order_type = self._coerce_order_type(
            str(payload.get("order_type") or payload.get("type") or (original.order_type.value if original else "market")),
            original,
        )
        order_id = str(payload.get("order_id") or payload.get("id") or payload.get("orderId") or "")
        if not order_id:
            raise ValueError("Order payload missing identifier")
        size_value = payload.get("size") or payload.get("base_size") or payload.get("filled_size")
        quantity = float(size_value or (original.quantity if original else 0.0))
        filled_value = payload.get("filled_size") or payload.get("executed_value") or 0.0
        try:
            filled = float(filled_value)
        except (TypeError, ValueError):
            filled = 0.0
        price_value = payload.get("price") or payload.get("limit_price") or (original.price if original else None)
        price = float(price_value) if price_value not in (None, "") else None
        avg_price_val = payload.get("average_filled_price") or payload.get("average_price")
        average_price = float(avg_price_val) if avg_price_val not in (None, "") else None
        status_value = str(payload.get("status") or "OPEN").upper()
        status = _STATUS_MAP.get(status_value, OrderStatus.OPEN)
        return Order(
            symbol=symbol,
            side=OrderSide(side),
            quantity=quantity if quantity > 0 else (original.quantity if original else 0.0),
            price=price,
            order_type=order_type,
            order_id=order_id,
            status=status,
            filled_quantity=filled,
            average_price=average_price,
        )

    def _cancel_endpoint(self, order_id: str) -> tuple[str, Dict[str, Any]]:
        return f"/orders/{order_id}", {}

    def _fetch_endpoint(self, order_id: str) -> tuple[str, Dict[str, Any]]:
        return f"/orders/{order_id}", {}

    def _open_orders_endpoint(self) -> tuple[str, Dict[str, Any]]:
        return "/orders/open", {}

    def _positions_endpoint(self) -> tuple[str, Dict[str, Any]]:
        return "/accounts", {}

    def _parse_positions(self, payload: Mapping[str, Any]) -> list[dict]:
        accounts = payload.get("accounts", [])
        positions: list[dict] = []
        for account in accounts or []:
            balance = account.get("available_balance") or {}
            if not isinstance(balance, Mapping):
                continue
            try:
                qty = float(balance.get("value", 0.0))
            except (TypeError, ValueError):
                qty = 0.0
            if not qty:
                continue
            asset = str(account.get("currency", "")).upper()
            positions.append({"symbol": asset, "qty": qty, "side": "long", "price": 0.0})
        return positions

    def _stream_url(self) -> str | None:
        if self._ws_factory is None:
            return None
        return self._stream_base

    def _handle_stream_message(self, payload: Mapping[str, Any]) -> None:
        message_type = str(payload.get("type") or "").lower()
        if message_type not in {"order_update", "orders"}:
            return
        order_payload = payload.get("order")
        if not isinstance(order_payload, Mapping):
            order_payload = payload
        order = self._parse_order(order_payload)
        with self._lock:
            if order.order_id:
                self._orders[order.order_id] = order

    def _coerce_order_type(self, value: str, original: Order | None) -> OrderType:
        raw = value.replace("-", "_").replace(" ", "_").strip().lower()
        mapped = _TYPE_ALIASES.get(raw)
        if mapped is not None:
            return mapped
        try:
            return OrderType(raw)
        except ValueError:
            return original.order_type if original is not None else OrderType.MARKET


__all__ = ["CoinbaseRESTConnector"]
