"""Bybit unified margin connector built on the REST/WebSocket abstraction."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from typing import Any, Dict, Mapping

from domain import Order, OrderSide, OrderStatus, OrderType

from .base import RESTWebSocketConnector


_STATUS_MAP = {
    "Created": OrderStatus.OPEN,
    "New": OrderStatus.OPEN,
    "PartiallyFilled": OrderStatus.PARTIALLY_FILLED,
    "Filled": OrderStatus.FILLED,
    "Cancelled": OrderStatus.CANCELLED,
    "Rejected": OrderStatus.REJECTED,
}


class BybitRESTConnector(RESTWebSocketConnector):
    """Implements order lifecycle operations against Bybit's v5 API."""

    def __init__(self, *, sandbox: bool = True, category: str = "linear", http_client=None, ws_factory=None) -> None:
        base_url = "https://api-testnet.bybit.com" if sandbox else "https://api.bybit.com"
        super().__init__(
            name="bybit",
            base_url=base_url,
            sandbox=sandbox,
            http_client=http_client,
            ws_factory=ws_factory,
            rate_limit=(50, 1.0),
        )
        self._category = category
        self._api_key = ""
        self._api_secret = ""

    def _resolve_credentials(self, credentials: Mapping[str, str] | None) -> Mapping[str, str]:
        supplied = {str(k).lower(): str(v) for k, v in (credentials or {}).items()}
        api_key = supplied.get("api_key") or os.getenv("BYBIT_API_KEY")
        api_secret = supplied.get("api_secret") or os.getenv("BYBIT_API_SECRET")
        recv_window = supplied.get("recv_window") or os.getenv("BYBIT_RECV_WINDOW") or "5000"
        if not api_key or not api_secret:
            raise ValueError("Bybit credentials must provide api_key and api_secret")
        self._api_key = api_key
        self._api_secret = api_secret
        return {"api_key": api_key, "api_secret": api_secret, "recv_window": recv_window}

    def _default_headers(self) -> Dict[str, str]:
        headers = super()._default_headers()
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
        timestamp = str(int(time.time() * 1000))
        recv_window = str(self._credentials.get("recv_window", "5000"))
        body = json.dumps(json_payload or {}, separators=(",", ":")) if json_payload else ""
        query = ""
        if params:
            query = "&".join(f"{key}={value}" for key, value in sorted(params.items()))
        payload = f"{timestamp}{self._api_key}{recv_window}{query}{body}"
        signature = hmac.new(self._api_secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
        headers.update(
            {
                "X-BAPI-API-KEY": self._api_key,
                "X-BAPI-SIGN": signature,
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-RECV-WINDOW": recv_window,
            }
        )
        return params, json_payload, headers

    # ------------------------------------------------------------------
    # REST endpoints
    def _order_endpoint(self) -> str:
        return "/v5/order/create"

    def _cancel_endpoint(self, order_id: str) -> tuple[str, Dict[str, Any]]:
        symbol = self._lookup_symbol(order_id)
        payload = {"category": self._category, "symbol": symbol, "orderId": order_id}
        return "/v5/order/cancel", payload

    def _fetch_endpoint(self, order_id: str) -> tuple[str, Dict[str, Any]]:
        symbol = self._lookup_symbol(order_id)
        params = {"category": self._category, "symbol": symbol, "orderId": order_id}
        return "/v5/order/realtime", params

    def _open_orders_endpoint(self) -> tuple[str, Dict[str, Any]]:
        return "/v5/order/realtime", {"category": self._category}

    def _positions_endpoint(self) -> tuple[str, Dict[str, Any]]:
        return "/v5/position/list", {"category": self._category}

    # ------------------------------------------------------------------
    # Payload transforms
    def _build_place_payload(self, order: Order, idempotency_key: str | None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "category": self._category,
            "symbol": order.symbol.upper(),
            "side": "Buy" if order.side == OrderSide.BUY else "Sell",
            "orderType": order.order_type.value.capitalize(),
            "qty": f"{order.quantity:.8f}",
        }
        if order.order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT} and order.price is not None:
            payload["price"] = f"{order.price:.8f}"
            payload["timeInForce"] = "GTC"
        if order.stop_price is not None:
            payload["triggerPrice"] = f"{order.stop_price:.8f}"
        if idempotency_key:
            payload["orderLinkId"] = idempotency_key
        return payload

    def _parse_order(self, payload: Mapping[str, Any], *, original: Order | None = None) -> Order:
        order_id = str(payload.get("orderId") or payload.get("orderId") or payload.get("orderID") or "")
        if not order_id and original is not None:
            order_id = original.order_id
        symbol = str(payload.get("symbol") or (original.symbol if original else ""))
        side = OrderSide(str(payload.get("side") or (original.side.value if original else "Buy")).lower())
        order_type_value = str(payload.get("orderType") or (original.order_type.value if original else "market"))
        try:
            order_type = OrderType(order_type_value.lower())
        except ValueError:
            order_type = original.order_type if original else OrderType.MARKET
        quantity = float(payload.get("qty") or payload.get("cumExecQty") or (original.quantity if original else 0))
        price = payload.get("price") or payload.get("avgPrice") or (original.price if original else None)
        avg_price = payload.get("avgPrice")
        average_price = float(avg_price) if avg_price not in (None, "") else None
        status_value = str(payload.get("orderStatus") or payload.get("orderStatus") or payload.get("status") or "New")
        status = _STATUS_MAP.get(status_value, OrderStatus.OPEN)
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=float(price) if price not in (None, "") else None,
            order_type=order_type,
            order_id=order_id,
            status=status,
            average_price=average_price,
        )
        return order

    def _parse_positions(self, payload: Mapping[str, Any]) -> list[dict]:
        positions: list[dict] = []
        for entry in payload.get("list", []):
            try:
                qty = float(entry.get("size", 0))
            except (TypeError, ValueError):
                qty = 0.0
            if not qty:
                continue
            entry_price = float(entry.get("avgPrice", 0) or 0)
            positions.append(
                {
                    "symbol": str(entry.get("symbol", "")).upper(),
                    "qty": qty,
                    "side": entry.get("side", "Buy").lower(),
                    "price": entry_price,
                }
            )
        return positions

    def _handle_stream_message(self, payload: Mapping[str, Any]) -> None:
        if "topic" not in payload:
            return
        data = payload.get("data") or []
        if not data:
            return
        order_payload = data[0]
        mapped = {
            "orderId": order_payload.get("orderId"),
            "symbol": order_payload.get("symbol"),
            "side": order_payload.get("side"),
            "orderType": order_payload.get("orderType"),
            "qty": order_payload.get("cumExecQty"),
            "orderStatus": order_payload.get("orderStatus"),
            "avgPrice": order_payload.get("avgPrice"),
        }
        order = self._parse_order(mapped)
        with self._lock:
            if order.order_id:
                self._orders[order.order_id] = order

    def _stream_url(self) -> str | None:
        if self._ws_factory is None:
            return None
        # Bybit unified margin streams share a single endpoint; listen key is optional
        suffix = "unifiedPublic" if self._category != "option" else "option"
        return f"wss://stream-testnet.bybit.com/v5/public/{suffix}" if self.sandbox else f"wss://stream.bybit.com/v5/public/{suffix}"

    def _lookup_symbol(self, order_id: str) -> str:
        with self._lock:
            order = self._orders.get(order_id)
        if order is None:
            raise ValueError(f"Symbol unknown for order_id={order_id}")
        return order.symbol
