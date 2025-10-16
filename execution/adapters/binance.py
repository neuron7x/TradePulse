# SPDX-License-Identifier: MIT
"""Production-grade Binance REST/WebSocket connector."""

from __future__ import annotations

import hmac
import hashlib
import os
import time
from typing import Any, Dict, Mapping
from urllib.parse import urlencode

from domain import Order, OrderSide, OrderStatus, OrderType

from .base import RESTWebSocketConnector


_STATUS_MAP = {
    "NEW": OrderStatus.OPEN,
    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
    "FILLED": OrderStatus.FILLED,
    "CANCELED": OrderStatus.CANCELLED,
    "PENDING_CANCEL": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED,
    "EXPIRED": OrderStatus.CANCELLED,
}

_TYPE_ALIASES: dict[str, OrderType] = {
    "market": OrderType.MARKET,
    "limit": OrderType.LIMIT,
    "limit_maker": OrderType.LIMIT,
    "stop": OrderType.STOP,
    "stop_loss": OrderType.STOP,
    "stop_market": OrderType.STOP,
    "take_profit": OrderType.STOP,
    "trailing_stop_market": OrderType.STOP,
    "stop_limit": OrderType.STOP_LIMIT,
    "stop_loss_limit": OrderType.STOP_LIMIT,
    "take_profit_limit": OrderType.STOP_LIMIT,
}


class BinanceRESTConnector(RESTWebSocketConnector):
    """Authenticated connector covering core Binance spot order flows."""

    def __init__(
        self,
        *,
        sandbox: bool = True,
        http_client=None,
        ws_factory=None,
    ) -> None:
        base_url = "https://testnet.binance.vision" if sandbox else "https://api.binance.com"
        stream_base = "wss://stream.binance.com:9443/ws"
        if sandbox:
            stream_base = "wss://testnet.binance.vision/ws"
        super().__init__(
            name="binance", base_url=base_url, sandbox=sandbox, http_client=http_client, ws_factory=ws_factory
        )
        self._stream_base = stream_base.rstrip("/")
        self._api_key = ""
        self._api_secret = ""
        self._listen_key: str | None = None

    # ------------------------------------------------------------------
    # Abstract hook implementations
    def _resolve_credentials(self, credentials: Mapping[str, str] | None) -> Mapping[str, str]:
        supplied = {str(k).lower(): str(v) for k, v in (credentials or {}).items()}
        api_key = supplied.get("api_key") or os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE_KEY")
        api_secret = supplied.get("api_secret") or os.getenv("BINANCE_API_SECRET") or os.getenv("BINANCE_SECRET")
        recv_window = supplied.get("recv_window") or os.getenv("BINANCE_RECV_WINDOW")
        if not api_key or not api_secret:
            raise ValueError("Binance credentials must provide api_key and api_secret")
        self._api_key = api_key
        self._api_secret = api_secret
        payload: Dict[str, str] = {"api_key": api_key, "api_secret": api_secret}
        if recv_window:
            payload["recv_window"] = str(recv_window)
        return payload

    def _default_headers(self) -> Dict[str, str]:
        headers = super()._default_headers()
        if self._api_key:
            headers["X-MBX-APIKEY"] = self._api_key
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
        params = dict(params)
        params.setdefault("timestamp", str(int(time.time() * 1000)))
        recv_window = self._credentials.get("recv_window") if hasattr(self, "_credentials") else None
        if recv_window and "recvWindow" not in params:
            params["recvWindow"] = str(recv_window)
        query = urlencode(sorted(params.items()))
        signature = hmac.new(
            self._api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,  # lgtm[py/weak-sensitive-data-hashing]
        ).hexdigest()
        params["signature"] = signature
        return params, None, headers

    def _order_endpoint(self) -> str:
        return "/api/v3/order"

    def _build_place_payload(self, order: Order, idempotency_key: str | None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "symbol": order.symbol.upper(),
            "side": order.side.value.upper(),
            "type": order.order_type.value.upper(),
            "quantity": f"{order.quantity:.10f}",
        }
        if order.order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT} and order.price is not None:
            payload["price"] = f"{order.price:.10f}"
            payload["timeInForce"] = "GTC"
        if order.stop_price is not None:
            payload["stopPrice"] = f"{order.stop_price:.10f}"
        if idempotency_key:
            payload["newClientOrderId"] = idempotency_key
        return payload

    def _parse_order(self, payload: Mapping[str, Any], *, original: Order | None = None) -> Order:
        symbol = str(payload.get("symbol") or (original.symbol if original else ""))
        if not symbol:
            raise ValueError("Order payload did not include symbol")
        side = str(payload.get("side") or payload.get("S") or (original.side.value if original else "buy")).lower()
        order_type = self._coerce_order_type(
            str(payload.get("type") or payload.get("o") or (original.order_type.value if original else "market")),
            original,
        )
        order_id = str(payload.get("orderId") or payload.get("i") or payload.get("order_id") or "")
        if not order_id:
            raise ValueError("Order payload missing identifier")
        quantity = float(payload.get("origQty") or payload.get("q") or (original.quantity if original else 0.0))
        if quantity <= 0:
            quantity = float(original.quantity if original else 0.0)
        price_value = payload.get("price") or payload.get("p") or (original.price if original else None)
        price = float(price_value) if price_value not in (None, "") else None
        filled = float(payload.get("executedQty") or payload.get("z") or payload.get("filledQty") or 0.0)
        cumulative_quote = payload.get("cummulativeQuoteQty") or payload.get("Z") or payload.get("cumulativeQuoteQty")
        avg_price = payload.get("avgPrice") or payload.get("ap")
        average_price = None
        if avg_price not in (None, ""):
            average_price = float(avg_price)
        elif filled and cumulative_quote not in (None, ""):
            try:
                average_price = float(cumulative_quote) / filled if filled else None
            except (TypeError, ValueError):
                average_price = None
        status_value = str(payload.get("status") or payload.get("X") or "NEW").upper()
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
        symbol = self._lookup_symbol(order_id)
        return self._order_endpoint(), {"symbol": symbol.upper(), "orderId": order_id}

    def _fetch_endpoint(self, order_id: str) -> tuple[str, Dict[str, Any]]:
        symbol = self._lookup_symbol(order_id)
        return self._order_endpoint(), {"symbol": symbol.upper(), "orderId": order_id}

    def _open_orders_endpoint(self) -> tuple[str, Dict[str, Any]]:
        return "/api/v3/openOrders", {}

    def _positions_endpoint(self) -> tuple[str, Dict[str, Any]]:
        return "/api/v3/account", {}

    def _parse_positions(self, payload: Mapping[str, Any]) -> list[dict]:
        balances = payload.get("balances", [])
        positions: list[dict] = []
        for balance in balances or []:
            asset = str(balance.get("asset", "")).upper()
            try:
                qty = float(balance.get("free", 0)) + float(balance.get("locked", 0))
            except (TypeError, ValueError):
                qty = 0.0
            if not qty:
                continue
            positions.append({"symbol": asset, "qty": qty, "side": "long", "price": 0.0})
        return positions

    def _stream_url(self) -> str | None:
        if self._ws_factory is None:
            return None
        response = self._request("POST", "/api/v3/userDataStream", params={}, signed=False)
        listen_key = response.get("listenKey")
        if not isinstance(listen_key, str) or not listen_key:
            raise ValueError("Binance userDataStream did not return listenKey")
        self._listen_key = listen_key
        return f"{self._stream_base}/{listen_key}"

    def _handle_stream_message(self, payload: Mapping[str, Any]) -> None:
        event = str(payload.get("e") or "").lower()
        if event != "executionreport":
            return
        mapped = {
            "symbol": payload.get("s"),
            "orderId": payload.get("i"),
            "side": payload.get("S"),
            "status": payload.get("X"),
            "executedQty": payload.get("z"),
            "cummulativeQuoteQty": payload.get("Z"),
            "price": payload.get("p") or payload.get("L"),
            "avgPrice": payload.get("ap"),
            "origQty": payload.get("q"),
            "type": payload.get("o"),
        }
        order = self._parse_order(mapped)
        with self._lock:
            if order.order_id:
                self._orders[order.order_id] = order

    # ------------------------------------------------------------------
    # Helpers
    def _lookup_symbol(self, order_id: str) -> str:
        with self._lock:
            order = self._orders.get(order_id)
        if order is None:
            raise ValueError(f"Symbol unknown for order_id={order_id}")
        return order.symbol

    def _coerce_order_type(self, value: str, original: Order | None) -> OrderType:
        raw = value.replace("-", "_").replace(" ", "_").strip().lower()
        mapped = _TYPE_ALIASES.get(raw)
        if mapped is not None:
            return mapped
        try:
            return OrderType(raw)
        except ValueError:
            return original.order_type if original is not None else OrderType.MARKET


__all__ = ["BinanceRESTConnector"]
