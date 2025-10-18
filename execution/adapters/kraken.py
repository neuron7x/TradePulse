# SPDX-License-Identifier: MIT
"""Kraken spot trading REST/WebSocket connector."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
from typing import Any, Dict, Iterable, Mapping
from urllib.parse import urlencode

from domain import Order, OrderSide, OrderStatus, OrderType

from .base import RESTWebSocketConnector
from .plugin import (
    AdapterCheckResult,
    AdapterContract,
    AdapterDiagnostic,
    AdapterPlugin,
)

_STATUS_MAP = {
    "pending": OrderStatus.PENDING,
    "open": OrderStatus.OPEN,
    "closed": OrderStatus.FILLED,
    "partial": OrderStatus.PARTIALLY_FILLED,
    "canceled": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    "expired": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED,
}

_TYPE_ALIASES: dict[str, OrderType] = {
    "market": OrderType.MARKET,
    "limit": OrderType.LIMIT,
    "stop-loss": OrderType.STOP,
    "take-profit": OrderType.STOP,
    "stop-limit": OrderType.STOP_LIMIT,
    "take-profit-limit": OrderType.STOP_LIMIT,
}


class KrakenRESTConnector(RESTWebSocketConnector):
    """Authenticated Kraken connector covering core spot order flows."""

    def __init__(
        self,
        *,
        sandbox: bool = True,
        http_client=None,
        ws_factory=None,
    ) -> None:
        base_url = "https://api.kraken.com"
        stream_base = "wss://ws-auth.kraken.com/v2"
        if sandbox:
            base_url = "https://api.sandbox.kraken.com"
            stream_base = "wss://ws-auth.sandbox.kraken.com/v2"
        super().__init__(
            name="kraken",
            base_url=base_url,
            sandbox=sandbox,
            http_client=http_client,
            ws_factory=ws_factory,
            rate_limit=(60, 10.0),
        )
        self._stream_base = stream_base.rstrip("/")
        self._api_key = ""
        self._api_secret = b""
        self._otp: str | None = None

    # ------------------------------------------------------------------
    # RESTWebSocketConnector hooks
    def _resolve_credentials(
        self, credentials: Mapping[str, str] | None
    ) -> Mapping[str, str]:
        supplied = {str(k).lower(): str(v) for k, v in (credentials or {}).items()}
        api_key = supplied.get("api_key") or os.getenv("KRAKEN_API_KEY")
        api_secret = supplied.get("api_secret") or os.getenv("KRAKEN_API_SECRET")
        otp = supplied.get("otp") or os.getenv("KRAKEN_OTP")
        if not api_key or not api_secret:
            raise ValueError("Kraken credentials must provide api_key and api_secret")
        try:
            decoded_secret = base64.b64decode(api_secret)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ValueError("Kraken api_secret must be base64 encoded") from exc
        self._api_key = api_key
        self._api_secret = decoded_secret
        self._otp = otp or None
        payload: Dict[str, str] = {"api_key": api_key, "api_secret": api_secret}
        if otp:
            payload["otp"] = otp
        return payload

    def _default_headers(self) -> Dict[str, str]:
        headers = super()._default_headers()
        if self._api_key:
            headers["API-Key"] = self._api_key
        if self._otp:
            headers["API-OTP"] = self._otp
        return headers

    def _sign_request(
        self,
        method: str,
        path: str,
        *,
        params: Dict[str, Any],
        json_payload: Dict[str, Any] | None,
        headers: Dict[str, str],
    ) -> tuple[Dict[str, Any], Dict[str, Any] | None, Dict[str, str], Any | None]:
        method = method.upper()
        if method != "POST":
            return params, json_payload, headers, None
        nonce = str(int(time.time() * 1000))
        payload = dict(params)
        payload.setdefault("nonce", nonce)
        postdata = urlencode(payload)
        sha256_hash = hashlib.sha256((payload["nonce"] + postdata).encode("utf-8")).digest()
        signature = hmac.new(
            self._api_secret,
            path.encode("utf-8") + sha256_hash,
            hashlib.sha512,
        ).digest()
        headers = dict(headers)
        headers["Content-Type"] = "application/x-www-form-urlencoded; charset=utf-8"
        headers["API-Sign"] = base64.b64encode(signature).decode("utf-8")
        return {}, None, headers, payload

    def _order_endpoint(self) -> str:
        return "/0/private/AddOrder"

    def _build_place_payload(
        self, order: Order, idempotency_key: str | None
    ) -> Dict[str, Any]:
        order_type = order.order_type
        kraken_type = order_type.value.replace("_", "-")
        if order_type is OrderType.STOP:
            kraken_type = "stop-loss"
        elif order_type is OrderType.STOP_LIMIT:
            kraken_type = "stop-limit"
        payload: Dict[str, Any] = {
            "pair": order.symbol.upper(),
            "type": order.side.value,
            "ordertype": kraken_type,
            "volume": f"{order.quantity:.10f}",
        }
        if order.price is not None and order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT}:
            payload["price"] = f"{order.price:.10f}"
        if order.stop_price is not None:
            payload["price2" if order_type is OrderType.STOP_LIMIT else "price"] = (
                f"{order.stop_price:.10f}"
            )
        if idempotency_key:
            payload["userref"] = idempotency_key
        return payload

    def _parse_order(
        self, payload: Mapping[str, Any], *, original: Order | None = None
    ) -> Order:
        data = self._extract_order_payload(payload)
        symbol = str(data.get("pair") or data.get("symbol") or "").upper()
        if not symbol and original is not None:
            symbol = original.symbol
        if not symbol:
            raise ValueError("Order payload missing symbol")
        side_value = str(data.get("type") or data.get("side") or "buy").lower()
        order_type = self._coerce_order_type(str(data.get("ordertype") or data.get("type")), original)
        raw_id = data.get("ordertxid") or data.get("order_id") or data.get("txid")
        order_id = ""
        if isinstance(raw_id, Iterable) and not isinstance(raw_id, (str, bytes)):
            txids_list = [str(item) for item in raw_id if item]
            order_id = txids_list[0] if txids_list else ""
        elif raw_id not in (None, ""):
            order_id = str(raw_id)
        if not order_id and original is not None and original.order_id:
            order_id = original.order_id
        if not order_id:
            raise ValueError("Order payload missing identifier")
        quantity_value = data.get("vol") or data.get("volume")
        quantity = float(quantity_value) if quantity_value not in (None, "") else (
            original.quantity if original else 0.0
        )
        filled_value = data.get("vol_exec") or data.get("filled")
        filled_quantity = (
            float(filled_value)
            if filled_value not in (None, "")
            else (original.filled_quantity if original else 0.0)
        )
        price_value = data.get("price") or data.get("limitprice") or data.get("avg_price")
        price = float(price_value) if price_value not in (None, "") else None
        avg_price_value = data.get("avg_price") or data.get("price")
        average_price = None
        if avg_price_value not in (None, ""):
            try:
                candidate = float(avg_price_value)
            except (TypeError, ValueError):
                candidate = None
            if candidate and candidate > 0:
                average_price = candidate
        status_value = str(data.get("status") or data.get("state") or "open").lower()
        status = _STATUS_MAP.get(status_value, OrderStatus.OPEN)
        if filled_quantity and filled_quantity < quantity and status is OrderStatus.FILLED:
            status = OrderStatus.PARTIALLY_FILLED
        return Order(
            symbol=symbol,
            side=OrderSide(side_value),
            quantity=quantity,
            price=price,
            order_type=order_type,
            order_id=order_id,
            status=status,
            filled_quantity=filled_quantity,
            average_price=average_price,
        )

    def _cancel_endpoint(self, order_id: str) -> tuple[str, Dict[str, Any]]:
        return "/0/private/CancelOrder", {"txid": order_id}

    def _fetch_endpoint(self, order_id: str) -> tuple[str, Dict[str, Any]]:
        return "/0/private/QueryOrders", {"txid": order_id}

    def _open_orders_endpoint(self) -> tuple[str, Dict[str, Any]]:
        return "/0/private/OpenOrders", {}

    def _positions_endpoint(self) -> tuple[str, Dict[str, Any]]:
        return "/0/private/Balance", {}

    def _parse_positions(self, payload: Mapping[str, Any]) -> list[dict]:
        result = payload.get("result") if isinstance(payload.get("result"), Mapping) else payload
        positions: list[dict] = []
        if isinstance(result, Mapping):
            for asset, value in result.items():
                try:
                    qty = float(value)
                except (TypeError, ValueError):
                    continue
                if qty <= 0:
                    continue
                positions.append(
                    {
                        "symbol": str(asset).upper(),
                        "qty": qty,
                        "side": "long" if qty >= 0 else "short",
                        "price": 0.0,
                    }
                )
        return positions

    def _stream_url(self) -> str | None:
        if self._ws_factory is None:
            return None
        token_response = self._request(
            "POST", "/0/private/GetWebSocketsToken", params={}, signed=True
        )
        token_payload = token_response.get("result") if isinstance(token_response, Mapping) else {}
        token = None
        if isinstance(token_payload, Mapping):
            token = token_payload.get("token")
        if not isinstance(token, str) or not token:
            raise ValueError("Kraken user stream token request failed")
        return f"{self._stream_base}?token={token}"

    def _handle_stream_message(self, payload: Mapping[str, Any]) -> None:
        event = str(payload.get("event") or payload.get("type") or "").lower()
        if event not in {"execution", "order", "trade"}:
            return
        order_payload = payload.get("order")
        if isinstance(order_payload, Mapping):
            parsed = self._parse_order(order_payload)
        else:
            parsed = self._parse_order(payload)
        with self._lock:
            if parsed.order_id:
                self._orders[parsed.order_id] = parsed

    # ------------------------------------------------------------------
    # Overrides to align with Kraken API semantics
    def cancel_order(self, order_id: str) -> bool:  # type: ignore[override]
        path, payload = self._cancel_endpoint(order_id)
        response = self._request("POST", path, params=payload, signed=True)
        result = response.get("result") if isinstance(response, Mapping) else {}
        if isinstance(result, Mapping) and int(result.get("count", 1)) <= 0:
            return False
        with self._lock:
            if order_id in self._orders:
                self._orders[order_id].cancel()
        return True

    def fetch_order(self, order_id: str) -> Order:  # type: ignore[override]
        path, payload = self._fetch_endpoint(order_id)
        response = self._request("POST", path, params=payload, signed=True)
        parsed = self._parse_order(response, original=self._orders.get(order_id))
        with self._lock:
            self._orders[parsed.order_id or order_id] = parsed
        return parsed

    def open_orders(self) -> list[Order]:  # type: ignore[override]
        path, payload = self._open_orders_endpoint()
        response = self._request("POST", path, params=payload, signed=True)
        result = response.get("result") if isinstance(response, Mapping) else {}
        orders_payload: list[Mapping[str, Any]] = []
        if isinstance(result, Mapping):
            open_orders = result.get("open")
            if isinstance(open_orders, Mapping):
                for order_id, data in open_orders.items():
                    if not isinstance(data, Mapping):
                        continue
                    enriched = dict(data)
                    enriched.setdefault("ordertxid", order_id)
                    orders_payload.append(enriched)
        orders: list[Order] = []
        for entry in orders_payload:
            orders.append(self._parse_order(entry))
        with self._lock:
            for order in orders:
                if order.order_id:
                    self._orders[order.order_id] = order
        return orders

    def get_positions(self) -> list[dict]:  # type: ignore[override]
        path, payload = self._positions_endpoint()
        response = self._request("POST", path, params=payload, signed=True)
        return self._parse_positions(response)

    # ------------------------------------------------------------------
    # Helpers
    def _extract_order_payload(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if "result" in payload and isinstance(payload["result"], Mapping):
            result = payload["result"]
            if "descr" in result or "txid" in result:
                return result
            if result:
                first_key, first_value = next(iter(result.items()))
                if isinstance(first_value, Mapping):
                    data = dict(first_value)
                    data.setdefault("ordertxid", first_key)
                    return data
        descr = payload.get("descr")
        if isinstance(descr, Mapping):
            merged = dict(payload)
            merged.update(descr)
            return merged
        return payload

    def _coerce_order_type(self, value: str, original: Order | None) -> OrderType:
        raw = value.replace("_", "-").strip().lower()
        mapped = _TYPE_ALIASES.get(raw)
        if mapped is not None:
            return mapped
        try:
            return OrderType(raw.replace("-", "_"))
        except ValueError:
            return original.order_type if original is not None else OrderType.MARKET


def _self_test() -> AdapterDiagnostic:
    checks = []
    try:
        connector = KrakenRESTConnector(sandbox=True)
        checks.append(
            AdapterCheckResult(
                name="instantiate",
                status="passed",
                detail="Connector instantiated with sandbox configuration",
            )
        )
        if not connector.sandbox:
            raise AssertionError("Connector sandbox flag not set")
        checks.append(
            AdapterCheckResult(
                name="sandbox-flag",
                status="passed",
                detail="Sandbox mode enabled by default",
            )
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        checks.append(AdapterCheckResult(name="instantiate", status="failed", detail=str(exc)))
    return AdapterDiagnostic(adapter_id="kraken.spot", checks=tuple(checks))


PLUGIN = AdapterPlugin(
    contract=AdapterContract(
        identifier="kraken.spot",
        name="Kraken Spot",
        provider="Kraken",
        version="1.0.0",
        description="Kraken spot trading connector using REST and authenticated WebSocket APIs.",
        transports={
            "rest": "https://api.kraken.com",
            "websocket": "wss://ws-auth.kraken.com/v2",
        },
        supports_sandbox=True,
        required_credentials=("api_key", "api_secret"),
        optional_credentials=("otp",),
        capabilities={
            "order_types": ["market", "limit", "stop", "stop_limit"],
            "supports_streaming": True,
            "supports_positions": True,
        },
        metadata={
            "sandbox_rest": "https://api.sandbox.kraken.com",
            "sandbox_websocket": "wss://ws-auth.sandbox.kraken.com/v2",
        },
    ),
    factory=KrakenRESTConnector,
    implementation=KrakenRESTConnector,
    self_test=_self_test,
    module=__name__,
)


__all__ = ["KrakenRESTConnector", "PLUGIN"]
