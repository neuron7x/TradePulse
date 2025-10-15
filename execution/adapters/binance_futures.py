"""Binance USD-M futures connector leveraging the REST/WebSocket base class."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from domain import Order, OrderStatus

from .binance import BinanceRESTConnector, _STATUS_MAP


class BinanceFuturesRESTConnector(BinanceRESTConnector):
    """Extends the spot connector with futures-specific endpoints."""

    def __init__(self, *, sandbox: bool = True, hedge_mode: bool = False, http_client=None, ws_factory=None) -> None:
        super().__init__(sandbox=sandbox, http_client=http_client, ws_factory=ws_factory)
        self.name = "binance-futures"
        self._base_url = "https://testnet.binancefuture.com" if sandbox else "https://fapi.binance.com"
        self._stream_base = "wss://stream.binancefuture.com/ws" if sandbox else "wss://fstream.binance.com/ws"
        self._hedge_mode = hedge_mode

    # ------------------------------------------------------------------
    # Endpoint overrides
    def _order_endpoint(self) -> str:
        return "/fapi/v1/order"

    def _open_orders_endpoint(self) -> tuple[str, Dict[str, Any]]:
        return "/fapi/v1/openOrders", {}

    def _cancel_endpoint(self, order_id: str) -> tuple[str, Dict[str, Any]]:  # type: ignore[override]
        symbol = self._lookup_symbol(order_id)
        return self._order_endpoint(), {"symbol": symbol.upper(), "orderId": order_id}

    def _fetch_endpoint(self, order_id: str) -> tuple[str, Dict[str, Any]]:  # type: ignore[override]
        symbol = self._lookup_symbol(order_id)
        return self._order_endpoint(), {"symbol": symbol.upper(), "orderId": order_id}

    def _positions_endpoint(self) -> tuple[str, Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if self._hedge_mode:
            params["dualSidePosition"] = "true"
        return "/fapi/v2/positionRisk", params

    def _stream_url(self) -> str | None:
        response = self._request("POST", "/fapi/v1/listenKey", params={}, signed=False)
        listen_key = response.get("listenKey")
        if not isinstance(listen_key, str) or not listen_key:
            raise ValueError("Failed to negotiate Binance futures listen key")
        self._listen_key = listen_key
        return f"{self._stream_base}/{listen_key}"

    # ------------------------------------------------------------------
    # Payload adjustments
    def _build_place_payload(self, order: Order, idempotency_key: str | None) -> Dict[str, Any]:  # type: ignore[override]
        payload = super()._build_place_payload(order, idempotency_key)
        payload.setdefault("newOrderRespType", "RESULT")
        return payload

    def _parse_positions(self, payload: Mapping[str, Any]) -> list[dict]:  # type: ignore[override]
        positions: list[dict] = []
        for position in payload or []:
            try:
                quantity = float(position.get("positionAmt", 0))
            except (TypeError, ValueError):
                quantity = 0.0
            if not quantity:
                continue
            entry_price = float(position.get("entryPrice", 0) or 0)
            positions.append(
                {
                    "symbol": str(position.get("symbol", "")).upper(),
                    "qty": quantity,
                    "side": "long" if quantity >= 0 else "short",
                    "price": entry_price,
                }
            )
        return positions

    def _parse_order(self, payload: Mapping[str, Any], *, original: Order | None = None) -> Order:  # type: ignore[override]
        order = super()._parse_order(payload, original=original)
        status_value = str(payload.get("status") or "NEW").upper()
        order.status = _STATUS_MAP.get(status_value, OrderStatus.OPEN)
        return order
