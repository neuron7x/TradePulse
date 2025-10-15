"""FastAPI application exposing operational TradePulse system primitives."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from application.system import TradePulseSystem
from application.trading import order_to_dto
from domain import Order, OrderSide, OrderType, Position


def _read_version() -> str:
    version_file = Path(__file__).resolve().parents[2] / "VERSION"
    try:
        raw = version_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "0.0.0"
    return raw or "0.0.0"


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class StatusResponse(BaseModel):
    """Service health metadata exposed via the REST API."""

    status: str = Field(..., description="Operational state of the system.")
    uptime_seconds: float = Field(..., ge=0.0, description="Service uptime in seconds.")
    version: str = Field(..., min_length=1, description="Semantic version of the deployment.")


class PositionSnapshot(BaseModel):
    """Normalised representation of a trading position."""

    symbol: str = Field(..., min_length=1)
    quantity: float = Field(..., description="Signed quantity representing exposure.")
    entry_price: float | None = Field(
        default=None, description="Average entry price where available."
    )
    current_price: float | None = Field(
        default=None, description="Latest mark price associated with the position."
    )
    unrealized_pnl: float | None = Field(
        default=None, description="Unrealised profit and loss inferred from venue data."
    )

    model_config = ConfigDict(populate_by_name=True)


class PositionsResponse(BaseModel):
    """Envelope containing active positions across venues."""

    positions: Sequence[PositionSnapshot] = Field(default_factory=list)


class OrderRequest(BaseModel):
    """Request payload describing an order submission."""

    symbol: str = Field(..., min_length=1)
    side: OrderSide
    order_type: OrderType = Field(default=OrderType.MARKET)
    quantity: float = Field(..., gt=0.0)
    price: float | None = Field(default=None, gt=0.0)
    venue: str | None = Field(
        default=None,
        description="Target venue identifier. Defaults to the first configured connector.",
    )
    client_order_id: str | None = Field(
        default=None,
        max_length=128,
        description="Optional idempotency key forwarded to the execution adapter.",
    )

    @model_validator(mode="before")
    def _validate_price(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            raw_type = data.get("order_type")
            price = data.get("price")
            try:
                order_type = OrderType(raw_type)
            except Exception:
                order_type = None
            priced_types = {OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT}
            if order_type in priced_types and price is None:
                raise ValueError("price must be supplied for priced order types")
        return data


class OrderResponse(BaseModel):
    """Subset of order lifecycle data returned to REST clients."""

    order_id: str | None = Field(default=None, description="Venue supplied order identifier.")
    status: str = Field(..., description="Latest known status of the order.")
    filled_quantity: float = Field(..., ge=0.0)
    average_price: float | None = Field(default=None, ge=0.0)


@dataclass(slots=True)
class _PositionNormaliser:
    """Translate heterogeneous connector payloads into :class:`PositionSnapshot`."""

    def normalise(self, payload: Mapping[str, Any]) -> PositionSnapshot:
        symbol = str(
            payload.get("symbol")
            or payload.get("instrument")
            or payload.get("pair")
            or payload.get("asset")
            or "unknown"
        )

        quantity = _coerce_float(
            payload.get("quantity")
            or payload.get("qty")
            or payload.get("size")
            or payload.get("positionAmt")
            or payload.get("position")
        ) or 0.0

        entry_price = _coerce_float(
            payload.get("entry_price")
            or payload.get("entryPrice")
            or payload.get("avg_entry_price")
            or payload.get("average_price")
            or payload.get("avgPrice")
        )

        current_price = _coerce_float(
            payload.get("current_price")
            or payload.get("currentPrice")
            or payload.get("mark_price")
            or payload.get("markPrice")
            or payload.get("market_price")
            or payload.get("price")
        )

        unrealized = _coerce_float(
            payload.get("unrealized_pnl")
            or payload.get("unrealizedPnl")
            or payload.get("unrealizedPnL")
            or payload.get("pnl")
        )

        if unrealized is None and entry_price is not None and current_price is not None:
            try:
                position = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=abs(entry_price) if quantity else 0.0,
                )
                position.mark_to_market(abs(current_price))
                unrealized = position.unrealized_pnl
                entry_price = position.entry_price or entry_price
                current_price = position.current_price or current_price
            except ValueError:
                # Fall back to raw values when payload cannot be coerced into Position.
                pass

        return PositionSnapshot(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pnl=unrealized,
        )


class SystemAccess:
    """Coordinator exposing TradePulse system capabilities over REST."""

    def __init__(self, system: TradePulseSystem) -> None:
        self._system = system
        self._started_at = datetime.now(timezone.utc)
        self._lock = asyncio.Lock()
        self._normaliser = _PositionNormaliser()
        self._connected: set[str] = set()
        self._version = _read_version()

    @property
    def version(self) -> str:
        return self._version

    def status(self) -> StatusResponse:
        uptime = max(
            0.0,
            (datetime.now(timezone.utc) - self._started_at).total_seconds(),
        )
        kill_switch = self._system.risk_manager.kill_switch.is_triggered()
        status_value = "halted" if kill_switch else "running"
        return StatusResponse(status=status_value, uptime_seconds=uptime, version=self._version)

    def _default_venue(self) -> str:
        names = self._system.connector_names
        if not names:
            raise RuntimeError("TradePulseSystem has no configured execution venues")
        return names[0]

    def _ensure_connected(self, venue: str) -> None:
        key = venue.lower()
        if key in self._connected:
            return
        connector = self._system.get_connector(venue)
        credentials = self._system.connector_credentials(venue)
        connector.connect(credentials)
        self._connected.add(key)

    async def list_positions(self) -> PositionsResponse:
        snapshots: list[PositionSnapshot] = []
        for venue in self._system.connector_names:
            connector = self._system.get_connector(venue)
            self._ensure_connected(venue)
            try:
                raw_positions = connector.get_positions()
            except Exception as exc:  # pragma: no cover - defensive
                raise HTTPException(
                    status.HTTP_502_BAD_GATEWAY,
                    detail=f"Failed to fetch positions from venue '{venue}': {exc}",
                ) from exc
            for payload in raw_positions:
                try:
                    snapshot = self._normaliser.normalise(payload)
                except Exception as exc:  # pragma: no cover - defensive
                    raise HTTPException(
                        status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to normalise position payload from '{venue}': {exc}",
                    ) from exc
                snapshots.append(snapshot)
        return PositionsResponse(positions=snapshots)

    async def place_order(self, request: OrderRequest) -> OrderResponse:
        venue = request.venue or self._default_venue()
        connector = self._system.get_connector(venue)
        async with self._lock:
            self._ensure_connected(venue)
            risk_price = request.price or 1.0
            if risk_price <= 0:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="reference price must be positive",
                )
            try:
                self._system.risk_manager.validate_order(
                    request.symbol,
                    request.side.value,
                    request.quantity,
                    risk_price,
                )
            except Exception as exc:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(exc),
                ) from exc

            order = Order(
                symbol=request.symbol,
                side=request.side,
                quantity=request.quantity,
                price=request.price,
                order_type=request.order_type,
            )

            placed: Order
            try:
                placed = connector.place_order(order, idempotency_key=request.client_order_id)
            except Exception as exc:
                raise HTTPException(
                    status.HTTP_502_BAD_GATEWAY,
                    detail=f"Connector rejected order submission: {exc}",
                ) from exc

            if placed.filled_quantity > 0:
                fill_price = placed.average_price or placed.price
                if fill_price:
                    try:
                        self._system.risk_manager.register_fill(
                            placed.symbol,
                            placed.side.value,
                            placed.filled_quantity,
                            float(fill_price),
                        )
                    except Exception:  # pragma: no cover - defensive
                        # Risk updates are best-effort; surfacing the order response takes precedence.
                        pass

            dto = order_to_dto(placed)
            return OrderResponse(
                order_id=dto.get("order_id"),
                status=str(dto.get("status", placed.status.value)),
                filled_quantity=float(dto.get("filled_quantity", placed.filled_quantity)),
                average_price=_coerce_float(dto.get("average_price")),
            )


def _get_access(request: Request) -> SystemAccess:
    access = getattr(request.app.state, "system_access", None)
    if not isinstance(access, SystemAccess):  # pragma: no cover - defensive
        raise RuntimeError("SystemAccess not initialised on application state")
    return access


def create_system_app(system: TradePulseSystem) -> FastAPI:
    """Instantiate a FastAPI app exposing TradePulse system endpoints."""

    access = SystemAccess(system)
    app = FastAPI(title="TradePulse System API", version=access.version)
    app.state.system_access = access

    @app.get("/api/v1/status", response_model=StatusResponse, tags=["system"])
    async def read_status(access: SystemAccess = Depends(_get_access)) -> StatusResponse:
        return access.status()

    @app.get("/api/v1/positions", response_model=PositionsResponse, tags=["system"])
    async def read_positions(
        access: SystemAccess = Depends(_get_access),
    ) -> PositionsResponse:
        return await access.list_positions()

    @app.post(
        "/api/v1/orders",
        response_model=OrderResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["system"],
    )
    async def submit_order(
        payload: OrderRequest,
        access: SystemAccess = Depends(_get_access),
    ) -> OrderResponse:
        return await access.place_order(payload)

    return app


__all__ = [
    "SystemAccess",
    "create_system_app",
    "OrderRequest",
    "OrderResponse",
    "PositionsResponse",
    "PositionSnapshot",
    "StatusResponse",
]

