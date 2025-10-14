"""FastAPI application exposing online feature computation and forecasting."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable
from weakref import WeakKeyDictionary

import numpy as np
import pandas as pd
from aiolimiter import AsyncLimiter
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from analytics.signals.pipeline import FeaturePipelineConfig, SignalFeaturePipeline
from application.trading import signal_to_dto
from domain import Signal, SignalAction
from execution.risk import RiskLimits, RiskManager
from src.admin.remote_control import TokenAuthenticator, create_remote_control_router
from src.audit.audit_logger import AuditLogger
from src.risk.risk_manager import RiskManagerFacade


@dataclass(slots=True)
class _CacheEntry:
    """In-memory TTL cache entry."""

    payload: BaseModel
    expires_at: datetime
    etag: str


class TTLCache:
    """A tiny in-memory cache with TTL semantics for small payloads."""

    def __init__(self, ttl_seconds: int = 30, max_entries: int = 256) -> None:
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._entries: dict[str, _CacheEntry] = {}
        self._lock = asyncio.Lock()

    @property
    def ttl_seconds(self) -> int:
        return self._ttl

    async def get(self, key: str) -> _CacheEntry | None:
        async with self._lock:
            entry = self._entries.get(key)
            now = datetime.now(timezone.utc)
            if entry is None:
                return None
            if entry.expires_at <= now:
                del self._entries[key]
                return None
            return entry

    async def set(self, key: str, payload: BaseModel, etag: str) -> None:
        async with self._lock:
            if len(self._entries) >= self._max_entries:
                # Drop the stalest entry deterministically (smallest expiry).
                oldest_key = min(self._entries, key=lambda item: self._entries[item].expires_at)
                self._entries.pop(oldest_key, None)
            expires = datetime.now(timezone.utc) + timedelta(seconds=self._ttl)
            self._entries[key] = _CacheEntry(payload=payload, expires_at=expires, etag=etag)


def _ensure_timezone(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


class MarketBar(BaseModel):
    """Representation of a single OHLCV bar for online inference."""

    timestamp: datetime = Field(..., description="Timestamp of the bar in ISO 8601 format.")
    open: float | None = Field(None, description="Opening price for the interval.")
    high: float = Field(..., description="High price for the interval.")
    low: float = Field(..., description="Low price for the interval.")
    close: float = Field(..., description="Close price for the interval.")
    volume: float | None = Field(
        None, description="Traded volume for the bar. Optional for illiquid venues."
    )
    bid_volume: float | None = Field(
        default=None,
        alias="bidVolume",
        description="Bid-side queue volume for microstructure features.",
    )
    ask_volume: float | None = Field(
        default=None,
        alias="askVolume",
        description="Ask-side queue volume for microstructure features.",
    )
    signed_volume: float | None = Field(
        default=None,
        alias="signedVolume",
        description="Signed volume (buy-sell imbalance).",
    )

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    @model_validator(mode="after")
    def _normalise_timezone(self) -> "MarketBar":
        object.__setattr__(self, "timestamp", _ensure_timezone(self.timestamp))
        return self

    def as_record(self) -> dict[str, Any]:
        record = self.model_dump(by_alias=False, exclude_none=True)
        record["timestamp"] = self.timestamp
        return record


class FeatureRequest(BaseModel):
    """Payload describing the series to transform into features."""

    symbol: str = Field(..., min_length=1, description="Instrument identifier.")
    bars: list[MarketBar] = Field(..., min_length=1, description="Ordered price bars.")

    def to_frame(self) -> pd.DataFrame:
        records = [bar.as_record() for bar in self.bars]
        frame = pd.DataFrame.from_records(records)
        frame.sort_values("timestamp", inplace=True)
        frame.set_index("timestamp", inplace=True)
        frame.index = pd.to_datetime(frame.index, utc=True)
        return frame


class FeatureResponse(BaseModel):
    """Response containing the most recent feature snapshot."""

    symbol: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    features: dict[str, float]

    model_config = ConfigDict(populate_by_name=True)


class PredictionRequest(FeatureRequest):
    """Prediction request payload, optionally specifying a forecast horizon."""

    horizon_seconds: int = Field(
        300,
        ge=60,
        le=3600,
        description="Prediction horizon in seconds for contextual metadata.",
    )


class PredictionResponse(BaseModel):
    """Response representing the generated trading signal."""

    symbol: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    horizon_seconds: int
    score: float = Field(..., description="Composite alpha score driving the action.")
    signal: dict[str, Any]

    model_config = ConfigDict(populate_by_name=True)


class OnlineSignalForecaster:
    """Wraps the feature pipeline and lightweight heuristics for live inference."""

    def __init__(self, pipeline: SignalFeaturePipeline | None = None) -> None:
        self._pipeline = pipeline or SignalFeaturePipeline(FeaturePipelineConfig())

    def compute_features(self, payload: FeatureRequest) -> pd.DataFrame:
        frame = payload.to_frame()
        features = self._pipeline.transform(frame)
        return features

    def latest_feature_vector(self, features: pd.DataFrame) -> pd.Series:
        if features.empty:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No features computed")
        latest = features.iloc[-1].dropna()
        if latest.empty:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Insufficient data to derive features",
            )
        return latest

    def derive_signal(self, symbol: str, latest: pd.Series, horizon_seconds: int) -> tuple[Signal, float]:
        # Compute a simple composite alpha score using a handful of stable metrics.
        macd = float(latest.get("macd", 0.0))
        rsi = float(latest.get("rsi", 50.0))
        ret_1 = float(latest.get("return_1", 0.0))
        volatility_20 = float(latest.get("volatility_20", 0.0))
        queue_imbalance = float(latest.get("queue_imbalance", 0.0))

        score = 0.0
        score += np.tanh(macd) * 0.4
        score += ((rsi - 50.0) / 50.0) * 0.3
        score += np.tanh(ret_1 * 100.0) * 0.2
        score += np.tanh(queue_imbalance) * 0.1
        score -= abs(volatility_20) * 0.05

        threshold = 0.15
        if score > threshold:
            action = SignalAction.BUY
        elif score < -threshold:
            action = SignalAction.SELL
        else:
            action = SignalAction.HOLD

        confidence = float(min(1.0, max(0.0, abs(score) / 0.75)))
        signal = Signal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            rationale=(
                "Composite heuristic based on MACD, RSI, return momentum and book imbalance"
            ),
            metadata={
                "score": score,
                "horizon_seconds": horizon_seconds,
            },
        )
        return signal, score


def _hash_payload(prefix: str, payload: BaseModel) -> str:
    data = json.dumps(payload.model_dump(mode="json"), sort_keys=True, default=str)
    digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def create_app(
    *,
    rate_limiter: AsyncLimiter | None = None,
    cache: TTLCache | None = None,
    forecaster_factory: Callable[[], OnlineSignalForecaster] | None = None,
    admin_token: str | None = None,
    audit_secret: str | None = None,
) -> FastAPI:
    """Build the FastAPI application with configured dependencies."""

    limiter = rate_limiter or AsyncLimiter(max_rate=60, time_period=60)
    ttl_cache = cache or TTLCache(ttl_seconds=30, max_entries=512)
    forecaster_provider = forecaster_factory or (lambda: OnlineSignalForecaster())
    forecaster = forecaster_provider()

    resolved_admin_token = admin_token or os.environ.get("TRADEPULSE_ADMIN_TOKEN")
    resolved_audit_secret = audit_secret or os.environ.get("TRADEPULSE_AUDIT_SECRET")
    missing = []
    if not resolved_admin_token:
        missing.append("TRADEPULSE_ADMIN_TOKEN")
    if not resolved_audit_secret:
        missing.append("TRADEPULSE_AUDIT_SECRET")
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "Missing required secret(s): {}. Provide them via create_app parameters or environment variables.".format(
                joined
            )
        )

    risk_manager_facade = RiskManagerFacade(RiskManager(RiskLimits()))
    audit_logger = AuditLogger(secret=resolved_audit_secret)
    authenticator = TokenAuthenticator(token=resolved_admin_token)

    limiter_rate = limiter.max_rate
    limiter_period = limiter.time_period
    loop_limiters: WeakKeyDictionary[asyncio.AbstractEventLoop, AsyncLimiter] = WeakKeyDictionary()

    app = FastAPI(
        title="TradePulse Online Inference API",
        description=(
            "Production-ready endpoints for computing feature vectors and generating "
            "lightweight trading signals from streaming market data."
        ),
        version="0.1.0",
        contact={
            "name": "TradePulse Platform Team",
            "url": "https://github.com/neuron7x/TradePulse",
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT",
        },
        openapi_tags=[
            {"name": "health", "description": "Operational endpoints"},
            {"name": "features", "description": "Feature engineering APIs"},
            {"name": "predictions", "description": "Signal forecasting APIs"},
        ],
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"]
    )

    app.include_router(create_remote_control_router(risk_manager_facade, audit_logger, authenticator))
    app.state.risk_manager = risk_manager_facade.risk_manager
    app.state.audit_logger = audit_logger

    async def enforce_rate_limit() -> None:
        loop = asyncio.get_running_loop()
        loop_limiter = loop_limiters.get(loop)
        if loop_limiter is None:
            loop_limiter = AsyncLimiter(max_rate=limiter_rate, time_period=limiter_period)
            loop_limiters[loop] = loop_limiter
        async with loop_limiter:
            return None

    def get_forecaster() -> OnlineSignalForecaster:
        return forecaster

    @app.middleware("http")
    async def add_cache_headers(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        response = await call_next(request)
        response.headers.setdefault("Cache-Control", f"public, max-age={ttl_cache.ttl_seconds}")
        return response

    @app.get("/health", tags=["health"], summary="Health probe")
    async def health_check() -> dict[str, str]:
        return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

    @app.post(
        "/features",
        response_model=FeatureResponse,
        tags=["features"],
        summary="Generate the latest engineered feature vector",
    )
    async def compute_features(
        payload: FeatureRequest,
        response: Response,
        _: None = Depends(enforce_rate_limit),
        predictor: OnlineSignalForecaster = Depends(get_forecaster),
    ) -> FeatureResponse:
        cache_key = _hash_payload("features", payload)
        cached = await ttl_cache.get(cache_key)
        if cached is not None:
            response.headers["X-Cache-Status"] = "hit"
            response.headers["ETag"] = cached.etag
            return cached.payload  # type: ignore[return-value]

        features = predictor.compute_features(payload)
        latest = predictor.latest_feature_vector(features)
        feature_dict = {k: float(v) for k, v in latest.items()}
        body = FeatureResponse(symbol=payload.symbol, features=feature_dict)
        etag = hashlib.sha256(json.dumps(feature_dict, sort_keys=True).encode("utf-8")).hexdigest()
        await ttl_cache.set(cache_key, body, etag)
        response.headers["X-Cache-Status"] = "miss"
        response.headers["ETag"] = etag
        return body

    @app.post(
        "/predictions",
        response_model=PredictionResponse,
        tags=["predictions"],
        summary="Produce a trading signal for the latest bar",
    )
    async def generate_prediction(
        payload: PredictionRequest,
        response: Response,
        _: None = Depends(enforce_rate_limit),
        predictor: OnlineSignalForecaster = Depends(get_forecaster),
    ) -> PredictionResponse:
        cache_key = _hash_payload("predictions", payload)
        cached = await ttl_cache.get(cache_key)
        if cached is not None:
            response.headers["X-Cache-Status"] = "hit"
            response.headers["ETag"] = cached.etag
            return cached.payload  # type: ignore[return-value]

        features = predictor.compute_features(payload)
        latest = predictor.latest_feature_vector(features)
        signal, score = predictor.derive_signal(payload.symbol, latest, payload.horizon_seconds)
        body = PredictionResponse(
            symbol=payload.symbol,
            horizon_seconds=payload.horizon_seconds,
            score=score,
            signal=signal_to_dto(signal),
        )
        etag = hashlib.sha256(json.dumps(body.model_dump(), sort_keys=True, default=str).encode("utf-8")).hexdigest()
        await ttl_cache.set(cache_key, body, etag)
        response.headers["X-Cache-Status"] = "miss"
        response.headers["ETag"] = etag
        return body

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.status_code,
                    "message": exc.detail,
                    "path": request.url.path,
                }
            },
        )

    return app


app = create_app()

__all__ = [
    "app",
    "create_app",
    "FeatureRequest",
    "FeatureResponse",
    "PredictionRequest",
    "PredictionResponse",
    "OnlineSignalForecaster",
    "TTLCache",
]
