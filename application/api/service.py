"""FastAPI application exposing online feature computation and forecasting."""

from __future__ import annotations

import asyncio
import hashlib
import json
from json import JSONDecodeError
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from analytics.signals.pipeline import FeaturePipelineConfig, SignalFeaturePipeline
from application.api.security import get_api_security_settings, verify_request_identity
from application.api.rate_limit import SlidingWindowRateLimiter, build_rate_limiter
from application.settings import AdminApiSettings, ApiRateLimitSettings, ApiSecuritySettings
from application.trading import signal_to_dto
from domain import Signal, SignalAction
from execution.risk import RiskLimits, RiskManager
from src.admin.remote_control import AdminIdentity, AdminRateLimiter, create_remote_control_router
from src.audit.audit_logger import AuditLogger, HttpAuditSink
from src.risk.risk_manager import RiskManagerFacade
from src.security import SecretNotFoundError, TokenAuthenticator, get_secret_manager


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
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="No features computed"
            )

        latest_row = features.iloc[-1]

        required_macd_columns = ("macd", "macd_signal", "macd_histogram")
        missing_columns = [col for col in required_macd_columns if col not in latest_row.index]
        if missing_columns:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Missing MACD features: {', '.join(sorted(missing_columns))}",
            )

        invalid_columns = [
            col
            for col in required_macd_columns
            if pd.isna(latest_row[col]) or not np.isfinite(float(latest_row[col]))
        ]
        if invalid_columns:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unavailable MACD features: {', '.join(sorted(invalid_columns))}",
            )

        latest = latest_row.dropna()
        if latest.empty:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Insufficient data to derive features",
            )
        return latest

    def derive_signal(
        self, symbol: str, latest: pd.Series, horizon_seconds: int
    ) -> tuple[Signal, float]:
        # Compute a simple composite alpha score using a handful of stable metrics.
        macd = float(latest["macd"])
        macd_signal_line = float(latest["macd_signal"])
        macd_histogram = float(latest["macd_histogram"])
        rsi = float(latest.get("rsi", 50.0))
        ret_1 = float(latest.get("return_1", 0.0))
        volatility_20 = float(latest.get("volatility_20", 0.0))
        queue_imbalance = float(latest.get("queue_imbalance", 0.0))

        # The heuristic emphasises MACD structure: the raw MACD trend captures the
        # dominant fast/slow EMA divergence, the crossover term highlights whether
        # MACD is leading or lagging the signal line, and the histogram term scales
        # the magnitude of the divergence to reward strong momentum while
        # suppressing noise. RSI, short-term returns, and order-book imbalance are
        # complementary momentum and flow signals, while realised volatility acts
        # as a risk haircut.
        contributions: dict[str, float] = {
            "macd_trend": np.tanh(macd) * 0.28,
            "macd_crossover": np.tanh(macd - macd_signal_line) * 0.24,
            "macd_histogram": np.tanh(macd_histogram * 2.0) * 0.18,
            "rsi_bias": ((rsi - 50.0) / 50.0) * 0.12,
            "return_momentum": np.tanh(ret_1 * 120.0) * 0.1,
            "order_flow": np.tanh(queue_imbalance) * 0.06,
            "volatility_risk": -abs(volatility_20) * 0.04,
        }

        score = sum(contributions.values())

        threshold = 0.12
        if score > threshold:
            action = SignalAction.BUY
        elif score < -threshold:
            action = SignalAction.SELL
        else:
            action = SignalAction.HOLD

        confidence = float(min(1.0, max(0.0, abs(score) / 0.85)))
        signal = Signal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            rationale=(
                "Composite heuristic weighting MACD trend, crossover momentum, histogram strength, RSI, returns, and book imbalance"
            ),
            metadata={
                "score": score,
                "horizon_seconds": horizon_seconds,
                "component_contributions": contributions,
                "macd_component_explanations": {
                    "macd_trend": "Measures overall EMA divergence; positive values indicate bullish acceleration.",
                    "macd_crossover": "Rewards MACD leading the signal line; negative values highlight bearish crossovers.",
                    "macd_histogram": "Scales the magnitude of MACD vs signal separation to favour decisive momentum.",
                },
            },
        )
        return signal, score


def _hash_payload(prefix: str, payload: BaseModel) -> str:
    data = json.dumps(payload.model_dump(mode="json"), sort_keys=True, default=str)
    digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


class PayloadGuardMiddleware(BaseHTTPMiddleware):
    """Inspect incoming JSON payloads for size and suspicious content."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body_bytes: int,
        suspicious_keys: set[str],
        suspicious_substrings: tuple[str, ...],
    ) -> None:
        super().__init__(app)
        self._max_body_bytes = max_body_bytes
        self._suspicious_keys = {key.lower() for key in suspicious_keys}
        self._suspicious_substrings = tuple(sub.lower() for sub in suspicious_substrings)

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.method in {"POST", "PUT", "PATCH"}:
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    length_value = int(content_length)
                except ValueError:
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={"detail": "Invalid Content-Length header."},
                    )
                if length_value > self._max_body_bytes:
                    return JSONResponse(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        content={"detail": "Request body exceeds configured limit."},
                    )

            body = await request.body()
            if len(body) > self._max_body_bytes:
                return JSONResponse(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    content={"detail": "Request body exceeds configured limit."},
                )

            content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type in {"application/json", "application/problem+json", ""}:
                if body:
                    try:
                        parsed = json.loads(body)
                    except JSONDecodeError:
                        return JSONResponse(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            content={"detail": "Malformed JSON payload."},
                        )
                    if not isinstance(parsed, (dict, list)):
                        return JSONResponse(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            content={"detail": "Unsupported JSON payload structure."},
                        )
                    if self._is_suspicious(parsed):
                        return JSONResponse(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            content={"detail": "Suspicious payload rejected."},
                        )
                request._body = body  # type: ignore[attr-defined]

        return await call_next(request)

    def _is_suspicious(self, payload: object) -> bool:
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(key, str) and key.lower() in self._suspicious_keys:
                    return True
                if self._is_suspicious(value):
                    return True
            return False
        if isinstance(payload, list):
            return any(self._is_suspicious(item) for item in payload)
        if isinstance(payload, str):
            lowered = payload.lower()
            return any(token in lowered for token in self._suspicious_substrings)
        return False


def _resolve_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        for part in forwarded_for.split(","):
            candidate = part.strip().split()[0]
            if candidate:
                return candidate

    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()

    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def create_app(
    *,
    rate_limiter: SlidingWindowRateLimiter | None = None,
    cache: TTLCache | None = None,
    forecaster_factory: Callable[[], OnlineSignalForecaster] | None = None,
    settings: AdminApiSettings | None = None,
    rate_limit_settings: ApiRateLimitSettings | None = None,
    security_settings: ApiSecuritySettings | None = None,
) -> FastAPI:
    """Build the FastAPI application with configured dependencies.

    Args:
        rate_limiter: Optional AsyncLimiter controlling request throughput for
            inference endpoints.
        cache: Shared cache instance for feature and prediction responses.
        forecaster_factory: Callable returning the forecaster implementation.
        settings: Administrative configuration backing the kill-switch API. When
            omitted the values are loaded from :class:`AdminApiSettings` using
            environment variables.
    """

    resolved_rate_settings = rate_limit_settings or ApiRateLimitSettings()
    limiter = rate_limiter or build_rate_limiter(resolved_rate_settings)
    ttl_cache = cache or TTLCache(ttl_seconds=30, max_entries=512)
    forecaster_provider = forecaster_factory or (lambda: OnlineSignalForecaster())
    forecaster = forecaster_provider()

    try:
        resolved_settings = settings or AdminApiSettings()
    except ValidationError as exc:  # pragma: no cover - defensive branch
        alias_map = {
            "audit_secret_id": "TRADEPULSE_AUDIT_SECRET_ID",
            "AUDIT_SECRET_ID": "TRADEPULSE_AUDIT_SECRET_ID",
            "admin_token_id": "TRADEPULSE_ADMIN_TOKEN_ID",
            "ADMIN_TOKEN_ID": "TRADEPULSE_ADMIN_TOKEN_ID",
            "admin_subject": "TRADEPULSE_ADMIN_SUBJECT",
            "ADMIN_SUBJECT": "TRADEPULSE_ADMIN_SUBJECT",
            "admin_rate_limit_max_attempts": "TRADEPULSE_ADMIN_RATE_LIMIT_MAX_ATTEMPTS",
            "ADMIN_RATE_LIMIT_MAX_ATTEMPTS": "TRADEPULSE_ADMIN_RATE_LIMIT_MAX_ATTEMPTS",
            "admin_rate_limit_interval_seconds": "TRADEPULSE_ADMIN_RATE_LIMIT_INTERVAL_SECONDS",
            "ADMIN_RATE_LIMIT_INTERVAL_SECONDS": "TRADEPULSE_ADMIN_RATE_LIMIT_INTERVAL_SECONDS",
            "audit_webhook_url": "TRADEPULSE_AUDIT_WEBHOOK_URL",
            "AUDIT_WEBHOOK_URL": "TRADEPULSE_AUDIT_WEBHOOK_URL",
        }
        missing = [
            alias_map.get(error["loc"][0], error["loc"][0])
            for error in exc.errors()
            if error.get("type") == "missing"
        ]
        joined = ", ".join(sorted(set(missing))) or "configuration values"
        raise RuntimeError(
            (
                "Missing required secret(s): {}. Provide them via AdminApiSettings or environment variables."
            ).format(joined)
        ) from exc

    try:
        resolved_security_settings = security_settings or get_api_security_settings()
    except ValidationError as exc:
        alias_map = {
            "oauth2_issuer": "TRADEPULSE_OAUTH2_ISSUER",
            "oauth2_audience": "TRADEPULSE_OAUTH2_AUDIENCE",
            "oauth2_jwks_uri": "TRADEPULSE_OAUTH2_JWKS_URI",
        }
        missing = [
            alias_map.get(error["loc"][0], error["loc"][0])
            for error in exc.errors()
            if error.get("type") == "missing"
        ]
        joined = ", ".join(sorted(set(missing))) or "OAuth configuration values"
        raise RuntimeError(("Missing required OAuth configuration: {}.").format(joined)) from exc
    if security_settings is not None:
        setattr(get_api_security_settings, "_instance", resolved_security_settings)
    require_bearer = verify_request_identity()
    require_bearer_with_mtls = verify_request_identity(require_client_certificate=True)
    secret_manager = get_secret_manager()
    audit_secret_id = resolved_settings.audit_secret_id
    try:
        secret_manager.register_secret(audit_secret_id)
    except SecretNotFoundError as exc:  # pragma: no cover - environment misconfiguration
        raise RuntimeError(
            f"Missing required secret material for {audit_secret_id}. Configure TRADEPULSE_AUDIT_SECRET_ID correctly."
        ) from exc
    admin_token_id = resolved_settings.admin_token_id
    token_authenticator: TokenAuthenticator | None = None
    if admin_token_id:
        try:
            secret_manager.register_secret(admin_token_id)
        except SecretNotFoundError as exc:  # pragma: no cover - environment misconfiguration
            raise RuntimeError(
                (
                    f"Missing required administrative token secret for {admin_token_id}. Configure "
                    "TRADEPULSE_ADMIN_TOKEN_ID correctly."
                )
            ) from exc
        token_authenticator = TokenAuthenticator(secret_manager, admin_token_id)
    rate_limit_max_attempts = resolved_settings.admin_rate_limit_max_attempts
    rate_limit_interval = resolved_settings.admin_rate_limit_interval_seconds

    audit_sink = None
    if resolved_settings.audit_webhook_url is not None:
        audit_sink = HttpAuditSink(str(resolved_settings.audit_webhook_url))

    risk_manager_facade = RiskManagerFacade(RiskManager(RiskLimits()))
    audit_logger = AuditLogger(
        secret_manager=secret_manager,
        secret_id=audit_secret_id,
        sink=audit_sink,
    )
    admin_rate_limiter = AdminRateLimiter(
        max_attempts=int(rate_limit_max_attempts),
        interval_seconds=float(rate_limit_interval),
    )

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
        allow_headers=["*"],
    )

    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(resolved_security_settings.trusted_hosts),
    )
    app.add_middleware(
        PayloadGuardMiddleware,
        max_body_bytes=int(resolved_security_settings.max_request_bytes),
        suspicious_keys=set(resolved_security_settings.suspicious_json_keys),
        suspicious_substrings=tuple(resolved_security_settings.suspicious_json_substrings),
    )

    app.include_router(
        create_remote_control_router(
            risk_manager_facade,
            audit_logger,
            identity_dependency=require_bearer_with_mtls,
            rate_limiter=admin_rate_limiter,
            token_authenticator=token_authenticator,
        )
    )
    app.state.risk_manager = risk_manager_facade.risk_manager
    app.state.audit_logger = audit_logger
    app.state.secret_manager = secret_manager
    if token_authenticator is not None:
        app.state.token_authenticator = token_authenticator
    app.state.admin_rate_limiter = admin_rate_limiter
    app.state.client_rate_limiter = limiter
    app.state.rate_limit_settings = resolved_rate_settings

    async def enforce_rate_limit(
        request: Request,
        identity: AdminIdentity = Depends(require_bearer),
    ) -> AdminIdentity:
        ip_address = _resolve_ip(request)
        await limiter.check(subject=identity.subject, ip_address=ip_address)
        return identity

    def get_forecaster() -> OnlineSignalForecaster:
        return forecaster

    def _append_vary_header(response: Response, value: str) -> None:
        existing = response.headers.get("Vary")
        if not existing:
            response.headers["Vary"] = value
            return
        values = {entry.strip() for entry in existing.split(",") if entry.strip()}
        if value not in values:
            response.headers["Vary"] = f"{existing}, {value}"

    @app.middleware("http")
    async def add_cache_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/admin") or response.status_code >= 400:
            response.headers["Cache-Control"] = "no-store"
            response.headers.setdefault("Pragma", "no-cache")
            _append_vary_header(response, "Authorization")
        else:
            response.headers["Cache-Control"] = f"private, max-age={ttl_cache.ttl_seconds}"
        _append_vary_header(response, "Accept")
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
        _identity: AdminIdentity = Depends(enforce_rate_limit),
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
        _identity: AdminIdentity = Depends(enforce_rate_limit),
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
        etag = hashlib.sha256(
            json.dumps(body.model_dump(), sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
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
