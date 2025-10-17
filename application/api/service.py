"""FastAPI application exposing online feature computation and forecasting."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from json import JSONDecodeError
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from time import perf_counter
from typing import Any, Awaitable, Callable, Mapping, Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from analytics.signals.pipeline import FeaturePipelineConfig, SignalFeaturePipeline
from application.api.security import get_api_security_settings, verify_request_identity
from application.api.rate_limit import RateLimiterSnapshot, SlidingWindowRateLimiter, build_rate_limiter
from application.settings import AdminApiSettings, ApiRateLimitSettings, ApiSecuritySettings
from application.trading import signal_to_dto
from core.utils.metrics import MetricsCollector, get_metrics_collector
from domain import Signal, SignalAction
from execution.risk import (
    PostgresKillSwitchStateStore,
    RiskLimits,
    RiskManager,
    SQLiteKillSwitchStateStore,
)
from observability.health import HealthServer
from src.admin.remote_control import (
    AdminIdentity,
    AdminRateLimiter,
    AdminRateLimiterSnapshot,
    create_remote_control_router,
)
from src.audit.audit_logger import AuditLogger, HttpAuditSink
from src.risk.risk_manager import RiskManagerFacade


@dataclass(slots=True)
class _CacheEntry:
    """In-memory TTL cache entry."""

    payload: BaseModel
    expires_at: datetime
    etag: str


@dataclass(slots=True)
class CacheSnapshot:
    """Observability snapshot of :class:`TTLCache` occupancy."""

    entries: int
    max_entries: int
    ttl_seconds: int


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

    async def snapshot(self) -> CacheSnapshot:
        """Return cache occupancy metrics for readiness probes."""

        async with self._lock:
            now = datetime.now(timezone.utc)
            expired = [key for key, entry in self._entries.items() if entry.expires_at <= now]
            for key in expired:
                self._entries.pop(key, None)
            return CacheSnapshot(
                entries=len(self._entries),
                max_entries=self._max_entries,
                ttl_seconds=self._ttl,
            )


@dataclass(slots=True)
class DependencyProbeResult:
    """Normalised representation of dependency readiness."""

    healthy: bool
    detail: str | None = None
    data: dict[str, Any] | None = None


class ComponentHealth(BaseModel):
    """Health status for an individual subsystem."""

    healthy: bool
    status: Literal["operational", "degraded", "failed"]
    detail: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Structured response for the readiness probe."""

    status: Literal["ready", "degraded", "failed"]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    components: dict[str, ComponentHealth]


DependencyProbe = Callable[[], Awaitable[DependencyProbeResult | bool | dict[str, Any]] | DependencyProbeResult | bool | dict[str, Any]]


def _coerce_dependency_result(value: DependencyProbeResult | bool | dict[str, Any]) -> DependencyProbeResult:
    """Normalise supported dependency probe return values."""

    if isinstance(value, DependencyProbeResult):
        return value
    if isinstance(value, dict):
        healthy = bool(value.get("healthy", False))
        detail = value.get("detail") or value.get("message")
        data = {
            key: payload
            for key, payload in value.items()
            if key not in {"healthy", "detail", "message"}
        }
        return DependencyProbeResult(healthy=healthy, detail=detail, data=data or None)
    return DependencyProbeResult(healthy=bool(value))


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
    dependency_probes: Mapping[str, DependencyProbe] | None = None,
    health_server: HealthServer | None = None,
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
    dependency_probe_map: dict[str, DependencyProbe] = dict(dependency_probes or {})

    try:
        resolved_settings = settings or AdminApiSettings()
    except ValidationError as exc:  # pragma: no cover - defensive branch
        alias_map = {
            "audit_secret": "TRADEPULSE_AUDIT_SECRET",
            "AUDIT_SECRET": "TRADEPULSE_AUDIT_SECRET",
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
    audit_sink = None
    if resolved_settings.audit_webhook_url is not None:
        audit_sink = HttpAuditSink(str(resolved_settings.audit_webhook_url))

    secret_manager = resolved_settings.build_secret_manager(
        audit_logger_factory=lambda manager: AuditLogger(
            secret_resolver=manager.provider("audit_secret"),
            sink=audit_sink,
        )
    )
    require_bearer = verify_request_identity()
    require_bearer_with_mtls = verify_request_identity(require_client_certificate=True)
    audit_secret_provider = secret_manager.provider("audit_secret")
    rate_limit_max_attempts = resolved_settings.admin_rate_limit_max_attempts
    rate_limit_interval = resolved_settings.admin_rate_limit_interval_seconds

    audit_logger = secret_manager.audit_logger
    if audit_logger is None:
        audit_logger = AuditLogger(secret_resolver=audit_secret_provider, sink=audit_sink)

    kill_switch_store_settings = resolved_settings.kill_switch_postgres
    if kill_switch_store_settings is not None:
        kill_switch_store = PostgresKillSwitchStateStore(
            str(kill_switch_store_settings.dsn),
            tls=kill_switch_store_settings.tls,
            pool_min_size=int(kill_switch_store_settings.min_pool_size),
            pool_max_size=int(kill_switch_store_settings.max_pool_size),
            acquire_timeout=(
                float(kill_switch_store_settings.acquire_timeout_seconds)
                if kill_switch_store_settings.acquire_timeout_seconds is not None
                else None
            ),
            connect_timeout=float(kill_switch_store_settings.connect_timeout_seconds),
            statement_timeout_ms=int(kill_switch_store_settings.statement_timeout_ms),
            max_retries=int(kill_switch_store_settings.max_retries),
            retry_interval=float(kill_switch_store_settings.retry_interval_seconds),
            backoff_multiplier=float(kill_switch_store_settings.backoff_multiplier),
        )
    else:
        kill_switch_store = SQLiteKillSwitchStateStore(resolved_settings.kill_switch_store_path)
    risk_manager_facade = RiskManagerFacade(
        RiskManager(RiskLimits(), kill_switch_store=kill_switch_store)
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
        )
    )
    app.state.risk_manager = risk_manager_facade.risk_manager
    app.state.audit_logger = audit_logger
    app.state.secret_manager = secret_manager
    app.state.admin_rate_limiter = admin_rate_limiter
    app.state.client_rate_limiter = limiter
    app.state.rate_limit_settings = resolved_rate_settings
    app.state.ttl_cache = ttl_cache
    app.state.dependency_probes = dependency_probe_map
    app.state.health_server = health_server
    metrics_registry = None
    try:  # Lazy import to avoid hard dependency during tests without prometheus_client
        from prometheus_client import REGISTRY as prometheus_registry  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        metrics_registry = None
    else:
        metrics_registry = prometheus_registry

    metrics_module = __import__("core.utils.metrics", fromlist=["MetricsCollector"])
    metrics = get_metrics_collector(metrics_registry)
    if metrics_registry is not None and getattr(metrics, "registry", None) is None:
        metrics = metrics_module.MetricsCollector(metrics_registry)
        setattr(metrics_module, "_collector", metrics)
    app.state.metrics = metrics

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

    @app.get("/health", tags=["health"], summary="Health probe", response_model=HealthResponse)
    async def health_check(response: Response) -> HealthResponse:
        overall_start = perf_counter()
        metrics: MetricsCollector | None = getattr(app.state, "metrics", None)
        components: dict[str, ComponentHealth] = {}

        risk_manager: RiskManager = app.state.risk_manager
        kill_switch = risk_manager.kill_switch
        kill_engaged = kill_switch.is_triggered()
        kill_metrics = {"kill_switch_engaged": kill_engaged}
        if kill_switch.reason:
            kill_metrics["reason"] = kill_switch.reason
        kill_detail = kill_switch.reason if kill_engaged and kill_switch.reason else None
        components["risk_manager"] = ComponentHealth(
            healthy=not kill_engaged,
            status="operational" if not kill_engaged else "failed",
            detail=kill_detail,
            metrics=kill_metrics,
        )

        cache_snapshot = await ttl_cache.snapshot()
        cache_utilisation = (
            cache_snapshot.entries / cache_snapshot.max_entries if cache_snapshot.max_entries else 0.0
        )
        cache_metrics = {
            "entries": cache_snapshot.entries,
            "max_entries": cache_snapshot.max_entries,
            "ttl_seconds": cache_snapshot.ttl_seconds,
            "utilization": round(cache_utilisation, 4),
        }
        cache_healthy = cache_snapshot.entries < cache_snapshot.max_entries
        components["inference_cache"] = ComponentHealth(
            healthy=cache_healthy,
            status="operational" if cache_healthy else "degraded",
            metrics=cache_metrics,
        )

        client_snapshot: RateLimiterSnapshot = limiter.snapshot()
        client_metrics = {
            "backend": client_snapshot.backend,
            "tracked_keys": client_snapshot.tracked_keys,
            "max_utilization": (
                round(client_snapshot.max_utilization, 4) if client_snapshot.max_utilization is not None else None
            ),
            "saturated_keys": list(client_snapshot.saturated_keys),
            "default_policy": {
                "max_requests": resolved_rate_settings.default_policy.max_requests,
                "window_seconds": resolved_rate_settings.default_policy.window_seconds,
            },
        }
        client_healthy = True
        client_status = "operational"
        if client_snapshot.max_utilization is not None and client_snapshot.max_utilization >= 0.9:
            client_healthy = False
            client_status = "degraded"
        if client_snapshot.saturated_keys:
            client_healthy = False
            client_status = "degraded"
        components["client_rate_limiter"] = ComponentHealth(
            healthy=client_healthy,
            status=client_status,
            metrics=client_metrics,
        )

        admin_snapshot: AdminRateLimiterSnapshot = await admin_rate_limiter.snapshot()
        admin_metrics = {
            "tracked_identifiers": admin_snapshot.tracked_identifiers,
            "max_attempts": admin_snapshot.max_attempts,
            "interval_seconds": admin_snapshot.interval_seconds,
            "max_utilization": round(admin_snapshot.max_utilization, 4),
            "saturated_identifiers": list(admin_snapshot.saturated_identifiers),
        }
        admin_healthy = admin_snapshot.max_utilization < 1.0 and not admin_snapshot.saturated_identifiers
        components["admin_rate_limiter"] = ComponentHealth(
            healthy=admin_healthy,
            status="operational" if admin_healthy else "degraded",
            metrics=admin_metrics,
        )

        dependency_failures = False
        for name, probe in dependency_probe_map.items():
            start = perf_counter()
            try:
                result = probe()
                if inspect.isawaitable(result):
                    result = await result
                elapsed_ms = (perf_counter() - start) * 1000
            except Exception as exc:  # pragma: no cover - defensive
                elapsed_ms = (perf_counter() - start) * 1000
                dependency_failures = True
                components[f"dependency:{name}"] = ComponentHealth(
                    healthy=False,
                    status="failed",
                    detail=str(exc),
                    metrics={"latency_ms": round(elapsed_ms, 2)},
                )
                continue

            normalised = _coerce_dependency_result(result)
            metrics = dict(normalised.data or {})
            metrics["latency_ms"] = round(elapsed_ms, 2)
            status_value = "operational" if normalised.healthy else "failed"
            if not normalised.healthy:
                dependency_failures = True
            components[f"dependency:{name}"] = ComponentHealth(
                healthy=normalised.healthy,
                status=status_value,
                detail=normalised.detail,
                metrics=metrics,
            )

        severity = "ready"
        if any(component.status == "failed" for component in components.values()) or dependency_failures or kill_engaged:
            severity = "failed"
        elif any(component.status == "degraded" for component in components.values()):
            severity = "degraded"

        health_payload = HealthResponse(status=severity, components=components)

        probe_status = status.HTTP_200_OK if severity == "ready" else status.HTTP_503_SERVICE_UNAVAILABLE
        response.status_code = probe_status

        health_state: HealthServer | None = app.state.health_server
        if health_state is not None:
            health_state.set_live(True)
            health_state.set_ready(severity == "ready")
            for name, component in components.items():
                health_state.update_component(name, component.healthy, component.detail)

        if metrics and metrics.enabled:
            duration = perf_counter() - overall_start
            metrics.observe_health_check_latency("api.overall", duration)
            metrics.set_health_check_status("api.overall", severity == "ready")
            for name, component in components.items():
                metrics.set_health_check_status(f"component.{name}", component.healthy)

        return health_payload

    @app.get(
        "/metrics",
        tags=["health"],
        summary="Prometheus metrics",
        response_class=PlainTextResponse,
    )
    async def prometheus_metrics() -> PlainTextResponse:
        metrics: MetricsCollector | None = getattr(app.state, "metrics", None)
        if metrics is None:
            metrics = get_metrics_collector()

        if not metrics.enabled:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Prometheus metrics are disabled")

        payload = metrics.render_prometheus()
        return PlainTextResponse(
            payload,
            media_type="text/plain; version=0.0.4; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

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
    "CacheSnapshot",
    "ComponentHealth",
    "HealthResponse",
    "DependencyProbe",
    "DependencyProbeResult",
]
