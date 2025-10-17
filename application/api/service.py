"""FastAPI application exposing online feature computation and forecasting."""

from __future__ import annotations

import asyncio
import asyncio
import hashlib
import inspect
import json
from json import JSONDecodeError
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from http import HTTPStatus
from time import perf_counter
from typing import Any, Awaitable, Callable, Mapping, Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.exceptions import RequestValidationError
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


class ApiErrorCode(str, Enum):
    """Stable error codes returned by the public HTTP API."""

    BAD_REQUEST = "ERR_BAD_REQUEST"
    AUTH_REQUIRED = "ERR_AUTH_REQUIRED"
    FORBIDDEN = "ERR_FORBIDDEN"
    NOT_FOUND = "ERR_NOT_FOUND"
    RATE_LIMIT = "ERR_RATE_LIMIT"
    VALIDATION_FAILED = "ERR_VALIDATION_FAILED"
    UNPROCESSABLE = "ERR_UNPROCESSABLE"
    INTERNAL = "ERR_INTERNAL"
    FEATURES_EMPTY = "ERR_FEATURES_EMPTY"
    FEATURES_MISSING = "ERR_FEATURES_MISSING"
    FEATURES_INVALID = "ERR_FEATURES_INVALID"
    FEATURES_FILTER_MISMATCH = "ERR_FEATURES_FILTER_MISMATCH"
    INVALID_CURSOR = "ERR_INVALID_CURSOR"
    INVALID_CONFIDENCE = "ERR_INVALID_CONFIDENCE"
    PREDICTIONS_FILTER_MISMATCH = "ERR_PREDICTIONS_FILTER_MISMATCH"


DEFAULT_ERROR_CODES: dict[int, ApiErrorCode] = {
    status.HTTP_400_BAD_REQUEST: ApiErrorCode.BAD_REQUEST,
    status.HTTP_401_UNAUTHORIZED: ApiErrorCode.AUTH_REQUIRED,
    status.HTTP_403_FORBIDDEN: ApiErrorCode.FORBIDDEN,
    status.HTTP_404_NOT_FOUND: ApiErrorCode.NOT_FOUND,
    status.HTTP_422_UNPROCESSABLE_CONTENT: ApiErrorCode.VALIDATION_FAILED,
    status.HTTP_429_TOO_MANY_REQUESTS: ApiErrorCode.RATE_LIMIT,
    status.HTTP_500_INTERNAL_SERVER_ERROR: ApiErrorCode.INTERNAL,
    status.HTTP_503_SERVICE_UNAVAILABLE: ApiErrorCode.INTERNAL,
}


@dataclass(slots=True, frozen=True)
class FeatureQueryParams:
    """Query parameters driving feature pagination and filtering."""

    limit: int
    cursor: datetime | None
    start_at: datetime | None
    end_at: datetime | None
    feature_prefix: str | None
    feature_keys: tuple[str, ...]

    def cache_fragment(self) -> dict[str, Any]:
        return {
            "limit": self.limit,
            "cursor": self.cursor.isoformat() if self.cursor else None,
            "start_at": self.start_at.isoformat() if self.start_at else None,
            "end_at": self.end_at.isoformat() if self.end_at else None,
            "feature_prefix": self.feature_prefix,
            "feature_keys": list(self.feature_keys),
        }


@dataclass(slots=True, frozen=True)
class PredictionQueryParams:
    """Query parameters for prediction pagination and filtering."""

    limit: int
    cursor: datetime | None
    start_at: datetime | None
    end_at: datetime | None
    actions: tuple[SignalAction, ...]
    min_confidence: float | None

    def cache_fragment(self) -> dict[str, Any]:
        return {
            "limit": self.limit,
            "cursor": self.cursor.isoformat() if self.cursor else None,
            "start_at": self.start_at.isoformat() if self.start_at else None,
            "end_at": self.end_at.isoformat() if self.end_at else None,
            "actions": [action.value for action in self.actions],
            "min_confidence": self.min_confidence,
        }


def _parse_datetime_param(name: str, raw: str | None) -> datetime | None:
    if raw is None:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": ApiErrorCode.INVALID_CURSOR.value,
                "message": f"Invalid {name} value; expected ISO 8601 format.",
                "meta": {"parameter": name, "value": raw},
            },
        ) from exc
    return _ensure_timezone(parsed)


def _parse_confidence_param(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": ApiErrorCode.INVALID_CONFIDENCE.value,
                "message": "min_confidence must be a floating point number between 0 and 1.",
                "meta": {"parameter": "min_confidence", "value": raw},
            },
        ) from exc
    if not 0.0 <= value <= 1.0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": ApiErrorCode.INVALID_CONFIDENCE.value,
                "message": "min_confidence must be within [0.0, 1.0].",
                "meta": {"parameter": "min_confidence", "value": raw},
            },
        )
    return value


def get_feature_query_params(
    limit: int = Query(1, ge=1, le=500, description="Number of feature snapshots to return."),
    cursor: str | None = Query(None, description="Pagination cursor (exclusive) encoded as ISO 8601 timestamp."),
    start_at: str | None = Query(None, alias="startAt", description="Filter snapshots on or after this timestamp."),
    end_at: str | None = Query(None, alias="endAt", description="Filter snapshots on or before this timestamp."),
    feature_prefix: str | None = Query(None, alias="featurePrefix", description="Return only feature keys with the provided prefix."),
    feature: list[str] | None = Query(None, alias="feature", description="Specific feature keys to include."),
) -> FeatureQueryParams:
    feature_keys: tuple[str, ...] = tuple(dict.fromkeys(feature or []))
    return FeatureQueryParams(
        limit=limit,
        cursor=_parse_datetime_param("cursor", cursor),
        start_at=_parse_datetime_param("start_at", start_at),
        end_at=_parse_datetime_param("end_at", end_at),
        feature_prefix=feature_prefix,
        feature_keys=feature_keys,
    )


def get_prediction_query_params(
    limit: int = Query(1, ge=1, le=500, description="Number of predictions to return."),
    cursor: str | None = Query(None, description="Pagination cursor (exclusive) encoded as ISO 8601 timestamp."),
    start_at: str | None = Query(None, alias="startAt", description="Return predictions generated at or after this time."),
    end_at: str | None = Query(None, alias="endAt", description="Return predictions generated at or before this time."),
    action: list[str] | None = Query(None, alias="action", description="Filter predictions by signal action."),
    min_confidence: str | None = Query(None, alias="minConfidence", description="Minimum signal confidence to include."),
) -> PredictionQueryParams:
    actions: list[SignalAction] = []
    for value in action or []:
        try:
            actions.append(SignalAction(value))
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "code": ApiErrorCode.UNPROCESSABLE.value,
                    "message": f"Unsupported action filter '{value}'.",
                    "meta": {"parameter": "action", "value": value},
                },
            ) from exc
    return PredictionQueryParams(
        limit=limit,
        cursor=_parse_datetime_param("cursor", cursor),
        start_at=_parse_datetime_param("start_at", start_at),
        end_at=_parse_datetime_param("end_at", end_at),
        actions=tuple(actions),
        min_confidence=_parse_confidence_param(min_confidence),
    )


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


class PaginationMeta(BaseModel):
    """Pagination envelope used by collection responses."""

    cursor: datetime | None = None
    next_cursor: datetime | None = None
    limit: int = 0
    returned: int = 0

    model_config = ConfigDict(populate_by_name=True)


class FeatureFilters(BaseModel):
    """Echoed filter parameters for feature responses."""

    start_at: datetime | None = None
    end_at: datetime | None = None
    feature_prefix: str | None = None
    feature_keys: tuple[str, ...] = Field(default_factory=tuple)

    model_config = ConfigDict(populate_by_name=True)


class FeatureSnapshot(BaseModel):
    """Single feature vector at a specific timestamp."""

    timestamp: datetime
    features: dict[str, float]

    model_config = ConfigDict(populate_by_name=True)


class FeatureResponse(BaseModel):
    """Response containing the most recent feature snapshot."""

    symbol: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    features: dict[str, float] = Field(default_factory=dict)
    items: list[FeatureSnapshot] = Field(default_factory=list)
    pagination: PaginationMeta = Field(default_factory=PaginationMeta)
    filters: FeatureFilters = Field(default_factory=FeatureFilters)

    model_config = ConfigDict(populate_by_name=True)


class PredictionRequest(FeatureRequest):
    """Prediction request payload, optionally specifying a forecast horizon."""

    horizon_seconds: int = Field(
        300,
        ge=60,
        le=3600,
        description="Prediction horizon in seconds for contextual metadata.",
    )


class PredictionFilters(BaseModel):
    """Echoed filter parameters for prediction responses."""

    start_at: datetime | None = None
    end_at: datetime | None = None
    actions: tuple[SignalAction, ...] = Field(default_factory=tuple)
    min_confidence: float | None = None

    model_config = ConfigDict(populate_by_name=True)


class PredictionSnapshot(BaseModel):
    """Snapshot of a derived signal at a point in time."""

    timestamp: datetime
    score: float
    signal: dict[str, Any]

    model_config = ConfigDict(populate_by_name=True)


class PredictionResponse(BaseModel):
    """Response representing the generated trading signal."""

    symbol: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    horizon_seconds: int
    score: float | None = Field(
        default=None,
        description="Composite alpha score driving the primary action.",
    )
    signal: dict[str, Any] | None = Field(
        default=None, description="Primary trading signal at the head of the page."
    )
    items: list[PredictionSnapshot] = Field(default_factory=list)
    pagination: PaginationMeta = Field(default_factory=PaginationMeta)
    filters: PredictionFilters = Field(default_factory=PredictionFilters)

    model_config = ConfigDict(populate_by_name=True)


class OnlineSignalForecaster:
    """Wraps the feature pipeline and lightweight heuristics for live inference."""

    def __init__(self, pipeline: SignalFeaturePipeline | None = None) -> None:
        self._pipeline = pipeline or SignalFeaturePipeline(FeaturePipelineConfig())

    def compute_features(self, payload: FeatureRequest) -> pd.DataFrame:
        frame = payload.to_frame()
        features = self._pipeline.transform(frame)
        return features

    def _normalise_feature_row(self, row: pd.Series, *, strict: bool) -> pd.Series | None:
        required_macd_columns = ("macd", "macd_signal", "macd_histogram")
        missing_columns = [col for col in required_macd_columns if col not in row.index]
        if missing_columns:
            if strict:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail={
                        "code": ApiErrorCode.FEATURES_MISSING.value,
                        "message": f"Missing MACD features: {', '.join(sorted(missing_columns))}",
                    },
                )
            return None

        invalid_columns = [
            col
            for col in required_macd_columns
            if pd.isna(row[col]) or not np.isfinite(float(row[col]))
        ]
        if invalid_columns:
            if strict:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail={
                        "code": ApiErrorCode.FEATURES_INVALID.value,
                        "message": f"Unavailable MACD features: {', '.join(sorted(invalid_columns))}",
                    },
                )
            return None

        cleaned = row.dropna()
        if cleaned.empty:
            if strict:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail={
                        "code": ApiErrorCode.FEATURES_INVALID.value,
                        "message": "Insufficient data to derive features",
                    },
                )
            return None
        return cleaned

    def normalised_feature_rows(
        self, features: pd.DataFrame, *, strict: bool = False
    ) -> list[tuple[datetime, pd.Series]]:
        rows: list[tuple[datetime, pd.Series]] = []
        for timestamp, raw in features.iterrows():
            normalised = self._normalise_feature_row(raw, strict=strict)
            if normalised is None:
                continue
            python_ts = timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp
            rows.append((_ensure_timezone(python_ts), normalised))
        return rows

    def latest_feature_vector(self, features: pd.DataFrame) -> pd.Series:
        if features.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": ApiErrorCode.FEATURES_EMPTY.value,
                    "message": "No features computed",
                },
            )

        latest_row = features.iloc[-1]
        normalised = self._normalise_feature_row(latest_row, strict=True)
        if normalised is None:  # pragma: no cover - strict=True ensures non-None
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "code": ApiErrorCode.FEATURES_INVALID.value,
                    "message": "Insufficient data to derive features",
                },
            )
        return normalised

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


def _filter_feature_frame(
    features: pd.DataFrame,
    *,
    start_at: datetime | None,
    end_at: datetime | None,
) -> pd.DataFrame:
    frame = features
    if start_at is not None:
        frame = frame[frame.index >= start_at]
    if end_at is not None:
        frame = frame[frame.index <= end_at]
    return frame


def _paginate_frame(
    frame: pd.DataFrame, *, limit: int, cursor: datetime | None
) -> tuple[pd.DataFrame, datetime | None]:
    ordered = frame.sort_index(ascending=False)
    if cursor is not None:
        ordered = ordered[ordered.index < cursor]
    page = ordered.iloc[:limit]
    if page.empty:
        return page, None
    next_cursor_ts = page.index[-1]
    if hasattr(next_cursor_ts, "to_pydatetime"):
        next_cursor = _ensure_timezone(next_cursor_ts.to_pydatetime())
    else:
        next_cursor = _ensure_timezone(next_cursor_ts)
    return page, next_cursor


def _filter_feature_values(
    feature_vector: pd.Series,
    *,
    feature_prefix: str | None,
    feature_keys: tuple[str, ...],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for key in sorted(feature_vector.index):
        if feature_prefix is not None and not key.startswith(feature_prefix):
            continue
        if feature_keys and key not in feature_keys:
            continue
        values[key] = float(feature_vector[key])
    return values


def _hash_payload(prefix: str, payload: BaseModel, extra: Mapping[str, Any] | None = None) -> str:
    body = payload.model_dump(mode="json")
    if extra:
        body["__query__"] = extra
    data = json.dumps(body, sort_keys=True, default=str)
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
        version="0.2.0",
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
    metrics_collector = get_metrics_collector(metrics_registry)
    if metrics_registry is not None and getattr(metrics_collector, "registry", None) is None:
        refreshed_metrics = metrics_module.MetricsCollector(metrics_registry)
        metrics_collector.__dict__.update(refreshed_metrics.__dict__)
        setattr(metrics_module, "_collector", metrics_collector)
    app.state.metrics = metrics_collector

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
            component_metrics = dict(normalised.data or {})
            component_metrics["latency_ms"] = round(elapsed_ms, 2)
            status_value = "operational" if normalised.healthy else "failed"
            if not normalised.healthy:
                dependency_failures = True
            components[f"dependency:{name}"] = ComponentHealth(
                healthy=normalised.healthy,
                status=status_value,
                detail=normalised.detail,
                metrics=component_metrics,
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

        if metrics_collector and metrics_collector.enabled:
            duration = perf_counter() - overall_start
            metrics_collector.observe_health_check_latency("api.overall", duration)
            metrics_collector.set_health_check_status("api.overall", severity == "ready")
            for name, component in components.items():
                metrics_collector.set_health_check_status(f"component.{name}", component.healthy)

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
        query: FeatureQueryParams = Depends(get_feature_query_params),
        _identity: AdminIdentity = Depends(enforce_rate_limit),
        predictor: OnlineSignalForecaster = Depends(get_forecaster),
    ) -> FeatureResponse:
        cache_key = _hash_payload("features", payload, query.cache_fragment())
        cached = await ttl_cache.get(cache_key)
        if cached is not None:
            response.headers["X-Cache-Status"] = "hit"
            response.headers["ETag"] = cached.etag
            return cached.payload  # type: ignore[return-value]

        features = predictor.compute_features(payload)
        filtered = _filter_feature_frame(
            features,
            start_at=query.start_at,
            end_at=query.end_at,
        )
        ordered = filtered.sort_index(ascending=False)
        if query.cursor is not None:
            ordered = ordered[ordered.index < query.cursor]

        snapshots: list[FeatureSnapshot] = []
        next_cursor: datetime | None = None
        last_position: int | None = None
        last_timestamp: pd.Timestamp | datetime | None = None

        for position, (row_timestamp, raw_vector) in enumerate(ordered.iterrows()):
            last_position = position
            last_timestamp = row_timestamp
            normalised = predictor._normalise_feature_row(raw_vector, strict=False)
            if normalised is None:
                continue
            values = _filter_feature_values(
                normalised,
                feature_prefix=query.feature_prefix,
                feature_keys=query.feature_keys,
            )
            if not values:
                continue
            python_ts = (
                row_timestamp.to_pydatetime()
                if hasattr(row_timestamp, "to_pydatetime")
                else row_timestamp
            )
            snapshots.append(
                FeatureSnapshot(timestamp=_ensure_timezone(python_ts), features=values)
            )
            if len(snapshots) >= query.limit:
                break

        if last_timestamp is not None and last_position is not None:
            remaining = ordered.iloc[last_position + 1 :]
            if not remaining.empty:
                python_ts = (
                    last_timestamp.to_pydatetime()
                    if hasattr(last_timestamp, "to_pydatetime")
                    else last_timestamp
                )
                next_cursor = _ensure_timezone(python_ts)

        if not snapshots:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": ApiErrorCode.FEATURES_FILTER_MISMATCH.value,
                    "message": "No feature snapshots matched the requested filters.",
                },
            )

        pagination = PaginationMeta(
            cursor=query.cursor,
            next_cursor=next_cursor,
            limit=query.limit,
            returned=len(snapshots),
        )
        filters = FeatureFilters(
            start_at=query.start_at,
            end_at=query.end_at,
            feature_prefix=query.feature_prefix,
            feature_keys=query.feature_keys,
        )
        feature_dict = snapshots[0].features if snapshots else {}
        body = FeatureResponse(
            symbol=payload.symbol,
            features=feature_dict,
            items=snapshots,
            pagination=pagination,
            filters=filters,
        )
        etag = hashlib.sha256(
            json.dumps(body.model_dump(), sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
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
        query: PredictionQueryParams = Depends(get_prediction_query_params),
        _identity: AdminIdentity = Depends(enforce_rate_limit),
        predictor: OnlineSignalForecaster = Depends(get_forecaster),
    ) -> PredictionResponse:
        cache_key = _hash_payload("predictions", payload, query.cache_fragment())
        cached = await ttl_cache.get(cache_key)
        if cached is not None:
            response.headers["X-Cache-Status"] = "hit"
            response.headers["ETag"] = cached.etag
            return cached.payload  # type: ignore[return-value]

        features = predictor.compute_features(payload)
        filtered = _filter_feature_frame(
            features,
            start_at=query.start_at,
            end_at=query.end_at,
        )
        ordered = filtered.sort_index(ascending=False)
        if query.cursor is not None:
            ordered = ordered[ordered.index < query.cursor]

        predictions: list[PredictionSnapshot] = []
        next_cursor: datetime | None = None
        last_position: int | None = None
        last_timestamp: pd.Timestamp | datetime | None = None

        for position, (row_timestamp, raw_vector) in enumerate(ordered.iterrows()):
            last_position = position
            last_timestamp = row_timestamp
            normalised = predictor._normalise_feature_row(raw_vector, strict=False)
            if normalised is None:
                continue
            signal, score = predictor.derive_signal(
                payload.symbol, normalised, payload.horizon_seconds
            )
            if query.actions and signal.action not in query.actions:
                continue
            if (
                query.min_confidence is not None
                and float(signal.confidence) < query.min_confidence
            ):
                continue
            python_ts = (
                row_timestamp.to_pydatetime()
                if hasattr(row_timestamp, "to_pydatetime")
                else row_timestamp
            )
            predictions.append(
                PredictionSnapshot(
                    timestamp=_ensure_timezone(python_ts),
                    score=score,
                    signal=signal_to_dto(signal),
                )
            )
            if len(predictions) >= query.limit:
                break

        if last_timestamp is not None and last_position is not None:
            remaining = ordered.iloc[last_position + 1 :]
            if not remaining.empty:
                python_ts = (
                    last_timestamp.to_pydatetime()
                    if hasattr(last_timestamp, "to_pydatetime")
                    else last_timestamp
                )
                next_cursor = _ensure_timezone(python_ts)

        if not predictions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": ApiErrorCode.PREDICTIONS_FILTER_MISMATCH.value,
                    "message": "No predictions matched the requested filters.",
                },
            )

        pagination = PaginationMeta(
            cursor=query.cursor,
            next_cursor=next_cursor,
            limit=query.limit,
            returned=len(predictions),
        )
        filters = PredictionFilters(
            start_at=query.start_at,
            end_at=query.end_at,
            actions=query.actions,
            min_confidence=query.min_confidence,
        )
        head = predictions[0] if predictions else None
        body = PredictionResponse(
            symbol=payload.symbol,
            horizon_seconds=payload.horizon_seconds,
            score=head.score if head else None,
            signal=head.signal if head else None,
            items=predictions,
            pagination=pagination,
            filters=filters,
        )
        etag = hashlib.sha256(
            json.dumps(body.model_dump(), sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        await ttl_cache.set(cache_key, body, etag)
        response.headers["X-Cache-Status"] = "miss"
        response.headers["ETag"] = etag
        return body

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": {
                    "code": ApiErrorCode.VALIDATION_FAILED.value,
                    "message": "Invalid request payload.",
                    "path": request.url.path,
                    "meta": {"errors": exc.errors()},
                }
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        default_code = DEFAULT_ERROR_CODES.get(exc.status_code, ApiErrorCode.INTERNAL).value
        detail = exc.detail
        message: str | None = None
        meta: Any | None = None
        code = default_code
        if isinstance(detail, dict):
            code = str(detail.get("code") or default_code)
            message = detail.get("message") or detail.get("detail")
            meta = detail.get("meta")
        elif isinstance(detail, str):
            message = detail
        if not message:
            try:
                message = HTTPStatus(exc.status_code).phrase
            except ValueError:  # pragma: no cover - defensive
                message = "An error occurred"

        error_content: dict[str, Any] = {
            "code": code,
            "message": message,
            "path": request.url.path,
        }
        if meta is not None:
            error_content["meta"] = meta
        return JSONResponse(status_code=exc.status_code, content={"error": error_content})

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
