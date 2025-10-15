"""Sliding window rate limiter utilities for TradePulse APIs."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Protocol

from fastapi import HTTPException, status

from application.settings import ApiRateLimitSettings, RateLimitPolicy

__all__ = [
    "RateLimiterBackend",
    "InMemorySlidingWindowBackend",
    "RedisSlidingWindowBackend",
    "SlidingWindowRateLimiter",
    "build_rate_limiter",
]


class RateLimiterBackend(Protocol):
    """Protocol describing a sliding window limiter backend."""

    async def hit(self, key: str, *, limit: int, window_seconds: float) -> int:
        """Register a hit for *key* and return the number of requests in the window."""


class InMemorySlidingWindowBackend:
    """Simple asyncio-aware sliding window backend for single-instance deployments."""

    def __init__(self) -> None:
        self._records: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    async def hit(self, key: str, *, limit: int, window_seconds: float) -> int:
        loop = asyncio.get_running_loop()
        now = loop.time()
        async with self._lock:
            bucket = self._records.setdefault(key, deque())
            threshold = now - window_seconds
            while bucket and bucket[0] <= threshold:
                bucket.popleft()
            bucket.append(now)
            return len(bucket)


class RedisSlidingWindowBackend:
    """Redis based backend suitable for horizontally scaled deployments."""

    def __init__(self, client, *, key_prefix: str = "tradepulse:rate") -> None:  # type: ignore[no-untyped-def]
        self._client = client
        self._prefix = key_prefix.rstrip(":")

    async def hit(self, key: str, *, limit: int, window_seconds: float) -> int:
        redis_key = f"{self._prefix}:{key}"
        now = time.time()
        window_start = now - window_seconds
        pipeline = self._client.pipeline()
        pipeline.zremrangebyscore(redis_key, 0, window_start)
        pipeline.zadd(redis_key, {str(now): now})
        pipeline.zcard(redis_key)
        pipeline.expire(redis_key, int(window_seconds) + 1)
        _, _, count, _ = await pipeline.execute()
        return int(count)


class SlidingWindowRateLimiter:
    """Coordinator that selects policies and delegates hit tracking."""

    def __init__(
        self,
        backend: RateLimiterBackend,
        settings: ApiRateLimitSettings,
    ) -> None:
        self._backend = backend
        self._settings = settings

    async def check(self, *, subject: str | None, ip_address: str | None) -> None:
        policy, key = self._resolve_policy(subject, ip_address)
        count = await self._backend.hit(key, limit=policy.max_requests, window_seconds=policy.window_seconds)
        if count > policy.max_requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for this client.",
            )

    def _resolve_policy(self, subject: str | None, ip_address: str | None) -> tuple[RateLimitPolicy, str]:
        if subject:
            specific = self._settings.client_policies.get(subject)
            if specific is not None:
                return specific, f"subject:{subject}"
            return self._settings.default_policy, f"subject:{subject}"
        if ip_address:
            policy = self._settings.unauthenticated_policy or self._settings.default_policy
            return policy, f"ip:{ip_address}"
        return self._settings.default_policy, "anonymous"


def build_rate_limiter(settings: ApiRateLimitSettings) -> SlidingWindowRateLimiter:
    """Instantiate a rate limiter using the most appropriate backend."""

    if settings.redis_url is not None:
        try:
            from redis.asyncio import from_url
        except ModuleNotFoundError as exc:  # pragma: no cover - defensive guard
            raise RuntimeError(
                "Redis-backed rate limiting requires the 'redis' package to be installed."
            ) from exc

        client = from_url(str(settings.redis_url), encoding="utf-8", decode_responses=False)
        backend: RateLimiterBackend = RedisSlidingWindowBackend(client, key_prefix=settings.redis_key_prefix)
    else:
        backend = InMemorySlidingWindowBackend()

    return SlidingWindowRateLimiter(backend, settings)
