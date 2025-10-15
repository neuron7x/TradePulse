# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from src.security.secret_manager import (
    SecretManager,
    SecretNotFoundError,
    SecretProvider,
    SecretValue,
    set_secret_manager,
)


@dataclass
class SecretManagerHarness:
    manager: SecretManager
    provider: "_InMemorySecretProvider"
    clock: "_MutableClock"
    metrics: list[tuple[str, dict[str, str]]]


class _MutableClock:
    def __init__(self, start: datetime | None = None) -> None:
        self._current = start or datetime(2025, 1, 1, tzinfo=timezone.utc)

    def advance(self, delta: timedelta) -> None:
        self._current += delta

    def __call__(self) -> datetime:
        return self._current


class _InMemorySecretProvider(SecretProvider):
    def __init__(self, clock: Callable[[], datetime]) -> None:
        self._clock = clock
        self._secrets: dict[str, SecretValue] = {}

    def set_secret(
        self,
        secret_id: str,
        value: str,
        *,
        ttl: timedelta = timedelta(hours=1),
        version: str | None = None,
    ) -> None:
        expires = self._clock() + ttl
        fingerprint = version or hashlib.sha256(f"{secret_id}:{value}:{expires.isoformat()}".encode("utf-8")).hexdigest()
        self._secrets[secret_id] = SecretValue(value=value, version=fingerprint, expires_at=expires)

    def fetch(self, secret_id: str) -> SecretValue:
        try:
            value = self._secrets[secret_id]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise SecretNotFoundError(secret_id) from exc
        return SecretValue(value=value.value, version=value.version, expires_at=value.expires_at)

    def secret_exists(self, secret_id: str) -> bool:
        return secret_id in self._secrets


@pytest.fixture()
def secret_manager_harness() -> SecretManagerHarness:
    clock = _MutableClock()
    provider = _InMemorySecretProvider(clock)
    metrics: list[tuple[str, dict[str, str]]] = []

    def _capture(event: str, attributes: dict[str, str]) -> None:
        metrics.append((event, dict(attributes)))

    manager = SecretManager(provider, refresh_margin=timedelta(seconds=30), clock=clock, metrics_hook=_capture)
    manager.start()
    set_secret_manager(manager)
    harness = SecretManagerHarness(manager=manager, provider=provider, clock=clock, metrics=metrics)
    try:
        yield harness
    finally:
        manager.stop()
        set_secret_manager(None)


@pytest.fixture(autouse=True)
def _set_seed():
    np.random.seed(42)
    yield
    np.random.seed(42)


@pytest.fixture
def sin_wave() -> np.ndarray:
    t = np.linspace(0, 4 * np.pi, 512, endpoint=False)
    return np.sin(t)


@pytest.fixture
def brownian_motion() -> np.ndarray:
    steps = np.random.normal(0, 1, size=4096)
    return np.cumsum(steps)


@pytest.fixture
def uniform_series() -> np.ndarray:
    return np.linspace(-1.0, 1.0, 300)


@pytest.fixture
def peaked_series() -> np.ndarray:
    data = np.zeros(200)
    data[:100] = -0.5
    data[100:] = 0.5
    return data


@pytest.fixture
def price_dataframe(tmp_path) -> pd.DataFrame:
    prices = np.linspace(100.0, 110.0, 50)
    df = pd.DataFrame({"price": prices})
    csv_path = tmp_path / "prices.csv"
    df.to_csv(csv_path, index=False)
    return pd.read_csv(csv_path)
