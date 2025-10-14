# SPDX-License-Identifier: MIT
"""High-performance indicator pipeline orchestration utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Mapping
from weakref import finalize as _finalize

import numpy as np

from ..utils.memory import ArrayPool
from .base import BaseFeature


@dataclass(slots=True)
class PipelineResult:
    """Container with both feature values and the shared input buffer."""

    values: Mapping[str, Any]
    buffer: np.ndarray
    _cleanup: Callable[[], None] | None = field(default=None, repr=False)
    _finalizer: Any | None = field(default=None, repr=False)

    def release(self) -> None:
        """Return the buffer to the originating pool (idempotent)."""

        if self._cleanup is not None:
            cleanup = self._cleanup
            self._cleanup = None
            if self._finalizer is not None:
                self._finalizer.detach()
                self._finalizer = None
            cleanup()

    def __del__(self) -> None:  # pragma: no cover - best-effort safety net
        self.release()


class IndicatorPipeline:
    """Execute a sequence of indicators using a shared float32 buffer."""

    def __init__(
        self,
        features: Sequence[BaseFeature],
        *,
        dtype: np.dtype | str = np.float32,
        pool: ArrayPool | None = None,
    ) -> None:
        if not features:
            raise ValueError("IndicatorPipeline requires at least one feature")
        self._features = tuple(features)
        self._dtype = np.dtype(dtype)
        self._pool = pool or ArrayPool(self._dtype)

    @property
    def features(self) -> tuple[BaseFeature, ...]:
        return self._features

    def _prepare_buffer(
        self, data: np.ndarray | Sequence[float]
    ) -> tuple[np.ndarray, bool]:
        array = np.asarray(data)
        borrowed = False
        if array.dtype != self._dtype or not array.flags.c_contiguous:
            buffer = self._pool.acquire(array.shape, dtype=self._dtype)
            np.copyto(buffer, array, casting="unsafe")
            array = buffer
            borrowed = True
        return array, borrowed

    def run(self, data: np.ndarray | Sequence[float], **kwargs: Any) -> PipelineResult:
        buffer, borrowed = self._prepare_buffer(data)
        values: dict[str, Any] = {}
        for feature in self._features:
            result = feature.transform(buffer, **kwargs)
            values[result.name] = result.value

        cleanup: Callable[[], None] | None = None
        if borrowed:
            cleanup = partial(self._pool.release, buffer)

        finalizer = None
        if cleanup is not None:
            finalizer = _finalize(buffer, cleanup)

        result = PipelineResult(
            values=values, buffer=buffer, _cleanup=cleanup, _finalizer=finalizer
        )
        return result


__all__ = ["IndicatorPipeline", "PipelineResult"]
