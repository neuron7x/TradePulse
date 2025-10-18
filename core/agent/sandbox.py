# SPDX-License-Identifier: MIT
"""Sandbox execution utilities for isolating strategy evaluations."""

from __future__ import annotations

import dataclasses
import math
import os
import time
from multiprocessing.connection import Connection, wait
from typing import Any, Dict

try:  # pragma: no cover - ``resource`` is unavailable on Windows
    import resource
except ModuleNotFoundError:  # pragma: no cover - handled gracefully in runtime
    resource = None  # type: ignore[assignment]

from multiprocessing import get_context


@dataclasses.dataclass(frozen=True)
class SandboxLimits:
    """Resource governance configuration for :class:`StrategySandbox`."""

    cpu_time_seconds: float | None = 2.0
    """Soft CPU time cap enforced with :func:`resource.setrlimit` when available."""

    wall_time_seconds: float | None = 5.0
    """Maximum wall clock time before the sandbox process is terminated."""

    memory_bytes: int | None = 512 * 1024 * 1024
    """Address space limit applied via ``RLIMIT_AS`` when supported."""

    nice_base: int = 0
    """Base niceness increment to apply to sandbox processes (positive lowers priority)."""

    nice_step: int = 1
    """Incremental niceness applied per-priority level passed to :meth:`StrategySandbox.run`."""


@dataclasses.dataclass(frozen=True)
class SandboxResult:
    """Serializable payload returned from sandbox execution."""

    strategy: Any
    score: float


class StrategySandboxError(RuntimeError):
    """Raised when a strategy fails within the sandbox environment."""

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.__cause__ = cause


class StrategySandbox:
    """Execute strategies in isolated, resource-governed subprocesses."""

    def __init__(self, *, limits: SandboxLimits | None = None, start_method: str = "spawn") -> None:
        self._limits = limits or SandboxLimits()
        self._ctx = get_context(start_method)

    # ------------------------------------------------------------------
    def run(self, strategy: Any, data: Any, *, priority: int = 0) -> SandboxResult:
        """Execute *strategy* against *data* inside an isolated sandbox.

        ``priority`` is an integer where larger positive values lower the CPU
        scheduling priority of the sandbox process. Negative values request
        higher priority and will be honoured when permitted by the OS.
        """

        parent_conn, child_conn = self._ctx.Pipe(duplex=False)
        limits = self._limits

        process = self._ctx.Process(
            target=_sandbox_worker,
            args=(child_conn, strategy, data, limits, priority),
            name=f"strategy-sandbox-{getattr(strategy, 'name', 'unknown')}",
        )
        process.daemon = False
        process.start()
        child_conn.close()

        try:
            payload = self._wait_for_payload(parent_conn, limits.wall_time_seconds)
        finally:
            parent_conn.close()

        if payload is None:
            self._terminate_process(process)
            raise StrategySandboxError(
                "Strategy execution timed out in sandbox", cause=TimeoutError()
            )

        process.join(timeout=0.0)
        if process.is_alive():
            self._terminate_process(process)

        status = payload.get("status")
        if status != "ok":
            error = payload.get("error")
            message = payload.get("message", "Strategy sandbox failed")
            if isinstance(error, BaseException):
                raise StrategySandboxError(message, cause=error) from error
            raise StrategySandboxError(message)

        result: SandboxResult = payload["result"]
        return result

    # ------------------------------------------------------------------
    def _wait_for_payload(
        self, conn: Connection, timeout: float | None
    ) -> Dict[str, Any] | None:
        if timeout is not None:
            timeout = max(0.0, float(timeout))

        start = time.monotonic()
        while True:
            remaining: float | None
            if timeout is None:
                remaining = None
            else:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    return None
                remaining = timeout - elapsed

            ready = wait([conn], timeout=remaining)
            if not ready:
                return None
            try:
                payload = conn.recv()
            except EOFError:
                return None
            if isinstance(payload, dict):
                return payload
            # Defensive: ignore unexpected payloads.

    def _terminate_process(self, process) -> None:
        try:
            if process.is_alive():
                process.kill()
        finally:
            process.join(timeout=0.0)


def _sandbox_worker(
    conn: Connection,
    strategy: Any,
    data: Any,
    limits: SandboxLimits,
    priority: int,
) -> None:
    try:
        _apply_limits(limits, priority)
        score = float(strategy.simulate_performance(data))
        result = SandboxResult(strategy=strategy, score=score)
        conn.send({"status": "ok", "result": result})
    except BaseException as exc:  # pragma: no cover - defensive guard
        conn.send({"status": "error", "error": exc, "message": str(exc)})
    finally:
        conn.close()


def _apply_limits(limits: SandboxLimits, priority: int) -> None:
    _set_priority(limits, priority)
    if resource is None:
        return

    if limits.cpu_time_seconds is not None:
        cpu_seconds = max(1, int(math.ceil(limits.cpu_time_seconds)))
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        except (ValueError, OSError):  # pragma: no cover - defensive fallback
            pass

    if limits.memory_bytes is not None:
        memory = max(1, int(limits.memory_bytes))
        for limit_name in ("RLIMIT_AS", "RLIMIT_DATA"):
            if not hasattr(resource, limit_name):
                continue
            try:
                limit = getattr(resource, limit_name)
                resource.setrlimit(limit, (memory, memory))
            except (ValueError, OSError):  # pragma: no cover - defensive fallback
                continue


def _set_priority(limits: SandboxLimits, priority: int) -> None:
    try:
        increment = limits.nice_base + limits.nice_step * priority
        if increment:
            os.nice(increment)
    except (AttributeError, OSError):  # pragma: no cover - unsupported platform
        pass


__all__ = [
    "SandboxLimits",
    "SandboxResult",
    "StrategySandbox",
    "StrategySandboxError",
]

