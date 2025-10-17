"""Benchmark TradePulse numeric accelerators against NumPy and Numba backends."""

from __future__ import annotations

import argparse
import math
from time import perf_counter
from typing import Callable, Iterable, Sequence

import numpy as np

from core.accelerators.numeric import (
    convolve_numpy_backend,
    convolve_python_backend,
    convolve_rust_backend,
    numpy_available,
    quantiles_numpy_backend,
    quantiles_python_backend,
    quantiles_rust_backend,
    rust_available,
    sliding_windows_numpy_backend,
    sliding_windows_python_backend,
    sliding_windows_rust_backend,
)

try:  # pragma: no cover - optional dependency for benchmarking
    from numba import njit
except Exception:  # pragma: no cover - best-effort import
    njit = None  # type: ignore[assignment]
    NUMBA_AVAILABLE = False
else:  # pragma: no cover - executed when numba is available
    NUMBA_AVAILABLE = True


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _numba_sliding_windows(
        data: np.ndarray, window: int, step: int
    ) -> np.ndarray:  # pragma: no cover - compiled at runtime
        if window <= 0 or step <= 0:
            raise ValueError("window and step must be positive")
        n = data.shape[0]
        if n < window:
            return np.empty((0, window), dtype=np.float64)
        windows = (n - window) // step + 1
        result = np.empty((windows, window), dtype=np.float64)
        for i in range(windows):
            start = i * step
            for j in range(window):
                result[i, j] = data[start + j]
        return result

    @njit(cache=True)
    def _numba_quantiles(
        data: np.ndarray, probabilities: np.ndarray
    ) -> np.ndarray:  # pragma: no cover - compiled at runtime
        n = data.shape[0]
        if n == 0:
            return np.empty(probabilities.shape[0], dtype=np.float64) * np.nan
        sorted_data = np.sort(data)
        result = np.empty(probabilities.shape[0], dtype=np.float64)
        for i in range(probabilities.shape[0]):
            p = probabilities[i]
            if not math.isfinite(p) or p < 0.0 or p > 1.0:
                raise ValueError("probabilities must lie within [0, 1]")
            position = p * (n - 1)
            lower = int(math.floor(position))
            upper = int(math.ceil(position))
            if lower == upper:
                result[i] = sorted_data[lower]
            else:
                weight = position - lower
                result[i] = (
                    sorted_data[lower]
                    + (sorted_data[upper] - sorted_data[lower]) * weight
                )
        return result

    @njit(cache=True)
    def _numba_full_convolution(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        result_length = signal.shape[0] + kernel.shape[0] - 1
        result = np.zeros(result_length, dtype=np.float64)
        for i in range(signal.shape[0]):
            for j in range(kernel.shape[0]):
                result[i + j] += signal[i] * kernel[j]
        return result

    @njit(cache=True)
    def _numba_convolve(
        signal: np.ndarray, kernel: np.ndarray, mode_code: int
    ) -> np.ndarray:
        full = _numba_full_convolution(signal, kernel)
        n = signal.shape[0]
        m = kernel.shape[0]
        if mode_code == 0:  # full
            return full
        if mode_code == 1:  # same
            target = n if n >= m else m
            pad = (full.shape[0] - target) // 2
            return full[pad : pad + target]
        # valid
        if n >= m:
            length = n - m + 1
            start = m - 1
            end = start + length
        else:
            length = m - n + 1
            start = n - 1
            end = start + length
        return full[start:end]

    NUMBA_MODE_MAP = {"full": 0, "same": 1, "valid": 2}
else:
    NUMBA_MODE_MAP: dict[str, int] = {}


BenchmarkCallable = Callable[[], object]


def _consume(result: object) -> float:
    if isinstance(result, np.ndarray):
        return float(result.sum())
    if isinstance(result, (list, tuple)):
        total = 0.0
        stack: list[Iterable[float] | float] = [result]  # type: ignore[list-item]
        while stack:
            current = stack.pop()
            if isinstance(current, (list, tuple)):
                stack.extend(current)  # type: ignore[arg-type]
            else:
                total += float(current)
        return total
    return 0.0


def _benchmark(name: str, func: BenchmarkCallable, repeat: int, warmup: int) -> float:
    for _ in range(warmup):
        _consume(func())
    samples: list[float] = []
    for _ in range(repeat):
        start = perf_counter()
        _consume(func())
        samples.append(perf_counter() - start)
    return min(samples)


def _register_sliding_windows(
    data: np.ndarray, window: int, step: int
) -> list[tuple[str, BenchmarkCallable]]:
    registrations: list[tuple[str, BenchmarkCallable]] = [
        ("python", lambda: sliding_windows_python_backend(data, window, step)),
    ]
    if numpy_available():
        registrations.append(
            ("numpy", lambda: sliding_windows_numpy_backend(data, window, step))
        )
    if NUMBA_AVAILABLE:
        registrations.append(
            ("numba", lambda: _numba_sliding_windows(data, window, step))
        )
    if rust_available():
        registrations.append(
            ("rust", lambda: sliding_windows_rust_backend(data, window, step))
        )
    return registrations


def _register_quantiles(
    data: np.ndarray, probabilities: Sequence[float]
) -> list[tuple[str, BenchmarkCallable]]:
    registrations: list[tuple[str, BenchmarkCallable]] = [
        ("python", lambda: quantiles_python_backend(data, probabilities)),
    ]
    if numpy_available():
        registrations.append(
            ("numpy", lambda: quantiles_numpy_backend(data, probabilities))
        )
    if NUMBA_AVAILABLE:
        probs_np = np.asarray(list(probabilities), dtype=np.float64)
        registrations.append(("numba", lambda: _numba_quantiles(data, probs_np)))
    if rust_available():
        registrations.append(
            ("rust", lambda: quantiles_rust_backend(data, probabilities))
        )
    return registrations


def _register_convolution(
    signal: np.ndarray, kernel: np.ndarray, mode: str
) -> list[tuple[str, BenchmarkCallable]]:
    registrations: list[tuple[str, BenchmarkCallable]] = [
        ("python", lambda: convolve_python_backend(signal, kernel, mode=mode)),
    ]
    if numpy_available():
        registrations.append(
            ("numpy", lambda: convolve_numpy_backend(signal, kernel, mode=mode))
        )
    if NUMBA_AVAILABLE:
        mode_code = NUMBA_MODE_MAP[mode]
        registrations.append(
            ("numba", lambda: _numba_convolve(signal, kernel, mode_code))
        )
    if rust_available():
        registrations.append(
            ("rust", lambda: convolve_rust_backend(signal, kernel, mode=mode))
        )
    return registrations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size", type=int, default=200_000, help="Length of the random data vector"
    )
    parser.add_argument("--window", type=int, default=64, help="Sliding window size")
    parser.add_argument("--step", type=int, default=4, help="Sliding window step")
    parser.add_argument(
        "--mode",
        choices=("full", "same", "valid"),
        default="same",
        help="Convolution mode to benchmark",
    )
    parser.add_argument("--repeat", type=int, default=5, help="Benchmark repetitions")
    parser.add_argument(
        "--warmup", type=int, default=1, help="Warmup iterations before timing"
    )
    args = parser.parse_args()

    if not numpy_available():
        raise SystemExit("NumPy is required to run the numeric benchmarks")

    rng = np.random.default_rng(2024)
    data = rng.normal(size=args.size).astype(np.float64)
    probabilities = (0.1, 0.5, 0.9)
    kernel_size = max(3, args.window // 2)
    kernel = rng.normal(size=kernel_size).astype(np.float64)

    suites = {
        "sliding_windows": _register_sliding_windows(data, args.window, args.step),
        "quantiles": _register_quantiles(data, probabilities),
        "convolve": _register_convolution(data, kernel, args.mode),
    }

    print("TradePulse numeric accelerator benchmarks")
    print(f"  data size: {args.size}")
    print(f"  window:    {args.window}")
    print(f"  step:      {args.step}")
    print(f"  mode:      {args.mode}")
    print(f"  numpy:     {'yes' if numpy_available() else 'no'}")
    print(f"  rust:      {'yes' if rust_available() else 'no'}")
    print(f"  numba:     {'yes' if NUMBA_AVAILABLE else 'no'}\n")

    for suite_name, registrations in suites.items():
        print(f"== {suite_name} ==")
        for backend, func in registrations:
            try:
                elapsed = _benchmark(
                    f"{suite_name}:{backend}", func, args.repeat, args.warmup
                )
            except Exception as exc:
                print(f"  {backend:>8}: error ({exc})")
                continue
            throughput = args.size / elapsed / 1e6 if elapsed > 0 else float("inf")
            print(
                f"  {backend:>8}: {elapsed * 1e3:8.3f} ms  ({throughput:8.3f} M items/s)"
            )
        print()

    print(
        "Tip: run with `python bench/bench_numeric_accelerators.py --help` for options."
    )


if __name__ == "__main__":
    main()
