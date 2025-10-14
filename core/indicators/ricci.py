# SPDX-License-Identifier: MIT
"""Ricci curvature-based structural stress indicators for price graphs.

This module turns price histories into discrete graphs and computes
Ollivier–Ricci curvature to characterise structural fragility. The approach
follows the geometric market diagnostics documented in ``docs/indicators.md``
and the resilience monitoring playbooks in ``docs/risk_ml_observability.md``.
By embedding curvature metrics into the feature stack, we meet the governance
requirement that core risk signals expose interpretable topology, as detailed
in ``docs/documentation_governance.md``.

Key dependencies include optional NetworkX (with an in-repo fallback) for graph
manipulation, NumPy for numerical work, and SciPy for Wasserstein distances when
available. The module records telemetry using ``core.utils`` helpers to satisfy
traceability expectations laid out in ``docs/quality_gates.md``.
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Literal

import numpy as np

from ..utils.logging import get_logger
from ..utils.metrics import get_metrics_collector
from .base import BaseFeature, FeatureResult

_logger = get_logger(__name__)
_metrics = get_metrics_collector()


def _log_debug_enabled() -> bool:
    base_logger = getattr(_logger, "logger", None)
    check = getattr(base_logger, "isEnabledFor", None)
    return bool(check and check(logging.DEBUG))

try:
    import networkx as nx
except Exception:  # pragma: no cover - fallback for lightweight environments
    class _SimpleGraph:
        def __init__(self) -> None:
            self._adj: dict[int, dict[int, float]] = {}

        def add_node(self, node: int) -> None:
            self._adj.setdefault(int(node), {})

        def add_nodes_from(self, nodes: Iterable[int]) -> None:
            for node in nodes:
                self.add_node(int(node))

        def add_edge(self, u: int, v: int, weight: float | None = None) -> None:
            self.add_node(int(u))
            self.add_node(int(v))
            w = float(weight if weight is not None else 1.0)
            self._adj[int(u)][int(v)] = w
            self._adj[int(v)][int(u)] = w

        def neighbors(self, node: int) -> Iterable[int]:
            return tuple(self._adj.get(int(node), ()))

        def degree(
            self,
            node: int | None = None,
            weight: str | None = None,
        ) -> Iterable[tuple[int, float]] | float:
            if node is None:
                if weight:
                    return tuple((n, sum(neigh.values())) for n, neigh in self._adj.items())
                return tuple((n, len(neigh)) for n, neigh in self._adj.items())
            neigh = self._adj.get(int(node), {})
            return sum(neigh.values()) if weight else len(neigh)

        def nodes(self) -> Iterable[int]:
            return tuple(self._adj.keys())

        def number_of_edges(self) -> int:
            return sum(len(neigh) for neigh in self._adj.values()) // 2

        def edges(
            self,
            data: bool = False,
        ) -> Iterable[tuple[int, int] | tuple[int, int, dict[str, float]]]:
            seen: set[tuple[int, int]] = set()
            for u, neigh in self._adj.items():
                for v in neigh:
                    edge = (min(u, v), max(u, v))
                    if edge not in seen:
                        seen.add(edge)
                        if data:
                            yield (edge[0], edge[1], {"weight": neigh[v]})
                        else:
                            yield edge

        def has_edge(self, u: int, v: int) -> bool:
            return int(v) in self._adj.get(int(u), set())

        def number_of_nodes(self) -> int:
            return len(self._adj)

        def shortest_path_length(
            self,
            source: int,
            target: int,
            weight: str | None = None,
        ) -> float:
            import heapq

            if source == target:
                return 0.0
            distances = {source: 0.0}
            heap: list[tuple[float, int]] = [(0.0, source)]
            while heap:
                dist, node = heapq.heappop(heap)
                if node == target:
                    return dist
                if dist > distances.get(node, float("inf")):
                    continue
                for neigh, w in self._adj.get(node, {}).items():
                    step = w if weight else 1.0
                    nd = dist + step
                    if nd < distances.get(neigh, float("inf")):
                        distances[neigh] = nd
                        heapq.heappush(heap, (nd, neigh))
            return float("inf")

        def get_edge_data(
            self,
            u: int,
            v: int,
            default: dict[str, float] | None = None,
        ) -> dict[str, float] | None:
            weight = self._adj.get(int(u), {}).get(int(v))
            if weight is None:
                return default
            return {"weight": weight}

    class _NXModule:  # pragma: no cover
        Graph = _SimpleGraph

    nx = _NXModule()  # type: ignore[assignment]

try:  # pragma: no cover - SciPy optional
    from scipy.spatial.distance import wasserstein_distance as W1
except Exception:  # pragma: no cover
    W1 = None


def build_price_graph(prices: np.ndarray, delta: float = 0.005) -> nx.Graph:
    """Quantize price levels into nodes; connect adjacent active levels."""
    p = np.asarray(prices, dtype=float)
    mask = np.isfinite(p)
    if mask.sum() < 2:
        return nx.Graph()
    p = p[mask]
    base = p[0]
    scale = float(abs(base))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    levels = np.round((p - base) / (scale * delta)).astype(int)
    G = nx.Graph()
    for i, lv in enumerate(levels):
        G.add_node(int(lv))
        if i > 0:
            weight = float(abs(p[i] - p[i - 1])) + 1.0
            G.add_edge(int(levels[i - 1]), int(lv), weight=weight)
    return G

def local_distribution(G: nx.Graph, node: int, radius: int = 1) -> np.ndarray:
    """Return degree-weighted distribution over neighbors within radius."""
    neigh = [n for n in G.neighbors(node)]
    if not neigh:  # pragma: no cover - defensive guard for isolated nodes
        return np.array([1.0])
    weights = []
    for n in neigh:
        data = G.get_edge_data(node, n, default={"weight": 1.0})
        weight = float(data.get("weight", 1.0))
        if not np.isfinite(weight):
            weight = 1.0
        weights.append(weight)
    w_arr = np.asarray(weights, dtype=float)
    total = w_arr.sum()
    if total == 0:  # pragma: no cover - degenerate weights
        return np.full(len(neigh), 1.0 / len(neigh))
    return w_arr / total

def ricci_curvature_edge(G: nx.Graph, x: int, y: int) -> float:
    """Ollivier–Ricci curvature κ(x,y) = 1 - W1(μ_x, μ_y)/d(x,y) for unweighted graphs."""
    if not G.has_edge(x, y):  # pragma: no cover - caller ensures edge exists
        return 0.0
    mu_x = local_distribution(G, x)
    mu_y = local_distribution(G, y)
    # for simple comparison, map distributions to common support by padding
    m = max(len(mu_x), len(mu_y))
    a = np.pad(mu_x, (0, m-len(mu_x)))
    b = np.pad(mu_y, (0, m-len(mu_y)))
    d_xy = _shortest_path_length_safe(G, x, y)
    if not np.isfinite(d_xy) or d_xy <= 0:
        return 0.0
    if W1 is None:
        warnings.warn(
            "SciPy unavailable; using discrete Wasserstein approximation for Ricci curvature",
            RuntimeWarning,
            stacklevel=2,
        )
    dist = W1(a, b) if W1 is not None else _w1_fallback(a, b)
    return float(1.0 - dist / d_xy)


def _shortest_path_length_safe(G: nx.Graph, x: int, y: int) -> float:
    """Return a robust shortest path length that tolerates bad edge weights."""

    def _call_shortest_path(graph: nx.Graph, weight: str | None) -> float:
        if hasattr(graph, "shortest_path_length"):
            return graph.shortest_path_length(x, y, weight=weight)
        return nx.shortest_path_length(graph, x, y, weight=weight)

    try:
        return float(_call_shortest_path(G, "weight"))
    except ValueError as exc:
        if _log_debug_enabled():
            _logger.debug(
                "ricci.shortest_path: falling back to unweighted distance for edge (%s,%s)",
                x,
                y,
                error=str(exc),
            )
        try:
            return float(_call_shortest_path(G, None))
        except Exception:
            return float("inf")

def mean_ricci(
    G: nx.Graph,
    *,
    chunk_size: int | None = None,
    use_float32: bool = False,
    parallel: Literal["none", "async"] = "none",
    max_workers: int | None = None,
) -> float:
    """Compute the mean Ollivier–Ricci curvature of a price graph.

    Args:
        G: Input graph whose edge weights encode price transition costs.
        chunk_size: Optional batch size for edge iteration to bound memory usage.
        use_float32: When ``True``, accumulate in ``float32`` as a performance
            trade-off.
        parallel: Execution strategy. Set to ``"async"`` to evaluate edges via an
            asyncio-backed thread pool, mirroring the scaling guidance in
            ``docs/execution.md``.
        max_workers: Upper bound for the thread pool when ``parallel`` is async.

    Returns:
        Mean curvature value across all edges. ``0.0`` is returned for empty
        graphs.

    Raises:
        RuntimeError: Propagated if asynchronous execution fails to initialise a
            loop.

    Notes:
        High positive curvature implies tightly connected price states, while
        negative curvature indicates dispersion—a signal cross-referenced by the
        monitoring blueprint in ``docs/risk_ml_observability.md``.
    """
    with _logger.operation(
        "mean_ricci",
        edges=G.number_of_edges(),
        nodes=G.number_of_nodes(),
        chunk_size=chunk_size,
        use_float32=use_float32,
        parallel=parallel,
    ):
        if G.number_of_edges() == 0:
            return 0.0

        edges = list(G.edges())

        # Chunked processing for large graphs
        if chunk_size is not None and len(edges) > chunk_size:
            dtype = np.float32 if use_float32 else float
            curvatures = []

            for i in range(0, len(edges), chunk_size):
                chunk_edges = edges[i:i + chunk_size]
                chunk_curv = [ricci_curvature_edge(G, u, v) for u, v in chunk_edges]
                curvatures.extend(chunk_curv)

            return float(np.mean(np.array(curvatures, dtype=dtype)))

        # Standard processing
        if parallel == "async":
            curv = _run_ricci_async(G, edges, max_workers)
        else:
            curv = [ricci_curvature_edge(G, u, v) for u, v in edges]
        dtype = np.float32 if use_float32 else float
        if not curv:  # pragma: no cover - empty graph handled above
            return 0.0
        arr = np.array(curv, dtype=dtype)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:  # pragma: no cover - defensive guard
            return 0.0
        return float(np.mean(arr))

def _run_ricci_async(
    G: nx.Graph,
    edges: list[tuple[int, int]],
    max_workers: int | None,
) -> list[float]:
    async def _runner() -> list[float]:
        loop = asyncio.get_running_loop()
        executor: ThreadPoolExecutor | None = None
        try:
            if max_workers is not None:
                executor = ThreadPoolExecutor(max_workers=max_workers)
            futures = [
                loop.run_in_executor(executor, ricci_curvature_edge, G, int(u), int(v))
                for u, v in edges
            ]
            return await asyncio.gather(*futures)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    try:
        return asyncio.run(_runner())
    except RuntimeError as exc:
        if "event loop is running" not in str(exc):
            raise
        new_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(_runner())
        finally:
            asyncio.set_event_loop(None)
            new_loop.close()


def _w1_fallback(a, b):
    import numpy as _np
    # normalize
    a = _np.asarray(a, dtype=float)
    b = _np.asarray(b, dtype=float)
    a = a / (a.sum() + 1e-12)
    b = b / (b.sum() + 1e-12)
    cdfa = _np.cumsum(a)
    cdfb = _np.cumsum(b)
    return float(_np.abs(cdfa - cdfb).sum()) / len(a)


class MeanRicciFeature(BaseFeature):
    """Feature wrapper for mean Ollivier–Ricci curvature.

    The feature converts a univariate price series into a quantised graph using
    :func:`build_price_graph` and then averages edge-level curvature. This is the
    production-ready implementation referenced in ``docs/indicators.md`` and the
    ``docs/risk_ml_observability.md`` control blueprint.

    Attributes are configured via the constructor to align the feature with
    portfolio monitoring guidelines (see ``docs/monitoring.md``).
    """

    def __init__(
        self,
        delta: float = 0.005,
        *,
        chunk_size: int | None = None,
        use_float32: bool = False,
        parallel_async: bool = False,
        max_workers: int | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize mean Ricci curvature feature.

        Args:
            delta: Price quantization granularity (default: 0.005)
            chunk_size: Process edges in chunks for large graphs (default: None)
            use_float32: Use float32 precision for memory efficiency (default: False)
            parallel_async: Execute curvature computations concurrently via
                asyncio thread pools (default: False)
            max_workers: Optional cap for the async worker pool (default: None)
            name: Optional custom name (default: "mean_ricci")
        """
        super().__init__(name or "mean_ricci")
        self.delta = float(delta)
        self.chunk_size = chunk_size
        self.use_float32 = use_float32
        self.parallel_async = parallel_async
        self.max_workers = max_workers

    def transform(self, data: np.ndarray, **_: Any) -> FeatureResult:
        """Compute mean Ricci curvature of price graph."""

        with _metrics.measure_feature_transform(self.name, "ricci"):
            G = build_price_graph(data, delta=self.delta)
            value = mean_ricci(
                G,
                chunk_size=self.chunk_size,
                use_float32=self.use_float32,
                parallel="async" if self.parallel_async else "none",
                max_workers=self.max_workers,
            )
            _metrics.record_feature_value(self.name, value)
            metadata: dict[str, Any] = {
                "delta": self.delta,
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
            }
            if self.use_float32:
                metadata["use_float32"] = True
            if self.chunk_size is not None:
                metadata["chunk_size"] = self.chunk_size
            if self.parallel_async:
                metadata["parallel"] = "async"
            if self.max_workers is not None:
                metadata["max_workers"] = self.max_workers
            return FeatureResult(name=self.name, value=value, metadata=metadata)



__all__ = [
    "build_price_graph",
    "local_distribution",
    "ricci_curvature_edge",
    "mean_ricci",
    "MeanRicciFeature",
]

