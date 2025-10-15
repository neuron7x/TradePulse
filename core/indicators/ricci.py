# SPDX-License-Identifier: MIT
"""Ricci curvature-based structural stress indicators for price graphs.

This module turns price histories into discrete graphs and computes
Ollivier–Ricci curvature to characterise structural fragility. The approach
follows the geometric market diagnostics documented in ``docs/indicators.md``
and the resilience monitoring playbooks in ``docs/risk_ml_observability.md``.
By embedding curvature metrics into the feature stack, we meet the governance
requirement that core risk signals expose interpretable topology, as detailed
in ``docs/documentation_governance.md``.

Upstream data arrives from the ingestion pipeline via indicator callers, while
downstream consumers include the feature engineering stack, execution risk
monitoring, and CLI diagnostics in ``interfaces/cli.py``. Key dependencies
include optional NetworkX (with an in-repo fallback) for graph manipulation,
NumPy for numerical work, and SciPy for Wasserstein distances when available.
The module records telemetry using ``core.utils`` helpers to satisfy
traceability expectations laid out in ``docs/quality_gates.md`` and to
coordinate with the governance guardrails documented in ``docs/monitoring.md``.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import warnings
from collections import deque
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

_scipy_linprog = None
_scipy_optimize_spec = importlib.util.find_spec("scipy.optimize")
if _scipy_optimize_spec is not None:  # pragma: no cover - SciPy optional
    from scipy.optimize import linprog as _scipy_linprog


def build_price_graph(prices: np.ndarray, delta: float = 0.005) -> nx.Graph:
    """Quantise a price path into a level graph.

    Args:
        prices: One-dimensional array of strictly positive prices representing a
            single asset history.
        delta: Relative price increment controlling the resolution of quantised
            levels as documented in ``docs/indicators.md``.

    Returns:
        nx.Graph: Undirected graph whose nodes correspond to discretised price
        levels and whose edges capture successive transitions weighted by price
        deltas.

    Examples:
        >>> prices = np.array([100.0, 100.5, 101.0])
        >>> G = build_price_graph(prices, delta=0.01)
        >>> sorted(G.nodes())
        [-0, 1, 2]

    Notes:
        Non-finite prices are removed before quantisation in accordance with the
        cleansing contract in ``docs/documentation_governance.md``. Empty or
        degenerate series yield an empty graph, keeping downstream curvature
        computations stable.
    """
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

def local_distribution(
    G: nx.Graph, node: Any, radius: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Return neighbour identifiers and their normalised transition mass.

    Args:
        G: Graph produced by :func:`build_price_graph` or a compatible structure.
        node: Node identifier whose neighbourhood distribution is required.
        radius: Currently unused, reserved for future extensions involving
            multi-hop neighbourhoods.

    Returns:
        tuple[np.ndarray, np.ndarray]: Pair of arrays ``(indices, weights)``
        where ``indices`` contains neighbour node identifiers and ``weights``
        contains the corresponding transition probabilities. When ``node`` has
        no neighbours, the function returns ``([node], [1.0])`` to represent a
        self-stay mass.

    Notes:
        Edge weights are sanitised to remain finite, matching the governance
        requirements of ``docs/quality_gates.md``. The returned probability
        vector always sums to one within floating-point tolerance.
    """

    neigh = list(G.neighbors(node))
    if not neigh:  # pragma: no cover - defensive guard for isolated nodes
        return (np.array([node], dtype=object), np.array([1.0], dtype=float))

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
        probs = np.full(len(neigh), 1.0 / len(neigh))
    else:
        probs = w_arr / total
    return np.asarray(neigh, dtype=object), probs


def ricci_curvature_edge(G: nx.Graph, x: Any, y: Any) -> float:
    """Evaluate the Ollivier–Ricci curvature for a specific edge.

    Args:
        G: Graph describing price transitions.
        x: Source node identifier.
        y: Target node identifier.

    Returns:
        float: Curvature value in the range ``(-∞, 1]`` where negative values
        denote dispersion and positive values indicate clustering.

    Notes:
        The implementation normalises discrete neighbourhood measures and solves
        the resulting optimal transport problem over their shared support using
        a SciPy linear programme when available or an internal min-cost flow
        fallback otherwise. Shortest-path calculations are hardened through
        :func:`_shortest_path_length_safe`, aligning with the numerical stability
        guidance in ``docs/monitoring.md``.
    """
    if not G.has_edge(x, y):  # pragma: no cover - caller ensures edge exists
        return 0.0
    neigh_x, mass_x = local_distribution(G, x)
    neigh_y, mass_y = local_distribution(G, y)
    if neigh_x.size == 0 or neigh_y.size == 0:
        return 0.0
    d_xy = _shortest_path_length_safe(G, x, y)
    if not np.isfinite(d_xy) or d_xy <= 0:
        return 0.0
    cost_matrix = np.empty((neigh_x.size, neigh_y.size), dtype=float)
    finite_mask = np.zeros_like(cost_matrix, dtype=bool)
    for i, src in enumerate(neigh_x):
        for j, dst in enumerate(neigh_y):
            dist = _shortest_path_length_safe(G, src, dst)
            if np.isfinite(dist):
                cost_matrix[i, j] = float(dist)
                finite_mask[i, j] = True
            else:
                cost_matrix[i, j] = np.inf
    if not finite_mask.any(axis=1).all() or not finite_mask.any(axis=0).all():
        return 0.0
    finite_values = cost_matrix[finite_mask]
    if finite_values.size == 0:
        return 0.0
    max_cost = float(np.max(finite_values))
    if not np.isfinite(max_cost) or max_cost <= 0.0:
        fill_value = 1.0
    else:
        fill_value = max(1.0, max_cost * 10.0)
    cost_matrix = np.where(finite_mask, cost_matrix, fill_value)
    transport_cost = _optimal_transport_distance(cost_matrix, mass_x, mass_y)
    return float(1.0 - transport_cost / d_xy)


def _shortest_path_length_safe(G: nx.Graph, x: Any, y: Any) -> float:
    """Return a robust shortest-path distance tolerant to malformed weights.

    Args:
        G: Graph describing price transitions.
        x: Source node identifier.
        y: Target node identifier.

    Returns:
        float: Weighted path length, falling back to unweighted distance when
        weight metadata is invalid. ``inf`` is returned when no path exists.

    Notes:
        The helper logs debug diagnostics when weight issues arise, which feed
        into the risk observability pipeline in ``docs/risk_ml_observability.md``.
    """

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
        float: Mean curvature value across all edges. ``0.0`` is returned for
        empty graphs.

    Raises:
        RuntimeError: Propagated if asynchronous execution fails to initialise a
            loop.

    Notes:
        High positive curvature implies tightly connected price states, while
        negative curvature indicates dispersion—a signal cross-referenced by the
        monitoring blueprint in ``docs/risk_ml_observability.md``. ``float32``
        accumulation is recommended only when graphs exceed ~50k edges; otherwise
        ``float64`` provides more stable averages.
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
    edges: list[tuple[Any, Any]],
    max_workers: int | None,
) -> list[float]:
    """Evaluate curvature across edges concurrently using asyncio threads."""
    async def _runner() -> list[float]:
        loop = asyncio.get_running_loop()
        executor: ThreadPoolExecutor | None = None
        try:
            if max_workers is not None:
                executor = ThreadPoolExecutor(max_workers=max_workers)
            futures = [
                loop.run_in_executor(executor, ricci_curvature_edge, G, u, v)
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


def _optimal_transport_distance(
    cost_matrix: np.ndarray, supply: np.ndarray, demand: np.ndarray
) -> float:
    """Compute the optimal transport cost between two discrete measures."""

    supply = np.asarray(supply, dtype=float)
    demand = np.asarray(demand, dtype=float)
    supply = np.clip(supply, 0.0, None)
    demand = np.clip(demand, 0.0, None)
    if supply.sum() == 0 or demand.sum() == 0:
        return 0.0
    supply = supply / supply.sum()
    demand = demand / demand.sum()

    if _scipy_linprog is not None:
        result = _solve_transport_scipy(cost_matrix, supply, demand)
        if result is not None:
            return result
        warnings.warn(
            "SciPy linear programming solver failed; using internal min-cost transport",
            RuntimeWarning,
            stacklevel=2,
        )
    return _solve_transport_vam(cost_matrix, supply, demand)


def _solve_transport_scipy(
    cost_matrix: np.ndarray, supply: np.ndarray, demand: np.ndarray
) -> float | None:
    """Solve the transport problem with SciPy's linear programming solver."""

    if _scipy_linprog is None:
        return None
    m, n = cost_matrix.shape
    c = cost_matrix.reshape(m * n)
    A_eq = np.zeros((m + n, m * n), dtype=float)
    for i in range(m):
        A_eq[i, i * n : (i + 1) * n] = 1.0
    for j in range(n):
        A_eq[m + j, j::n] = 1.0
    b_eq = np.concatenate([supply, demand])
    bounds = [(0.0, None)] * (m * n)
    try:
        result = _scipy_linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    except Exception:  # pragma: no cover - SciPy solver edge cases
        return None
    if result.success:
        return float(result.fun)
    return None


_TRANSPORT_TOL = 1e-12


def _solve_transport_vam(
    cost_matrix: np.ndarray, supply: np.ndarray, demand: np.ndarray
) -> float:
    """Solve the transport problem via VAM + MODI heuristics."""

    supply_rem = supply.copy()
    demand_rem = demand.copy()
    m, n = cost_matrix.shape
    allocation = np.zeros((m, n), dtype=float)
    active_rows = {i for i in range(m) if supply_rem[i] > _TRANSPORT_TOL}
    active_cols = {j for j in range(n) if demand_rem[j] > _TRANSPORT_TOL}

    while active_rows and active_cols:
        penalties: list[tuple[float, str, int, int]] = []
        for i in list(active_rows):
            candidates = [
                (cost_matrix[i, j], j)
                for j in active_cols
                if demand_rem[j] > _TRANSPORT_TOL
            ]
            if not candidates:
                continue
            candidates.sort()
            penalty = (candidates[1][0] - candidates[0][0]) if len(candidates) > 1 else candidates[0][0]
            penalties.append((penalty, "row", i, candidates[0][1]))
        for j in list(active_cols):
            candidates = [
                (cost_matrix[i, j], i)
                for i in active_rows
                if supply_rem[i] > _TRANSPORT_TOL
            ]
            if not candidates:
                continue
            candidates.sort()
            penalty = (candidates[1][0] - candidates[0][0]) if len(candidates) > 1 else candidates[0][0]
            penalties.append((penalty, "col", j, candidates[0][1]))
        if not penalties:
            break
        _, axis, idx, partner = max(penalties, key=lambda entry: (entry[0], entry[1] == "row"))
        if axis == "row":
            i, j = idx, partner
        else:
            j, i = idx, partner
        amount = min(supply_rem[i], demand_rem[j])
        allocation[i, j] += amount
        supply_rem[i] -= amount
        demand_rem[j] -= amount
        if supply_rem[i] <= _TRANSPORT_TOL:
            active_rows.discard(i)
        if demand_rem[j] <= _TRANSPORT_TOL:
            active_cols.discard(j)

    for i in range(m):
        if supply_rem[i] > _TRANSPORT_TOL:
            j = int(np.argmin(cost_matrix[i]))
            amount = supply_rem[i]
            allocation[i, j] += amount
            supply_rem[i] = 0.0
            demand_rem[j] = max(demand_rem[j] - amount, 0.0)
    for j in range(n):
        if demand_rem[j] > _TRANSPORT_TOL:
            i = int(np.argmin(cost_matrix[:, j]))
            amount = demand_rem[j]
            allocation[i, j] += amount
            demand_rem[j] = 0.0
            supply_rem[i] = max(supply_rem[i] - amount, 0.0)

    basis = _ensure_transport_basis(allocation, cost_matrix)

    while True:
        u, v = _compute_transport_potentials(cost_matrix, basis)
        reduced = cost_matrix - (u[:, None] + v[None, :])
        mask = ~basis
        if not np.any(mask):
            break
        min_val = float(np.min(np.where(mask, reduced, np.inf)))
        if not np.isfinite(min_val) or min_val >= -_TRANSPORT_TOL:
            break
        entering = np.unravel_index(
            int(np.argmin(np.where(mask, reduced, np.inf))), cost_matrix.shape
        )
        cycle = _find_transport_cycle(basis, entering)
        if cycle is None:
            basis[entering] = True
            basis = _ensure_transport_basis(allocation, cost_matrix)
            continue
        theta_candidates = [
            allocation[i, j]
            for idx, (i, j) in enumerate(cycle)
            if idx % 2 == 1
        ]
        positive_candidates = [val for val in theta_candidates if val > _TRANSPORT_TOL]
        theta = min(positive_candidates) if positive_candidates else min(theta_candidates, default=0.0)
        for idx, (i, j) in enumerate(cycle):
            if idx % 2 == 0:
                allocation[i, j] += theta
            else:
                allocation[i, j] -= theta
                if allocation[i, j] <= _TRANSPORT_TOL:
                    allocation[i, j] = 0.0
                    if (i, j) != entering:
                        basis[i, j] = False
        basis[entering] = True
        basis = _ensure_transport_basis(allocation, cost_matrix)

    return float(np.sum(allocation * cost_matrix))


def _ensure_transport_basis(allocation: np.ndarray, cost_matrix: np.ndarray) -> np.ndarray:
    """Ensure the set of basic variables spans all rows and columns."""

    basis = allocation > _TRANSPORT_TOL
    m, n = allocation.shape
    for i in range(m):
        if not basis[i].any():
            j = int(np.argmin(cost_matrix[i]))
            basis[i, j] = True
    for j in range(n):
        if not basis[:, j].any():
            i = int(np.argmin(cost_matrix[:, j]))
            basis[i, j] = True
    while basis.sum() < (m + n - 1):
        mask = np.where(basis, np.inf, cost_matrix)
        idx = np.unravel_index(int(np.argmin(mask)), allocation.shape)
        basis[idx] = True
    return basis


def _compute_transport_potentials(
    cost_matrix: np.ndarray, basis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute dual potentials for the transport problem."""

    m, n = cost_matrix.shape
    u = np.full(m, np.nan)
    v = np.full(n, np.nan)
    u[0] = 0.0
    queue: deque[tuple[str, int]] = deque([("row", 0)])
    while queue:
        axis, idx = queue.popleft()
        if axis == "row":
            i = idx
            for j in range(n):
                if basis[i, j]:
                    if np.isnan(v[j]):
                        v[j] = cost_matrix[i, j] - u[i]
                        queue.append(("col", j))
        else:
            j = idx
            for i in range(m):
                if basis[i, j]:
                    if np.isnan(u[i]):
                        u[i] = cost_matrix[i, j] - v[j]
                        queue.append(("row", i))
    u = np.nan_to_num(u, nan=0.0)
    v = np.nan_to_num(v, nan=0.0)
    return u, v


def _find_transport_cycle(basis: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]] | None:
    """Construct the alternating cycle created by introducing ``start``."""

    m, n = basis.shape
    i0, j0 = start
    adjacency: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for i in range(m):
        adjacency[("row", i)] = []
    for j in range(n):
        adjacency[("col", j)] = []
    for i in range(m):
        for j in range(n):
            if basis[i, j]:
                adjacency[("row", i)].append(("col", j))
                adjacency[("col", j)].append(("row", i))

    start_node = ("row", int(i0))
    target_node = ("col", int(j0))
    queue: deque[tuple[str, int]] = deque([start_node])
    prev: dict[tuple[str, int], tuple[str, int] | None] = {start_node: None}
    found = False
    while queue:
        node = queue.popleft()
        if node == target_node:
            found = True
            break
        for neigh in adjacency[node]:
            if neigh not in prev:
                prev[neigh] = node
                queue.append(neigh)
    if not found:
        return None

    path: list[tuple[str, int]] = []
    node = target_node
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()

    cycle: list[tuple[int, int]] = [start]
    for idx in range(len(path) - 1):
        a = path[idx]
        b = path[idx + 1]
        if a[0] == "row" and b[0] == "col":
            cycle.append((a[1], b[1]))
        elif a[0] == "col" and b[0] == "row":
            cycle.append((b[1], a[1]))
    cycle.append(start)
    return cycle


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
        """Initialise the feature configuration.

        Args:
            delta: Price quantisation granularity.
            chunk_size: Process edges in chunks for large graphs.
            use_float32: Use ``float32`` precision for memory efficiency.
            parallel_async: Execute curvature computations concurrently via
                asyncio thread pools.
            max_workers: Optional cap for the async worker pool when
                ``parallel_async`` is enabled.
            name: Optional custom name.
        """
        super().__init__(name or "mean_ricci")
        self.delta = float(delta)
        self.chunk_size = chunk_size
        self.use_float32 = use_float32
        self.parallel_async = parallel_async
        self.max_workers = max_workers

    def transform(self, data: np.ndarray, **_: Any) -> FeatureResult:
        """Compute mean Ricci curvature of the price graph.

        Args:
            data: Price array used to build the underlying graph.
            **_: Additional keyword arguments (ignored).

        Returns:
            FeatureResult: Mean curvature value and metadata about graph size.
        """

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

