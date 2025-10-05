# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseFeature

try:
    import networkx as nx  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency guard
    nx = None  # type: ignore
    _NETWORKX_IMPORT_ERROR = exc
else:  # pragma: no cover - executed when dependency available
    _NETWORKX_IMPORT_ERROR = None

try:
    from scipy.spatial.distance import wasserstein_distance as W1
except Exception:  # pragma: no cover - SciPy optional
    W1 = None

__all__ = [
    "build_price_graph",
    "local_distribution",
    "ricci_curvature_edge",
    "RicciCurvatureFeature",
    "MeanRicciFeature",
    "mean_ricci",
]


def _require_networkx() -> Any:
    if nx is None:
        raise ImportError(
            "networkx is required for Ricci curvature features"
        ) from _NETWORKX_IMPORT_ERROR
    return nx


def build_price_graph(prices: np.ndarray, delta: float = 0.005):
    """Quantize price levels into nodes; connect adjacent active levels."""
    nx_mod = _require_networkx()
    p = np.asarray(prices, dtype=float)
    base = p[0]
    levels = np.round((p - base) / (base * delta)).astype(int)
    graph = nx_mod.Graph()
    for i, lv in enumerate(levels):
        graph.add_node(int(lv))
        if i > 0:
            graph.add_edge(int(levels[i - 1]), int(lv), weight=1.0)
    return graph


def local_distribution(G, node: int, radius: int = 1) -> np.ndarray:
    """Return degree-weighted distribution over neighbors within radius."""
    nx_mod = _require_networkx()
    neigh = [n for n in nx_mod.neighbors(G, node)]
    if not neigh:
        return np.array([1.0])
    deg = np.array([nx_mod.degree(G, n) for n in neigh], dtype=float)
    p = deg / deg.sum()
    return p


def ricci_curvature_edge(G, x: int, y: int) -> float:
    """Ollivier-Ricci curvature `kappa(x,y)` for unweighted graphs."""
    nx_mod = _require_networkx()
    if not nx_mod.has_edge(G, x, y):
        return 0.0
    mu_x = local_distribution(G, x)
    mu_y = local_distribution(G, y)
    m = max(len(mu_x), len(mu_y))
    a = np.pad(mu_x, (0, m - len(mu_x)))
    b = np.pad(mu_y, (0, m - len(mu_y)))
    d_xy = 1.0
    dist = W1(a, b) if W1 is not None else _w1_fallback(a, b)
    return float(1.0 - dist / d_xy)


class RicciCurvatureFeature(BaseFeature):
    """Edge-level Ollivier-Ricci curvature."""

    def __init__(self, nodes: tuple[int, int]) -> None:
        if len(nodes) != 2:
            raise ValueError("RicciCurvatureFeature expects a pair of node ids")
        x, y = int(nodes[0]), int(nodes[1])
        super().__init__(
            name="ricci_curvature",
            params={"nodes": (x, y)},
            description="Curvature between two nodes within a price graph.",
        )
        self._nodes = (x, y)

    def transform(self, graph) -> float:
        _require_networkx()
        x, y = self._nodes
        return ricci_curvature_edge(graph, x, y)


class MeanRicciFeature(BaseFeature):
    """Average Ricci curvature across all edges."""

    def __init__(self) -> None:
        super().__init__(
            name="mean_ricci",
            params={},
            description="Mean Ollivier-Ricci curvature across every edge of the graph.",
        )

    def transform(self, graph) -> float:
        nx_mod = _require_networkx()
        if nx_mod.number_of_edges(graph) == 0:
            return 0.0
        curvatures = [ricci_curvature_edge(graph, u, v) for u, v in nx_mod.edges(graph)]
        return float(np.mean(curvatures))


def mean_ricci(G) -> float:
    return MeanRicciFeature().transform(G)


def _w1_fallback(a, b):
    import numpy as _np

    a = _np.asarray(a, dtype=float)
    b = _np.asarray(b, dtype=float)
    a = a / (a.sum() + 1e-12)
    b = b / (b.sum() + 1e-12)
    cdfa = _np.cumsum(a)
    cdfb = _np.cumsum(b)
    return float(_np.abs(cdfa - cdfb).sum()) / len(a)
