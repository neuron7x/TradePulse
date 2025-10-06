# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .base import BaseFeature, FeatureResult

try:
    import networkx as nx
except Exception:  # pragma: no cover - fallback for lightweight environments
    class _SimpleGraph:
        def __init__(self) -> None:
            self._adj: dict[int, set[int]] = {}

        def add_node(self, node: int) -> None:
            self._adj.setdefault(int(node), set())

        def add_nodes_from(self, nodes: Iterable[int]) -> None:
            for node in nodes:
                self.add_node(int(node))

        def add_edge(self, u: int, v: int, weight: float | None = None) -> None:
            self.add_node(int(u))
            self.add_node(int(v))
            self._adj[int(u)].add(int(v))
            self._adj[int(v)].add(int(u))

        def neighbors(self, node: int) -> Iterable[int]:
            return tuple(self._adj.get(int(node), ()))

        def degree(self, node: int | None = None) -> Iterable[tuple[int, int]] | int:
            if node is None:
                return tuple((n, len(neigh)) for n, neigh in self._adj.items())
            return len(self._adj.get(int(node), ()))

        def nodes(self) -> Iterable[int]:
            return tuple(self._adj.keys())

        def number_of_edges(self) -> int:
            return sum(len(neigh) for neigh in self._adj.values()) // 2

        def edges(self) -> Iterable[tuple[int, int]]:
            seen: set[tuple[int, int]] = set()
            for u, neigh in self._adj.items():
                for v in neigh:
                    edge = (min(u, v), max(u, v))
                    if edge not in seen:
                        seen.add(edge)
                        yield edge

        def has_edge(self, u: int, v: int) -> bool:
            return int(v) in self._adj.get(int(u), set())

        def number_of_nodes(self) -> int:
            return len(self._adj)

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
    base = p[0]
    levels = np.round((p - base) / (base * delta)).astype(int)
    G = nx.Graph()
    for i, lv in enumerate(levels):
        G.add_node(int(lv))
        if i>0:
            G.add_edge(int(levels[i-1]), int(lv), weight=1.0)
    return G

def local_distribution(G: nx.Graph, node: int, radius: int = 1) -> np.ndarray:
    """Return degree-weighted distribution over neighbors within radius."""
    neigh = [n for n in G.neighbors(node)]
    if not neigh:
        return np.array([1.0])
    deg = np.array([G.degree(n) for n in neigh], dtype=float)
    p = deg / deg.sum()
    return p

def ricci_curvature_edge(G: nx.Graph, x: int, y: int) -> float:
    """Ollivier–Ricci curvature κ(x,y) = 1 - W1(μ_x, μ_y)/d(x,y) for unweighted graphs."""
    if not G.has_edge(x, y):
        return 0.0
    mu_x = local_distribution(G, x)
    mu_y = local_distribution(G, y)
    # for simple comparison, map distributions to common support by padding
    m = max(len(mu_x), len(mu_y))
    a = np.pad(mu_x, (0, m-len(mu_x)))
    b = np.pad(mu_y, (0, m-len(mu_y)))
    d_xy = 1.0  # unweighted graph
    dist = W1(a, b) if W1 is not None else _w1_fallback(a, b)
    return float(1.0 - dist / d_xy)

def mean_ricci(G: nx.Graph) -> float:
    if G.number_of_edges() == 0:
        return 0.0
    curv = [ricci_curvature_edge(G, u, v) for u, v in G.edges()]
    return float(np.mean(curv))

def _w1_fallback(a, b):
    import numpy as _np
    # normalize
    a = _np.asarray(a, dtype=float); b = _np.asarray(b, dtype=float)
    a = a / (a.sum() + 1e-12); b = b / (b.sum() + 1e-12)
    cdfa = _np.cumsum(a); cdfb = _np.cumsum(b)
    return float(_np.abs(cdfa - cdfb).sum()) / len(a)


class MeanRicciFeature(BaseFeature):
    """Feature computing mean Ollivier–Ricci curvature of a price graph."""

    def __init__(self, delta: float = 0.005, *, name: str | None = None) -> None:
        super().__init__(name or "mean_ricci")
        self.delta = float(delta)

    def transform(self, data: np.ndarray, **_: Any) -> FeatureResult:
        G = build_price_graph(data, delta=self.delta)
        value = mean_ricci(G)
        metadata = {"delta": self.delta, "nodes": G.number_of_nodes()}
        return FeatureResult(name=self.name, value=value, metadata=metadata)


__all__ = [
    "build_price_graph",
    "local_distribution",
    "ricci_curvature_edge",
    "mean_ricci",
    "MeanRicciFeature",
]

