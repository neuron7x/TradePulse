# SPDX-License-Identifier: MIT
from __future__ import annotations
import numpy as np
import networkx as nx
try:
    from scipy.spatial.distance import wasserstein_distance as W1
except Exception:
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

