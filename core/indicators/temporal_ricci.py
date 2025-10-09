
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# --------- Lightweight Graph (no networkx) ---------
class LightGraph:
    def __init__(self, n: int):
        self.n = n
        self.adj: List[Dict[int, float]] = [dict() for _ in range(n)]  # neighbor -> weight
        self._edges_cache: Optional[List[Tuple[int, int]]] = None

    def add_edge(self, i: int, j: int, w: float = 1.0):
        if i == j:
            return
        weight = float(w)
        self.adj[i][j] = weight
        self.adj[j][i] = weight
        self._edges_cache = None

    def edges(self) -> List[Tuple[int,int]]:
        if self._edges_cache is None:
            seen = set()
            edges: List[Tuple[int, int]] = []
            for i in range(self.n):
                for j in self.adj[i].keys():
                    e = (min(i, j), max(i, j))
                    if e not in seen:
                        seen.add(e)
                        edges.append(e)
            self._edges_cache = edges
        return list(self._edges_cache)

    def neighbors(self, i: int) -> List[int]:
        return list(self.adj[i].keys())

    def num_edges(self) -> int:
        return len(self.edges())

    def number_of_nodes(self) -> int:
        return self.n

    def number_of_edges(self) -> int:
        return self.num_edges()

    def is_connected(self) -> bool:
        if self.n == 0:
            return True
        seen = {0}
        q: deque[int] = deque([0])
        while q:
            v = q.popleft()
            for u in self.adj[v].keys():
                if u not in seen:
                    seen.add(u)
                    q.append(u)
        # consider nodes with any adjacency or isolated; treat graph with no edges as connected
        if self.num_edges() == 0:
            return True
        # Only consider nodes that participate in at least one edge
        active = set()
        for i in range(self.n):
            if self.adj[i]:
                active.add(i)
        return active.issubset(seen)

    def shortest_path_length(self, s: int, t: int) -> int:
        if s == t:
            return 0
        visited = {s}
        q: deque[Tuple[int, int]] = deque([(s, 0)])
        while q:
            v, d = q.popleft()
            for u in self.adj[v].keys():
                if u == t:
                    return d + 1
                if u not in visited:
                    visited.add(u)
                    q.append((u, d + 1))
        return int(1e9)

# --------- Data models ---------
@dataclass
class GraphSnapshot:
    graph: LightGraph
    timestamp: pd.Timestamp
    price_levels: np.ndarray
    ricci_curvatures: Dict[Tuple[int,int], float]
    avg_curvature: float

@dataclass
class TemporalRicciResult:
    temporal_curvature: float
    topological_transition_score: float
    graph_snapshots: List[GraphSnapshot]
    structural_stability: float
    edge_persistence: float

# --------- Ollivier-Ricci (proxy without SciPy) ---------
class OllivierRicciCurvatureLite:
    """
    κ(x,y) ≈ 1 - TV(μ_x, μ_y) / d(x,y),
    where TV(p,q)=0.5 * ||p-q||_1; d is unweighted shortest-path length.
    This is a computationally cheap proxy of Ollivier curvature.
    """
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def _iter_neighbors(self, G: LightGraph, node: int) -> list[int]:
        neighbors_iter = G.neighbors(node)
        if isinstance(neighbors_iter, list):
            return neighbors_iter
        return list(neighbors_iter)

    def _lazy_rw(self, G: LightGraph, node: int) -> Dict[int, float]:
        neigh = self._iter_neighbors(G, node)
        if not neigh:
            return {node: 1.0}
        dist = {node: self.alpha}
        p = (1.0 - self.alpha) / len(neigh)
        for v in neigh:
            dist[v] = p
        return dist

    def _shortest_path_length(self, G: LightGraph, source: int, target: int) -> int:
        if hasattr(G, "shortest_path_length"):
            try:
                return int(G.shortest_path_length(source, target))  # type: ignore[attr-defined]
            except TypeError:
                pass

        from collections import deque

        visited = {source}
        queue: deque[tuple[int, int]] = deque([(source, 0)])
        while queue:
            node, dist = queue.popleft()
            for nbr in self._iter_neighbors(G, node):
                if nbr == target:
                    return dist + 1
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, dist + 1))
        return int(1e9)

    def edge_curvature(self, G: LightGraph, edge: Tuple[int,int]) -> float:
        x, y = edge
        mu_x = self._lazy_rw(G, x)
        mu_y = self._lazy_rw(G, y)

        # align supports
        support = sorted(set(mu_x.keys()) | set(mu_y.keys()))
        px = np.array([mu_x.get(k,0.0) for k in support], dtype=float)
        py = np.array([mu_y.get(k,0.0) for k in support], dtype=float)

        # total variation distance
        tv = 0.5 * np.abs(px - py).sum()

        dxy = self._shortest_path_length(G, x, y)
        if dxy <= 0 or dxy >= 1e9:
            return 0.0
        kappa = 1.0 - (tv / dxy)
        return float(kappa)

    def all_curvatures(self, G: LightGraph) -> Dict[Tuple[int,int], float]:
        curv = {}
        for e in G.edges():
            curv[e] = self.edge_curvature(G, e)
        return curv

    # Backwards-compatible API expected by the test-suite ---------------------
    def compute_edge_curvature(self, G: LightGraph, edge: Tuple[int, int]) -> float:
        return self.edge_curvature(G, edge)

    def compute_all_curvatures(self, G: LightGraph) -> Dict[Tuple[int, int], float]:
        return self.all_curvatures(G)

# --------- Build graph from price levels ---------
class PriceLevelGraph:
    def __init__(self, n_levels: int = 20, connection_threshold: float = 0.1):
        self.n_levels = n_levels
        self.connection_threshold = connection_threshold

    def build(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> LightGraph:
        prices = np.asarray(prices, dtype=float)
        n = self.n_levels
        pmin, pmax = float(prices.min()), float(prices.max())
        if pmax == pmin:
            pmax = pmin + 1e-6
        levels = np.linspace(pmin, pmax, n)
        idx = np.clip(np.digitize(prices, levels) - 1, 0, n-1)

        G = LightGraph(n)
        if len(idx) < 2:
            return G
        weights = np.ones(len(idx) - 1, dtype=float)
        if volumes is not None:
            raw = np.asarray(volumes, dtype=float)
            if raw.size == len(idx):
                raw = raw[:-1]
            elif raw.size != len(idx) - 1:
                raise ValueError("volumes length must be len(prices) - 1")
            weights = np.maximum(raw, 0.0)
        transitions = np.column_stack([idx[:-1], idx[1:]]).astype(int)
        mask = transitions[:, 0] != transitions[:, 1]
        transitions = transitions[mask]
        weights = weights[mask]
        if transitions.size == 0:
            return G
        matrix = np.zeros((n, n), dtype=float)
        np.add.at(matrix, (transitions[:, 0], transitions[:, 1]), weights)
        np.add.at(matrix, (transitions[:, 1], transitions[:, 0]), weights)
        max_weight = matrix.max()
        if max_weight <= 0:
            return G
        matrix /= max_weight
        i_idx, j_idx = np.where(matrix > self.connection_threshold)
        for a, b in zip(i_idx, j_idx):
            if a < b:
                G.add_edge(int(a), int(b), matrix[a, b])
        return G

# --------- Temporal analyzer ---------
class TemporalRicciAnalyzer:
    def __init__(
        self,
        window_size: int = 100,
        n_snapshots: int = 10,
        n_levels: int = 20,
        *,
        retain_history: bool = True,
        connection_threshold: float = 0.1,
    ):
        self.window_size = window_size
        self.n_snapshots = n_snapshots
        self.n_levels = n_levels
        self.ricci = OllivierRicciCurvatureLite(alpha=0.5)
        self.builder = PriceLevelGraph(n_levels=n_levels, connection_threshold=connection_threshold)
        self.retain_history = retain_history
        self.connection_threshold = connection_threshold
        self.history: deque[GraphSnapshot] = deque(maxlen=n_snapshots)

    def _snapshot(self, prices: np.ndarray, volumes: Optional[np.ndarray], ts: pd.Timestamp) -> GraphSnapshot:
        G = self.builder.build(prices, volumes)
        curv = self.ricci.all_curvatures(G)
        avg_k = float(np.mean(list(curv.values()))) if curv else 0.0
        return GraphSnapshot(graph=G, timestamp=ts, price_levels=prices, ricci_curvatures=curv, avg_curvature=avg_k)

    def _temporal_curvature(self) -> float:
        if len(self.history) < 2: return 0.0
        vals = []
        for i in range(len(self.history)-1):
            a = self.history[i]
            b = self.history[i+1]
            # normalized edge symmetric difference
            Ea, Eb = set(a.graph.edges()), set(b.graph.edges())
            union = len(Ea | Eb)
            if union == 0:
                ged_norm = 0.0
            else:
                sym = len(Ea ^ Eb)
                ged_norm = sym / union
            delta_k = abs(b.avg_curvature - a.avg_curvature)
            vals.append(-(ged_norm + delta_k))  # more negative -> more change
        return float(np.mean(vals))

    def _transition_score(self) -> float:
        if len(self.history) < 3: return 0.0
        metrics = []
        for s in self.history:
            E = s.graph.num_edges()
            # average degree over active nodes
            degs = [len(s.graph.neighbors(i)) for i in range(s.graph.n)]
            active = [d for d in degs if d > 0]
            avg_deg = float(np.mean(active)) if active else 0.0
            metrics.append([E, avg_deg, s.avg_curvature])
        M = np.array(metrics, dtype=float)
        D = np.abs(np.diff(M, axis=0))
        if D.size == 0:
            return 0.0
        maxd = D.max(axis=0)
        maxd[maxd == 0] = 1.0
        Dn = D / maxd
        avg_jump = float(np.mean(Dn))
        beta = 8.0
        score = 1.0 / (1.0 + np.exp(-beta * (avg_jump - 0.15)))
        curvatures = np.array([s.avg_curvature for s in self.history], dtype=float)
        if curvatures.size >= 2:
            curvature_component = float(np.clip(np.std(np.diff(curvatures)), 0.0, 1.0))
        else:
            curvature_component = 0.0
        vol_series = [np.std(np.diff(s.price_levels)) for s in self.history if len(s.price_levels) >= 2]
        if len(vol_series) >= 2:
            vol_diff = np.mean(vol_series[-2:]) - np.mean(vol_series[:-2]) if len(vol_series) > 2 else vol_series[-1] - vol_series[0]
            volatility_component = float(np.clip(vol_diff, 0.0, 1.0))
        else:
            volatility_component = 0.0
        return float(np.clip(score + 0.2 * curvature_component + 0.2 * volatility_component, 0.0, 1.0))

    def _stability(self) -> float:
        if len(self.history) < 2: return 1.0
        sims = []
        for i in range(len(self.history)-1):
            Ea = set(self.history[i].graph.edges())
            Eb = set(self.history[i+1].graph.edges())
            if len(Ea | Eb) == 0: sims.append(1.0)
            else: sims.append(len(Ea & Eb) / len(Ea | Eb))
        return float(np.mean(sims))

    def _persistence(self) -> float:
        if len(self.history) < 2: return 1.0
        edge_sets = [set(s.graph.edges()) for s in self.history]
        all_edges = set().union(*edge_sets) if edge_sets else set()
        if not all_edges: return 1.0
        persistent = set.intersection(*edge_sets) if edge_sets else set()
        return float(len(persistent) / len(all_edges))

    def analyze(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        volume_col: Optional[str] = "volume",
        *,
        reset_history: bool = False,
    ) -> "TemporalRicciResult":
        if df.empty or price_col not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column and not be empty")
        if reset_history or not self.retain_history:
            self.history.clear()
        elif self.history and df.index[0] <= self.history[-1].timestamp:
            warnings.warn(
                "TemporalRicciAnalyzer received non-monotonic timestamps; resetting history buffer",
                RuntimeWarning,
                stacklevel=2,
            )
            self.history.clear()
        N = len(df)
        if self.n_snapshots <= 1:
            step = max(self.window_size, N)
        else:
            step = max(1, (N - self.window_size) // (self.n_snapshots - 1))
        for i in range(0, max(1, N - self.window_size + 1), step):
            seg = df.iloc[i:i+self.window_size]
            if len(seg) < self.window_size:
                continue
            prices = seg[price_col].astype(float).values
            volumes = seg[volume_col].astype(float).values if (volume_col and volume_col in seg.columns) else None
            ts = seg.index[-1]
            snap = self._snapshot(prices, volumes, ts)
            self.history.append(snap)
        k_temporal = self._temporal_curvature()
        trans = self._transition_score()
        stab = self._stability()
        pers = self._persistence()
        return TemporalRicciResult(temporal_curvature=k_temporal, topological_transition_score=trans, graph_snapshots=list(self.history), structural_stability=stab, edge_persistence=pers)


# Backwards compatible aliases expected by the public API
OllivierRicciCurvature = OllivierRicciCurvatureLite
PriceLevelGraphBuilder = PriceLevelGraph
