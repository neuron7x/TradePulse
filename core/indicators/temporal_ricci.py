
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from collections import deque, defaultdict

# --------- Lightweight Graph (no networkx) ---------
class LightGraph:
    def __init__(self, n: int):
        self.n = n
        self.adj = [dict() for _ in range(n)]  # neighbor -> weight

    def add_edge(self, i: int, j: int, w: float = 1.0):
        if i == j: return
        self.adj[i][j] = w
        self.adj[j][i] = w

    def edges(self) -> List[Tuple[int,int]]:
        seen = set()
        E = []
        for i in range(self.n):
            for j in self.adj[i].keys():
                e = (min(i,j), max(i,j))
                if e not in seen:
                    seen.add(e); E.append(e)
        return E

    def neighbors(self, i: int) -> List[int]:
        return list(self.adj[i].keys())

    def num_edges(self) -> int:
        return len(self.edges())

    def number_of_nodes(self) -> int:
        return self.n

    def number_of_edges(self) -> int:
        return self.num_edges()

    def is_connected(self) -> bool:
        if self.n == 0: return True
        # BFS
        seen = set([0])
        q = [0]
        while q:
            v = q.pop(0)
            for u in self.adj[v].keys():
                if u not in seen:
                    seen.add(u); q.append(u)
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
        if s == t: return 0
        visited = set([s])
        q = [(s,0)]
        while q:
            v,d = q.pop(0)
            for u in self.adj[v].keys():
                if u == t: return d+1
                if u not in visited:
                    visited.add(u)
                    q.append((u, d+1))
        return int(1e9)  # effectively inf

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

    def _lazy_rw(self, G: LightGraph, node: int) -> Dict[int, float]:
        neigh = G.neighbors(node)
        if not neigh:
            return {node: 1.0}
        dist = {node: self.alpha}
        p = (1.0 - self.alpha) / len(neigh)
        for v in neigh:
            dist[v] = p
        return dist

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

        dxy = G.shortest_path_length(x, y)
        if dxy <= 0 or dxy >= 1e9:
            return 0.0
        kappa = 1.0 - (tv / dxy)
        return float(kappa)

    def all_curvatures(self, G: LightGraph) -> Dict[Tuple[int,int], float]:
        curv = {}
        for e in G.edges():
            curv[e] = self.edge_curvature(G, e)
        return curv

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
        counts = np.zeros((n,n), dtype=float)
        for i in range(len(idx)-1):
            a, b = int(idx[i]), int(idx[i+1])
            w = float(volumes[i]) if volumes is not None else 1.0
            counts[a,b] += w; counts[b,a] += w
        if counts.max() > 0:
            counts /= counts.max()
        # connect edges above threshold
        for i in range(n):
            for j in range(i+1, n):
                if counts[i,j] > self.connection_threshold:
                    G.add_edge(i, j, counts[i,j])
        return G

# --------- Temporal analyzer ---------
class TemporalRicciAnalyzer:
    def __init__(self, window_size: int = 100, n_snapshots: int = 10, n_levels: int = 20):
        self.window_size = window_size
        self.n_snapshots = n_snapshots
        self.n_levels = n_levels
        self.ricci = OllivierRicciCurvatureLite(alpha=0.5)
        self.builder = PriceLevelGraph(n_levels=n_levels)
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
        if len(self.history) < 3:
            return 0.0
        metrics = []
        for snapshot in self.history:
            edge_count = snapshot.graph.num_edges()
            degrees = [len(snapshot.graph.neighbors(i)) for i in range(snapshot.graph.n)]
            active = [deg for deg in degrees if deg > 0]
            avg_degree = float(np.mean(active)) if active else 0.0
            metrics.append([edge_count, avg_degree, snapshot.avg_curvature])
        matrix = np.array(metrics, dtype=float)
        diffs = np.abs(np.diff(matrix, axis=0))
        if diffs.size == 0:
            return 0.0
        max_per_feature = diffs.max(axis=0)
        max_per_feature[max_per_feature == 0] = 1.0
        normalized = diffs / max_per_feature
        avg_jump = float(np.mean(normalized))
        beta = 8.0
        score = 1.0 / (1.0 + np.exp(-beta * (avg_jump - 0.15)))
        curvatures = np.array([snap.avg_curvature for snap in self.history], dtype=float)
        if curvatures.size >= 2:
            curvature_component = float(np.clip(np.std(np.diff(curvatures)), 0.0, 1.0))
        else:
            curvature_component = 0.0
        vol_series = [np.std(np.diff(snap.price_levels)) for snap in self.history if len(snap.price_levels) >= 2]
        if len(vol_series) >= 2:
            if len(vol_series) > 2:
                volatility_component = float(np.clip(np.mean(vol_series[-2:]) - np.mean(vol_series[:-2]), 0.0, 1.0))
            else:
                volatility_component = float(np.clip(vol_series[-1] - vol_series[0], 0.0, 1.0))
        else:
            volatility_component = 0.0
        combined = score + 0.2 * curvature_component + 0.2 * volatility_component
        return float(np.clip(combined, 0.0, 1.0))

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

    def analyze(self, df: pd.DataFrame, price_col: str = "close", volume_col: Optional[str] = "volume") -> "TemporalRicciResult":
        if df.empty or price_col not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column and not be empty")
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

# Backwards compatible aliases
OllivierRicciCurvature = OllivierRicciCurvatureLite
PriceLevelGraphBuilder = PriceLevelGraph

