# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import ricci as _ricci

mean_ricci = _ricci.mean_ricci
ricci_curvature_edge = _ricci.ricci_curvature_edge
nx = _ricci.nx

try:  # pragma: no cover - SciPy is optional
    from scipy.spatial.distance import jensenshannon as _jsd
except Exception:  # pragma: no cover
    _jsd = None


@dataclass
class GraphSnapshot:
    """Snapshot of a price-level graph at a specific timestamp."""

    graph: "nx.Graph"
    timestamp: np.datetime64
    price_levels: np.ndarray
    ricci_curvatures: Dict[Tuple[int, int], float]
    avg_curvature: float

    def edge_set(self) -> set[Tuple[int, int]]:
        return {tuple(sorted(edge)) for edge in self.graph.edges()}

    def degree_distribution(self) -> np.ndarray:
        try:
            degree_items = list(self.graph.degree())  # type: ignore[arg-type]
        except TypeError:  # Fallback graph returns callable without iteration support
            nodes = getattr(self.graph, "nodes", lambda: tuple())()
            degree_items = [(node, self.graph.degree(node)) for node in nodes]
        degrees = np.array([deg for _, deg in degree_items], dtype=float)
        if degrees.size == 0:
            return np.array([1.0])
        total = degrees.sum()
        if total == 0.0:
            return np.array([1.0])
        distribution = degrees / total
        return distribution


@dataclass
class TemporalRicciResult:
    """Temporal Ricci curvature analysis result."""

    temporal_curvature: float
    topological_transition_score: float
    graph_snapshots: List[GraphSnapshot]
    structural_stability: float
    edge_persistence: float


class PriceLevelGraphBuilder:
    """Construct a graph that represents transitions between price levels."""

    def __init__(self, n_levels: int = 20, connection_threshold: float = 0.1):
        self.n_levels = int(n_levels)
        self.connection_threshold = float(connection_threshold)

    def build(
        self,
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None,
    ) -> "nx.Graph":
        if nx is None:  # pragma: no cover - safeguard for environments without networkx
            raise RuntimeError("networkx is required for temporal Ricci analysis")

        values = np.asarray(prices, dtype=float)
        if values.size < 2:
            G = nx.Graph()
            for node in range(self.n_levels):
                G.add_node(node)
            return G

        price_min = float(values.min())
        price_max = float(values.max())
        if np.isclose(price_min, price_max):
            price_max = price_min + 1e-6

        levels = np.linspace(price_min, price_max, self.n_levels)
        indices = np.digitize(values, levels, right=False) - 1
        indices = np.clip(indices, 0, self.n_levels - 1)

        transition_counts = np.zeros((self.n_levels, self.n_levels), dtype=float)
        weights = (
            np.asarray(volumes, dtype=float)
            if volumes is not None
            else np.ones(values.size - 1, dtype=float)
        )

        for idx in range(values.size - 1):
            i = int(indices[idx])
            j = int(indices[idx + 1])
            w = float(weights[idx])
            transition_counts[i, j] += w
            transition_counts[j, i] += w

        G = nx.Graph()
        for node in range(self.n_levels):
            G.add_node(node)

        max_count = float(transition_counts.max())
        if max_count <= 0:
            return G

        transition_counts /= max_count
        for i in range(self.n_levels):
            for j in range(i + 1, self.n_levels):
                weight = transition_counts[i, j]
                if weight > self.connection_threshold:
                    G.add_edge(i, j, weight=weight)

        return G


class TemporalRicciAnalyzer:
    """Temporal Ricci curvature analysis with topological transition detection."""

    def __init__(
        self,
        window_size: int = 256,
        n_snapshots: int = 10,
        n_levels: int = 20,
        connection_threshold: float = 0.1,
    ) -> None:
        if n_snapshots < 1:
            raise ValueError("n_snapshots must be at least 1")

        self.window_size = int(window_size)
        self.n_snapshots = int(n_snapshots)
        self.builder = PriceLevelGraphBuilder(
            n_levels=n_levels, connection_threshold=connection_threshold
        )

    def analyze(
        self,
        df: Optional[pd.DataFrame] = None,
        *,
        price_series: Optional[np.ndarray] = None,
        volume_series: Optional[np.ndarray] = None,
        price_col: str = "close",
        volume_col: Optional[str] = "volume",
    ) -> TemporalRicciResult:

        if nx is None:
            raise RuntimeError("networkx is required for temporal Ricci analysis")

        if df is not None:
            data = df
            if not isinstance(data.index, pd.DatetimeIndex):
                data = data.copy()
                data.index = pd.to_datetime(data.index)
            prices = data[price_col].to_numpy()
            volumes = data[volume_col].to_numpy() if volume_col and volume_col in data else None
            timestamps = data.index.to_numpy()
        else:
            if price_series is None:
                raise ValueError("Either df or price_series must be provided")
            prices = np.asarray(price_series, dtype=float)
            volumes = (
                np.asarray(volume_series, dtype=float)
                if volume_series is not None
                else None
            )
            timestamps = np.arange(prices.size, dtype=int)

        if prices.size < self.window_size or self.window_size <= 1:
            empty_result = TemporalRicciResult(
                temporal_curvature=0.0,
                topological_transition_score=0.0,
                graph_snapshots=[],
                structural_stability=1.0,
                edge_persistence=1.0,
            )
            return empty_result

        starts = self._snapshot_starts(prices.size)
        snapshots: List[GraphSnapshot] = []

        for start in starts:
            end = start + self.window_size
            window_prices = prices[start:end]
            window_volumes = volumes[start : end - 1] if volumes is not None else None
            timestamp = timestamps[end - 1]

            graph = self.builder.build(window_prices, window_volumes)
            curvatures = {
                tuple(sorted((u, v))): ricci_curvature_edge(graph, u, v)
                for u, v in graph.edges()
            }
            avg_curvature = mean_ricci(graph)

            snapshot = GraphSnapshot(
                graph=graph,
                timestamp=timestamp,
                price_levels=window_prices,
                ricci_curvatures=curvatures,
                avg_curvature=avg_curvature,
            )
            snapshots.append(snapshot)

        temporal_curvature = self._temporal_curvature(snapshots)
        transition_score = self._transition_score(snapshots)
        stability = self._structural_stability(snapshots)
        persistence = self._edge_persistence(snapshots)

        return TemporalRicciResult(
            temporal_curvature=temporal_curvature,
            topological_transition_score=transition_score,
            graph_snapshots=snapshots,
            structural_stability=stability,
            edge_persistence=persistence,
        )

    def _snapshot_starts(self, length: int) -> Sequence[int]:
        if length <= self.window_size:
            return [0]

        step = max(1, (length - self.window_size) // max(1, self.n_snapshots - 1))
        starts = list(range(0, length - self.window_size + 1, step))
        if starts[-1] != length - self.window_size:
            starts.append(length - self.window_size)
        if len(starts) > self.n_snapshots:
            starts = starts[: self.n_snapshots]
            if starts[-1] != length - self.window_size:
                starts[-1] = length - self.window_size
        return starts

    def _temporal_curvature(self, snapshots: Sequence[GraphSnapshot]) -> float:
        if len(snapshots) < 2:
            return 0.0

        changes: List[float] = []
        for prev, curr in zip(snapshots[:-1], snapshots[1:]):
            edges_prev = prev.edge_set()
            edges_curr = curr.edge_set()
            union = edges_prev | edges_curr
            symmetric = union - (edges_prev & edges_curr)
            if not union:
                edge_change = 0.0
            else:
                edge_change = len(symmetric) / len(union)

            delta_kappa = abs(curr.avg_curvature - prev.avg_curvature)
            changes.append(-(edge_change + delta_kappa))

        return float(np.mean(changes)) if changes else 0.0

    def _transition_score(self, snapshots: Sequence[GraphSnapshot]) -> float:
        if len(snapshots) < 3:
            return 0.0

        metrics = []
        edge_changes: List[float] = []
        curvature_jumps: List[float] = []
        for snap in snapshots:
            n_edges = snap.graph.number_of_edges()
            n_nodes = snap.graph.number_of_nodes()
            avg_degree = 2.0 * n_edges / n_nodes if n_nodes > 0 else 0.0
            avg_kappa = snap.avg_curvature
            entropy = self._degree_entropy(snap)
            metrics.append([n_edges, avg_degree, avg_kappa, entropy])

        for prev, curr in zip(snapshots[:-1], snapshots[1:]):
            edges_prev = prev.edge_set()
            edges_curr = curr.edge_set()
            union = edges_prev | edges_curr
            if not union:
                edge_changes.append(0.0)
            else:
                symmetric = union - (edges_prev & edges_curr)
                edge_changes.append(len(symmetric) / len(union))
            curvature_jumps.append(abs(curr.avg_curvature - prev.avg_curvature))

        metrics_arr = np.asarray(metrics, dtype=float)
        diffs = np.abs(np.diff(metrics_arr, axis=0))
        if not np.isfinite(diffs).any():
            return 0.0

        max_vals = np.max(diffs, axis=0)
        max_vals[max_vals == 0.0] = 1.0
        normalized = diffs / max_vals
        jump = np.linalg.norm(normalized, axis=1) / np.sqrt(normalized.shape[1])
        jump_score = float(np.mean(jump)) if jump.size > 0 else 0.0
        structure_score = float(np.mean(edge_changes)) if edge_changes else 0.0
        curvature_score = (
            float(1.0 - np.exp(-np.mean(curvature_jumps))) if curvature_jumps else 0.0
        )
        score = 0.5 * jump_score + 0.3 * structure_score + 0.2 * curvature_score
        return float(np.clip(score, 0.0, 1.0))

    def _structural_stability(self, snapshots: Sequence[GraphSnapshot]) -> float:
        if len(snapshots) < 2:
            return 1.0

        similarities: List[float] = []
        for prev, curr in zip(snapshots[:-1], snapshots[1:]):
            edges_prev = prev.edge_set()
            edges_curr = curr.edge_set()
            if not edges_prev and not edges_curr:
                similarities.append(1.0)
                continue
            union = edges_prev | edges_curr
            if not union:
                similarities.append(1.0)
                continue
            intersection = edges_prev & edges_curr
            similarities.append(len(intersection) / len(union))

        return float(np.mean(similarities)) if similarities else 1.0

    def _edge_persistence(self, snapshots: Sequence[GraphSnapshot]) -> float:
        if len(snapshots) < 2:
            return 1.0

        edge_sets = [snap.edge_set() for snap in snapshots if snap.edge_set()]
        if not edge_sets:
            return 1.0

        persistent = set.intersection(*edge_sets)
        union = set.union(*edge_sets)
        if not union:
            return 1.0

        return float(len(persistent) / len(union))

    def _degree_entropy(self, snapshot: GraphSnapshot) -> float:
        degrees = snapshot.degree_distribution()
        if degrees.size <= 1:
            return 0.0

        if _jsd is not None:
            base = np.full_like(degrees, 1.0 / degrees.size)
            divergence = float(_jsd(degrees, base, base=2.0))
        else:
            divergence = float(self._jensen_shannon_fallback(degrees))
        return divergence

    @staticmethod
    def _jensen_shannon_fallback(p: np.ndarray) -> float:
        probs = np.asarray(p, dtype=float)
        probs = probs / (probs.sum() + 1e-12)
        uniform = np.full_like(probs, 1.0 / probs.size)
        m = 0.5 * (probs + uniform)
        kl_pm = _safe_entropy(probs, m)
        kl_um = _safe_entropy(uniform, m)
        return 0.5 * (kl_pm + kl_um) / np.log(2.0)


def _safe_entropy(p: np.ndarray, q: np.ndarray) -> float:
    mask = (p > 0) & (q > 0)
    if not np.any(mask):
        return 0.0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


__all__ = [
    "GraphSnapshot",
    "TemporalRicciResult",
    "PriceLevelGraphBuilder",
    "TemporalRicciAnalyzer",
]

