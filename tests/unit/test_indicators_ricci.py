# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from core.indicators.ricci import (
    MeanRicciFeature,
    build_price_graph,
    local_distribution,
    mean_ricci,
    ricci_curvature_edge,
)


def test_build_price_graph_connects_consecutive_levels() -> None:
    prices = np.array([100, 101, 102, 103], dtype=float)
    graph = build_price_graph(prices, delta=0.01)
    assert graph.number_of_nodes() >= 3
    assert graph.number_of_edges() >= 2


def test_local_distribution_normalizes_probabilities() -> None:
    prices = np.array([100, 101, 101.5, 102], dtype=float)
    graph = build_price_graph(prices, delta=0.005)
    edges = list(graph.edges())
    assert edges, "Graph must have edges for distribution test"
    node = edges[0][0]
    probs = local_distribution(graph, node)
    assert abs(probs.sum() - 1.0) < 1e-9


def test_ricci_curvature_bounded_between_minus_one_and_one() -> None:
    prices = np.array([100, 100.5, 101.0, 101.5, 102.0], dtype=float)
    graph = build_price_graph(prices, delta=0.005)
    edges = list(graph.edges())
    for u, v in edges:
        kappa = ricci_curvature_edge(graph, u, v)
        assert -1.0 <= kappa <= 1.0, f"Curvature {kappa} outside bounds"


def test_mean_ricci_feature_matches_function() -> None:
    prices = np.linspace(100.0, 105.0, 20)
    feature = MeanRicciFeature(delta=0.01)
    result = feature.transform(prices)
    graph = build_price_graph(prices, delta=0.01)
    assert result.name == "mean_ricci"
    assert result.metadata["nodes"] == graph.number_of_nodes()
    assert result.value == pytest.approx(mean_ricci(graph), rel=1e-9)
