# SPDX-License-Identifier: MIT
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.indicators import ricci as ricci_module
from core.indicators.ricci import (
    MeanRicciFeature,
    build_price_graph,
    local_distribution,
    mean_ricci,
    ricci_curvature_edge,
)


def test_mean_ricci_feature_returns_metadata():
    prices = np.cumsum(np.random.randn(256)) + 100
    feature = MeanRicciFeature(delta=0.01)
    result = feature(prices)
    assert "nodes" in result.metadata
    assert result.metadata["delta"] == 0.01


def test_build_price_graph_connects_consecutive_levels():
    prices = np.array([100, 101, 102, 101.5])
    graph = build_price_graph(prices, delta=0.01)
    assert graph.number_of_nodes() >= 3
    assert graph.number_of_edges() >= 2


def test_local_distribution_returns_probability():
    prices = np.array([100, 100.5, 101.0, 101.5])
    graph = build_price_graph(prices, delta=0.01)
    nodes_iter = graph.nodes() if hasattr(graph, "nodes") else graph._adj.keys()  # type: ignore[attr-defined]
    node = next(iter(nodes_iter))
    distribution = local_distribution(graph, node)
    assert np.isclose(distribution.sum(), 1.0)


def test_local_distribution_returns_unity_for_isolated_node():
    prices = np.array([100.0])
    graph = build_price_graph(prices, delta=0.01)
    nodes_iter = graph.nodes() if hasattr(graph, "nodes") else graph._adj.keys()  # type: ignore[attr-defined]
    node = next(iter(nodes_iter))
    distribution = local_distribution(graph, node)
    assert np.array_equal(distribution, np.array([1.0]))


def test_ricci_curvature_falls_back_without_wasserstein(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ricci_module, "W1", None)
    prices = np.array([100, 101, 102, 103])
    graph = build_price_graph(prices, delta=0.01)
    edges = list(graph.edges())
    assert edges
    curvature = ricci_curvature_edge(graph, *edges[0])
    assert -1.0 <= curvature <= 1.0


def test_ricci_curvature_zero_when_no_edge():
    prices = np.array([100, 101, 102])
    graph = build_price_graph(prices, delta=0.01)
    graph.add_node(999)
    assert ricci_curvature_edge(graph, list(graph.edges())[0][0], 999) == 0.0


def test_mean_ricci_graph_helpers():
    prices = np.array([100, 101, 102, 101.5])
    graph = build_price_graph(prices)
    value = mean_ricci(graph)
    assert isinstance(value, float)


def test_mean_ricci_returns_zero_for_graph_without_edges():
    graph = build_price_graph(np.array([100.0]))
    assert mean_ricci(graph) == 0.0
