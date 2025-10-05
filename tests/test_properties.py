# SPDX-License-Identifier: MIT
import pytest
import numpy as np

nx = pytest.importorskip("networkx")

from core.indicators.kuramoto import kuramoto_order
from core.indicators.ricci import MeanRicciFeature, RicciCurvatureFeature, build_price_graph, mean_ricci


def test_R_bounds_property():
    ph = np.random.uniform(-np.pi, np.pi, size=512)
    R = kuramoto_order(ph)
    assert 0.0 <= R <= 1.0


def test_mean_ricci_feature_matches_function():
    prices = np.linspace(100, 102, 20)
    graph = build_price_graph(prices)
    feature = MeanRicciFeature()
    assert feature.metadata()["name"] == "mean_ricci"
    assert feature.transform(graph) == mean_ricci(graph)


def test_edge_ricci_feature_handles_missing_edge():
    graph = nx.path_graph(4)
    feature = RicciCurvatureFeature(nodes=(0, 3))
    assert feature.transform(graph) == 0.0
