# SPDX-License-Identifier: MIT
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.indicators.ricci import MeanRicciFeature, build_price_graph, mean_ricci


def test_mean_ricci_feature_returns_metadata():
    prices = np.cumsum(np.random.randn(256)) + 100
    feature = MeanRicciFeature(delta=0.01)
    result = feature(prices)
    assert "nodes" in result.metadata
    assert result.metadata["delta"] == 0.01


def test_mean_ricci_graph_helpers():
    prices = np.array([100, 101, 102, 101.5])
    graph = build_price_graph(prices)
    value = mean_ricci(graph)
    assert isinstance(value, float)
