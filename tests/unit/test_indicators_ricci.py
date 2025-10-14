# SPDX-License-Identifier: MIT
"""Unit tests for Ollivier-Ricci curvature indicators.

This module tests the MeanRicciFeature class and related functions for computing
Ollivier-Ricci curvature on price graphs. MeanRicciFeature now supports optional
optimization parameters (use_float32, chunk_size) which are conditionally
included in metadata only when explicitly enabled.

Tests verify:
- Price graph construction is correct
- Ricci curvature calculations are within valid bounds
- Metadata contains required keys (delta, nodes, edges)
- Optional metadata fields appear only when parameters are enabled
- Performance optimizations preserve accuracy
"""
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
    """Test MeanRicciFeature with default parameters produces expected metadata."""
    prices = np.linspace(100.0, 105.0, 20)
    feature = MeanRicciFeature(delta=0.01)
    result = feature.transform(prices)
    graph = build_price_graph(prices, delta=0.01)
    assert result.name == "mean_ricci"
    # With default parameters, metadata should contain delta, nodes, edges only
    assert "delta" in result.metadata
    assert "nodes" in result.metadata
    assert "edges" in result.metadata
    assert result.metadata["nodes"] == graph.number_of_nodes()
    assert result.metadata["edges"] == graph.number_of_edges()
    # Optional flags should not be present with default settings
    assert "use_float32" not in result.metadata
    assert "chunk_size" not in result.metadata
    assert result.value == pytest.approx(mean_ricci(graph), rel=1e-9)


def test_mean_ricci_feature_metadata_contains_required_keys() -> None:
    """Test that MeanRicciFeature metadata always contains required keys."""
    prices = np.linspace(100.0, 105.0, 20)
    feature = MeanRicciFeature(delta=0.005)
    result = feature.transform(prices)

    # Required keys must always be present
    assert "delta" in result.metadata
    assert "nodes" in result.metadata
    assert "edges" in result.metadata
    assert result.metadata["delta"] == 0.005

    # With default settings, only required keys should be present
    assert set(result.metadata.keys()) == {"delta", "nodes", "edges"}


def test_mean_ricci_feature_with_float32_adds_metadata() -> None:
    """Test that use_float32 parameter adds metadata when enabled."""
    prices = np.linspace(100.0, 105.0, 20)
    feature = MeanRicciFeature(delta=0.01, use_float32=True)
    result = feature.transform(prices)

    # Required keys
    assert "delta" in result.metadata
    assert "nodes" in result.metadata
    assert "edges" in result.metadata

    # Optional optimization flag should be present when enabled
    assert "use_float32" in result.metadata
    assert result.metadata["use_float32"] is True

    # Verify computation still works correctly
    assert isinstance(result.value, float)
    assert np.isfinite(result.value)


def test_mean_ricci_feature_with_chunk_size_adds_metadata() -> None:
    """Test that chunk_size parameter adds metadata when enabled."""
    prices = np.linspace(100.0, 105.0, 30)
    feature = MeanRicciFeature(delta=0.01, chunk_size=10)
    result = feature.transform(prices)

    # Required keys
    assert "delta" in result.metadata
    assert "nodes" in result.metadata
    assert "edges" in result.metadata

    # Optional optimization flag should be present when enabled
    assert "chunk_size" in result.metadata
    assert result.metadata["chunk_size"] == 10

    # Verify computation still works correctly
    assert isinstance(result.value, float)
    assert np.isfinite(result.value)


def test_mean_ricci_feature_with_combined_optimizations() -> None:
    """Test MeanRicciFeature with both float32 and chunk_size enabled."""
    prices = np.linspace(100.0, 105.0, 30)
    feature = MeanRicciFeature(delta=0.01, use_float32=True, chunk_size=10)
    result = feature.transform(prices)

    # All keys should be present
    assert "delta" in result.metadata
    assert "nodes" in result.metadata
    assert "edges" in result.metadata
    assert "use_float32" in result.metadata
    assert "chunk_size" in result.metadata

    assert result.metadata["use_float32"] is True
    assert result.metadata["chunk_size"] == 10

    # Verify value is computed
    assert isinstance(result.value, float)
    assert np.isfinite(result.value)


def test_mean_ricci_feature_float32_preserves_accuracy() -> None:
    """Test that float32 optimization doesn't significantly change results."""
    prices = np.linspace(100.0, 105.0, 30)

    feature_64 = MeanRicciFeature(delta=0.01, use_float32=False)
    feature_32 = MeanRicciFeature(delta=0.01, use_float32=True)

    result_64 = feature_64.transform(prices)
    result_32 = feature_32.transform(prices)

    # Skip comparison if graph has no edges
    if result_64.metadata["edges"] == 0:
        pytest.skip("Graph has no edges")

    # Results should be close (allow reasonable tolerance for float32)
    assert (
        abs(result_64.value - result_32.value) < 0.5
    ), f"Float32 and float64 Ricci values differ too much: {result_64.value} vs {result_32.value}"


def test_mean_ricci_feature_chunk_size_behavior() -> None:
    """Test that chunk_size affects processing of graphs with many edges."""
    # Create data that produces a graph with many edges
    prices = np.linspace(100.0, 110.0, 50)

    feature_unchunked = MeanRicciFeature(delta=0.005)
    feature_chunked = MeanRicciFeature(delta=0.005, chunk_size=5)

    result_unchunked = feature_unchunked.transform(prices)
    result_chunked = feature_chunked.transform(prices)

    # Skip if graph has no edges
    if result_unchunked.metadata["edges"] == 0:
        pytest.skip("Graph has no edges")

    # Both should produce valid results
    assert np.isfinite(result_unchunked.value)
    assert np.isfinite(result_chunked.value)

    # Results should be identical (chunking doesn't change computation, just order)
    assert abs(result_unchunked.value - result_chunked.value) < 0.01

    # Metadata should reflect the difference
    assert "chunk_size" not in result_unchunked.metadata
    assert result_chunked.metadata["chunk_size"] == 5
