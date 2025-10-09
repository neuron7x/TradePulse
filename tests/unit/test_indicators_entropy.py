# SPDX-License-Identifier: MIT
"""Unit tests for entropy-based market uncertainty indicators.

This module tests the EntropyFeature and DeltaEntropyFeature classes,
including their metadata handling. These features now support optional
optimization parameters (use_float32, chunk_size) which are conditionally
included in metadata only when explicitly enabled.

Tests verify:
- Core entropy calculations are correct
- Metadata contains required keys (e.g., 'bins')
- Optional metadata fields appear only when parameters are enabled
- Edge cases like empty arrays and extreme values are handled properly
- Performance optimizations don't change core behavior significantly
"""
from __future__ import annotations

import numpy as np
import pytest

from core.indicators.entropy import (
    DeltaEntropyFeature,
    EntropyFeature,
    delta_entropy,
    entropy,
)


def test_entropy_uniform_distribution_matches_log_bins(uniform_series: np.ndarray) -> None:
    bins = 20
    result = entropy(uniform_series, bins=bins)
    expected = np.log(bins)
    assert abs(result - expected) < 0.15, f"Entropy {result} deviates from log(bins) {expected}"


def test_entropy_degenerate_distribution_near_zero() -> None:
    series = np.ones(128)
    result = entropy(series, bins=10)
    assert result < 1e-9


def test_entropy_handles_extreme_values_and_non_finite() -> None:
    series = np.array(
        [
            0.0,
            0.0,
            1.0,
            np.finfo(float).max,
            -np.finfo(float).max / 10,
            np.nan,
            np.inf,
            -np.inf,
        ]
    )
    result = entropy(series, bins=16)
    assert np.isfinite(result)
    assert result >= 0.0


def test_entropy_of_empty_series_is_zero() -> None:
    assert entropy(np.array([])) == 0.0


def test_delta_entropy_requires_two_windows(peaked_series: np.ndarray) -> None:
    short_series = peaked_series[:100]
    assert delta_entropy(short_series, window=80) == 0.0


def test_delta_entropy_detects_spread_change() -> None:
    first = np.zeros(80)
    second = np.linspace(-1.0, 1.0, 80)
    series = np.concatenate([first, second])
    result = delta_entropy(series, window=80)
    assert result > 0.0, "Delta entropy should increase when distribution widens"


def test_entropy_feature_wraps_indicator(uniform_series: np.ndarray) -> None:
    """Test EntropyFeature with default parameters produces minimal metadata."""
    feature = EntropyFeature(bins=15, name="custom_entropy")
    outcome = feature.transform(uniform_series)
    assert outcome.name == "custom_entropy"
    # With default parameters, only 'bins' should be in metadata
    assert outcome.metadata == {"bins": 15}
    expected = entropy(uniform_series, bins=15)
    assert outcome.value == pytest.approx(expected, rel=1e-12)


def test_delta_entropy_feature_metadata(peaked_series: np.ndarray) -> None:
    """Test DeltaEntropyFeature metadata structure."""
    feature = DeltaEntropyFeature(window=40, bins_range=(5, 25))
    outcome = feature.transform(peaked_series)
    assert outcome.name == "delta_entropy"
    assert outcome.metadata == {"window": 40, "bins_range": (5, 25)}
    expected = delta_entropy(peaked_series, window=40, bins_range=(5, 25))
    assert outcome.value == pytest.approx(expected, rel=1e-12)


def test_entropy_feature_metadata_contains_required_keys(uniform_series: np.ndarray) -> None:
    """Test that EntropyFeature metadata always contains required keys.
    
    This test verifies that the 'bins' key is always present in metadata,
    regardless of whether optional optimization parameters are used.
    """
    feature = EntropyFeature(bins=20)
    outcome = feature.transform(uniform_series)
    
    # Required key must always be present
    assert "bins" in outcome.metadata
    assert outcome.metadata["bins"] == 20
    
    # With default settings, only 'bins' should be present
    assert set(outcome.metadata.keys()) == {"bins"}


def test_entropy_feature_with_float32_adds_metadata(uniform_series: np.ndarray) -> None:
    """Test that use_float32 parameter adds metadata when enabled."""
    feature = EntropyFeature(bins=20, use_float32=True)
    outcome = feature.transform(uniform_series)
    
    # Required keys
    assert "bins" in outcome.metadata
    assert outcome.metadata["bins"] == 20
    
    # Optional optimization flag should be present when enabled
    assert "use_float32" in outcome.metadata
    assert outcome.metadata["use_float32"] is True
    
    # Verify computation still works correctly
    expected = entropy(uniform_series, bins=20, use_float32=True)
    assert outcome.value == pytest.approx(expected, rel=1e-6)


def test_entropy_feature_with_chunk_size_adds_metadata(uniform_series: np.ndarray) -> None:
    """Test that chunk_size parameter adds metadata when enabled."""
    feature = EntropyFeature(bins=20, chunk_size=50)
    outcome = feature.transform(uniform_series)
    
    # Required keys
    assert "bins" in outcome.metadata
    assert outcome.metadata["bins"] == 20
    
    # Optional optimization flag should be present when enabled
    assert "chunk_size" in outcome.metadata
    assert outcome.metadata["chunk_size"] == 50
    
    # Verify computation still works correctly
    expected = entropy(uniform_series, bins=20, chunk_size=50)
    # Chunked processing may have slight differences
    assert abs(outcome.value - expected) < 0.5


def test_entropy_feature_with_combined_optimizations(uniform_series: np.ndarray) -> None:
    """Test EntropyFeature with both float32 and chunk_size enabled."""
    feature = EntropyFeature(bins=25, use_float32=True, chunk_size=40)
    outcome = feature.transform(uniform_series)
    
    # All keys should be present
    assert "bins" in outcome.metadata
    assert "use_float32" in outcome.metadata
    assert "chunk_size" in outcome.metadata
    
    assert outcome.metadata["bins"] == 25
    assert outcome.metadata["use_float32"] is True
    assert outcome.metadata["chunk_size"] == 40
    
    # Verify value is computed
    assert isinstance(outcome.value, float)
    assert np.isfinite(outcome.value)
    assert outcome.value >= 0.0


def test_entropy_feature_float32_preserves_accuracy(uniform_series: np.ndarray) -> None:
    """Test that float32 optimization doesn't significantly change results."""
    feature_64 = EntropyFeature(bins=30, use_float32=False)
    feature_32 = EntropyFeature(bins=30, use_float32=True)
    
    result_64 = feature_64.transform(uniform_series)
    result_32 = feature_32.transform(uniform_series)
    
    # Results should be very close (within float32 precision tolerance)
    assert abs(result_64.value - result_32.value) < 0.1, \
        f"Float32 and float64 results differ too much: {result_64.value} vs {result_32.value}"


def test_entropy_feature_chunk_size_behavior() -> None:
    """Test that chunk_size affects processing of large arrays."""
    # Create a large array
    large_data = np.random.randn(10000)
    
    feature_unchunked = EntropyFeature(bins=50)
    feature_chunked = EntropyFeature(bins=50, chunk_size=1000)
    
    result_unchunked = feature_unchunked.transform(large_data)
    result_chunked = feature_chunked.transform(large_data)
    
    # Both should produce valid results
    assert np.isfinite(result_unchunked.value)
    assert np.isfinite(result_chunked.value)
    
    # Results should be reasonably close (chunking uses weighted averaging)
    assert abs(result_unchunked.value - result_chunked.value) < 1.0
    
    # Metadata should reflect the difference
    assert "chunk_size" not in result_unchunked.metadata
    assert result_chunked.metadata["chunk_size"] == 1000
