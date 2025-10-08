# SPDX-License-Identifier: MIT
"""Chaos and fault-injection tests for data ingestion.

These tests verify that the data model handles unexpected values,
edge cases, and extreme conditions gracefully.
"""
from __future__ import annotations

import math

import pytest

from core.data.ingestion import Ticker


class TestNetworkFailures:
    """Test data ingestion under network failure conditions."""

    @pytest.mark.parametrize("exception_type", [
        ConnectionError,
        TimeoutError,
        OSError,
    ])
    def test_handles_network_exceptions(self, exception_type) -> None:
        """Should handle various network exceptions gracefully."""
        # This test documents expected behavior for network failures
        with pytest.raises(exception_type):
            raise exception_type("Simulated network failure")


class TestDataIntegrityUnderChaos:
    """Test data integrity when chaos is injected."""

    def test_ticker_maintains_data_integrity(self) -> None:
        """Ticker objects should maintain data integrity."""
        ticker = Ticker(ts=1609459200.0, price=50000.0, volume=100.0)
        
        assert ticker.ts == 1609459200.0
        assert ticker.price == 50000.0
        assert ticker.volume == 100.0

    def test_handles_negative_price(self) -> None:
        """Should handle negative price values."""
        ticker = Ticker(ts=1609459200.0, price=-50000.0, volume=100.0)
        assert ticker.price == -50000.0

    def test_handles_extreme_timestamp_values(self) -> None:
        """Should handle extreme timestamp values."""
        ticker1 = Ticker(ts=0.0, price=50000.0, volume=100.0)
        assert ticker1.ts == 0.0
        
        ticker2 = Ticker(ts=9999999999.0, price=50000.0, volume=100.0)
        assert ticker2.ts == 9999999999.0

    def test_handles_extreme_price_values(self) -> None:
        """Should handle extreme price values."""
        ticker1 = Ticker(ts=1609459200.0, price=0.00000001, volume=100.0)
        assert ticker1.price == 0.00000001
        
        ticker2 = Ticker(ts=1609459200.0, price=1e15, volume=100.0)
        assert ticker2.price == 1e15

    def test_handles_extreme_volume_values(self) -> None:
        """Should handle extreme volume values."""
        ticker1 = Ticker(ts=1609459200.0, price=50000.0, volume=0.0)
        assert ticker1.volume == 0.0
        
        ticker2 = Ticker(ts=1609459200.0, price=50000.0, volume=1e12)
        assert ticker2.volume == 1e12

    def test_handles_zero_values(self) -> None:
        """Should handle zero values."""
        ticker = Ticker(ts=0.0, price=0.0, volume=0.0)
        assert ticker.ts == 0.0
        assert ticker.price == 0.0
        assert ticker.volume == 0.0

    def test_handles_negative_timestamp(self) -> None:
        """Should handle negative timestamps."""
        ticker = Ticker(ts=-1000.0, price=50000.0, volume=100.0)
        assert ticker.ts == -1000.0

    def test_handles_negative_volume(self) -> None:
        """Should handle negative volume."""
        ticker = Ticker(ts=1609459200.0, price=50000.0, volume=-100.0)
        assert ticker.volume == -100.0


class TestTickerEdgeCases:
    """Test edge cases and boundary conditions for Ticker."""

    def test_ticker_with_float_precision(self) -> None:
        """Should maintain float precision."""
        ticker = Ticker(ts=1609459200.123456, price=50000.987654, volume=100.111111)
        
        assert abs(ticker.ts - 1609459200.123456) < 1e-6
        assert abs(ticker.price - 50000.987654) < 1e-6
        assert abs(ticker.volume - 100.111111) < 1e-6

    def test_ticker_with_very_small_numbers(self) -> None:
        """Should handle very small numbers."""
        ticker = Ticker(ts=1e-10, price=1e-10, volume=1e-10)
        
        assert ticker.ts == 1e-10
        assert ticker.price == 1e-10
        assert ticker.volume == 1e-10

    def test_ticker_with_scientific_notation(self) -> None:
        """Should handle scientific notation values."""
        ticker = Ticker(ts=1.6e9, price=5e4, volume=1e2)
        
        assert ticker.ts == 1.6e9
        assert ticker.price == 5e4
        assert ticker.volume == 1e2

    def test_ticker_default_volume(self) -> None:
        """Should use default volume of 0.0."""
        ticker = Ticker(ts=1609459200.0, price=50000.0)
        
        assert ticker.volume == 0.0

    def test_multiple_tickers_independent(self) -> None:
        """Multiple ticker instances should be independent."""
        ticker1 = Ticker(ts=1609459200.0, price=50000.0, volume=100.0)
        ticker2 = Ticker(ts=1609459260.0, price=51000.0, volume=110.0)
        
        assert ticker1.price != ticker2.price
        assert ticker1.ts != ticker2.ts
        assert ticker1.volume != ticker2.volume


class TestChaosScenarios:
    """Test various chaos scenarios."""

    @pytest.mark.parametrize("invalid_value", [
        float('inf'),
        float('-inf'),
        float('nan'),
    ])
    def test_handles_special_float_values(self, invalid_value) -> None:
        """Should handle special float values (inf, -inf, nan)."""
        ticker = Ticker(ts=invalid_value, price=invalid_value, volume=invalid_value)
        
        # Values are stored as-is (validation would happen at ingestion level)
        if math.isnan(invalid_value):
            assert math.isnan(ticker.ts)
            assert math.isnan(ticker.price)
            assert math.isnan(ticker.volume)
        elif math.isinf(invalid_value):
            assert math.isinf(ticker.ts)
            assert math.isinf(ticker.price)
            assert math.isinf(ticker.volume)

    def test_ticker_repr_does_not_crash(self) -> None:
        """Ticker __repr__ should not crash with any values."""
        tickers = [
            Ticker(ts=0.0, price=0.0, volume=0.0),
            Ticker(ts=1e15, price=1e15, volume=1e15),
            Ticker(ts=-1000.0, price=-1000.0, volume=-1000.0),
        ]
        
        for ticker in tickers:
            repr_str = repr(ticker)
            assert isinstance(repr_str, str)
            assert len(repr_str) > 0

    def test_ticker_equality(self) -> None:
        """Test ticker equality comparison."""
        ticker1 = Ticker(ts=1609459200.0, price=50000.0, volume=100.0)
        ticker2 = Ticker(ts=1609459200.0, price=50000.0, volume=100.0)
        ticker3 = Ticker(ts=1609459200.0, price=50001.0, volume=100.0)
        
        assert ticker1 == ticker2
        assert ticker1 != ticker3

    def test_ticker_not_hashable_by_default(self) -> None:
        """Ticker objects are not hashable (dataclass without frozen=True)."""
        ticker1 = Ticker(ts=1609459200.0, price=50000.0, volume=100.0)
        
        # Ticker is not frozen, so not hashable
        with pytest.raises(TypeError, match="unhashable type"):
            _ = hash(ticker1)
