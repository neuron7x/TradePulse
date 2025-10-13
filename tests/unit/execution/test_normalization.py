# SPDX-License-Identifier: MIT
"""Unit tests for exchange lot and tick mapping in the normalizer."""

from __future__ import annotations

import pytest

from execution.normalization import NormalizationError, SymbolNormalizer, SymbolSpecification


def _build_normalizer() -> SymbolNormalizer:
    specs = {
        "ETHUSDT": SymbolSpecification(
            symbol="ETHUSDT",
            min_qty=0.01,
            min_notional=10.0,
            step_size=0.01,
            tick_size=0.05,
        )
    }
    symbol_map = {"ETH-USD": "ETHUSDT", "ethusd": "ETHUSDT"}
    return SymbolNormalizer(symbol_map=symbol_map, specifications=specs)


def test_symbol_normalizer_alias_rounding() -> None:
    normalizer = _build_normalizer()

    spec = normalizer.specification("eth_usd")
    assert spec is not None and spec.symbol == "ETHUSDT"

    rounded_qty = normalizer.round_quantity("ETH-USD", 1.234)
    assert rounded_qty == pytest.approx(1.23)

    rounded_price = normalizer.round_price("ethusd", 2010.027)
    assert rounded_price == pytest.approx(2010.05)


def test_symbol_normalizer_alias_validation_constraints() -> None:
    normalizer = _build_normalizer()

    with pytest.raises(NormalizationError):
        normalizer.validate("ETH_USD", 0.005, 2000.0)

    with pytest.raises(NormalizationError):
        normalizer.validate("ETH-USD", 0.01, 900.0)

    normalizer.validate("ETHUSDT", 0.02, 2000.0)
