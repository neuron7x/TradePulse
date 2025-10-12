"""Tests for data catalog normalization helpers."""

import enum

import pytest

from core.data.catalog import normalize_symbol, normalize_venue


class InstrumentKind(enum.Enum):
    SPOT = "spot"
    DERIVATIVE = "derivative"


@pytest.mark.parametrize(
    "raw, expected",
    [
        (" binance ", "BINANCE"),
        ("binance global", "BINANCE"),
        ("POLYGON.io", "POLYGON"),
    ],
)
def test_normalize_venue_known_alias(raw: str, expected: str) -> None:
    """Known venue aliases are canonicalised to their configured names."""

    assert normalize_venue(raw) == expected


def test_normalize_venue_unknown_is_uppercased() -> None:
    """Unknown venues are trimmed and upper-cased for consistent casing."""

    assert normalize_venue(" custom-exchange ") == "CUSTOM-EXCHANGE"


def test_normalize_venue_rejects_empty_string() -> None:
    """An empty venue string should raise a ``ValueError`` for clarity."""

    with pytest.raises(ValueError):
        normalize_venue("  ")


@pytest.mark.parametrize(
    "raw, expected",
    [
        (" btcusdt ", "BTC/USDT"),
        ("eth_usd", "ETH/USD"),
        ("SOL-USDT", "SOL/USDT"),
    ],
)
def test_normalize_symbol_spot_aliases(raw: str, expected: str) -> None:
    """Various spot aliases normalise to ``BASE/QUOTE`` representation."""

    assert normalize_symbol(raw) == expected


def test_normalize_symbol_single_leg_preserved() -> None:
    """Single leg instruments keep their canonical ticker without separators."""

    assert normalize_symbol(" aapl ") == "AAPL"


def test_normalize_symbol_derivative_suffix_detected() -> None:
    """Derivative suffixes trigger ``-`` separated canonical representation."""

    assert normalize_symbol("btc_usdt-perp") == "BTC-USDT-PERP"


def test_normalize_symbol_derivative_hint_overrides_suffix() -> None:
    """An explicit derivative hint forces dash separated output for pairs."""

    assert normalize_symbol("btc/usdt", instrument_type_hint="future") == "BTC-USDT"


def test_normalize_symbol_enum_hint_respected() -> None:
    """Enum hints behave like their value when inferring the instrument type."""

    assert (
        normalize_symbol(
            "ethusd", instrument_type_hint=InstrumentKind.SPOT
        )
        == "ETH/USD"
    )


def test_normalize_symbol_rejects_empty_string() -> None:
    """An empty symbol should raise a ``ValueError`` for caller mistakes."""

    with pytest.raises(ValueError):
        normalize_symbol("   ")
