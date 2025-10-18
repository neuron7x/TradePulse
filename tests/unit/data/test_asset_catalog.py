from datetime import datetime, timezone

import pytest

from core.data.asset_catalog import AssetCatalog, AssetRecord, AssetStatus
from core.data.models import InstrumentType


def test_asset_catalog_register_and_resolve() -> None:
    catalog = AssetCatalog()
    record = catalog.create_asset(
        asset_id="btc",
        name="Bitcoin",
        primary_symbol="btcusdt",
        instrument_type=InstrumentType.SPOT,
        venue_symbols={"binance": "BTCUSDT", "coinbase": "btc-usd"},
    )

    assert record.primary_symbol == "BTC/USDT"
    assert catalog.resolve(" btcusdt ").asset_id == "btc"
    assert catalog.resolve("BTCUSD", venue="coinbase").asset_id == "btc"
    assert catalog.get_display_symbol("btc", venue="binance") == "BTC/USDT"
    assert catalog.get_display_symbol("btc") == "BTC/USDT"


def test_asset_catalog_update_symbol_tracks_history() -> None:
    catalog = AssetCatalog()
    catalog.create_asset(
        asset_id="sol",
        name="Solana",
        primary_symbol="solusd",
    )

    catalog.update_primary_symbol("sol", "SOLUSDC")

    assert catalog.resolve("SOLUSDC").asset_id == "sol"
    assert catalog.resolve("solusd").asset_id == "sol"
    with pytest.raises(KeyError):
        catalog.resolve("SOLUSD", include_historical=False)


def test_asset_catalog_synchronize_venue_symbol_records_history() -> None:
    catalog = AssetCatalog()
    catalog.create_asset(
        asset_id="btcp",
        name="BTC Perpetual",
        primary_symbol="btcusdt",
        instrument_type=InstrumentType.FUTURES,
        venue_symbols={"binance": "btcusdt"},
    )

    catalog.synchronize_venue_symbol("btcp", "binance", "btcusdt_perp")

    assert catalog.resolve("BTCUSDT_PERP", venue="binance").asset_id == "btcp"
    assert catalog.get_display_symbol("btcp", venue="binance") == "BTC-USDT-PERP"
    # Historical alias remains discoverable without the venue context.
    assert catalog.resolve("BTCUSDT").asset_id == "btcp"


def test_asset_catalog_mark_delisted_and_reactivate() -> None:
    catalog = AssetCatalog()
    catalog.create_asset(
        asset_id="aapl",
        name="Apple Inc.",
        primary_symbol="aapl",
    )

    when = datetime(2024, 1, 1, tzinfo=timezone.utc)
    catalog.mark_delisted("aapl", when=when)

    record = catalog.get("aapl")
    assert record.status == AssetStatus.DELISTED
    assert record.delisted_at == when

    catalog.mark_active("aapl")
    assert record.status == AssetStatus.ACTIVE
    assert record.delisted_at is None


def test_asset_catalog_update_name_trimmed() -> None:
    catalog = AssetCatalog()
    catalog.create_asset(
        asset_id="eth",
        name="Ethereum",
        primary_symbol="ethusd",
    )

    catalog.update_name("eth", "  Ethereum Network  ")

    assert catalog.get("eth").name == "Ethereum Network"


def test_asset_catalog_set_name_rejects_blank() -> None:
    catalog = AssetCatalog()
    catalog.create_asset(asset_id="eth", name="Ethereum", primary_symbol="ethusd")
    with pytest.raises(ValueError):
        catalog.get("eth").set_name("   ")


def test_asset_catalog_historical_ambiguity_raises() -> None:
    catalog = AssetCatalog()
    catalog.create_asset(
        asset_id="asset1",
        name="Asset One",
        primary_symbol="aaausd",
    )
    catalog.update_primary_symbol("asset1", "aaausdc")

    catalog.create_asset(
        asset_id="asset2",
        name="Asset Two",
        primary_symbol="bbbusd",
    )
    catalog.synchronize_venue_symbol("asset2", "binance", "AAaUsd")
    catalog.synchronize_venue_symbol("asset2", "binance", "BBBUSD")

    with pytest.raises(LookupError):
        catalog.resolve("AAaUsd")


def test_asset_catalog_prevents_duplicate_registration() -> None:
    catalog = AssetCatalog()
    catalog.create_asset(asset_id="id1", name="Asset One", primary_symbol="aaa/usd")
    with pytest.raises(ValueError, match="already registered"):
        catalog.create_asset(asset_id="id1", name="Duplicate", primary_symbol="bbb/usd")


def test_asset_catalog_rejects_symbol_conflicts() -> None:
    catalog = AssetCatalog()
    catalog.create_asset(asset_id="first", name="First", primary_symbol="abc/usd")
    with pytest.raises(ValueError, match="already registered"):
        catalog.create_asset(asset_id="second", name="Second", primary_symbol="abc/usd")


def test_asset_record_validates_non_empty_fields() -> None:
    catalog = AssetCatalog()
    with pytest.raises(ValueError):
        catalog.create_asset(asset_id="   ", name="Asset", primary_symbol="sym")
    with pytest.raises(ValueError):
        catalog.create_asset(asset_id="asset", name="   ", primary_symbol="sym")


def test_asset_catalog_update_primary_symbol_noop_returns_same_record() -> None:
    catalog = AssetCatalog()
    record = catalog.create_asset(asset_id="asset", name="Asset", primary_symbol="sym")
    updated = catalog.update_primary_symbol("asset", record.primary_symbol)
    assert updated is record


def test_asset_catalog_constructor_registers_initial_assets() -> None:
    record = AssetRecord(asset_id="seed", name="Seed", primary_symbol="seed/usd")
    catalog = AssetCatalog([record])
    assert catalog.get("seed").primary_symbol == "SEED/USD"


def test_asset_catalog_synchronize_venue_symbol_no_change_returns_asset() -> None:
    catalog = AssetCatalog()
    catalog.create_asset(
        asset_id="asset", name="Asset", primary_symbol="aaa/usd", venue_symbols={"binance": "AAA/USDT"}
    )
    result = catalog.synchronize_venue_symbol("asset", "binance", "AAA/USDT")
    assert result is catalog.get("asset")


def test_asset_catalog_resolve_prefers_venue_assignment() -> None:
    catalog = AssetCatalog()
    catalog.create_asset(
        asset_id="asset", name="Asset", primary_symbol="aaa/usd", venue_symbols={"binance": "AAA/USDT"}
    )
    resolved = catalog.resolve("AAA/USDT", venue="binance")
    assert resolved.asset_id == "asset"


def test_asset_catalog_assets_filter_by_status() -> None:
    catalog = AssetCatalog()
    catalog.create_asset(asset_id="a", name="A", primary_symbol="a/usd")
    catalog.create_asset(asset_id="b", name="B", primary_symbol="b/usd")
    catalog.mark_delisted("b")
    active_ids = {asset.asset_id for asset in catalog.assets(status=AssetStatus.ACTIVE)}
    assert active_ids == {"a"}

