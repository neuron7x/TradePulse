"""Resampling utilities bridging tick, L1, OHLCV and order book data."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd


def _ensure_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("frame must have a DatetimeIndex")
    if frame.index.tz is None:
        frame = frame.tz_localize("UTC")
    return frame.sort_index()


def resample_ticks_to_l1(
    ticks: pd.DataFrame,
    *,
    freq: str,
    price_col: str = "price",
    size_col: str = "size",
) -> pd.DataFrame:
    """Resample tick data to L1 snapshots by forward filling the last quote."""

    ticks = _ensure_datetime_index(ticks)
    grouped = ticks[[price_col, size_col]].resample(freq)
    l1 = grouped.last().ffill()
    l1.columns = pd.Index(["mid_price", "last_size"])
    return l1


def resample_l1_to_ohlcv(
    l1: pd.DataFrame,
    *,
    freq: str,
    price_col: str = "mid_price",
    size_col: str = "last_size",
) -> pd.DataFrame:
    """Aggregate L1 quotes into OHLCV bars."""

    l1 = _ensure_datetime_index(l1)
    grouped = l1.resample(freq)
    ohlc = grouped[price_col].ohlc()
    volume = grouped[size_col].sum(min_count=1)
    ohlc["volume"] = volume
    return ohlc.dropna(how="all")


def align_timeframes(frames: Mapping[str, pd.DataFrame], *, reference: str) -> Dict[str, pd.DataFrame]:
    """Align multiple timeframes to the ``reference`` calendar."""

    if reference not in frames:
        raise ValueError("reference timeframe missing")
    ref_index = _ensure_datetime_index(frames[reference]).index
    aligned: Dict[str, pd.DataFrame] = {}
    for name, frame in frames.items():
        frame = _ensure_datetime_index(frame)
        aligned[name] = frame.reindex(ref_index, method="pad")
    return aligned


def resample_order_book(
    levels: pd.DataFrame,
    *,
    freq: str,
    bid_cols: Iterable[str],
    ask_cols: Iterable[str],
    bid_price_col: str,
    ask_price_col: str,
) -> pd.DataFrame:
    """Resample level-2 order book snapshots preserving imbalance metrics."""

    levels = _ensure_datetime_index(levels)
    bid_cols = tuple(bid_cols)
    ask_cols = tuple(ask_cols)
    if not bid_cols or not ask_cols:
        raise ValueError("bid_cols and ask_cols must not be empty")
    if bid_price_col not in levels.columns or ask_price_col not in levels.columns:
        raise KeyError("bid_price_col and ask_price_col must exist in the frame")

    grouped = levels.resample(freq)
    bids = grouped[list(bid_cols)].mean()
    asks = grouped[list(ask_cols)].mean()
    bid_total = bids.sum(axis=1)
    ask_total = asks.sum(axis=1)
    denom = (bid_total + ask_total).replace(0.0, np.nan)
    out = pd.concat({"bids": bids, "asks": asks}, axis=1)

    best_bid = grouped[bid_price_col].last().ffill()
    best_ask = grouped[ask_price_col].last().ffill()
    numerator = best_bid * ask_total + best_ask * bid_total
    microprice = (numerator / denom).fillna((best_bid + best_ask) / 2)
    imbalance = ((bid_total - ask_total) / denom).fillna(0.0)

    out["microprice"] = microprice
    out["imbalance"] = imbalance
    return out


__all__ = [
    "align_timeframes",
    "resample_l1_to_ohlcv",
    "resample_order_book",
    "resample_ticks_to_l1",
]

