# SPDX-License-Identifier: MIT
"""Parsers for Binance spot depth feeds."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Mapping, Sequence

from ..models import OrderBookDiff, OrderBookSnapshot, PriceLevel


def _levels(entries: Iterable[Sequence[str | float]]) -> tuple[PriceLevel, ...]:
    return tuple(PriceLevel.from_raw(price, quantity) for price, quantity in entries)


def parse_snapshot(
    payload: Mapping[str, object],
    *,
    instrument: str,
    ts_arrival: datetime,
    source: str = "binance",
) -> OrderBookSnapshot:
    last_update_id = int(payload["lastUpdateId"])
    bids = _levels(payload.get("bids", []))
    asks = _levels(payload.get("asks", []))
    raw_event = payload.get("E")
    if raw_event is None:
        ts_event = ts_arrival
    else:
        event_value = float(raw_event)
        if event_value > 1e12:  # millisecond precision
            event_value /= 1_000
        ts_event = datetime.fromtimestamp(event_value, tz=timezone.utc)
    if ts_arrival.tzinfo is None:
        raise ValueError("ts_arrival must be timezone aware")
    return OrderBookSnapshot(
        instrument=instrument,
        sequence=last_update_id,
        bids=bids,
        asks=asks,
        ts_event=ts_event,
        ts_arrival=ts_arrival,
        source=source,
    )


def parse_diff(
    payload: Mapping[str, object],
    *,
    ts_arrival: datetime,
    source: str = "binance",
) -> OrderBookDiff:
    instrument = str(payload["s"])
    first_update = int(payload["U"])
    final_update = int(payload["u"])
    event_time = datetime.fromtimestamp(int(payload["E"]) / 1_000, tz=timezone.utc)
    bids = _levels(payload.get("b", []))
    asks = _levels(payload.get("a", []))
    if ts_arrival.tzinfo is None:
        raise ValueError("ts_arrival must be timezone aware")
    return OrderBookDiff(
        instrument=instrument,
        sequence_start=first_update,
        sequence_end=final_update,
        bids=bids,
        asks=asks,
        ts_event=event_time,
        ts_arrival=ts_arrival,
        source=source,
    )
