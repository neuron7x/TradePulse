# SPDX-License-Identifier: MIT
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

try:
    from binance.websocket.spot.websocket_client import SpotWebsocketClient as BinanceWS
except Exception:  # pragma: no cover - optional dependency
    BinanceWS = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class Ticker:
    ts: float
    price: float
    volume: float = 0.0


class BinanceStreamHandle:
    def __init__(self, ws: BinanceWS) -> None:  # type: ignore[name-defined]
        self._ws = ws
        self._active = False

    def start(self, *, symbol: str, interval: str, callback: Callable[[dict], None]) -> None:
        self._active = True
        self._ws.start()
        self._ws.kline(symbol=symbol.lower(), id=1, interval=interval, callback=callback)

    def close(self) -> None:
        if self._active:
            self._ws.stop()
            self._active = False

    def __enter__(self) -> "BinanceStreamHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class DataIngestor:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret

    def historical_csv(self, path: str, on_tick: Callable[[Ticker], None], *, required_fields: Iterable[str] = ("ts", "price")) -> None:
        missing: list[str] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV file must include a header row")
            missing = [field for field in required_fields if field not in reader.fieldnames]
            if missing:
                raise ValueError(f"CSV missing required columns: {', '.join(missing)}")
            for row_number, row in enumerate(reader, start=2):
                try:
                    ts = float(row["ts"])
                    price = float(row["price"])
                    volume = float(row.get("volume", 0.0) or 0.0)
                except (TypeError, ValueError) as exc:
                    logger.warning("Skipping malformed row %s in %s: %s", row_number, path, exc)
                    continue
                on_tick(Ticker(ts=ts, price=price, volume=volume))

    def binance_ws(self, symbol: str, on_tick: Callable[[Ticker], None], *, interval: str = "1m") -> object:
        if BinanceWS is None:
            raise RuntimeError("python-binance is not installed")

        ws = BinanceWS()  # type: ignore[operator]
        handle = BinanceStreamHandle(ws)

        def _callback(message: dict) -> None:
            kline = message.get("k")
            if not kline:
                return
            try:
                tick = Ticker(ts=float(kline["T"]) / 1000.0, price=float(kline["c"]), volume=float(kline.get("v", 0.0)))
            except (TypeError, ValueError) as exc:
                logger.warning("Failed to parse websocket payload: %s", exc)
                return
            on_tick(tick)

        handle.start(symbol=symbol, interval=interval, callback=_callback)
        setattr(ws, "stream_handle", handle)
        return ws
