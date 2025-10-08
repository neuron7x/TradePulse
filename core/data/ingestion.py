# SPDX-License-Identifier: MIT
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Protocol

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


class _TickerHandler(Protocol):
    def __call__(self, tick: Ticker) -> None: ...


class DataIngestor:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret

    @staticmethod
    def _normalise_path(path: str | Path) -> Path:
        if isinstance(path, Path):
            return path
        return Path(path)

    def historical_csv(
        self,
        path: str | Path,
        on_tick: _TickerHandler,
        *,
        required_fields: Iterable[str] = ("ts", "price"),
    ) -> None:
        file_path = self._normalise_path(path)
        missing: list[str] = []
        with file_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV file must include a header row")
            missing = [field for field in required_fields if field not in reader.fieldnames]
            if missing:
                raise ValueError(f"CSV missing required columns: {', '.join(missing)}")
            for row_number, row in enumerate(reader, start=2):
                if not row:
                    continue
                try:
                    ts_raw = row.get("ts")
                    price_raw = row.get("price")
                    if ts_raw is None or price_raw is None:
                        raise ValueError("row is missing required values")
                    ts = float(ts_raw)
                    price = float(price_raw)
                    volume = float(row.get("volume", 0.0) or 0.0)
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        "Skipping malformed row %s in %s: %s", row_number, file_path, exc
                    )
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
