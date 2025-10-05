# SPDX-License-Identifier: MIT
from __future__ import annotations
import time, json, threading
from dataclasses import dataclass
from typing import Callable, Optional, Any, Dict, List

try:
    from binance.client import Client as BinanceClient
    from binance.websocket.spot.websocket_client import SpotWebsocketClient as BinanceWS
except Exception:
    BinanceClient = None
    BinanceWS = None

@dataclass
class Ticker:
    ts: float
    price: float
    volume: float

class DataIngestor:
    def __init__(self, api_key: Optional[str]=None, api_secret: Optional[str]=None):
        self.api_key = api_key
        self.api_secret = api_secret

    def historical_csv(self, path: str, on_tick: Callable[[Ticker], None]):
        import csv
        with open(path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                on_tick(Ticker(ts=float(row["ts"]), price=float(row["price"]), volume=float(row.get("volume",0))))

    def binance_ws(self, symbol: str, on_tick: Callable[[Ticker], None]):
        if BinanceWS is None:
            raise RuntimeError("python-binance not installed")
        ws = BinanceWS()
        def cb(msg):
            if "e" in msg and "k" in msg:  # kline
                k = msg["k"]
                on_tick(Ticker(ts=k["T"]/1000.0, price=float(k["c"]), volume=float(k["v"])))
        ws.start()
        ws.kline(symbol=symbol.lower(), id=1, interval="1m", callback=cb)
        return ws  # caller should .stop()
