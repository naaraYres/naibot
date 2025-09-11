from __future__ import annotations
from src.indicators.technical import RSI
from src.strategies.base_strategy import Strategy, Signal


class RSIBands(Strategy):
def __init__(self, period: int = 14, lo: float = 30.0, hi: float = 70.0):
self.rsi = RSI(period)
self.lo = lo
self.hi = hi


def on_tick(self, ts: int, price: float) -> Signal:
v = self.rsi.update(price)
if v is None:
return Signal(None, "warmup")
if v < self.lo:
return Signal("BUY", f"RSI {v:.1f}< {self.lo}")
if v > self.hi:
return Signal("SELL", f"RSI {v:.1f}> {self.hi}")
return Signal(None, "mid-range")