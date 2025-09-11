from __future__ import annotations
from src.indicators.technical import EMA
from src.strategies.base_strategy import Strategy, Signal


class EMACrossover(Strategy):
def __init__(self, fast: int = 9, slow: int = 21):
self.ema_f = EMA(fast)
self.ema_s = EMA(slow)
self.prev_state = None # "bull" | "bear" | None


def on_tick(self, ts: int, price: float) -> Signal:
f = self.ema_f.update(price)
s = self.ema_s.update(price)
if f is None or s is None:
return Signal(None, "warmup")
state = "bull" if f > s else "bear"
if self.prev_state is None:
self.prev_state = state
return Signal(None, "init")
# Señal solo cuando hay cruce
if state != self.prev_state:
self.prev_state = state
if state == "bull":
return Signal("BUY", "EMA fast>slow")
else:
return Signal("SELL", "EMA fast<slow")
return Signal(None, "hold")