# Placeholder: aquí integrarías bybit API
from __future__ import annotations
from typing import Dict
from .base_exchange import BaseExchange


class BybitExchange(BaseExchange):
def connect(self):
raise NotImplementedError("Bybit live no implementado (usa paper mode)")


def place_order(self, side: str, qty: float, symbol: str) -> Dict:
raise NotImplementedError("Bybit live no implementado (usa paper mode)")