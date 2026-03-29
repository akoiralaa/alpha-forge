
from __future__ import annotations

import time
from dataclasses import dataclass, field

@dataclass
class SymbolStaleness:
    last_tick_ns: int = 0
    is_stale: bool = False
    orders_cancelled: int = 0

class StaleDataMonitor:

    def __init__(self, stale_threshold_ms: float = 500.0):
        self.stale_threshold_ns = int(stale_threshold_ms * 1_000_000)
        self._symbols: dict[int, SymbolStaleness] = {}

    def on_tick(self, symbol_id: int, timestamp_ns: int = 0):
        now = timestamp_ns or time.time_ns()
        if symbol_id not in self._symbols:
            self._symbols[symbol_id] = SymbolStaleness()
        self._symbols[symbol_id].last_tick_ns = now
        self._symbols[symbol_id].is_stale = False

    def check_staleness(self, now_ns: int = 0) -> list[int]:
        now = now_ns or time.time_ns()
        newly_stale = []
        for sid, state in self._symbols.items():
            if state.last_tick_ns == 0:
                continue
            elapsed = now - state.last_tick_ns
            if elapsed > self.stale_threshold_ns and not state.is_stale:
                state.is_stale = True
                newly_stale.append(sid)
        return newly_stale

    def is_stale(self, symbol_id: int) -> bool:
        state = self._symbols.get(symbol_id)
        return state.is_stale if state else False

    def is_order_allowed(self, symbol_id: int) -> bool:
        return not self.is_stale(symbol_id)

    def get_stale_symbols(self) -> list[int]:
        return [sid for sid, s in self._symbols.items() if s.is_stale]

    def record_cancellation(self, symbol_id: int, count: int = 1):
        if symbol_id in self._symbols:
            self._symbols[symbol_id].orders_cancelled += count
