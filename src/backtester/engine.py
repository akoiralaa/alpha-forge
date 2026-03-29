"""Tick-by-tick deterministic backtester.

NOT a vectorized engine. One tick in, one state update out.
Same C++ FeatureEngine binary as live — zero logic drift.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Ensure C++ module is importable
build_dir = Path(__file__).resolve().parent.parent / "cpp" / "build"
sys.path.insert(0, str(build_dir))

from src.backtester.execution import LatencyModel, SimulatedExecution
from src.backtester.types import (
    BacktestResult,
    BookSnapshot,
    InstrumentSpec,
    OrderIntent,
    OrderType,
    Position,
    Side,
    SimFill,
)

# Type alias for signal function
# Takes MSV dict and returns Optional[OrderIntent]
SignalFn = Callable[[object, dict[int, Position]], Optional[OrderIntent]]


class Backtester:
    """Deterministic tick-by-tick backtester.

    Replay loop:
    1. Load ticks in chronological order by capture_time_ns
    2. Call engine.on_tick(tick)
    3. Pass MSV to signal function
    4. If signal → simulate fill
    5. If fill → engine.on_fill(fill), update positions
    6. Record state
    """

    def __init__(
        self,
        feature_engine,
        signal_fn: SignalFn,
        execution: Optional[SimulatedExecution] = None,
        instrument_specs: Optional[dict[int, InstrumentSpec]] = None,
        max_position_per_symbol: int = 1000,
    ):
        self.engine = feature_engine
        self.signal_fn = signal_fn
        self.execution = execution or SimulatedExecution()
        self.specs = instrument_specs or {}
        self.max_pos = max_position_per_symbol

        # State
        self.positions: dict[int, Position] = {}
        self.pnl_history: list[float] = []
        self.fills: list[SimFill] = []
        self.total_cost_bps: float = 0.0
        self.total_latency_ns: int = 0

    def get_position(self, symbol_id: int) -> Position:
        if symbol_id not in self.positions:
            self.positions[symbol_id] = Position(symbol_id=symbol_id)
        return self.positions[symbol_id]

    def get_spec(self, symbol_id: int) -> InstrumentSpec:
        if symbol_id not in self.specs:
            self.specs[symbol_id] = InstrumentSpec(symbol_id=symbol_id)
        return self.specs[symbol_id]

    def _current_equity(self) -> float:
        total = 0.0
        for pos in self.positions.values():
            total += pos.realized_pnl + pos.unrealized_pnl
        return total

    def run(self, ticks: list) -> BacktestResult:
        """Run backtest on a list of C++ Tick objects.

        Ticks must be sorted by capture_time_ns.
        """
        self.engine.reset()
        self.positions.clear()
        self.pnl_history.clear()
        self.fills.clear()
        self.total_cost_bps = 0.0
        self.total_latency_ns = 0

        for tick in ticks:
            # 1. Process tick through feature engine
            msv = self.engine.on_tick(tick)

            # 2. Update position marks
            mid = (tick.bid + tick.ask) / 2.0
            if mid <= 0:
                mid = tick.last_price
            pos = self.get_position(tick.symbol_id)
            pos.update_mark(mid)

            # 3. If valid MSV, generate signal
            if msv.valid:
                order = self.signal_fn(msv, self.positions)

                if order is not None:
                    # Check position limits
                    current_pos = self.get_position(order.symbol_id)
                    new_qty = current_pos.quantity
                    if order.side == Side.BUY:
                        new_qty += order.size
                    else:
                        new_qty -= order.size

                    if abs(new_qty) <= self.max_pos:
                        # 4. Build book snapshot from current tick
                        book = BookSnapshot(
                            symbol_id=tick.symbol_id,
                            timestamp_ns=tick.capture_time_ns,
                            bid=tick.bid,
                            ask=tick.ask,
                            bid_size=tick.bid_size,
                            ask_size=tick.ask_size,
                            last_price=tick.last_price,
                            vpin=msv.vpin if not np.isnan(msv.vpin) else 0.0,
                        )

                        # 5. Simulate fill
                        spec = self.get_spec(order.symbol_id)
                        fill = self.execution.simulate_fill(order, book, spec)

                        if fill is not None:
                            # 6. Apply fill
                            current_pos.apply_fill(fill)

                            # Deduct transaction costs from realized PnL
                            if fill.cost_bps > 0:
                                notional = fill.fill_price * fill.fill_size
                                cost_dollars = notional * fill.cost_bps / 10000.0
                                current_pos.realized_pnl -= cost_dollars

                            self.fills.append(fill)
                            self.total_cost_bps += fill.cost_bps
                            self.total_latency_ns += fill.latency_ns

            # 7. Record equity
            self.pnl_history.append(self._current_equity())

        # Build result
        pnl_arr = np.array(self.pnl_history) if self.pnl_history else np.array([0.0])
        result = BacktestResult(
            pnl_series=pnl_arr,
            n_trades=len(self.fills),
            total_cost_bps=self.total_cost_bps,
            avg_latency_ns=(self.total_latency_ns / len(self.fills)
                            if self.fills else 0.0),
            fills=list(self.fills),
        )
        result.compute_metrics()
        return result

    def run_numpy(
        self,
        symbol_ids: np.ndarray,
        exchange_time_ns: np.ndarray,
        capture_time_ns: np.ndarray,
        bids: np.ndarray,
        asks: np.ndarray,
        bid_sizes: np.ndarray,
        ask_sizes: np.ndarray,
        last_prices: np.ndarray,
        last_sizes: np.ndarray,
    ) -> BacktestResult:
        """Run backtest from numpy arrays (avoids creating Python Tick objects)."""
        # Import here to avoid circular
        try:
            from engine import Tick
        except ImportError:
            from _onebrain_cpp import Tick

        n = len(symbol_ids)
        ticks = []
        for i in range(n):
            t = Tick()
            t.symbol_id = int(symbol_ids[i])
            t.exchange_time_ns = int(exchange_time_ns[i])
            t.capture_time_ns = int(capture_time_ns[i])
            t.bid = float(bids[i])
            t.ask = float(asks[i])
            t.bid_size = int(bid_sizes[i])
            t.ask_size = int(ask_sizes[i])
            t.last_price = float(last_prices[i])
            t.last_size = int(last_sizes[i])
            ticks.append(t)
        return self.run(ticks)
