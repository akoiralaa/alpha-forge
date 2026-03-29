
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.execution.broker import BrokerInterface, PaperBroker
from src.execution.wal import OrderState, WriteAheadLog

class BreakType(Enum):
    QUANTITY_MISMATCH = "QUANTITY_MISMATCH"
    MISSING_INTERNAL = "MISSING_INTERNAL"    # broker has position, we don't
    MISSING_BROKER = "MISSING_BROKER"        # we have position, broker doesn't
    CASH_MISMATCH = "CASH_MISMATCH"

@dataclass
class ReconciliationBreak:
    break_type: BreakType
    symbol_id: int
    internal_qty: float
    broker_qty: float
    difference: float
    timestamp_ns: int

@dataclass
class ReconciliationReport:
    timestamp_ns: int
    is_clean: bool
    breaks: list[ReconciliationBreak] = field(default_factory=list)
    internal_positions: dict[int, float] = field(default_factory=dict)
    broker_positions: dict[int, float] = field(default_factory=dict)
    tolerance: float = 1e-6

class Reconciler:

    def __init__(
        self,
        broker: BrokerInterface | PaperBroker,
        wal: WriteAheadLog,
        tolerance: float = 1e-6,
    ):
        self.broker = broker
        self.wal = wal
        self.tolerance = tolerance
        self._reports: list[ReconciliationReport] = []

    def compute_internal_positions(self) -> dict[int, float]:
        positions: dict[int, float] = {}
        state = self.wal.replay()
        for order_id, entry in state.items():
            if entry.state == OrderState.FILLED and entry.filled_size > 0:
                sid = entry.symbol_id
                if sid == 0:  # skip kill switch events
                    continue
                current = positions.get(sid, 0.0)
                positions[sid] = current + entry.side * entry.filled_size
        # Remove zero positions
        return {k: v for k, v in positions.items() if abs(v) > self.tolerance}

    def reconcile(self) -> ReconciliationReport:
        internal = self.compute_internal_positions()
        broker = self.broker.get_positions()
        # Remove zero broker positions
        broker = {k: v for k, v in broker.items() if abs(v) > self.tolerance}

        breaks: list[ReconciliationBreak] = []
        now = time.time_ns()

        all_symbols = set(internal.keys()) | set(broker.keys())
        for sid in all_symbols:
            i_qty = internal.get(sid, 0.0)
            b_qty = broker.get(sid, 0.0)
            diff = abs(i_qty - b_qty)

            if diff <= self.tolerance:
                continue

            if sid not in internal:
                btype = BreakType.MISSING_INTERNAL
            elif sid not in broker:
                btype = BreakType.MISSING_BROKER
            else:
                btype = BreakType.QUANTITY_MISMATCH

            breaks.append(ReconciliationBreak(
                break_type=btype,
                symbol_id=sid,
                internal_qty=i_qty,
                broker_qty=b_qty,
                difference=diff,
                timestamp_ns=now,
            ))

        report = ReconciliationReport(
            timestamp_ns=now,
            is_clean=len(breaks) == 0,
            breaks=breaks,
            internal_positions=internal,
            broker_positions=broker,
            tolerance=self.tolerance,
        )
        self._reports.append(report)
        return report

    @property
    def last_report(self) -> Optional[ReconciliationReport]:
        return self._reports[-1] if self._reports else None

    @property
    def reports(self) -> list[ReconciliationReport]:
        return list(self._reports)
