
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from src.execution.broker import BrokerInterface, BrokerOrder, PaperBroker
from src.execution.wal import OrderState, WALEntry, WriteAheadLog

class KillLevel(IntEnum):
    NORMAL = 0
    CANCEL_ONLY = 1      # cancel all orders, block new ones
    FLATTEN = 2           # cancel + flatten positions
    DISCONNECT = 3        # cancel + flatten + disconnect

@dataclass
class KillSwitchEvent:
    timestamp_ns: int
    level: KillLevel
    reason: str
    orders_cancelled: int = 0
    positions_flattened: int = 0

class KillSwitch:

    def __init__(
        self,
        broker: BrokerInterface | PaperBroker,
        wal: WriteAheadLog,
        drawdown_auto_kill_pct: float = 0.15,
    ):
        self.broker = broker
        self.wal = wal
        self.drawdown_auto_kill_pct = drawdown_auto_kill_pct
        self.level = KillLevel.NORMAL
        self.armed = True
        self._events: list[KillSwitchEvent] = []
        self._blocked_since_ns: Optional[int] = None

    def activate(self, level: KillLevel, reason: str) -> KillSwitchEvent:
        if level <= self.level:
            # Already at this level or higher
            return KillSwitchEvent(
                timestamp_ns=time.time_ns(),
                level=self.level,
                reason=f"already at level {self.level.name}",
            )

        self.level = level
        self._blocked_since_ns = time.time_ns()
        event = KillSwitchEvent(
            timestamp_ns=time.time_ns(),
            level=level,
            reason=reason,
        )

        # Level 1+: cancel all open orders
        if level >= KillLevel.CANCEL_ONLY:
            event.orders_cancelled = self._cancel_all_orders()

        # Level 2+: flatten all positions
        if level >= KillLevel.FLATTEN:
            event.positions_flattened = self._flatten_all_positions()

        # Log to WAL
        self._log_kill_event(event)

        self._events.append(event)
        return event

    def check_drawdown(self, nav: float, peak_nav: float) -> Optional[KillSwitchEvent]:
        if peak_nav <= 0 or not self.armed:
            return None
        dd = (peak_nav - nav) / peak_nav
        if dd >= self.drawdown_auto_kill_pct and self.level < KillLevel.FLATTEN:
            return self.activate(
                KillLevel.FLATTEN,
                f"auto: drawdown {dd:.1%} >= {self.drawdown_auto_kill_pct:.1%}",
            )
        return None

    def is_order_allowed(self) -> bool:
        return self.level == KillLevel.NORMAL

    def reset(self, reason: str = "manual reset"):
        self.level = KillLevel.NORMAL
        self._blocked_since_ns = None
        self._events.append(KillSwitchEvent(
            timestamp_ns=time.time_ns(),
            level=KillLevel.NORMAL,
            reason=reason,
        ))

    def _cancel_all_orders(self) -> int:
        open_orders = self.wal.get_open_orders()
        cancelled = 0
        for entry in open_orders:
            if entry.broker_order_id:
                success = self.broker.cancel_order(entry.broker_order_id)
                if success:
                    # Log cancellation to WAL
                    cancel_entry = WALEntry(
                        sequence_id=0,
                        timestamp_ns=time.time_ns(),
                        order_id=entry.order_id,
                        state=OrderState.CANCELLED,
                        symbol_id=entry.symbol_id,
                        side=entry.side,
                        order_type=entry.order_type,
                        size=entry.size,
                        error_msg="kill_switch_cancel",
                    )
                    self.wal.append(cancel_entry)
                    cancelled += 1
        return cancelled

    def _flatten_all_positions(self) -> int:
        positions = self.broker.get_positions()
        flattened = 0
        for symbol_id, qty in positions.items():
            if abs(qty) < 1e-10:
                continue
            # Opposite side to flatten
            side = -1 if qty > 0 else 1
            order_id = f"KILL-{symbol_id}-{time.time_ns()}"
            order = BrokerOrder(
                order_id=order_id,
                symbol_id=symbol_id,
                side=side,
                size=int(abs(qty)),
                order_type="MARKET",
            )
            # Log to WAL before submitting
            wal_entry = WALEntry(
                sequence_id=0,
                timestamp_ns=time.time_ns(),
                order_id=order_id,
                state=OrderState.PENDING,
                symbol_id=symbol_id,
                side=side,
                order_type="MARKET",
                size=int(abs(qty)),
                metadata="kill_switch_flatten",
            )
            self.wal.append(wal_entry)

            broker_id = self.broker.submit_order(order)

            # Log submission
            submit_entry = WALEntry(
                sequence_id=0,
                timestamp_ns=time.time_ns(),
                order_id=order_id,
                state=OrderState.FILLED,
                symbol_id=symbol_id,
                side=side,
                order_type="MARKET",
                size=int(abs(qty)),
                filled_size=int(abs(qty)),
                broker_order_id=broker_id,
                metadata="kill_switch_flatten",
            )
            self.wal.append(submit_entry)
            flattened += 1

        return flattened

    def _log_kill_event(self, event: KillSwitchEvent):
        entry = WALEntry(
            sequence_id=0,
            timestamp_ns=event.timestamp_ns,
            order_id=f"KILL-EVENT-{event.timestamp_ns}",
            state=OrderState.ERROR,
            symbol_id=0,
            side=0,
            order_type="KILL_SWITCH",
            size=0,
            error_msg=f"level={event.level.name}: {event.reason}",
            metadata=f"cancelled={event.orders_cancelled},flattened={event.positions_flattened}",
        )
        self.wal.append(entry)

    @property
    def events(self) -> list[KillSwitchEvent]:
        return list(self._events)
