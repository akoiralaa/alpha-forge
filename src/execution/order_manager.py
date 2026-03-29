
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from src.execution.broker import BrokerFill, BrokerInterface, BrokerOrder, PaperBroker
from src.execution.kill_switch import KillSwitch
from src.execution.wal import OrderState, WALEntry, WriteAheadLog
from src.portfolio.risk import OrderIntent, Portfolio, PreTradeRiskCheck, RiskCheckResult

@dataclass
class ManagedOrder:
    order_id: str
    symbol_id: int
    side: int
    size: int
    order_type: str
    limit_price: Optional[float]
    state: OrderState
    broker_order_id: Optional[str] = None
    filled_size: int = 0
    avg_fill_price: float = 0.0
    error_msg: str = ""
    created_ns: int = 0
    updated_ns: int = 0

class OrderManager:

    def __init__(
        self,
        broker: BrokerInterface | PaperBroker,
        wal: WriteAheadLog,
        kill_switch: KillSwitch,
        risk_check: Optional[PreTradeRiskCheck] = None,
        portfolio: Optional[Portfolio] = None,
    ):
        self.broker = broker
        self.wal = wal
        self.kill_switch = kill_switch
        self.risk_check = risk_check
        self.portfolio = portfolio
        self._orders: dict[str, ManagedOrder] = {}

    def submit(
        self,
        symbol_id: int,
        side: int,
        size: int,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        current_price: float = 100.0,
        adv_20d: float = 1_000_000.0,
    ) -> tuple[ManagedOrder, Optional[RiskCheckResult]]:
        order_id = f"ORD-{uuid.uuid4().hex[:12]}"
        now = time.time_ns()

        managed = ManagedOrder(
            order_id=order_id,
            symbol_id=symbol_id,
            side=side,
            size=size,
            order_type=order_type,
            limit_price=limit_price,
            state=OrderState.PENDING,
            created_ns=now,
            updated_ns=now,
        )

        # Kill switch check
        if not self.kill_switch.is_order_allowed():
            managed.state = OrderState.REJECTED
            managed.error_msg = "kill_switch_active"
            self._log_state(managed)
            self._orders[order_id] = managed
            return managed, None

        # Pre-trade risk check
        risk_result = None
        if self.risk_check and self.portfolio:
            intent = OrderIntent(
                symbol_id=symbol_id,
                side=side,
                size=size,
                order_type=order_type,
                limit_price=limit_price,
                current_price=current_price,
                adv_20d=adv_20d,
                current_mid=current_price,
            )
            risk_result = self.risk_check.check(intent, self.portfolio)
            if not risk_result.passed:
                managed.state = OrderState.REJECTED
                managed.error_msg = f"risk:{risk_result.reason.value}"
                self._log_state(managed)
                self._orders[order_id] = managed
                return managed, risk_result

        # WAL: log PENDING
        self._log_state(managed)

        # Submit to broker
        broker_order = BrokerOrder(
            order_id=order_id,
            symbol_id=symbol_id,
            side=side,
            size=size,
            order_type=order_type,
            limit_price=limit_price,
        )
        try:
            broker_id = self.broker.submit_order(broker_order)
            managed.broker_order_id = broker_id
            managed.state = OrderState.SUBMITTED
        except Exception as e:
            managed.state = OrderState.ERROR
            managed.error_msg = str(e)

        managed.updated_ns = time.time_ns()
        self._log_state(managed)
        self._orders[order_id] = managed

        return managed, risk_result

    def on_fill(self, fill: BrokerFill):
        order = self._orders.get(fill.order_id)
        if order is None:
            return

        order.filled_size += fill.filled_size
        order.avg_fill_price = fill.avg_price
        order.broker_order_id = fill.broker_order_id
        order.updated_ns = time.time_ns()

        if fill.is_partial:
            order.state = OrderState.PARTIALLY_FILLED
        else:
            order.state = OrderState.FILLED

        self._log_state(order)

    def cancel(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order is None or order.state in (
            OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED
        ):
            return False

        if order.broker_order_id:
            success = self.broker.cancel_order(order.broker_order_id)
            if not success:
                return False

        order.state = OrderState.CANCELLED
        order.updated_ns = time.time_ns()
        self._log_state(order)
        return True

    def get_order(self, order_id: str) -> Optional[ManagedOrder]:
        return self._orders.get(order_id)

    def get_open_orders(self) -> list[ManagedOrder]:
        return [
            o for o in self._orders.values()
            if o.state in (OrderState.PENDING, OrderState.SUBMITTED, OrderState.PARTIALLY_FILLED)
        ]

    def get_filled_orders(self) -> list[ManagedOrder]:
        return [o for o in self._orders.values() if o.state == OrderState.FILLED]

    def _log_state(self, order: ManagedOrder):
        entry = WALEntry(
            sequence_id=0,
            timestamp_ns=order.updated_ns or time.time_ns(),
            order_id=order.order_id,
            state=order.state,
            symbol_id=order.symbol_id,
            side=order.side,
            order_type=order.order_type,
            size=order.size,
            limit_price=order.limit_price,
            filled_size=order.filled_size,
            avg_fill_price=order.avg_fill_price,
            broker_order_id=order.broker_order_id,
            error_msg=order.error_msg,
        )
        self.wal.append(entry)
