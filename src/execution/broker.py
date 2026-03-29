
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from src.execution.wal import OrderState, WALEntry, WriteAheadLog

@dataclass
class BrokerFill:
    order_id: str
    broker_order_id: str
    symbol_id: int
    side: int
    filled_size: int
    avg_price: float
    timestamp_ns: int
    commission: float = 0.0
    is_partial: bool = False

@dataclass
class BrokerOrder:
    order_id: str
    symbol_id: int
    side: int             # +1 buy, -1 sell
    size: int
    order_type: str = "MARKET"
    limit_price: Optional[float] = None
    time_in_force: str = "DAY"

class BrokerInterface(Protocol):

    def submit_order(self, order: BrokerOrder) -> str:
        ...

    def cancel_order(self, broker_order_id: str) -> bool:
        ...

    def get_positions(self) -> dict[int, float]:
        ...

    def get_cash(self) -> float:
        ...

    def is_connected(self) -> bool:
        ...

class PaperBroker:

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        fill_callback: Optional[Callable[[BrokerFill], None]] = None,
        slippage_bps: float = 1.0,
        commission_per_share: float = 0.005,
    ):
        self.cash = initial_cash
        self.positions: dict[int, float] = {}
        self.prices: dict[int, float] = {}       # symbol_id -> current price
        self._orders: dict[str, BrokerOrder] = {}
        self._fills: list[BrokerFill] = []
        self._fill_callback = fill_callback
        self.slippage_bps = slippage_bps
        self.commission_per_share = commission_per_share
        self._connected = True
        self._next_broker_id = 1

    def set_price(self, symbol_id: int, price: float):
        self.prices[symbol_id] = price

    def submit_order(self, order: BrokerOrder) -> str:
        broker_id = f"PAPER-{self._next_broker_id}"
        self._next_broker_id += 1
        self._orders[broker_id] = order

        price = self.prices.get(order.symbol_id, 100.0)

        if order.order_type == "LIMIT" and order.limit_price is not None:
            # Only fill if limit price is favorable
            if order.side == 1 and order.limit_price < price:
                return broker_id  # no fill, resting
            if order.side == -1 and order.limit_price > price:
                return broker_id  # no fill, resting
            price = order.limit_price

        # Apply slippage
        slip = price * self.slippage_bps / 10_000 * order.side
        fill_price = price + slip

        commission = self.commission_per_share * order.size

        # Update positions
        current = self.positions.get(order.symbol_id, 0)
        self.positions[order.symbol_id] = current + order.side * order.size

        # Update cash
        self.cash -= order.side * order.size * fill_price + commission

        fill = BrokerFill(
            order_id=order.order_id,
            broker_order_id=broker_id,
            symbol_id=order.symbol_id,
            side=order.side,
            filled_size=order.size,
            avg_price=fill_price,
            timestamp_ns=time.time_ns(),
            commission=commission,
        )
        self._fills.append(fill)

        if self._fill_callback:
            self._fill_callback(fill)

        return broker_id

    def cancel_order(self, broker_order_id: str) -> bool:
        if broker_order_id in self._orders:
            del self._orders[broker_order_id]
            return True
        return False

    def get_positions(self) -> dict[int, float]:
        return dict(self.positions)

    def get_cash(self) -> float:
        return self.cash

    def is_connected(self) -> bool:
        return self._connected

    @property
    def fills(self) -> list[BrokerFill]:
        return list(self._fills)
