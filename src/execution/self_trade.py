"""Self-trade prevention.

Blocks orders that would match against our own resting orders.
"""

from __future__ import annotations

from src.execution.broker import BrokerOrder


def would_self_match(new_order: BrokerOrder, open_orders: list[BrokerOrder]) -> bool:
    """Check if a new order would match against any existing open order.

    Returns True if the new order should be rejected to prevent self-trade.
    """
    for existing in open_orders:
        if existing.symbol_id != new_order.symbol_id:
            continue
        if existing.side == new_order.side:
            continue  # same side: no self-match risk

        # Both must be limit orders with prices to check crossing
        if existing.limit_price is None or new_order.limit_price is None:
            # Market orders always cross — self-trade risk
            if new_order.order_type == "MARKET" or existing.order_type == "MARKET":
                return True
            continue

        would_match = (
            (new_order.side == 1 and new_order.limit_price >= existing.limit_price)
            or (new_order.side == -1 and new_order.limit_price <= existing.limit_price)
        )
        if would_match:
            return True

    return False
