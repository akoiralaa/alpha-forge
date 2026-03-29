"""P&L attribution: decompose trade PnL into alpha, costs, and slippage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradeAttribution:
    trade_id: int
    symbol_id: int
    open_time_ns: int
    close_time_ns: int
    side: int           # +1 long, -1 short
    size: int
    alpha_pnl: float    # PnL from signal being directionally correct
    impact_cost: float
    timing_slippage: float
    spread_cost: float
    commission: float
    exchange_fee: float

    @property
    def total_pnl(self) -> float:
        return self.alpha_pnl - self.impact_cost - self.timing_slippage - self.spread_cost - self.commission - self.exchange_fee


def attribute_trade(
    trade_id: int,
    symbol_id: int,
    side: int,
    size: int,
    open_time_ns: int,
    close_time_ns: int,
    entry_mid: float,
    exit_mid: float,
    entry_fill: float,
    exit_fill: float,
    commission_per_share: float = 0.005,
    exchange_fee_per_share: float = 0.003,
) -> TradeAttribution:
    """Decompose a round-trip trade into alpha and cost components."""
    # Alpha: mid-to-mid return * side * size
    alpha_pnl = side * (exit_mid - entry_mid) * size

    # Impact: difference between fill and mid
    entry_impact = abs(entry_fill - entry_mid) * size
    exit_impact = abs(exit_fill - exit_mid) * size
    impact_cost = entry_impact + exit_impact

    # Timing slippage: any additional adverse movement
    # (simplified: lumped into impact)
    timing_slippage = 0.0

    # Spread: half spread at entry + half spread at exit (estimated)
    spread_cost = 0.0  # already in impact

    commission = commission_per_share * size * 2  # round trip
    exchange_fee = exchange_fee_per_share * size * 2

    return TradeAttribution(
        trade_id=trade_id,
        symbol_id=symbol_id,
        open_time_ns=open_time_ns,
        close_time_ns=close_time_ns,
        side=side,
        size=size,
        alpha_pnl=alpha_pnl,
        impact_cost=impact_cost,
        timing_slippage=timing_slippage,
        spread_cost=spread_cost,
        commission=commission,
        exchange_fee=exchange_fee,
    )
