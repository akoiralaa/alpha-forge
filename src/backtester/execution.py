
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.backtester.types import (
    BookSnapshot,
    InstrumentSpec,
    OrderIntent,
    OrderType,
    Side,
    SimFill,
)

@dataclass
class LatencyModel:
    mu_ns: float = 17.0       # log-space mean (~50ms median)
    sigma_ns: float = 0.5     # log-space std
    min_ns: int = 1_000_000   # 1ms floor
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42))

    def sample(self) -> int:
        if self.mu_ns == 0.0 and self.sigma_ns == 0.0:
            return 0
        raw = self.rng.lognormal(self.mu_ns, self.sigma_ns)
        return max(self.min_ns, int(raw))

    @classmethod
    def from_p50_ns(cls, p50_ns: int, sigma: float = 0.5,
                    seed: int = 42) -> LatencyModel:
        if p50_ns <= 0:
            # Zero latency mode
            return cls(mu_ns=0.0, sigma_ns=0.0, min_ns=0,
                       rng=np.random.default_rng(seed))
        mu = math.log(p50_ns)
        return cls(mu_ns=mu, sigma_ns=sigma, min_ns=1_000_000,
                   rng=np.random.default_rng(seed))

    @property
    def p50(self) -> int:
        if self.mu_ns == 0 and self.sigma_ns == 0:
            return 0
        return int(math.exp(self.mu_ns))

def compute_market_impact_bps(
    order_size: int,
    adv_20d: float,
    realized_vol_daily_bps: float,
    k: float = 0.5,
) -> float:
    if adv_20d <= 0:
        return 0.0
    return k * math.sqrt(order_size / adv_20d) * realized_vol_daily_bps

def compute_transaction_cost_bps(
    fill: SimFill,
    order: OrderIntent,
    spec: InstrumentSpec,
    held_overnight: bool = False,
) -> float:
    commission = spec.commission_bps

    # Spread cost (half spread per side)
    mid = fill.fill_price  # approximate
    spread_cost = 0.0  # already embedded in fill price

    # Exchange fees
    if order.order_type == OrderType.LIMIT:
        exchange_fee = spec.maker_rebate_bps  # negative = rebate
    else:
        exchange_fee = spec.taker_fee_bps

    impact = fill.impact_bps

    # Overnight financing
    financing = 0.0
    if held_overnight and fill.fill_price > 0:
        notional = fill.fill_price * fill.fill_size
        financing = notional * spec.overnight_rate_annual / 252.0
        if notional > 0:
            financing = financing / notional * 10000.0  # convert to bps

    return commission + spread_cost + exchange_fee + impact + financing

class SimulatedExecution:

    def __init__(
        self,
        latency_model: Optional[LatencyModel] = None,
        impact_k: float = 0.5,
        adverse_selection_base: float = 0.8,
        costs_enabled: bool = True,
        seed: int = 42,
    ):
        self.latency = latency_model or LatencyModel()
        self.impact_k = impact_k
        self.adverse_selection_base = adverse_selection_base
        self.costs_enabled = costs_enabled
        self.rng = np.random.default_rng(seed)

    def simulate_fill(
        self,
        order: OrderIntent,
        book: BookSnapshot,
        spec: InstrumentSpec,
    ) -> Optional[SimFill]:
        # 1. Sample latency
        latency_ns = self.latency.sample()
        fill_time_ns = order.signal_time_ns + latency_ns

        # 2. Compute market impact
        impact_bps = compute_market_impact_bps(
            order.size, spec.adv_20d, spec.realized_vol_daily_bps, self.impact_k
        )

        mid = book.mid
        half_spread = book.spread / 2.0

        if order.order_type == OrderType.MARKET:
            # Walk the book — simplified: fill at mid ± half_spread ± impact ± latency slippage
            impact_price = impact_bps / 10000.0 * mid

            # Latency causes additional adverse price movement
            # The longer we wait, the more the price moves against us
            latency_slippage_bps = 0.0
            if latency_ns > 0:
                latency_ms = latency_ns / 1_000_000.0
                # Slippage scales with sqrt(latency) * volatility
                latency_slippage_bps = 0.1 * math.sqrt(latency_ms) * (
                    spec.realized_vol_daily_bps / 100.0)
            latency_slippage = latency_slippage_bps / 10000.0 * mid

            if order.side == Side.BUY:
                fill_price = mid + half_spread + impact_price + latency_slippage
            else:
                fill_price = mid - half_spread - impact_price - latency_slippage

            fill = SimFill(
                symbol_id=order.symbol_id,
                side=order.side,
                fill_price=fill_price,
                fill_size=order.size,
                fill_time_ns=fill_time_ns,
                impact_bps=impact_bps,
                latency_ns=latency_ns,
            )

        elif order.order_type == OrderType.LIMIT:
            # Check if limit price is available
            if order.side == Side.BUY:
                if book.ask > order.limit_price:
                    return None  # price not available
                # Adverse selection: less likely to fill when VPIN is high
                fill_prob = self.adverse_selection_base * (1.0 - 0.5 * min(book.vpin, 1.0))
                if self.rng.random() > fill_prob:
                    return None
                fill_price = order.limit_price
            else:
                if book.bid < order.limit_price:
                    return None
                fill_prob = self.adverse_selection_base * (1.0 - 0.5 * min(book.vpin, 1.0))
                if self.rng.random() > fill_prob:
                    return None
                fill_price = order.limit_price

            fill = SimFill(
                symbol_id=order.symbol_id,
                side=order.side,
                fill_price=fill_price,
                fill_size=order.size,
                fill_time_ns=fill_time_ns,
                impact_bps=impact_bps,
                latency_ns=latency_ns,
            )
        else:
            return None

        # 3. Compute transaction costs
        if self.costs_enabled:
            cost = compute_transaction_cost_bps(fill, order, spec)
            fill = SimFill(
                symbol_id=fill.symbol_id,
                side=fill.side,
                fill_price=fill.fill_price,
                fill_size=fill.fill_size,
                fill_time_ns=fill.fill_time_ns,
                cost_bps=cost,
                impact_bps=fill.impact_bps,
                latency_ns=fill.latency_ns,
            )

        return fill
