"""Pre-trade risk checks, circuit breakers, and correlation cluster monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class RiskCheckReason(Enum):
    PASS = "PASS"
    MAX_POSITION = "MAX_POSITION"
    MAX_SECTOR_EXPOSURE = "MAX_SECTOR_EXPOSURE"
    MAX_GROSS_LEVERAGE = "MAX_GROSS_LEVERAGE"
    MAX_NET_EXPOSURE = "MAX_NET_EXPOSURE"
    INTRADAY_STOP = "INTRADAY_STOP"
    DRAWDOWN_LEVEL1 = "DRAWDOWN_LEVEL1"
    DRAWDOWN_LEVEL2 = "DRAWDOWN_LEVEL2"
    VPIN_HALT = "VPIN_HALT"
    FAT_FINGER_SIZE = "FAT_FINGER_SIZE"
    FAT_FINGER_PRICE = "FAT_FINGER_PRICE"
    STALE_DATA = "STALE_DATA"


@dataclass
class RiskCheckResult:
    passed: bool
    reason: RiskCheckReason = RiskCheckReason.PASS
    detail: str = ""


@dataclass
class Portfolio:
    """Portfolio state for risk checks."""
    nav: float = 1_000_000.0
    peak_nav: float = 1_000_000.0
    positions: Dict[int, float] = field(default_factory=dict)  # symbol_id -> dollar exposure
    sectors: Dict[int, str] = field(default_factory=dict)       # symbol_id -> sector
    current_vpin: float = 0.0
    daily_pnl_pct: float = 0.0
    halt_level: int = 0  # 0=normal, 1=level1, 2=level2(read-only)

    @property
    def gross_exposure(self) -> float:
        return sum(abs(v) for v in self.positions.values())

    @property
    def net_exposure(self) -> float:
        return sum(self.positions.values())

    @property
    def drawdown_pct(self) -> float:
        if self.peak_nav <= 0:
            return 0.0
        return (self.peak_nav - self.nav) / self.peak_nav

    def sector_net_exposure(self, sector: str) -> float:
        total = 0.0
        for sid, exp in self.positions.items():
            if self.sectors.get(sid) == sector:
                total += exp
        return total

    def update_nav(self, new_nav: float):
        self.nav = new_nav
        self.peak_nav = max(self.peak_nav, new_nav)


@dataclass
class OrderIntent:
    """Simplified order intent for risk checks."""
    symbol_id: int
    side: int  # +1 buy, -1 sell
    size: int
    order_type: str = "MARKET"  # MARKET or LIMIT
    limit_price: Optional[float] = None
    current_mid: float = 100.0
    adv_20d: float = 1_000_000.0
    current_price: float = 100.0


class PreTradeRiskCheck:
    """All pre-trade risk limits. Any FAIL blocks the order."""

    def __init__(
        self,
        max_position_pct_nav: float = 0.05,
        max_sector_net_exposure: float = 0.20,
        max_gross_leverage: float = 3.0,
        max_net_exposure: float = 1.0,
        intraday_stop_loss_pct: float = 0.015,
        drawdown_level1_pct: float = 0.10,
        drawdown_level2_pct: float = 0.20,
        vpin_passive_only_pct: float = 0.90,
        fat_finger_adv_pct: float = 0.10,
        fat_finger_price_pct: float = 0.05,
        stale_data_ms: float = 500.0,
    ):
        self.max_position_pct_nav = max_position_pct_nav
        self.max_sector_net_exposure = max_sector_net_exposure
        self.max_gross_leverage = max_gross_leverage
        self.max_net_exposure = max_net_exposure
        self.intraday_stop_loss_pct = intraday_stop_loss_pct
        self.drawdown_level1_pct = drawdown_level1_pct
        self.drawdown_level2_pct = drawdown_level2_pct
        self.vpin_passive_only_pct = vpin_passive_only_pct
        self.fat_finger_adv_pct = fat_finger_adv_pct
        self.fat_finger_price_pct = fat_finger_price_pct
        self.stale_data_ms = stale_data_ms

    def check(self, order: OrderIntent, portfolio: Portfolio) -> RiskCheckResult:
        """Run all pre-trade checks. Returns first failure or PASS."""

        # Drawdown circuit breaker — Level 2 blocks everything
        if portfolio.drawdown_pct >= self.drawdown_level2_pct:
            portfolio.halt_level = 2
            return RiskCheckResult(False, RiskCheckReason.DRAWDOWN_LEVEL2,
                                   f"drawdown={portfolio.drawdown_pct:.1%}")

        # Drawdown Level 1 — halt new entries, allow reduces
        if portfolio.drawdown_pct >= self.drawdown_level1_pct:
            portfolio.halt_level = 1
            current_pos = portfolio.positions.get(order.symbol_id, 0)
            is_reducing = (
                (order.side == -1 and current_pos > 0) or
                (order.side == 1 and current_pos < 0)
            )
            if not is_reducing:
                return RiskCheckResult(False, RiskCheckReason.DRAWDOWN_LEVEL1,
                                       f"drawdown={portfolio.drawdown_pct:.1%}")

        # VPIN halt — only passive orders allowed
        if portfolio.current_vpin >= self.vpin_passive_only_pct:
            if order.order_type == "MARKET":
                return RiskCheckResult(False, RiskCheckReason.VPIN_HALT,
                                       f"vpin={portfolio.current_vpin:.2f}")

        # Fat finger: size vs ADV
        if order.adv_20d > 0:
            size_pct = order.size / order.adv_20d
            if size_pct > self.fat_finger_adv_pct:
                return RiskCheckResult(False, RiskCheckReason.FAT_FINGER_SIZE,
                                       f"size={order.size}, adv={order.adv_20d}, "
                                       f"pct={size_pct:.2%}")

        # Fat finger: price vs mid
        if order.limit_price is not None and order.current_mid > 0:
            price_diff = abs(order.limit_price - order.current_mid) / order.current_mid
            if price_diff > self.fat_finger_price_pct:
                return RiskCheckResult(False, RiskCheckReason.FAT_FINGER_PRICE,
                                       f"limit={order.limit_price}, mid={order.current_mid}, "
                                       f"diff={price_diff:.2%}")

        # Max single position
        order_notional = order.size * order.current_price
        new_pos = portfolio.positions.get(order.symbol_id, 0) + order.side * order_notional
        if abs(new_pos) / portfolio.nav > self.max_position_pct_nav:
            return RiskCheckResult(False, RiskCheckReason.MAX_POSITION,
                                   f"position_pct={abs(new_pos)/portfolio.nav:.2%}")

        # Max gross leverage
        new_gross = portfolio.gross_exposure + order_notional
        if new_gross / portfolio.nav > self.max_gross_leverage:
            return RiskCheckResult(False, RiskCheckReason.MAX_GROSS_LEVERAGE,
                                   f"gross={new_gross/portfolio.nav:.2f}x")

        # Max net exposure
        new_net = portfolio.net_exposure + order.side * order_notional
        if abs(new_net) / portfolio.nav > self.max_net_exposure:
            return RiskCheckResult(False, RiskCheckReason.MAX_NET_EXPOSURE,
                                   f"net={abs(new_net)/portfolio.nav:.2f}x")

        # Intraday stop
        if abs(portfolio.daily_pnl_pct) > self.intraday_stop_loss_pct:
            return RiskCheckResult(False, RiskCheckReason.INTRADAY_STOP,
                                   f"daily_pnl={portfolio.daily_pnl_pct:.2%}")

        return RiskCheckResult(True)


# ── Correlation cluster monitor ──────────────────────────────

@dataclass
class ClusterViolation:
    members: list[int]
    current_exposure: float
    max_allowed: float
    excess: float


def cluster_exposure_check(
    positions: Dict[int, float],
    corr_matrix: pd.DataFrame,
    nav: float,
    cluster_threshold: float = 0.70,
    max_cluster_exposure_pct: float = 0.25,
) -> List[ClusterViolation]:
    """Find clusters of correlated assets exceeding exposure limits."""
    symbols = list(positions.keys())
    if len(symbols) < 2:
        return []

    # Build adjacency: corr > threshold
    violations = []
    visited = set()

    for i, s1 in enumerate(symbols):
        if s1 in visited:
            continue
        cluster = [s1]
        visited.add(s1)
        for j, s2 in enumerate(symbols):
            if s2 in visited or s1 == s2:
                continue
            if s1 in corr_matrix.index and s2 in corr_matrix.index:
                corr = corr_matrix.loc[s1, s2]
                if abs(corr) > cluster_threshold:
                    cluster.append(s2)
                    visited.add(s2)

        if len(cluster) > 1:
            exposure = sum(abs(positions.get(s, 0)) for s in cluster)
            max_allowed = nav * max_cluster_exposure_pct
            if exposure > max_allowed:
                violations.append(ClusterViolation(
                    members=cluster,
                    current_exposure=exposure,
                    max_allowed=max_allowed,
                    excess=exposure - max_allowed,
                ))

    return violations
