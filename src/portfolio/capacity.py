from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.backtester.execution import compute_market_impact_bps


@dataclass
class LiquiditySnapshot:
    symbol_id: int
    price: float
    adv_usd: float
    spread_bps: float
    realized_vol_daily: float
    participation_limit: float = 0.05
    impact_limit_bps: float = 15.0
    max_spread_bps: float = 12.0


@dataclass
class OrderCapacityEstimate:
    symbol_id: int
    order_notional_usd: float
    max_order_notional_usd: float
    impact_bps: float
    utilization: float
    remaining_capacity_usd: float
    can_trade: bool
    limiting_factor: str


@dataclass
class StrategyCapacityEstimate:
    strategy_name: str
    nav_capacity_usd: float
    utilization: float
    limiting_symbol: int | None
    weighted_impact_bps: float
    weighted_participation: float


class LiquidityCapacityModel:

    def __init__(
        self,
        impact_k: float = 0.5,
        min_adv_usd: float = 1_000_000.0,
    ):
        self.impact_k = impact_k
        self.min_adv_usd = min_adv_usd

    @staticmethod
    def _finite_or(value: float, default: float) -> float:
        if value is None:
            return default
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        return value if np.isfinite(value) else default

    @classmethod
    def _shares_from_notional(cls, order_notional_usd: float, price: float) -> float:
        order_notional_usd = cls._finite_or(order_notional_usd, 0.0)
        price = cls._finite_or(price, 0.0)
        if price <= 0:
            return 0.0
        return max(order_notional_usd / price, 0.0)

    @classmethod
    def _adv_shares(cls, snapshot: LiquiditySnapshot) -> float:
        price = cls._finite_or(snapshot.price, 0.0)
        adv_usd = cls._finite_or(snapshot.adv_usd, 0.0)
        if price <= 0:
            return 0.0
        return max(adv_usd, 0.0) / price

    @classmethod
    def _rv_bps(cls, snapshot: LiquiditySnapshot) -> float:
        realized_vol_daily = cls._finite_or(snapshot.realized_vol_daily, 1e-6)
        return max(realized_vol_daily, 1e-6) * 10_000.0

    def impact_bps(
        self,
        order_notional_usd: float,
        snapshot: LiquiditySnapshot,
    ) -> float:
        adv_shares = max(self._adv_shares(snapshot), 1e-12)
        order_shares = self._shares_from_notional(order_notional_usd, snapshot.price)
        return compute_market_impact_bps(
            int(max(order_shares, 0.0)),
            adv_shares,
            self._rv_bps(snapshot),
            self.impact_k,
        )

    def max_order_notional_usd(self, snapshot: LiquiditySnapshot) -> tuple[float, str]:
        adv_usd = max(self._finite_or(snapshot.adv_usd, self.min_adv_usd), self.min_adv_usd)
        participation_limit = max(self._finite_or(snapshot.participation_limit, 0.05), 1e-4)
        impact_limit_bps = self._finite_or(snapshot.impact_limit_bps, 15.0)
        spread_bps = self._finite_or(snapshot.spread_bps, 0.0)
        max_spread_bps = max(self._finite_or(snapshot.max_spread_bps, 12.0), 1e-6)
        by_participation = adv_usd * participation_limit

        rv_bps = self._rv_bps(snapshot)
        if rv_bps <= 1e-9 or impact_limit_bps <= 0:
            by_impact = by_participation
        else:
            ratio = impact_limit_bps / max(self.impact_k * rv_bps, 1e-12)
            by_impact = adv_usd * ratio * ratio

        spread_penalty = 1.0
        if spread_bps > max_spread_bps:
            spread_penalty = max_spread_bps / max(spread_bps, 1e-12)

        capped = min(by_participation, by_impact) * max(spread_penalty, 0.0)
        if capped == by_participation * max(spread_penalty, 0.0):
            return capped, "participation"
        return capped, "impact"

    def estimate_order(
        self,
        snapshot: LiquiditySnapshot,
        order_notional_usd: float,
    ) -> OrderCapacityEstimate:
        order_notional_usd = max(self._finite_or(order_notional_usd, 0.0), 0.0)
        max_notional, limiting = self.max_order_notional_usd(snapshot)
        impact = self.impact_bps(order_notional_usd, snapshot)
        utilization = (
            order_notional_usd / max(max_notional, 1e-12)
            if max_notional > 0 else float("inf")
        )
        remaining = max(max_notional - order_notional_usd, 0.0)
        can_trade = max_notional > 0 and utilization <= 1.0
        return OrderCapacityEstimate(
            symbol_id=snapshot.symbol_id,
            order_notional_usd=order_notional_usd,
            max_order_notional_usd=max_notional,
            impact_bps=impact,
            utilization=utilization,
            remaining_capacity_usd=remaining,
            can_trade=can_trade,
            limiting_factor=limiting,
        )

    def estimate_strategy_capacity(
        self,
        strategy_name: str,
        symbol_weights: Mapping[int, float],
        snapshots: Mapping[int, LiquiditySnapshot],
        nav_usd: float,
        turnover: float = 1.0,
    ) -> StrategyCapacityEstimate:
        if not symbol_weights:
            return StrategyCapacityEstimate(
                strategy_name=strategy_name,
                nav_capacity_usd=float("inf"),
                utilization=0.0,
                limiting_symbol=None,
                weighted_impact_bps=0.0,
                weighted_participation=0.0,
            )

        abs_weights = {
            sid: abs(weight)
            for sid, weight in symbol_weights.items()
            if abs(weight) > 1e-12 and sid in snapshots
        }
        if not abs_weights:
            return StrategyCapacityEstimate(
                strategy_name=strategy_name,
                nav_capacity_usd=float("inf"),
                utilization=0.0,
                limiting_symbol=None,
                weighted_impact_bps=0.0,
                weighted_participation=0.0,
            )

        total_weight = sum(abs_weights.values())
        limiting_symbol = None
        capacity_nav = float("inf")
        weighted_impact = 0.0
        weighted_participation = 0.0

        for sid, abs_weight in abs_weights.items():
            snap = snapshots[sid]
            flow_weight = abs_weight * max(turnover, 1e-6)
            if flow_weight <= 1e-12:
                continue

            order_notional = nav_usd * flow_weight
            estimate = self.estimate_order(snap, order_notional)
            max_nav_here = estimate.max_order_notional_usd / flow_weight
            if max_nav_here < capacity_nav:
                capacity_nav = max_nav_here
                limiting_symbol = sid

            weighted_impact += (abs_weight / total_weight) * estimate.impact_bps
            weighted_participation += (
                abs_weight / total_weight
            ) * min(order_notional / max(self._finite_or(snap.adv_usd, 0.0), 1e-12), 10.0)

        utilization = (
            nav_usd / max(capacity_nav, 1e-12)
            if np.isfinite(capacity_nav) and capacity_nav > 0 else 0.0
        )
        return StrategyCapacityEstimate(
            strategy_name=strategy_name,
            nav_capacity_usd=capacity_nav,
            utilization=utilization,
            limiting_symbol=limiting_symbol,
            weighted_impact_bps=weighted_impact,
            weighted_participation=weighted_participation,
        )
