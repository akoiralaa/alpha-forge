from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.portfolio.allocator import CentralRiskAllocator, StrategyExpectation
from src.portfolio.capacity import LiquidityCapacityModel, LiquiditySnapshot


class TestLiquidityCapacityModel:
    def test_larger_orders_increase_impact_and_utilization(self):
        model = LiquidityCapacityModel()
        snapshot = LiquiditySnapshot(
            symbol_id=1,
            price=100.0,
            adv_usd=50_000_000.0,
            spread_bps=2.0,
            realized_vol_daily=0.02,
        )

        small = model.estimate_order(snapshot, 250_000.0)
        large = model.estimate_order(snapshot, 2_500_000.0)

        assert large.impact_bps > small.impact_bps
        assert large.utilization > small.utilization

    def test_strategy_capacity_limited_by_tightest_symbol(self):
        model = LiquidityCapacityModel()
        snapshots = {
            1: LiquiditySnapshot(1, 100.0, 100_000_000.0, 1.5, 0.015),
            2: LiquiditySnapshot(2, 50.0, 8_000_000.0, 6.0, 0.03),
        }
        estimate = model.estimate_strategy_capacity(
            "cross_asset",
            symbol_weights={1: 0.30, 2: 0.20},
            snapshots=snapshots,
            nav_usd=10_000_000.0,
            turnover=0.50,
        )

        assert estimate.limiting_symbol == 2
        assert estimate.utilization > 0


class TestCentralRiskAllocator:
    def test_outperformer_gets_more_weight(self):
        allocator = CentralRiskAllocator([
            StrategyExpectation("winner", expected_return_annual=0.12, expected_vol_annual=0.16, base_weight=1.0),
            StrategyExpectation("loser", expected_return_annual=0.12, expected_vol_annual=0.16, base_weight=1.0),
        ])

        for _ in range(80):
            allocator.observe("winner", 0.0025)
            allocator.observe("loser", -0.0015)

        weights = allocator.target_weights()
        assert weights["winner"] > weights["loser"]

    def test_capacity_pressure_reduces_weight(self):
        allocator = CentralRiskAllocator([
            StrategyExpectation("spacious", expected_return_annual=0.12, expected_vol_annual=0.16, base_weight=1.0),
            StrategyExpectation("crowded", expected_return_annual=0.12, expected_vol_annual=0.16, base_weight=1.0),
        ])

        for _ in range(80):
            allocator.observe("spacious", 0.0015)
            allocator.observe("crowded", 0.0015)

        allocator.observe_capacity("spacious", utilization=0.35, impact_bps=4.0)
        allocator.observe_capacity("crowded", utilization=1.25, impact_bps=18.0)

        weights = allocator.target_weights()
        assert weights["spacious"] > weights["crowded"]
