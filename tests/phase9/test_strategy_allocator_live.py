from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.paper.engine import PaperConfig, PaperTradingEngine
from src.portfolio.allocator import CentralRiskAllocator, StrategyExpectation
from src.portfolio.capacity import LiquidityCapacityModel
from tests.phase9.test_paper_trading import _generate_trending_ticks


class TestStrategyAllocatorLive:
    def test_live_allocator_promotes_winner_and_updates_metrics(self):
        engine = PaperTradingEngine(
            PaperConfig(
                signal_threshold=0.05,
                risk_budget_per_position=0.0005,
                kelly_fraction=0.05,
                max_position_pct_nav=0.10,
            )
        )
        allocator = CentralRiskAllocator([
            StrategyExpectation("trend", expected_return_annual=0.12, expected_vol_annual=0.16, base_weight=1.0),
            StrategyExpectation("fade", expected_return_annual=0.08, expected_vol_annual=0.16, base_weight=1.0),
        ])

        def trend(symbol_id, price, eng):
            rets = eng.returns.get(symbol_id, [])
            if len(rets) < 5:
                return 0.0
            return 1.0 if np.mean(rets[-5:]) > 0 else -1.0

        def fade(symbol_id, price, eng):
            rets = eng.returns.get(symbol_id, [])
            if len(rets) < 5:
                return 0.0
            return -0.4 if np.mean(rets[-5:]) > 0 else 0.4

        engine.set_strategy_functions(
            {"trend": trend, "fade": fade},
            allocator=allocator,
            capacity_model=LiquidityCapacityModel(),
        )

        ticks = _generate_trending_ticks(1, n=220, drift=0.0012, noise=0.002)
        stats = engine.run_session(ticks)
        weights = allocator.target_weights()

        assert stats.orders_submitted > 0
        assert weights["trend"] > weights["fade"]
        assert engine.metrics.strategy_weight.labels(strategy_name="trend")._value.get() > (
            engine.metrics.strategy_weight.labels(strategy_name="fade")._value.get()
        )
        assert engine.metrics.strategy_capacity_utilization.labels(strategy_name="trend")._value.get() >= 0
