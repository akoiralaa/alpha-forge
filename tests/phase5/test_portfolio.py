"""Phase 5 tests — Portfolio, risk, sizing, HRP, attribution."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.portfolio.sizing import compute_position_size
from src.portfolio.hrp import hrp_weights
from src.portfolio.risk import (
    ClusterViolation,
    OrderIntent,
    Portfolio,
    PreTradeRiskCheck,
    RiskCheckReason,
    cluster_exposure_check,
)
from src.portfolio.attribution import TradeAttribution, attribute_trade


# ── Sizing ───────────────────────────────────────────────────

class TestSizing:
    def test_doubled_vol_halves_size(self):
        size_a = compute_position_size(1, 0.5, 1_000_000, 0.005, 100.0, 0.02)
        size_b = compute_position_size(1, 0.5, 1_000_000, 0.005, 100.0, 0.04)
        ratio = size_b / size_a if size_a != 0 else 0
        assert abs(ratio - 0.5) < 0.01

    def test_zero_signal_zero_size(self):
        assert compute_position_size(1, 0.0, 1_000_000, 0.005, 100.0, 0.02) == 0

    def test_negative_signal_negative_size(self):
        size = compute_position_size(1, -0.5, 1_000_000, 0.005, 100.0, 0.02)
        assert size < 0

    def test_kelly_constraint(self):
        # Small signal should be Kelly-constrained
        size_constrained = compute_position_size(
            1, 0.1, 1_000_000, 0.005, 100.0, 0.02, kelly_fraction=0.25)
        # Unconstrained vol-scaled size
        vol_scaled = int(1_000_000 * 0.005 / (100.0 * 0.02))
        assert abs(size_constrained) <= vol_scaled


# ── HRP ──────────────────────────────────────────────────────

class TestHRP:
    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(rng.standard_normal((100, 5)), columns=range(5))
        w = hrp_weights(returns)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_all_weights_positive(self):
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(rng.standard_normal((100, 5)), columns=range(5))
        w = hrp_weights(returns)
        assert (w >= 0).all()

    def test_correlated_cluster_gets_less_weight(self):
        rng = np.random.default_rng(42)
        n = 200
        # 5 correlated assets
        base = rng.standard_normal(n)
        corr_block = np.column_stack([base + rng.standard_normal(n) * 0.1 for _ in range(5)])
        # 5 independent assets
        indep_block = rng.standard_normal((n, 5))
        data = np.column_stack([corr_block, indep_block])
        returns = pd.DataFrame(data, columns=range(10))

        w = hrp_weights(returns)
        cluster_weight = w.iloc[:5].sum()
        assert cluster_weight < 0.55  # correlated group gets < 55%

    def test_single_asset(self):
        returns = pd.DataFrame(np.random.randn(50, 1), columns=[0])
        w = hrp_weights(returns)
        assert abs(w[0] - 1.0) < 1e-6


# ── Risk checks ──────────────────────────────────────────────

class TestPreTradeRisk:
    def test_pass_normal_order(self):
        check = PreTradeRiskCheck()
        port = Portfolio(nav=1_000_000)
        order = OrderIntent(symbol_id=1, side=1, size=100, current_price=100.0,
                            adv_20d=1_000_000)
        result = check.check(order, port)
        assert result.passed

    def test_fat_finger_size(self):
        check = PreTradeRiskCheck()
        port = Portfolio(nav=1_000_000)
        order = OrderIntent(symbol_id=1, side=1, size=110_000,
                            adv_20d=1_000_000, current_price=100.0)
        result = check.check(order, port)
        assert not result.passed
        assert result.reason == RiskCheckReason.FAT_FINGER_SIZE

    def test_fat_finger_price(self):
        check = PreTradeRiskCheck()
        port = Portfolio(nav=1_000_000)
        order = OrderIntent(symbol_id=1, side=1, size=100,
                            order_type="LIMIT", limit_price=106.0,
                            current_mid=100.0, current_price=100.0,
                            adv_20d=1_000_000)
        result = check.check(order, port)
        assert not result.passed
        assert result.reason == RiskCheckReason.FAT_FINGER_PRICE

    def test_vpin_blocks_market_allows_limit(self):
        check = PreTradeRiskCheck()
        port = Portfolio(nav=1_000_000, current_vpin=0.91)
        market_order = OrderIntent(symbol_id=1, side=1, size=100,
                                   order_type="MARKET", current_price=100.0,
                                   adv_20d=1_000_000)
        result = check.check(market_order, port)
        assert not result.passed
        assert result.reason == RiskCheckReason.VPIN_HALT

        limit_order = OrderIntent(symbol_id=1, side=1, size=100,
                                  order_type="LIMIT", limit_price=100.0,
                                  current_mid=100.0, current_price=100.0,
                                  adv_20d=1_000_000)
        result2 = check.check(limit_order, port)
        assert result2.passed

    def test_drawdown_level1(self):
        check = PreTradeRiskCheck()
        port = Portfolio(nav=900_000, peak_nav=1_000_000)  # 10% drawdown
        order = OrderIntent(symbol_id=1, side=1, size=100, current_price=100.0,
                            adv_20d=1_000_000)
        result = check.check(order, port)
        assert not result.passed
        assert result.reason == RiskCheckReason.DRAWDOWN_LEVEL1

    def test_drawdown_level2(self):
        check = PreTradeRiskCheck()
        port = Portfolio(nav=800_000, peak_nav=1_000_000)  # 20% drawdown
        order = OrderIntent(symbol_id=1, side=1, size=100, current_price=100.0,
                            adv_20d=1_000_000)
        result = check.check(order, port)
        assert not result.passed
        assert result.reason == RiskCheckReason.DRAWDOWN_LEVEL2

    def test_max_position(self):
        check = PreTradeRiskCheck(max_position_pct_nav=0.05)
        port = Portfolio(nav=1_000_000)
        # Order for 60K notional = 6% of NAV
        order = OrderIntent(symbol_id=1, side=1, size=600, current_price=100.0,
                            adv_20d=1_000_000)
        result = check.check(order, port)
        assert not result.passed
        assert result.reason == RiskCheckReason.MAX_POSITION


# ── Cluster exposure ─────────────────────────────────────────

class TestClusterExposure:
    def test_no_violation(self):
        positions = {1: 100_000, 2: 100_000}
        corr = pd.DataFrame([[1.0, 0.3], [0.3, 1.0]], index=[1, 2], columns=[1, 2])
        violations = cluster_exposure_check(positions, corr, nav=1_000_000)
        assert len(violations) == 0

    def test_correlated_violation(self):
        positions = {1: 200_000, 2: 200_000}
        corr = pd.DataFrame([[1.0, 0.8], [0.8, 1.0]], index=[1, 2], columns=[1, 2])
        violations = cluster_exposure_check(positions, corr, nav=1_000_000,
                                            cluster_threshold=0.7,
                                            max_cluster_exposure_pct=0.25)
        assert len(violations) > 0


# ── Attribution ──────────────────────────────────────────────

class TestAttribution:
    def test_profitable_long(self):
        attr = attribute_trade(
            trade_id=1, symbol_id=1, side=1, size=100,
            open_time_ns=0, close_time_ns=1000,
            entry_mid=100.0, exit_mid=105.0,
            entry_fill=100.01, exit_fill=104.99,
        )
        assert attr.alpha_pnl == 500.0  # (105-100)*100
        assert attr.total_pnl < attr.alpha_pnl  # costs reduce PnL

    def test_losing_short(self):
        attr = attribute_trade(
            trade_id=2, symbol_id=1, side=-1, size=100,
            open_time_ns=0, close_time_ns=1000,
            entry_mid=100.0, exit_mid=105.0,
            entry_fill=99.99, exit_fill=105.01,
        )
        assert attr.alpha_pnl == -500.0
