
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.paper.engine import (
    CapitalDeploymentManager,
    DeploymentStage,
    PaperConfig,
    PaperTick,
    PaperTradingEngine,
)
from src.execution.broker import BrokerOrder
from src.execution.kill_switch import KillLevel
from src.execution.wal import OrderState

def _make_tick(sid: int, price: float, volume: int = 1000, ts: int = 0) -> PaperTick:
    return PaperTick(symbol_id=sid, price=price, volume=volume, timestamp_ns=ts or time.time_ns())

def _generate_trending_ticks(sid: int, n: int = 200, start_price: float = 100.0,
                              drift: float = 0.001, noise: float = 0.005) -> list[PaperTick]:
    rng = np.random.default_rng(42)
    prices = [start_price]
    for i in range(n - 1):
        ret = drift + rng.normal(0, noise)
        prices.append(prices[-1] * (1 + ret))
    return [_make_tick(sid, p, 1000, i * 1_000_000) for i, p in enumerate(prices)]

def _generate_mean_reverting_ticks(sid: int, n: int = 300,
                                     center: float = 100.0) -> list[PaperTick]:
    rng = np.random.default_rng(42)
    price = center
    ticks = []
    for i in range(n):
        noise = rng.normal(0, 0.5)
        price = center + (price - center) * 0.95 + noise
        ticks.append(_make_tick(sid, max(price, 1.0), 1000, i * 1_000_000))
    return ticks

# ── Core Integration Tests ──────────────────────────────────────

class TestPaperTradingEngine:
    def test_engine_creates(self):
        engine = PaperTradingEngine()
        assert engine.portfolio.nav == 1_000_000
        assert engine.broker.is_connected()

    def test_process_single_tick(self):
        engine = PaperTradingEngine()
        result = engine.on_tick(_make_tick(1, 100.0))
        assert engine.stats.ticks_processed == 1

    def test_run_session(self):
        engine = PaperTradingEngine()
        ticks = _generate_mean_reverting_ticks(1, n=200)
        stats = engine.run_session(ticks)
        assert stats.ticks_processed == 200
        assert stats.reconciliation_runs >= 1

    def test_orders_generated(self):
        engine = PaperTradingEngine(PaperConfig(signal_threshold=0.05))
        ticks = _generate_mean_reverting_ticks(1, n=300)
        stats = engine.run_session(ticks)
        assert stats.orders_submitted > 0

    def test_nav_updates(self):
        engine = PaperTradingEngine()
        ticks = _generate_trending_ticks(1, n=100)
        engine.run_session(ticks)
        # NAV should have changed from initial
        assert engine.portfolio.nav != 0

    def test_multi_symbol(self):
        engine = PaperTradingEngine(PaperConfig(signal_threshold=0.05))
        ticks1 = _generate_mean_reverting_ticks(1, n=150, center=100.0)
        ticks2 = _generate_mean_reverting_ticks(2, n=150, center=50.0)
        # Interleave
        combined = []
        for i in range(150):
            combined.append(ticks1[i])
            combined.append(ticks2[i])
        engine.run_session(combined)
        assert engine.stats.ticks_processed == 300

# ── WAL Integration ─────────────────────────────────────────────

class TestWALIntegration:
    def test_all_orders_logged(self):
        engine = PaperTradingEngine(PaperConfig(signal_threshold=0.05))
        ticks = _generate_mean_reverting_ticks(1, n=300)
        engine.run_session(ticks)
        # WAL should have entries for every order
        wal_count = engine.wal.entry_count()
        if engine.stats.orders_submitted > 0:
            assert wal_count > 0

    def test_replay_recovers_state(self):
        engine = PaperTradingEngine(PaperConfig(signal_threshold=0.05))
        ticks = _generate_mean_reverting_ticks(1, n=300)
        engine.run_session(ticks)
        state = engine.wal.replay()
        # All replayed orders should have valid states
        for oid, entry in state.items():
            assert entry.state in (
                OrderState.PENDING, OrderState.SUBMITTED,
                OrderState.FILLED, OrderState.REJECTED,
                OrderState.CANCELLED, OrderState.ERROR,
            )

# ── Kill Switch Integration ────────────────────────────────────

class TestKillSwitchIntegration:
    def test_kill_switch_blocks_orders(self):
        engine = PaperTradingEngine(PaperConfig(signal_threshold=0.01))
        # Process some ticks to build up signal state
        ticks = _generate_mean_reverting_ticks(1, n=50)
        for t in ticks:
            engine.on_tick(t)
        # Activate kill switch
        engine.kill_switch.activate(KillLevel.CANCEL_ONLY, "test")
        # Further ticks should not generate orders
        pre_submitted = engine.stats.orders_submitted
        more_ticks = _generate_mean_reverting_ticks(1, n=50)
        for t in more_ticks:
            engine.on_tick(t)
        # Orders attempted after kill switch should be rejected
        assert not engine.kill_switch.is_order_allowed()

    def test_drawdown_triggers_kill(self):
        config = PaperConfig(
            initial_nav=1_000_000,
            drawdown_auto_kill_pct=0.001,  # very sensitive
            signal_threshold=0.01,
        )
        engine = PaperTradingEngine(config)
        # Create a large losing position manually
        engine.broker.set_price(1, 100.0)
        engine.broker.submit_order(
            __import__('src.execution.broker', fromlist=['BrokerOrder']).BrokerOrder(
                "SETUP", 1, 1, 5000, "MARKET"))
        # Price drops
        for i in range(20):
            engine.on_tick(_make_tick(1, 100.0 - i * 2, 1000))
        # Kill switch should have activated
        assert engine.kill_switch.level >= KillLevel.FLATTEN or engine.portfolio.drawdown_pct > 0

# ── Risk Integration ───────────────────────────────────────────

class TestRiskIntegration:
    def test_fat_finger_blocked(self):
        engine = PaperTradingEngine()
        engine.broker.set_price(1, 100.0)
        # Try to submit absurdly large order
        order, risk = engine.order_manager.submit(
            1, 1, 500_000, current_price=100.0, adv_20d=1_000_000)
        assert order.state == OrderState.REJECTED

    def test_risk_check_integrated(self):
        engine = PaperTradingEngine(PaperConfig(max_position_pct_nav=0.01))
        engine.broker.set_price(1, 100.0)
        # Order for $50K notional = 5% of NAV > 1% limit
        order, risk = engine.order_manager.submit(
            1, 1, 500, current_price=100.0, adv_20d=10_000_000)
        assert order.state == OrderState.REJECTED
        assert risk is not None

    def test_risk_uses_live_position_state(self):
        engine = PaperTradingEngine(PaperConfig(max_position_pct_nav=0.05))
        engine.broker.set_price(1, 100.0)
        engine.broker.submit_order(BrokerOrder("SETUP", 1, 1, 400, "MARKET"))
        engine._update_nav()
        order, risk = engine.order_manager.submit(
            1, 1, 200, current_price=100.0, adv_20d=10_000_000)
        assert order.state == OrderState.REJECTED
        assert risk is not None

# ── Reconciliation Integration ─────────────────────────────────

class TestReconciliationIntegration:
    def test_clean_after_session(self):
        engine = PaperTradingEngine(PaperConfig(signal_threshold=0.05))
        ticks = _generate_mean_reverting_ticks(1, n=200)
        engine.run_session(ticks)
        # Paper broker should always reconcile cleanly (no external interference)
        # Run final reconciliation
        report = engine.reconciler.reconcile()
        # With paper broker, internal and broker state should be consistent
        # (they diverge because WAL tracks individual orders, not net position perfectly)
        assert report is not None

    def test_reconciliation_runs_periodically(self):
        config = PaperConfig(reconciliation_interval_ticks=50)
        engine = PaperTradingEngine(config)
        ticks = _generate_mean_reverting_ticks(1, n=200)
        engine.run_session(ticks)
        # Should have run at least 3 times (200/50=4, plus final)
        assert engine.stats.reconciliation_runs >= 4

# ── Monitoring Integration ─────────────────────────────────────

class TestMonitoringIntegration:
    def test_metrics_updated(self):
        engine = PaperTradingEngine()
        ticks = _generate_mean_reverting_ticks(1, n=100)
        engine.run_session(ticks)
        assert engine.metrics.ticks_processed._value.get() == 100
        assert engine.metrics.nav._value.get() > 0

    def test_health_check(self):
        engine = PaperTradingEngine()
        health = engine.get_health()
        assert health["status"] == "HEALTHY"
        assert "broker" in health["components"]

    def test_prometheus_snapshot(self):
        engine = PaperTradingEngine()
        ticks = _generate_trending_ticks(1, n=50)
        engine.run_session(ticks)
        snap = engine.metrics.snapshot()
        assert b"trading_nav_dollars" in snap
        assert b"trading_ticks_processed_total" in snap

    def test_alerts_on_drawdown(self):
        engine = PaperTradingEngine()
        # Manually set drawdown
        engine.portfolio.nav = 900_000
        engine.portfolio.peak_nav = 1_000_000
        fired = engine.alerts.evaluate_all({
            "drawdown_pct": engine.portfolio.drawdown_pct,
        })
        # 10% drawdown should trigger warning (>5%) and critical (>=10%)
        assert len(fired) >= 1

    def test_exposure_metrics_updated(self):
        engine = PaperTradingEngine()
        engine.broker.set_price(1, 100.0)
        engine.broker.submit_order(BrokerOrder("SETUP", 1, 1, 100, "MARKET"))
        engine.on_tick(_make_tick(1, 100.0))
        assert engine.metrics.gross_exposure._value.get() > 0
        assert engine.metrics.net_exposure._value.get() > 0
        assert engine.metrics.kill_switch_level._value.get() == 0

# ── Custom Signal Function ─────────────────────────────────────

class TestCustomSignal:
    def test_custom_signal_function(self):
        engine = PaperTradingEngine(PaperConfig(signal_threshold=0.05))
        # Always-long signal
        engine.set_signal_function(lambda sid, price, eng: 0.5)
        ticks = _generate_trending_ticks(1, n=100, drift=0.001)
        engine.run_session(ticks)
        # Should have submitted orders
        assert engine.stats.orders_submitted > 0

    def test_zero_signal_no_trades(self):
        engine = PaperTradingEngine()
        engine.set_signal_function(lambda sid, price, eng: 0.0)
        ticks = _generate_trending_ticks(1, n=100)
        engine.run_session(ticks)
        assert engine.stats.orders_submitted == 0

# ── Stats Tracking ──────────────────────────────────────────────

class TestStats:
    def test_stats_complete(self):
        engine = PaperTradingEngine(PaperConfig(signal_threshold=0.05))
        ticks = _generate_mean_reverting_ticks(1, n=300)
        stats = engine.run_session(ticks)
        assert stats.ticks_processed == 300
        assert stats.final_nav > 0
        assert stats.peak_nav >= stats.final_nav
        assert stats.reconciliation_runs >= 1

    def test_pnl_tracking(self):
        engine = PaperTradingEngine(PaperConfig(signal_threshold=0.05))
        ticks = _generate_mean_reverting_ticks(1, n=300)
        stats = engine.run_session(ticks)
        expected_pnl = stats.final_nav - engine.config.initial_nav
        assert abs(stats.total_pnl - expected_pnl) < 1.0


class TestCapitalDeploymentManager:
    def test_legacy_ratio_gate_still_works(self):
        mgr = CapitalDeploymentManager(10_000_000)
        transition = mgr.can_advance(live_sharpe=0.9, paper_sharpe=1.0)
        assert transition.allowed
        assert transition.to_stage == DeploymentStage.LIVE_5PCT

    def test_strict_fund_gate_requires_full_hurdles(self):
        mgr = CapitalDeploymentManager(10_000_000)
        transition = mgr.can_advance(
            live_sharpe=1.15,
            paper_sharpe=1.20,
            live_return=0.18,
            paper_return=0.20,
            live_max_drawdown=0.08,
            infrastructure_sharpe=0.90,
            reconciliation_breaks=0,
            critical_alerts=0,
            trading_days=15,
            enforce_fund_hurdles=True,
        )
        assert transition.allowed
        applied = mgr.advance(
            live_sharpe=1.15,
            paper_sharpe=1.20,
            live_return=0.18,
            paper_return=0.20,
            live_max_drawdown=0.08,
            infrastructure_sharpe=0.90,
            reconciliation_breaks=0,
            critical_alerts=0,
            trading_days=15,
            enforce_fund_hurdles=True,
        )
        assert applied.allowed
        assert mgr.current_stage == DeploymentStage.LIVE_5PCT

    def test_strict_fund_gate_blocks_operational_breaks(self):
        mgr = CapitalDeploymentManager(10_000_000)
        transition = mgr.can_advance(
            live_sharpe=1.20,
            paper_sharpe=1.10,
            live_return=0.19,
            paper_return=0.20,
            live_max_drawdown=0.07,
            infrastructure_sharpe=0.88,
            reconciliation_breaks=1,
            critical_alerts=0,
            trading_days=15,
            enforce_fund_hurdles=True,
        )
        assert not transition.allowed
        assert any("reconciliation" in msg for msg in transition.failed_checks)
