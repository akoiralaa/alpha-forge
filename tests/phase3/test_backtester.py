
from __future__ import annotations

import json
import math
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pytest

build_dir = Path(__file__).resolve().parent.parent.parent / "src" / "cpp" / "build"
sys.path.insert(0, str(build_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from engine import FeatureEngine, FeatureEngineConfig, Tick
from src.backtester.engine import Backtester
from src.backtester.execution import (
    LatencyModel,
    SimulatedExecution,
    compute_market_impact_bps,
    compute_transaction_cost_bps,
)
from src.backtester.types import (
    BacktestResult,
    BookSnapshot,
    InstrumentSpec,
    LockBox,
    OrderIntent,
    OrderType,
    Position,
    Side,
    SimFill,
    StressScenario,
)
from src.backtester.walk_forward import (
    apply_stress_scenario,
    run_stress_tests,
    verify_embargo,
    walk_forward,
)
from tests.phase3.conftest import generate_tick_series, make_tick

# ── Position tests ───────────────────────────────────────────

class TestPosition:
    def test_buy_updates_quantity(self):
        pos = Position(symbol_id=1)
        fill = SimFill(symbol_id=1, side=Side.BUY, fill_price=100.0,
                       fill_size=10, fill_time_ns=0)
        pos.apply_fill(fill)
        assert pos.quantity == 10
        assert pos.avg_price == 100.0

    def test_sell_closes_position(self):
        pos = Position(symbol_id=1, quantity=10, avg_price=100.0)
        fill = SimFill(symbol_id=1, side=Side.SELL, fill_price=105.0,
                       fill_size=10, fill_time_ns=0)
        pos.apply_fill(fill)
        assert pos.quantity == 0
        assert pos.realized_pnl == 50.0  # (105-100) * 10

    def test_partial_close(self):
        pos = Position(symbol_id=1, quantity=100, avg_price=50.0)
        fill = SimFill(symbol_id=1, side=Side.SELL, fill_price=55.0,
                       fill_size=40, fill_time_ns=0)
        pos.apply_fill(fill)
        assert pos.quantity == 60
        assert pos.realized_pnl == 200.0  # (55-50)*40

    def test_unrealized_pnl(self):
        pos = Position(symbol_id=1, quantity=10, avg_price=100.0)
        pos.update_mark(110.0)
        assert pos.unrealized_pnl == 100.0

# ── Execution tests ──────────────────────────────────────────

class TestMarketImpact:
    def test_impact_increases_with_size(self):
        small = compute_market_impact_bps(100, 1_000_000, 100.0)
        large = compute_market_impact_bps(10_000, 1_000_000, 100.0)
        assert large > small

    def test_zero_adv(self):
        assert compute_market_impact_bps(100, 0, 100.0) == 0.0

class TestLatencyModel:
    def test_p50_construction(self):
        lm = LatencyModel.from_p50_ns(50_000_000)  # 50ms
        assert abs(lm.p50 - 50_000_000) <= 1  # int rounding
        samples = [lm.sample() for _ in range(100)]
        assert all(s >= lm.min_ns for s in samples)

    def test_zero_latency(self):
        lm = LatencyModel.from_p50_ns(0)
        assert lm.sample() == 0

class TestSimulatedExecution:
    def test_market_order_fills(self):
        exec_ = SimulatedExecution(
            latency_model=LatencyModel.from_p50_ns(0),
            costs_enabled=False,
        )
        order = OrderIntent(symbol_id=1, side=Side.BUY,
                            order_type=OrderType.MARKET, size=100,
                            signal_time_ns=1000)
        book = BookSnapshot(symbol_id=1, timestamp_ns=1000,
                            bid=99.99, ask=100.01, bid_size=500,
                            ask_size=500, last_price=100.0)
        spec = InstrumentSpec(symbol_id=1)
        fill = exec_.simulate_fill(order, book, spec)
        assert fill is not None
        assert fill.side == Side.BUY
        assert fill.fill_price > book.mid  # buy fills above mid

    def test_limit_order_no_fill_when_price_away(self):
        exec_ = SimulatedExecution(
            latency_model=LatencyModel.from_p50_ns(0),
        )
        order = OrderIntent(symbol_id=1, side=Side.BUY,
                            order_type=OrderType.LIMIT, size=100,
                            limit_price=99.0, signal_time_ns=1000)
        book = BookSnapshot(symbol_id=1, timestamp_ns=1000,
                            bid=99.99, ask=100.01, bid_size=500,
                            ask_size=500, last_price=100.0)
        spec = InstrumentSpec(symbol_id=1)
        fill = exec_.simulate_fill(order, book, spec)
        assert fill is None  # ask > limit, no fill

    def test_costs_applied(self):
        exec_ = SimulatedExecution(
            latency_model=LatencyModel.from_p50_ns(0),
            costs_enabled=True,
        )
        order = OrderIntent(symbol_id=1, side=Side.BUY,
                            order_type=OrderType.MARKET, size=100,
                            signal_time_ns=1000)
        book = BookSnapshot(symbol_id=1, timestamp_ns=1000,
                            bid=99.99, ask=100.01, bid_size=500,
                            ask_size=500, last_price=100.0)
        spec = InstrumentSpec(symbol_id=1, commission_bps=1.0,
                              taker_fee_bps=0.3)
        fill = exec_.simulate_fill(order, book, spec)
        assert fill is not None
        assert fill.cost_bps > 0

# ── Backtester engine tests ──────────────────────────────────

class TestBacktester:
    def _make_engine(self, warmup=50):
        cfg = FeatureEngineConfig()
        cfg.warmup_ticks = warmup
        return FeatureEngine(cfg)

    def test_no_signal_no_trades(self):
        engine = self._make_engine()
        bt = Backtester(
            feature_engine=engine,
            signal_fn=lambda msv, pos: None,
        )
        ticks = generate_tick_series(n=200, seed=1)
        result = bt.run(ticks)
        assert result.n_trades == 0

    def test_always_buy_generates_trades(self):
        engine = self._make_engine(warmup=10)

        def buy_signal(msv, positions):
            pos = positions.get(msv.symbol_id)
            if pos is None or pos.quantity == 0:
                return OrderIntent(
                    symbol_id=msv.symbol_id,
                    side=Side.BUY,
                    order_type=OrderType.MARKET,
                    size=10,
                    signal_time_ns=msv.timestamp_ns,
                )
            return None

        bt = Backtester(
            feature_engine=engine,
            signal_fn=buy_signal,
            execution=SimulatedExecution(
                latency_model=LatencyModel.from_p50_ns(0),
                costs_enabled=False,
            ),
        )
        ticks = generate_tick_series(n=200, seed=1)
        result = bt.run(ticks)
        assert result.n_trades >= 1

    def test_pnl_series_length(self):
        engine = self._make_engine()
        bt = Backtester(engine, signal_fn=lambda m, p: None)
        ticks = generate_tick_series(n=100, seed=1)
        result = bt.run(ticks)
        assert len(result.pnl_series) == 100

    def test_backtest_result_metrics(self):
        result = BacktestResult(pnl_series=np.cumsum(np.random.randn(1000)))
        result.compute_metrics()
        assert isinstance(result.sharpe, float)
        assert result.max_drawdown <= 0

# ── Walk-forward tests ───────────────────────────────────────

class TestWalkForward:
    def test_embargo_verification(self):
        assert verify_embargo(30, 20)
        assert verify_embargo(30, 30)
        assert not verify_embargo(10, 30)

    def test_walk_forward_produces_results(self):
        results = walk_forward(
            run_backtest_fn=lambda s, e: BacktestResult(
                pnl_series=np.array([0.0, 1.0]),
                sharpe=0.5,
                n_trades=10,
            ),
            train_fn=lambda s, e: None,
            start_date=date(2020, 1, 1),
            end_date=date(2022, 1, 1),
            train_window_days=180,
            test_window_days=60,
            embargo_days=30,
            step_days=60,
        )
        assert len(results) > 0
        for r in results:
            # Embargo gap exists
            assert r.embargo_range[0] < r.embargo_range[1]
            assert r.test_range[0] >= r.embargo_range[1]

# ── Lock box tests ───────────────────────────────────────────

class TestLockBox:
    def test_save_and_load(self, tmp_path):
        lb = LockBox(
            start_date=date(2024, 1, 1),
            end_date=date(2026, 1, 1),
            data_hash="abc123",
        )
        path = tmp_path / "lockbox.json"
        lb.save(path)

        loaded = LockBox.load(path)
        assert loaded.start_date == date(2024, 1, 1)
        assert loaded.access_count == 0

    def test_access_increments(self, tmp_path):
        lb = LockBox(start_date=date(2024, 1, 1), end_date=date(2026, 1, 1))
        path = tmp_path / "lockbox.json"
        lb.save(path)

        lb.access()
        assert lb.access_count == 1
        loaded = LockBox.load(path)
        assert loaded.access_count == 1

# ── Stress test scenarios ────────────────────────────────────

class TestStressScenarios:
    def test_liquidity_blackout_widens_spreads(self):
        ticks = generate_tick_series(n=500, seed=1)
        original_spreads = [t.ask - t.bid for t in ticks]
        stressed = apply_stress_scenario(ticks, StressScenario.LIQUIDITY_BLACKOUT)
        stressed_spreads = [t.ask - t.bid for t in stressed]
        # Some spreads should be wider
        assert max(stressed_spreads) > max(original_spreads)

    def test_gap_down_applies(self):
        ticks = generate_tick_series(n=100, seed=1)
        original_first = ticks[0].last_price
        stressed = apply_stress_scenario(ticks, StressScenario.GAP_DOWN_10PCT)
        assert stressed[0].last_price < original_first * 0.95

    def test_feed_outage_removes_ticks(self):
        ticks = generate_tick_series(n=500, seed=1)
        stressed = apply_stress_scenario(ticks, StressScenario.FEED_OUTAGE_5S)
        assert len(stressed) < len(ticks)

    def test_run_stress_tests_all_scenarios(self):
        cfg = FeatureEngineConfig()
        cfg.warmup_ticks = 10
        engine = FeatureEngine(cfg)
        bt = Backtester(
            feature_engine=engine,
            signal_fn=lambda m, p: None,  # no trades = no drawdown
            execution=SimulatedExecution(
                latency_model=LatencyModel.from_p50_ns(0),
                costs_enabled=False,
            ),
        )
        ticks = generate_tick_series(n=200, seed=1)
        results = run_stress_tests(bt, ticks, max_drawdown_pct=0.5)
        assert len(results) == 4  # 4 synthetic scenarios
        for name, (passed, dd) in results.items():
            assert isinstance(passed, bool)
