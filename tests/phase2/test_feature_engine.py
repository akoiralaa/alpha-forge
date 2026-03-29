
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src" / "cpp" / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from engine import (
    CircularBufferDouble,
    FeatureEngine,
    FeatureEngineConfig,
    MarketStateVector,
    Tick,
    WelfordAccumulator,
)
from tests.phase2.conftest import make_tick

# ── CircularBuffer tests ─────────────────────────────────────

class TestCircularBuffer:
    def test_push_and_access(self):
        buf = CircularBufferDouble(5)
        buf.push(1.0)
        buf.push(2.0)
        buf.push(3.0)
        assert buf[0] == 3.0
        assert buf[2] == 1.0
        assert buf.size() == 3

    def test_wrap_around(self):
        buf = CircularBufferDouble(3)
        for i in range(5):
            buf.push(float(i))
        assert buf.full()
        assert buf[0] == 4.0
        assert buf[2] == 2.0

    def test_clear(self):
        buf = CircularBufferDouble(5)
        buf.push(1.0)
        buf.clear()
        assert buf.empty()

# ── WelfordAccumulator tests ─────────────────────────────────

class TestWelford:
    def test_running_stats(self):
        w = WelfordAccumulator(0)
        data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        for x in data:
            w.update(x)
        assert abs(w.mean() - 5.0) < 1e-6
        assert abs(w.variance() - 4.571428) < 1e-3

    def test_windowed(self):
        w = WelfordAccumulator(5)
        for i in range(1, 6):
            w.update(float(i))
        assert abs(w.mean() - 3.0) < 1e-6
        w.update(6.0)
        assert abs(w.mean() - 4.0) < 0.2

    def test_zscore(self):
        w = WelfordAccumulator(0)
        for i in range(1, 101):
            w.update(float(i))
        assert abs(w.zscore(50.5)) < 0.01

    def test_empty_returns_nan(self):
        w = WelfordAccumulator(0)
        assert math.isnan(w.mean())
        assert math.isnan(w.variance())

# ── FeatureEngine tests ──────────────────────────────────────

class TestFeatureEngine:
    def test_warmup(self, engine):
        for i in range(9):
            msv = engine.on_tick(make_tick(ts_ns=i * 10**9, price=100.0 + i * 0.1))
            assert not msv.valid
        msv = engine.on_tick(make_tick(ts_ns=9 * 10**9, price=100.9))
        assert msv.valid

    def test_returns_positive_for_rising(self, engine):
        for i in range(20):
            engine.on_tick(make_tick(ts_ns=i * 10**9, price=100.0 + i * 0.5))
        msv = engine.on_tick(make_tick(ts_ns=20 * 10**9, price=110.5))
        assert msv.ret_1s > 0.0
        assert msv.ret_10s > 0.0

    def test_returns_negative_for_falling(self, engine):
        for i in range(20):
            engine.on_tick(make_tick(ts_ns=i * 10**9, price=200.0 - i * 0.5))
        msv = engine.on_tick(make_tick(ts_ns=20 * 10**9, price=189.0))
        assert msv.ret_1s < 0.0

    def test_multi_symbol_isolation(self, engine):
        for i in range(15):
            engine.on_tick(make_tick(symbol_id=1, ts_ns=i * 10**9, price=100.0 + i))
            engine.on_tick(make_tick(symbol_id=2, ts_ns=i * 10**9, price=200.0 - i))

        msv1 = engine.on_tick(make_tick(symbol_id=1, ts_ns=15 * 10**9, price=116.0))
        msv2 = engine.on_tick(make_tick(symbol_id=2, ts_ns=15 * 10**9, price=184.0))
        assert msv1.ret_1s > 0
        assert msv2.ret_1s < 0
        assert engine.num_symbols() == 2

    def test_zscores_populated(self, engine):
        for i in range(50):
            engine.on_tick(make_tick(ts_ns=i * 10**9, price=100.0 + (i % 10) * 0.1))
        msv = engine.on_tick(make_tick(ts_ns=50 * 10**9, price=105.0))
        assert not math.isnan(msv.zscore_20)
        assert not math.isnan(msv.zscore_100)

    def test_spread_features(self, engine):
        for i in range(15):
            engine.on_tick(make_tick(ts_ns=i * 10**9, price=100.0, spread=0.10))
        msv = engine.on_tick(make_tick(ts_ns=15 * 10**9, price=100.0, spread=0.10))
        assert msv.spread_bps > 0.0
        assert not math.isnan(msv.ewma_spread_fast)
        assert not math.isnan(msv.ewma_spread_slow)

    def test_ofi(self, engine):
        t1 = make_tick(ts_ns=10**9, price=100.0)
        t1.bid_size = 1000
        t1.ask_size = 1000
        engine.on_tick(t1)

        t2 = make_tick(ts_ns=2 * 10**9, price=100.0)
        t2.bid_size = 2000
        t2.ask_size = 1000
        msv = engine.on_tick(t2)
        assert msv.ofi > 0.0

    def test_deterministic(self, default_config):
        e1 = FeatureEngine(default_config)
        e2 = FeatureEngine(default_config)
        for i in range(100):
            price = 100.0 + math.sin(i * 0.1) * 5.0
            tick = make_tick(ts_ns=i * 10**9, price=price, size=100 + i)
            m1 = e1.on_tick(tick)
            m2 = e2.on_tick(tick)
        assert m1.ret_1s == m2.ret_1s
        assert m1.zscore_20 == m2.zscore_20
        assert m1.vol_60s == m2.vol_60s
        assert m1.ofi == m2.ofi

    def test_reset(self, engine):
        for i in range(15):
            engine.on_tick(make_tick(ts_ns=i * 10**9))
        assert engine.is_warmed_up(1)
        engine.reset()
        assert engine.num_symbols() == 0
        assert not engine.is_warmed_up(1)

    def test_volume_ratio(self):
        cfg = FeatureEngineConfig()
        cfg.warmup_ticks = 3
        cfg.vol_window_1d = 20
        engine = FeatureEngine(cfg)

        for i in range(20):
            engine.on_tick(make_tick(ts_ns=i * 10**9, size=1000))
        msv = engine.on_tick(make_tick(ts_ns=20 * 10**9, size=3000))
        assert not math.isnan(msv.volume_ratio_20)
        assert msv.volume_ratio_20 > 2.0

    def test_msv_fields_accessible(self, engine):
        for i in range(15):
            engine.on_tick(make_tick(ts_ns=i * 10**9, price=100.0 + i * 0.1))
        msv = engine.on_tick(make_tick(ts_ns=15 * 10**9, price=101.5))

        # Universal fields
        assert hasattr(msv, 'ret_1s')
        assert hasattr(msv, 'zscore_20')
        assert hasattr(msv, 'vol_60s')
        assert hasattr(msv, 'spread_bps')
        assert hasattr(msv, 'ofi')
        assert hasattr(msv, 'vpin')
        # Asset-specific (should be NaN by default)
        assert math.isnan(msv.earnings_surprise_z)
        assert math.isnan(msv.term_structure_slope)
        assert math.isnan(msv.carry_differential)

    def test_ticks_processed_count(self, engine):
        for i in range(25):
            engine.on_tick(make_tick(ts_ns=i * 10**9))
        assert engine.ticks_processed(1) == 25
        assert engine.ticks_processed(999) == 0  # unknown symbol
