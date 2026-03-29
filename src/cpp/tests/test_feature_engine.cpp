#include "onebrain/feature_engine.h"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace onebrain;

static bool is_finite(double x) {
    return !std::isnan(x) && !std::isinf(x);
}

static Tick make_tick(int32_t sym, int64_t ts_ns, double price, int64_t size = 100,
                      double spread = 0.02) {
    Tick t;
    t.symbol_id        = sym;
    t.exchange_time_ns = ts_ns;
    t.capture_time_ns  = ts_ns + 1000;
    t.bid              = price - spread / 2.0;
    t.ask              = price + spread / 2.0;
    t.bid_size         = 500;
    t.ask_size         = 500;
    t.last_price       = price;
    t.last_size        = size;
    return t;
}

void test_warmup_period() {
    FeatureEngineConfig cfg;
    cfg.warmup_ticks = 10;
    FeatureEngine engine(cfg);

    for (int i = 0; i < 9; i++) {
        auto msv = engine.on_tick(make_tick(1, i * 1000000000LL, 100.0 + i * 0.1));
        assert(!msv.valid);
    }
    // 10th tick should mark it valid
    auto msv = engine.on_tick(make_tick(1, 9000000000LL, 100.9));
    assert(msv.valid);
}

void test_returns_computed() {
    FeatureEngineConfig cfg;
    cfg.warmup_ticks = 5;
    FeatureEngine engine(cfg);

    // Push 20 ticks with increasing prices
    for (int i = 0; i < 20; i++) {
        engine.on_tick(make_tick(1, i * 1000000000LL, 100.0 + i * 0.5));
    }

    auto msv = engine.on_tick(make_tick(1, 20000000000LL, 110.5));
    assert(msv.valid);
    assert(is_finite(msv.ret_1s));
    assert(msv.ret_1s > 0.0);  // price went up
    assert(is_finite(msv.ret_10s));
    assert(msv.ret_10s > 0.0);
}

void test_zscores_populated() {
    FeatureEngineConfig cfg;
    cfg.warmup_ticks = 5;
    cfg.zscore_window_short = 10;
    FeatureEngine engine(cfg);

    // Push 50 ticks
    for (int i = 0; i < 50; i++) {
        engine.on_tick(make_tick(1, i * 1000000000LL, 100.0 + (i % 10) * 0.1));
    }

    auto msv = engine.on_tick(make_tick(1, 50000000000LL, 105.0));
    assert(is_finite(msv.zscore_20));
    assert(is_finite(msv.zscore_100));
    // zscore_500 may still be NaN if < 2 samples in 500-window (only 51 ticks)
}

void test_multi_symbol_isolation() {
    FeatureEngineConfig cfg;
    cfg.warmup_ticks = 5;
    FeatureEngine engine(cfg);

    // Feed sym 1 with rising prices
    for (int i = 0; i < 10; i++)
        engine.on_tick(make_tick(1, i * 1000000000LL, 100.0 + i));

    // Feed sym 2 with falling prices
    for (int i = 0; i < 10; i++)
        engine.on_tick(make_tick(2, i * 1000000000LL, 200.0 - i));

    assert(engine.num_symbols() == 2);
    assert(engine.ticks_processed(1) == 10);
    assert(engine.ticks_processed(2) == 10);

    auto msv1 = engine.on_tick(make_tick(1, 10000000000LL, 111.0));
    auto msv2 = engine.on_tick(make_tick(2, 10000000000LL, 189.0));

    assert(msv1.ret_1s > 0.0);  // rising
    assert(msv2.ret_1s < 0.0);  // falling
}

void test_spread_features() {
    FeatureEngineConfig cfg;
    cfg.warmup_ticks = 3;
    FeatureEngine engine(cfg);

    for (int i = 0; i < 10; i++) {
        auto msv = engine.on_tick(make_tick(1, i * 1000000000LL, 100.0, 100, 0.10));
    }

    auto msv = engine.on_tick(make_tick(1, 10000000000LL, 100.0, 100, 0.10));
    assert(msv.spread_bps > 0.0);
    assert(is_finite(msv.ewma_spread_fast));
    assert(is_finite(msv.ewma_spread_slow));
}

void test_ofi_direction() {
    FeatureEngineConfig cfg;
    cfg.warmup_ticks = 2;
    FeatureEngine engine(cfg);

    // First tick baseline
    Tick t1 = make_tick(1, 1000000000LL, 100.0);
    t1.bid_size = 1000; t1.ask_size = 1000;
    engine.on_tick(t1);

    // Second tick: bid size increases (buy pressure) — OFI should be positive
    Tick t2 = make_tick(1, 2000000000LL, 100.0);
    t2.bid_size = 2000; t2.ask_size = 1000;
    auto msv = engine.on_tick(t2);
    assert(msv.ofi > 0.0);
}

void test_reset() {
    FeatureEngineConfig cfg;
    cfg.warmup_ticks = 5;
    FeatureEngine engine(cfg);

    for (int i = 0; i < 10; i++)
        engine.on_tick(make_tick(1, i * 1000000000LL, 100.0));

    assert(engine.is_warmed_up(1));
    engine.reset();
    assert(engine.num_symbols() == 0);
    assert(!engine.is_warmed_up(1));
}

void test_reset_single_symbol() {
    FeatureEngineConfig cfg;
    cfg.warmup_ticks = 5;
    FeatureEngine engine(cfg);

    for (int i = 0; i < 10; i++) {
        engine.on_tick(make_tick(1, i * 1000000000LL, 100.0));
        engine.on_tick(make_tick(2, i * 1000000000LL, 200.0));
    }

    engine.reset_symbol(1);
    assert(!engine.is_warmed_up(1));
    assert(engine.is_warmed_up(2));
    assert(engine.num_symbols() == 1);
}

void test_same_binary_determinism() {
    // Same input sequence → same output. This is the zero-logic-drift check.
    FeatureEngineConfig cfg;
    cfg.warmup_ticks = 5;

    FeatureEngine engine1(cfg);
    FeatureEngine engine2(cfg);

    MarketStateVector last1, last2;
    for (int i = 0; i < 100; i++) {
        double price = 100.0 + std::sin(i * 0.1) * 5.0;
        Tick tick = make_tick(1, i * 1000000000LL, price, 100 + i);
        last1 = engine1.on_tick(tick);
        last2 = engine2.on_tick(tick);
    }

    // Every field must match exactly
    assert(last1.ret_1s == last2.ret_1s);
    assert(last1.ret_10s == last2.ret_10s);
    assert(last1.zscore_20 == last2.zscore_20);
    assert(last1.zscore_100 == last2.zscore_100);
    assert(last1.vol_60s == last2.vol_60s);
    assert(last1.ofi == last2.ofi);
    assert(last1.spread_bps == last2.spread_bps);
    assert(last1.ewma_spread_fast == last2.ewma_spread_fast);
}

void test_volume_ratio() {
    FeatureEngineConfig cfg;
    cfg.warmup_ticks = 3;
    cfg.vol_window_1d = 20;
    FeatureEngine engine(cfg);

    // Push 20 ticks with constant volume
    for (int i = 0; i < 20; i++)
        engine.on_tick(make_tick(1, i * 1000000000LL, 100.0, 1000));

    // Push one tick with 3x volume
    auto msv = engine.on_tick(make_tick(1, 20000000000LL, 100.0, 3000));
    assert(is_finite(msv.volume_ratio_20));
    assert(msv.volume_ratio_20 > 2.0);  // should be ~3x
}

int main() {
    test_warmup_period();
    test_returns_computed();
    test_zscores_populated();
    test_multi_symbol_isolation();
    test_spread_features();
    test_ofi_direction();
    test_reset();
    test_reset_single_symbol();
    test_same_binary_determinism();
    test_volume_ratio();

    std::cout << "All feature_engine tests passed." << std::endl;
    return 0;
}
