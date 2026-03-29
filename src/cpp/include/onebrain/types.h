#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

namespace onebrain {

struct Tick {
    int64_t  exchange_time_ns = 0;
    int64_t  capture_time_ns  = 0;
    int32_t  symbol_id        = 0;
    double   bid              = 0.0;
    double   ask              = 0.0;
    int64_t  bid_size         = 0;
    int64_t  ask_size         = 0;
    double   last_price       = 0.0;
    int64_t  last_size        = 0;
    uint8_t  trade_condition  = 0;
};

struct MarketStateVector {
    int32_t  symbol_id     = 0;
    int64_t  timestamp_ns  = 0;
    bool     valid         = false;

    // Universal features — all assets
    double   ret_1s            = std::numeric_limits<double>::quiet_NaN();
    double   ret_10s           = std::numeric_limits<double>::quiet_NaN();
    double   ret_60s           = std::numeric_limits<double>::quiet_NaN();
    double   ret_300s          = std::numeric_limits<double>::quiet_NaN();
    double   ret_1800s         = std::numeric_limits<double>::quiet_NaN();
    double   ret_1d            = std::numeric_limits<double>::quiet_NaN();
    double   vol_1s            = std::numeric_limits<double>::quiet_NaN();
    double   vol_10s           = std::numeric_limits<double>::quiet_NaN();
    double   vol_60s           = std::numeric_limits<double>::quiet_NaN();
    double   vol_300s          = std::numeric_limits<double>::quiet_NaN();
    double   vol_1d            = std::numeric_limits<double>::quiet_NaN();
    double   zscore_20         = std::numeric_limits<double>::quiet_NaN();
    double   zscore_100        = std::numeric_limits<double>::quiet_NaN();
    double   zscore_500        = std::numeric_limits<double>::quiet_NaN();
    double   ewma_spread_fast  = std::numeric_limits<double>::quiet_NaN();
    double   ewma_spread_slow  = std::numeric_limits<double>::quiet_NaN();
    double   ofi               = std::numeric_limits<double>::quiet_NaN();
    double   volume_ratio_20   = std::numeric_limits<double>::quiet_NaN();
    double   spread_bps        = std::numeric_limits<double>::quiet_NaN();
    double   vpin              = std::numeric_limits<double>::quiet_NaN();
    double   residual_momentum = std::numeric_limits<double>::quiet_NaN();

    // Asset-specific — NaN when not applicable
    double   earnings_surprise_z      = std::numeric_limits<double>::quiet_NaN();
    double   sector_relative_str      = std::numeric_limits<double>::quiet_NaN();
    double   short_interest_ratio     = std::numeric_limits<double>::quiet_NaN();
    double   analyst_revision_mom     = std::numeric_limits<double>::quiet_NaN();
    double   term_structure_slope     = std::numeric_limits<double>::quiet_NaN();
    double   roll_proximity_days      = std::numeric_limits<double>::quiet_NaN();
    double   open_interest_change_z   = std::numeric_limits<double>::quiet_NaN();
    double   carry_differential       = std::numeric_limits<double>::quiet_NaN();
    double   implied_vol_diff         = std::numeric_limits<double>::quiet_NaN();
    double   cross_rate_deviation     = std::numeric_limits<double>::quiet_NaN();
    double   yield_curve_slope        = std::numeric_limits<double>::quiet_NaN();
    double   credit_spread_z          = std::numeric_limits<double>::quiet_NaN();
    double   duration_adj_sensitivity = std::numeric_limits<double>::quiet_NaN();
};

struct Fill {
    int32_t  symbol_id    = 0;
    int64_t  timestamp_ns = 0;
    double   fill_price   = 0.0;
    int64_t  fill_qty     = 0;
    bool     is_buy       = true;
};

struct Cancel {
    int32_t  symbol_id    = 0;
    int64_t  timestamp_ns = 0;
    int64_t  order_id     = 0;
};

struct FeatureEngineConfig {
    int     warmup_ticks         = 500;
    int     zscore_window_short  = 20;
    int     zscore_window_mid    = 100;
    int     zscore_window_long   = 500;
    double  ewma_alpha_fast      = 0.10;
    double  ewma_alpha_slow      = 0.02;
    int     vpin_bucket_size     = 1000;
    int     vpin_window_buckets  = 50;
    int     vol_window_1d        = 390;
    int     benchmark_symbol_id  = -1;  // symbol_id for benchmark (e.g. SPY)
};

}  // namespace onebrain
