#pragma once

#include "circular_buffer.h"
#include "types.h"
#include "welford.h"

#include <cmath>
#include <unordered_map>
#include <vector>

namespace onebrain {

/// Per-symbol state maintained by the feature engine.
struct SymbolState {
    int32_t symbol_id = 0;

    // Price/return buffers (indexed by capture_time_ns)
    CircularBuffer<double>  mid_prices;      // mid-price history
    CircularBuffer<int64_t> timestamps_ns;   // capture_time_ns history
    CircularBuffer<double>  returns;         // log returns
    CircularBuffer<double>  spreads;         // bid-ask spread in bps
    CircularBuffer<int64_t> volumes;         // last_size per tick

    // Welford accumulators for z-scores at different windows
    WelfordAccumulator  welford_short;   // 20 ticks
    WelfordAccumulator  welford_mid;     // 100 ticks
    WelfordAccumulator  welford_long;    // 500 ticks

    // EWMA state
    double ewma_spread_fast = std::numeric_limits<double>::quiet_NaN();
    double ewma_spread_slow = std::numeric_limits<double>::quiet_NaN();

    // OFI (Order Flow Imbalance)
    double prev_bid       = 0.0;
    double prev_ask       = 0.0;
    int64_t prev_bid_size = 0;
    int64_t prev_ask_size = 0;

    // VPIN buckets
    CircularBuffer<double>  vpin_buy_vol;
    CircularBuffer<double>  vpin_sell_vol;
    double vpin_bucket_buy  = 0.0;
    double vpin_bucket_sell = 0.0;
    int    vpin_bucket_count = 0;

    // Volume tracking for volume ratio
    WelfordAccumulator volume_welford;

    // Day boundary tracking
    int64_t day_open_ns    = 0;
    double  day_open_price = std::numeric_limits<double>::quiet_NaN();

    int64_t ticks_processed = 0;

    explicit SymbolState(const FeatureEngineConfig& cfg)
        : mid_prices(cfg.zscore_window_long + 10)
        , timestamps_ns(cfg.zscore_window_long + 10)
        , returns(cfg.zscore_window_long + 10)
        , spreads(cfg.zscore_window_long + 10)
        , volumes(cfg.vol_window_1d)
        , welford_short(cfg.zscore_window_short)
        , welford_mid(cfg.zscore_window_mid)
        , welford_long(cfg.zscore_window_long)
        , vpin_buy_vol(cfg.vpin_window_buckets)
        , vpin_sell_vol(cfg.vpin_window_buckets)
        , volume_welford(cfg.vol_window_1d) {}
};

class FeatureEngine {
public:
    explicit FeatureEngine(const FeatureEngineConfig& cfg = {});

    /// Process one tick, update internal state, return Market State Vector.
    /// This is the SAME function used in backtest and live — zero logic drift.
    MarketStateVector on_tick(const Tick& tick);

    /// Batch process: takes vector of ticks, returns vector of MSVs.
    std::vector<MarketStateVector> on_tick_batch(const std::vector<Tick>& ticks);

    /// Batch process from raw arrays (no per-element Python→C++ conversion).
    /// Arrays must all be length n.
    std::vector<MarketStateVector> on_tick_batch_raw(
        int n,
        const int32_t*  symbol_ids,
        const int64_t*  exchange_time_ns,
        const int64_t*  capture_time_ns,
        const double*   bids,
        const double*   asks,
        const int64_t*  bid_sizes,
        const int64_t*  ask_sizes,
        const double*   last_prices,
        const int64_t*  last_sizes);

    /// Process a fill event (for execution features)
    void on_fill(const Fill& fill);

    /// Process a cancel event
    void on_cancel(const Cancel& cancel);

    /// Reset all state (e.g. start of new day)
    void reset();

    /// Reset state for a single symbol
    void reset_symbol(int32_t symbol_id);

    /// Get current MSV without processing a new tick
    MarketStateVector get_state(int32_t symbol_id) const;

    /// Check if symbol has enough data for valid features
    bool is_warmed_up(int32_t symbol_id) const;

    /// Get number of symbols tracked
    size_t num_symbols() const { return states_.size(); }

    /// Get ticks processed for a symbol
    int64_t ticks_processed(int32_t symbol_id) const;

    const FeatureEngineConfig& config() const { return cfg_; }

private:
    FeatureEngineConfig cfg_;
    std::unordered_map<int32_t, SymbolState> states_;

    SymbolState& get_or_create_state(int32_t symbol_id);

    // Feature computation helpers
    double compute_return(const SymbolState& st, int lookback_ticks) const;
    double compute_realized_vol(const SymbolState& st, int window) const;
    double compute_ofi(const SymbolState& st, const Tick& tick) const;
    double compute_vpin(SymbolState& st) const;
    double compute_volume_ratio(const SymbolState& st) const;
};

}  // namespace onebrain
