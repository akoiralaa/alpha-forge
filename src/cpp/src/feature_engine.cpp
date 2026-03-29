#include "onebrain/feature_engine.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace onebrain {

static constexpr int64_t NS_PER_SEC = 1'000'000'000LL;

FeatureEngine::FeatureEngine(const FeatureEngineConfig& cfg) : cfg_(cfg) {}

SymbolState& FeatureEngine::get_or_create_state(int32_t symbol_id) {
    auto it = states_.find(symbol_id);
    if (it == states_.end()) {
        auto [inserted, _] = states_.emplace(symbol_id, SymbolState(cfg_));
        inserted->second.symbol_id = symbol_id;
        return inserted->second;
    }
    return it->second;
}

MarketStateVector FeatureEngine::on_tick(const Tick& tick) {
    auto& st = get_or_create_state(tick.symbol_id);
    st.ticks_processed++;

    double mid = (tick.bid + tick.ask) / 2.0;
    if (tick.bid <= 0 || tick.ask <= 0) {
        mid = tick.last_price;
    }

    // Compute log return from previous mid
    double log_ret = 0.0;
    if (!st.mid_prices.empty()) {
        double prev_mid = st.mid_prices.newest();
        if (prev_mid > 0.0 && mid > 0.0) {
            log_ret = std::log(mid / prev_mid);
        }
    }

    // Spread in basis points
    double spread_bps = 0.0;
    if (mid > 0.0 && tick.ask > tick.bid) {
        spread_bps = (tick.ask - tick.bid) / mid * 10000.0;
    }

    // Push to buffers
    st.mid_prices.push(mid);
    st.timestamps_ns.push(tick.capture_time_ns);
    st.returns.push(log_ret);
    st.spreads.push(spread_bps);
    st.volumes.push(tick.last_size);

    // Update Welford accumulators with mid price
    st.welford_short.update(mid);
    st.welford_mid.update(mid);
    st.welford_long.update(mid);

    // Update EWMA spread
    if (std::isnan(st.ewma_spread_fast)) {
        st.ewma_spread_fast = spread_bps;
        st.ewma_spread_slow = spread_bps;
    } else {
        st.ewma_spread_fast = cfg_.ewma_alpha_fast * spread_bps +
                              (1.0 - cfg_.ewma_alpha_fast) * st.ewma_spread_fast;
        st.ewma_spread_slow = cfg_.ewma_alpha_slow * spread_bps +
                              (1.0 - cfg_.ewma_alpha_slow) * st.ewma_spread_slow;
    }

    // Update volume Welford
    st.volume_welford.update(static_cast<double>(tick.last_size));

    // OFI computation
    double ofi = compute_ofi(st, tick);
    st.prev_bid = tick.bid;
    st.prev_ask = tick.ask;
    st.prev_bid_size = tick.bid_size;
    st.prev_ask_size = tick.ask_size;

    // VPIN: classify trade as buy/sell using tick rule
    if (tick.last_size > 0) {
        bool is_buy = (tick.last_price >= mid);
        if (is_buy)
            st.vpin_bucket_buy += tick.last_size;
        else
            st.vpin_bucket_sell += tick.last_size;
        st.vpin_bucket_count += tick.last_size;

        if (st.vpin_bucket_count >= cfg_.vpin_bucket_size) {
            st.vpin_buy_vol.push(st.vpin_bucket_buy);
            st.vpin_sell_vol.push(st.vpin_bucket_sell);
            st.vpin_bucket_buy = 0.0;
            st.vpin_bucket_sell = 0.0;
            st.vpin_bucket_count = 0;
        }
    }

    // Day tracking
    if (st.day_open_ns == 0) {
        st.day_open_ns = tick.capture_time_ns;
        st.day_open_price = mid;
    }

    // Build MSV
    MarketStateVector msv;
    msv.symbol_id    = tick.symbol_id;
    msv.timestamp_ns = tick.capture_time_ns;
    msv.valid        = (st.ticks_processed >= cfg_.warmup_ticks);

    // Returns at various lookbacks (tick-based approximation of time intervals)
    msv.ret_1s    = compute_return(st, 1);
    msv.ret_10s   = compute_return(st, 10);
    msv.ret_60s   = compute_return(st, 60);
    msv.ret_300s  = compute_return(st, 300);
    msv.ret_1800s = compute_return(st, std::min(1800, static_cast<int>(st.mid_prices.size()) - 1));

    // Realized vol at various windows
    msv.vol_1s   = compute_realized_vol(st, 1);
    msv.vol_10s  = compute_realized_vol(st, 10);
    msv.vol_60s  = compute_realized_vol(st, 60);
    msv.vol_300s = compute_realized_vol(st, 300);
    msv.vol_1d   = compute_realized_vol(st, cfg_.vol_window_1d);

    // 1-day return from day open
    if (!std::isnan(st.day_open_price) && st.day_open_price > 0 && mid > 0) {
        msv.ret_1d = std::log(mid / st.day_open_price);
    }

    // Z-scores
    msv.zscore_20  = st.welford_short.zscore(mid);
    msv.zscore_100 = st.welford_mid.zscore(mid);
    msv.zscore_500 = st.welford_long.zscore(mid);

    // EWMA spreads
    msv.ewma_spread_fast = st.ewma_spread_fast;
    msv.ewma_spread_slow = st.ewma_spread_slow;

    // OFI
    msv.ofi = ofi;

    // Spread
    msv.spread_bps = spread_bps;

    // Volume ratio
    msv.volume_ratio_20 = compute_volume_ratio(st);

    // VPIN
    msv.vpin = compute_vpin(st);

    return msv;
}

std::vector<MarketStateVector> FeatureEngine::on_tick_batch(
        const std::vector<Tick>& ticks) {
    std::vector<MarketStateVector> results;
    results.reserve(ticks.size());
    for (const auto& tick : ticks) {
        results.push_back(on_tick(tick));
    }
    return results;
}

std::vector<MarketStateVector> FeatureEngine::on_tick_batch_raw(
        int n,
        const int32_t*  symbol_ids,
        const int64_t*  exchange_time_ns,
        const int64_t*  capture_time_ns,
        const double*   bids,
        const double*   asks,
        const int64_t*  bid_sizes,
        const int64_t*  ask_sizes,
        const double*   last_prices,
        const int64_t*  last_sizes) {
    std::vector<MarketStateVector> results;
    results.reserve(n);
    Tick tick;
    for (int i = 0; i < n; ++i) {
        tick.symbol_id        = symbol_ids[i];
        tick.exchange_time_ns = exchange_time_ns[i];
        tick.capture_time_ns  = capture_time_ns[i];
        tick.bid              = bids[i];
        tick.ask              = asks[i];
        tick.bid_size         = bid_sizes[i];
        tick.ask_size         = ask_sizes[i];
        tick.last_price       = last_prices[i];
        tick.last_size        = last_sizes[i];
        results.push_back(on_tick(tick));
    }
    return results;
}

void FeatureEngine::on_fill(const Fill& /*fill*/) {
    // Placeholder for execution-aware features in Phase 7
}

void FeatureEngine::on_cancel(const Cancel& /*cancel*/) {
    // Placeholder for execution-aware features in Phase 7
}

void FeatureEngine::reset() {
    states_.clear();
}

void FeatureEngine::reset_symbol(int32_t symbol_id) {
    states_.erase(symbol_id);
}

MarketStateVector FeatureEngine::get_state(int32_t symbol_id) const {
    MarketStateVector msv;
    msv.symbol_id = symbol_id;
    auto it = states_.find(symbol_id);
    if (it == states_.end()) return msv;

    const auto& st = it->second;
    msv.valid = (st.ticks_processed >= cfg_.warmup_ticks);
    if (!st.mid_prices.empty()) {
        double mid = st.mid_prices.newest();
        msv.zscore_20  = st.welford_short.zscore(mid);
        msv.zscore_100 = st.welford_mid.zscore(mid);
        msv.zscore_500 = st.welford_long.zscore(mid);
    }
    return msv;
}

bool FeatureEngine::is_warmed_up(int32_t symbol_id) const {
    auto it = states_.find(symbol_id);
    if (it == states_.end()) return false;
    return it->second.ticks_processed >= cfg_.warmup_ticks;
}

int64_t FeatureEngine::ticks_processed(int32_t symbol_id) const {
    auto it = states_.find(symbol_id);
    if (it == states_.end()) return 0;
    return it->second.ticks_processed;
}

double FeatureEngine::compute_return(const SymbolState& st, int lookback_ticks) const {
    if (static_cast<size_t>(lookback_ticks) >= st.mid_prices.size())
        return std::numeric_limits<double>::quiet_NaN();
    double current = st.mid_prices[0];
    double past    = st.mid_prices[lookback_ticks];
    if (past <= 0.0 || current <= 0.0)
        return std::numeric_limits<double>::quiet_NaN();
    return std::log(current / past);
}

double FeatureEngine::compute_realized_vol(const SymbolState& st, int window) const {
    int n = std::min(window, static_cast<int>(st.returns.size()));
    if (n < 2) return std::numeric_limits<double>::quiet_NaN();

    double sum = 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        double r = st.returns[i];
        sum += r;
        sum_sq += r * r;
    }
    double mean = sum / n;
    double var = (sum_sq / n) - (mean * mean);
    if (var < 0.0) var = 0.0;
    return std::sqrt(var);
}

double FeatureEngine::compute_ofi(const SymbolState& st, const Tick& tick) const {
    if (st.prev_bid <= 0.0) return 0.0;

    double delta_bid = 0.0;
    if (tick.bid > st.prev_bid)
        delta_bid = static_cast<double>(tick.bid_size);
    else if (tick.bid == st.prev_bid)
        delta_bid = static_cast<double>(tick.bid_size - st.prev_bid_size);
    else
        delta_bid = -static_cast<double>(st.prev_bid_size);

    double delta_ask = 0.0;
    if (tick.ask < st.prev_ask)
        delta_ask = static_cast<double>(tick.ask_size);
    else if (tick.ask == st.prev_ask)
        delta_ask = static_cast<double>(tick.ask_size - st.prev_ask_size);
    else
        delta_ask = -static_cast<double>(st.prev_ask_size);

    return delta_bid - delta_ask;  // positive = buy pressure
}

double FeatureEngine::compute_vpin(SymbolState& st) const {
    if (st.vpin_buy_vol.empty()) return std::numeric_limits<double>::quiet_NaN();

    size_t n = st.vpin_buy_vol.size();
    double total_imbalance = 0.0;
    double total_volume = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double buy  = st.vpin_buy_vol[i];
        double sell = st.vpin_sell_vol[i];
        total_imbalance += std::abs(buy - sell);
        total_volume += buy + sell;
    }
    if (total_volume < 1.0) return std::numeric_limits<double>::quiet_NaN();
    return total_imbalance / total_volume;
}

double FeatureEngine::compute_volume_ratio(const SymbolState& st) const {
    if (st.volumes.empty() || st.volumes.size() < 2)
        return std::numeric_limits<double>::quiet_NaN();

    double current = static_cast<double>(st.volumes[0]);
    double avg = st.volume_welford.mean();
    if (std::isnan(avg) || avg < 1.0)
        return std::numeric_limits<double>::quiet_NaN();
    return current / avg;
}

}  // namespace onebrain
