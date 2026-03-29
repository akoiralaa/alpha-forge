#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "onebrain/circular_buffer.h"
#include "onebrain/feature_engine.h"
#include "onebrain/types.h"
#include "onebrain/welford.h"

namespace py = pybind11;
using namespace onebrain;

PYBIND11_MODULE(_onebrain_cpp, m) {
    m.doc() = "One Brain C++20 Feature Engine";

    // ── Tick ───────────────────────────────────────────────────
    py::class_<Tick>(m, "Tick")
        .def(py::init<>())
        .def_readwrite("exchange_time_ns", &Tick::exchange_time_ns)
        .def_readwrite("capture_time_ns",  &Tick::capture_time_ns)
        .def_readwrite("symbol_id",        &Tick::symbol_id)
        .def_readwrite("bid",              &Tick::bid)
        .def_readwrite("ask",              &Tick::ask)
        .def_readwrite("bid_size",         &Tick::bid_size)
        .def_readwrite("ask_size",         &Tick::ask_size)
        .def_readwrite("last_price",       &Tick::last_price)
        .def_readwrite("last_size",        &Tick::last_size)
        .def_readwrite("trade_condition",  &Tick::trade_condition);

    // ── MarketStateVector ─────────────────────────────────────
    py::class_<MarketStateVector>(m, "MarketStateVector")
        .def(py::init<>())
        .def_readwrite("symbol_id",              &MarketStateVector::symbol_id)
        .def_readwrite("timestamp_ns",           &MarketStateVector::timestamp_ns)
        .def_readwrite("valid",                  &MarketStateVector::valid)
        .def_readwrite("ret_1s",                 &MarketStateVector::ret_1s)
        .def_readwrite("ret_10s",                &MarketStateVector::ret_10s)
        .def_readwrite("ret_60s",                &MarketStateVector::ret_60s)
        .def_readwrite("ret_300s",               &MarketStateVector::ret_300s)
        .def_readwrite("ret_1800s",              &MarketStateVector::ret_1800s)
        .def_readwrite("ret_1d",                 &MarketStateVector::ret_1d)
        .def_readwrite("vol_1s",                 &MarketStateVector::vol_1s)
        .def_readwrite("vol_10s",                &MarketStateVector::vol_10s)
        .def_readwrite("vol_60s",                &MarketStateVector::vol_60s)
        .def_readwrite("vol_300s",               &MarketStateVector::vol_300s)
        .def_readwrite("vol_1d",                 &MarketStateVector::vol_1d)
        .def_readwrite("zscore_20",              &MarketStateVector::zscore_20)
        .def_readwrite("zscore_100",             &MarketStateVector::zscore_100)
        .def_readwrite("zscore_500",             &MarketStateVector::zscore_500)
        .def_readwrite("ewma_spread_fast",       &MarketStateVector::ewma_spread_fast)
        .def_readwrite("ewma_spread_slow",       &MarketStateVector::ewma_spread_slow)
        .def_readwrite("ofi",                    &MarketStateVector::ofi)
        .def_readwrite("volume_ratio_20",        &MarketStateVector::volume_ratio_20)
        .def_readwrite("spread_bps",             &MarketStateVector::spread_bps)
        .def_readwrite("vpin",                   &MarketStateVector::vpin)
        .def_readwrite("residual_momentum",      &MarketStateVector::residual_momentum)
        .def_readwrite("earnings_surprise_z",    &MarketStateVector::earnings_surprise_z)
        .def_readwrite("sector_relative_str",    &MarketStateVector::sector_relative_str)
        .def_readwrite("short_interest_ratio",   &MarketStateVector::short_interest_ratio)
        .def_readwrite("analyst_revision_mom",   &MarketStateVector::analyst_revision_mom)
        .def_readwrite("term_structure_slope",   &MarketStateVector::term_structure_slope)
        .def_readwrite("roll_proximity_days",    &MarketStateVector::roll_proximity_days)
        .def_readwrite("open_interest_change_z", &MarketStateVector::open_interest_change_z)
        .def_readwrite("carry_differential",     &MarketStateVector::carry_differential)
        .def_readwrite("implied_vol_diff",       &MarketStateVector::implied_vol_diff)
        .def_readwrite("cross_rate_deviation",   &MarketStateVector::cross_rate_deviation)
        .def_readwrite("yield_curve_slope",      &MarketStateVector::yield_curve_slope)
        .def_readwrite("credit_spread_z",        &MarketStateVector::credit_spread_z)
        .def_readwrite("duration_adj_sensitivity", &MarketStateVector::duration_adj_sensitivity);

    // ── Fill / Cancel ─────────────────────────────────────────
    py::class_<Fill>(m, "Fill")
        .def(py::init<>())
        .def_readwrite("symbol_id",    &Fill::symbol_id)
        .def_readwrite("timestamp_ns", &Fill::timestamp_ns)
        .def_readwrite("fill_price",   &Fill::fill_price)
        .def_readwrite("fill_qty",     &Fill::fill_qty)
        .def_readwrite("is_buy",       &Fill::is_buy);

    py::class_<Cancel>(m, "Cancel")
        .def(py::init<>())
        .def_readwrite("symbol_id",    &Cancel::symbol_id)
        .def_readwrite("timestamp_ns", &Cancel::timestamp_ns)
        .def_readwrite("order_id",     &Cancel::order_id);

    // ── FeatureEngineConfig ───────────────────────────────────
    py::class_<FeatureEngineConfig>(m, "FeatureEngineConfig")
        .def(py::init<>())
        .def_readwrite("warmup_ticks",         &FeatureEngineConfig::warmup_ticks)
        .def_readwrite("zscore_window_short",  &FeatureEngineConfig::zscore_window_short)
        .def_readwrite("zscore_window_mid",    &FeatureEngineConfig::zscore_window_mid)
        .def_readwrite("zscore_window_long",   &FeatureEngineConfig::zscore_window_long)
        .def_readwrite("ewma_alpha_fast",      &FeatureEngineConfig::ewma_alpha_fast)
        .def_readwrite("ewma_alpha_slow",      &FeatureEngineConfig::ewma_alpha_slow)
        .def_readwrite("vpin_bucket_size",     &FeatureEngineConfig::vpin_bucket_size)
        .def_readwrite("vpin_window_buckets",  &FeatureEngineConfig::vpin_window_buckets)
        .def_readwrite("vol_window_1d",        &FeatureEngineConfig::vol_window_1d)
        .def_readwrite("benchmark_symbol_id",  &FeatureEngineConfig::benchmark_symbol_id);

    // ── CircularBuffer<double> ────────────────────────────────
    py::class_<CircularBuffer<double>>(m, "CircularBufferDouble")
        .def(py::init<size_t>())
        .def("push",     &CircularBuffer<double>::push)
        .def("__getitem__", &CircularBuffer<double>::operator[])
        .def("newest",   &CircularBuffer<double>::newest)
        .def("oldest",   &CircularBuffer<double>::oldest)
        .def("size",     &CircularBuffer<double>::size)
        .def("capacity", &CircularBuffer<double>::capacity)
        .def("full",     &CircularBuffer<double>::full)
        .def("empty",    &CircularBuffer<double>::empty)
        .def("clear",    &CircularBuffer<double>::clear);

    // ── WelfordAccumulator ────────────────────────────────────
    py::class_<WelfordAccumulator>(m, "WelfordAccumulator")
        .def(py::init<size_t>(), py::arg("window") = 0)
        .def("update",          &WelfordAccumulator::update)
        .def("mean",            &WelfordAccumulator::mean)
        .def("variance",        &WelfordAccumulator::variance)
        .def("stddev",          &WelfordAccumulator::stddev)
        .def("zscore",          &WelfordAccumulator::zscore)
        .def("effective_count", &WelfordAccumulator::effective_count)
        .def("window",          &WelfordAccumulator::window)
        .def("reset",           &WelfordAccumulator::reset);

    // ── FeatureEngine ─────────────────────────────────────────
    py::class_<FeatureEngine>(m, "FeatureEngine")
        .def(py::init<const FeatureEngineConfig&>(),
             py::arg("config") = FeatureEngineConfig())
        .def("on_tick",          &FeatureEngine::on_tick)
        .def("on_tick_batch",    &FeatureEngine::on_tick_batch)
        .def("on_tick_batch_numpy", [](FeatureEngine& self,
                py::array_t<int32_t>  symbol_ids,
                py::array_t<int64_t>  exchange_time_ns,
                py::array_t<int64_t>  capture_time_ns,
                py::array_t<double>   bids,
                py::array_t<double>   asks,
                py::array_t<int64_t>  bid_sizes,
                py::array_t<int64_t>  ask_sizes,
                py::array_t<double>   last_prices,
                py::array_t<int64_t>  last_sizes) {
            int n = static_cast<int>(symbol_ids.size());
            return self.on_tick_batch_raw(n,
                symbol_ids.data(),
                exchange_time_ns.data(),
                capture_time_ns.data(),
                bids.data(),
                asks.data(),
                bid_sizes.data(),
                ask_sizes.data(),
                last_prices.data(),
                last_sizes.data());
        })
        .def("on_tick_batch_numpy_last", [](FeatureEngine& self,
                py::array_t<int32_t>  symbol_ids,
                py::array_t<int64_t>  exchange_time_ns,
                py::array_t<int64_t>  capture_time_ns,
                py::array_t<double>   bids,
                py::array_t<double>   asks,
                py::array_t<int64_t>  bid_sizes,
                py::array_t<int64_t>  ask_sizes,
                py::array_t<double>   last_prices,
                py::array_t<int64_t>  last_sizes) -> MarketStateVector {
            // Process all ticks but only return the last MSV.
            // Avoids creating 1M Python objects for throughput benchmarking.
            int n = static_cast<int>(symbol_ids.size());
            auto* sym = symbol_ids.data();
            auto* ets = exchange_time_ns.data();
            auto* cts = capture_time_ns.data();
            auto* b   = bids.data();
            auto* a   = asks.data();
            auto* bs  = bid_sizes.data();
            auto* as  = ask_sizes.data();
            auto* lp  = last_prices.data();
            auto* ls  = last_sizes.data();
            MarketStateVector last;
            Tick tick;
            for (int i = 0; i < n; ++i) {
                tick.symbol_id        = sym[i];
                tick.exchange_time_ns = ets[i];
                tick.capture_time_ns  = cts[i];
                tick.bid              = b[i];
                tick.ask              = a[i];
                tick.bid_size         = bs[i];
                tick.ask_size         = as[i];
                tick.last_price       = lp[i];
                tick.last_size        = ls[i];
                last = self.on_tick(tick);
            }
            return last;
        })
        .def("on_fill",          &FeatureEngine::on_fill)
        .def("on_cancel",        &FeatureEngine::on_cancel)
        .def("reset",            &FeatureEngine::reset)
        .def("reset_symbol",     &FeatureEngine::reset_symbol)
        .def("get_state",        &FeatureEngine::get_state)
        .def("is_warmed_up",     &FeatureEngine::is_warmed_up)
        .def("num_symbols",      &FeatureEngine::num_symbols)
        .def("ticks_processed",  &FeatureEngine::ticks_processed)
        .def("config",           &FeatureEngine::config,
             py::return_value_policy::reference_internal);
}
