
from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
)

class TradingMetrics:

    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()

        # ── PnL & NAV ──────────────────────────────────────────
        self.nav = Gauge(
            "trading_nav_dollars", "Current NAV in dollars",
            registry=self.registry,
        )
        self.daily_pnl = Gauge(
            "trading_daily_pnl_dollars", "Daily realized + unrealized PnL",
            registry=self.registry,
        )
        self.drawdown_pct = Gauge(
            "trading_drawdown_pct", "Current drawdown from peak NAV",
            registry=self.registry,
        )
        self.peak_nav = Gauge(
            "trading_peak_nav_dollars", "Peak NAV watermark",
            registry=self.registry,
        )

        # ── Orders & Fills ──────────────────────────────────────
        self.orders_submitted = Counter(
            "trading_orders_submitted_total", "Total orders submitted",
            ["side", "order_type"],
            registry=self.registry,
        )
        self.orders_filled = Counter(
            "trading_orders_filled_total", "Total orders filled",
            ["side"],
            registry=self.registry,
        )
        self.orders_rejected = Counter(
            "trading_orders_rejected_total", "Total orders rejected",
            ["reason"],
            registry=self.registry,
        )
        self.fill_latency_ms = Histogram(
            "trading_fill_latency_ms", "Fill latency in milliseconds",
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
            registry=self.registry,
        )

        # ── Positions ──────────────────────────────────────────
        self.gross_exposure = Gauge(
            "trading_gross_exposure_dollars", "Gross dollar exposure",
            registry=self.registry,
        )
        self.net_exposure = Gauge(
            "trading_net_exposure_dollars", "Net dollar exposure",
            registry=self.registry,
        )
        self.position_count = Gauge(
            "trading_position_count", "Number of open positions",
            registry=self.registry,
        )

        # ── Risk ───────────────────────────────────────────────
        self.kill_switch_level = Gauge(
            "trading_kill_switch_level", "Current kill switch level (0-3)",
            registry=self.registry,
        )
        self.vpin = Gauge(
            "trading_vpin", "Current VPIN estimate",
            registry=self.registry,
        )
        self.risk_checks_failed = Counter(
            "trading_risk_checks_failed_total", "Risk check failures",
            ["reason"],
            registry=self.registry,
        )

        # ── Regime ─────────────────────────────────────────────
        self.regime_id = Gauge(
            "trading_regime_id", "Current regime ID",
            registry=self.registry,
        )
        self.regime_confidence = Gauge(
            "trading_regime_confidence", "Regime classification confidence",
            registry=self.registry,
        )
        self.regime_position_scale = Gauge(
            "trading_regime_position_scale", "Regime-based position scale factor",
            registry=self.registry,
        )

        # ── Data & System ──────────────────────────────────────
        self.ticks_processed = Counter(
            "trading_ticks_processed_total", "Total ticks processed",
            registry=self.registry,
        )
        self.tick_latency_us = Histogram(
            "trading_tick_latency_us", "Tick processing latency in microseconds",
            buckets=[10, 50, 100, 250, 500, 1000, 5000],
            registry=self.registry,
        )
        self.data_gaps = Counter(
            "trading_data_gaps_total", "Number of detected data gaps",
            registry=self.registry,
        )
        self.reconciliation_breaks = Counter(
            "trading_reconciliation_breaks_total", "Reconciliation breaks detected",
            registry=self.registry,
        )

        # ── Signals ────────────────────────────────────────────
        self.signal_value = Gauge(
            "trading_signal_value", "Current signal value",
            ["signal_name"],
            registry=self.registry,
        )
        self.strategy_weight = Gauge(
            "trading_strategy_weight", "Dynamic capital weight by strategy",
            ["strategy_name"],
            registry=self.registry,
        )
        self.strategy_realized_sharpe = Gauge(
            "trading_strategy_realized_sharpe", "Realized annualized Sharpe by strategy",
            ["strategy_name"],
            registry=self.registry,
        )
        self.strategy_expected_sharpe = Gauge(
            "trading_strategy_expected_sharpe", "Expected annualized Sharpe by strategy",
            ["strategy_name"],
            registry=self.registry,
        )
        self.strategy_performance_gap = Gauge(
            "trading_strategy_performance_gap", "Realized minus expected Sharpe by strategy",
            ["strategy_name"],
            registry=self.registry,
        )
        self.strategy_capacity_utilization = Gauge(
            "trading_strategy_capacity_utilization",
            "Capacity utilization by strategy (1.0 means at estimated limit)",
            ["strategy_name"],
            registry=self.registry,
        )
        self.strategy_avg_impact_bps = Gauge(
            "trading_strategy_avg_impact_bps", "Estimated average impact in bps by strategy",
            ["strategy_name"],
            registry=self.registry,
        )

        # ── HFT metrics ───────────────────────────────────────

        _ns_buckets = [100, 500, 1000, 5000, 10000, 50000, 100000]

        # Latency (nanosecond histograms)
        self.hft_tick_to_signal_latency_ns = Histogram(
            "hft_tick_to_signal_latency_ns",
            "Tick-to-signal latency in nanoseconds",
            ["symbol_id", "asset_class"],
            buckets=_ns_buckets,
            registry=self.registry,
        )
        self.hft_signal_to_order_latency_ns = Histogram(
            "hft_signal_to_order_latency_ns",
            "Signal-to-order latency in nanoseconds",
            ["symbol_id"],
            buckets=_ns_buckets,
            registry=self.registry,
        )
        self.hft_order_to_fill_latency_ns = Histogram(
            "hft_order_to_fill_latency_ns",
            "Order-to-fill latency in nanoseconds",
            ["symbol_id", "broker"],
            buckets=_ns_buckets,
            registry=self.registry,
        )

        # Throughput
        self.hft_ticks_processed_total = Counter(
            "hft_ticks_processed_total",
            "Total ticks processed per symbol",
            ["symbol_id"],
            registry=self.registry,
        )
        self.hft_orders_submitted_total = Counter(
            "hft_orders_submitted_total",
            "Total orders submitted per symbol",
            ["symbol_id", "side", "type"],
            registry=self.registry,
        )
        self.hft_fills_received_total = Counter(
            "hft_fills_received_total",
            "Total fills received per symbol",
            ["symbol_id", "side"],
            registry=self.registry,
        )

        # System health
        self.hft_ring_buffer_occupancy_pct = Gauge(
            "hft_ring_buffer_occupancy_pct",
            "Ring buffer occupancy percentage",
            ["buffer_name"],
            registry=self.registry,
        )
        self.hft_feature_staleness_ms = Gauge(
            "hft_feature_staleness_ms",
            "Feature staleness in milliseconds",
            ["symbol_id", "feature_name"],
            registry=self.registry,
        )
        self.hft_feed_connected = Gauge(
            "hft_feed_connected",
            "Feed connection status (0 or 1)",
            ["symbol_id"],
            registry=self.registry,
        )

        # Portfolio (HFT namespace)
        self.hft_portfolio_nav = Gauge(
            "hft_portfolio_nav", "Portfolio NAV",
            registry=self.registry,
        )
        self.hft_portfolio_pnl_realized = Gauge(
            "hft_portfolio_pnl_realized", "Realized PnL",
            registry=self.registry,
        )
        self.hft_portfolio_pnl_unrealized = Gauge(
            "hft_portfolio_pnl_unrealized", "Unrealized PnL",
            registry=self.registry,
        )
        self.hft_portfolio_drawdown_pct = Gauge(
            "hft_portfolio_drawdown_pct", "Portfolio drawdown percentage",
            registry=self.registry,
        )
        self.hft_position_size = Gauge(
            "hft_position_size", "Position size per symbol",
            ["symbol_id"],
            registry=self.registry,
        )
        self.hft_position_pnl = Gauge(
            "hft_position_pnl", "Position PnL per symbol",
            ["symbol_id"],
            registry=self.registry,
        )

        # Risk (HFT namespace)
        self.hft_vpin = Gauge(
            "hft_vpin", "VPIN per symbol",
            ["symbol_id"],
            registry=self.registry,
        )
        self.hft_regime_posterior = Gauge(
            "hft_regime_posterior", "Regime posterior probability",
            ["regime_name"],
            registry=self.registry,
        )
        self.hft_kill_switch_level = Gauge(
            "hft_kill_switch_level", "Kill switch level (0, 1, 2)",
            registry=self.registry,
        )
        self.hft_reconciliation_discrepancy = Gauge(
            "hft_reconciliation_discrepancy", "Reconciliation discrepancy per symbol",
            ["symbol_id"],
            registry=self.registry,
        )

        # Signal health
        self.hft_signal_sharpe_30d = Gauge(
            "hft_signal_sharpe_30d", "Signal 30-day rolling Sharpe",
            ["signal_name"],
            registry=self.registry,
        )
        self.hft_signal_sharpe_60d = Gauge(
            "hft_signal_sharpe_60d", "Signal 60-day rolling Sharpe",
            ["signal_name"],
            registry=self.registry,
        )
        self.hft_signal_weight = Gauge(
            "hft_signal_weight", "Signal combiner weight",
            ["signal_name"],
            registry=self.registry,
        )
        self.hft_feature_kl_divergence = Gauge(
            "hft_feature_kl_divergence", "Feature KL-divergence vs training distribution",
            ["feature_name"],
            registry=self.registry,
        )

    def snapshot(self) -> bytes:
        return generate_latest(self.registry)
