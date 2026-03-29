"""Phase 8 validation gate — Monitoring: metrics, health, alerting."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

results: list[tuple[str, bool, str]] = []


def gate(name: str, passed: bool, detail: str = ""):
    tag = "PASS" if passed else "FAIL"
    results.append((name, passed, detail))
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))


# ── 1. Prometheus metrics exposition ───────────────────────────

from src.monitoring.metrics import TradingMetrics

m = TradingMetrics()
m.nav.set(1_000_000)
m.drawdown_pct.set(0.05)
m.orders_submitted.labels(side="buy", order_type="MARKET").inc(10)
m.ticks_processed.inc(50_000)
m.fill_latency_ms.observe(42)

snap = m.snapshot()
has_nav = b"trading_nav_dollars" in snap
has_dd = b"trading_drawdown_pct 0.05" in snap
has_ticks = b"trading_ticks_processed_total 50000.0" in snap

gate("prometheus_exposition",
     has_nav and has_dd and has_ticks and len(snap) > 500,
     f"snapshot={len(snap)} bytes, nav={has_nav}, dd={has_dd}, ticks={has_ticks}")


# ── 2. Metric isolation ───────────────────────────────────────

m1 = TradingMetrics()
m2 = TradingMetrics()
m1.nav.set(100)
m2.nav.set(999)
isolated = m1.nav._value.get() == 100 and m2.nav._value.get() == 999

gate("metric_isolation",
     isolated,
     f"m1.nav={m1.nav._value.get()}, m2.nav={m2.nav._value.get()}")


# ── 3. Health check aggregation ───────────────────────────────

from src.monitoring.health import HealthChecker, HealthStatus

hc = HealthChecker()
hc.register("broker", lambda: (HealthStatus.HEALTHY, "connected"))
hc.register("data", lambda: (HealthStatus.HEALTHY, "streaming"))
hc.register("engine", lambda: (HealthStatus.HEALTHY, "running"))
healthy = hc.check()
all_healthy = healthy.is_healthy

hc2 = HealthChecker()
hc2.register("broker", lambda: (HealthStatus.HEALTHY, "ok"))
hc2.register("data", lambda: (HealthStatus.UNHEALTHY, "stale 30s"))
unhealthy = hc2.check()
detects_failure = not unhealthy.is_healthy and len(unhealthy.unhealthy_components) == 1

gate("health_check_aggregation",
     all_healthy and detects_failure,
     f"3_healthy={all_healthy}, detects_failure={detects_failure}")


# ── 4. Alert threshold firing ─────────────────────────────────

from src.monitoring.alerting import AlertManager, AlertRule, AlertSeverity

am = AlertManager()
am.add_rule(AlertRule("drawdown_pct", AlertSeverity.WARNING, 0.05, "gt"))
am.add_rule(AlertRule("drawdown_pct", AlertSeverity.CRITICAL, 0.10, "gt"))

no_fire = am.evaluate("drawdown_pct", 0.03)
warn_fire = am.evaluate("drawdown_pct", 0.07)
# Reset cooldown for critical test
for r in am._rules:
    r._last_fired_ns = 0
crit_fire = am.evaluate("drawdown_pct", 0.12)

gate("alert_threshold_firing",
     len(no_fire) == 0 and len(warn_fire) == 1 and len(crit_fire) == 2,
     f"below=0, warn={len(warn_fire)}, crit={len(crit_fire)}")


# ── 5. Alert cooldown ────────────────────────────────────────

am2 = AlertManager()
am2.add_rule(AlertRule("x", AlertSeverity.WARNING, 0, "gt",
                       cooldown_ns=10_000_000_000))  # 10s
first = am2.evaluate("x", 1)
second = am2.evaluate("x", 2)  # suppressed by cooldown

gate("alert_cooldown",
     len(first) == 1 and len(second) == 0,
     f"first={len(first)}, suppressed={len(second)}")


# ── 6. Alert handler dispatch ────────────────────────────────

received = []
am3 = AlertManager()
am3.add_rule(AlertRule("metric", AlertSeverity.CRITICAL, 100, "gt"))
am3.add_handler(lambda a: received.append(a))
am3.evaluate("metric", 150)
handler_ok = len(received) == 1 and received[0].severity == AlertSeverity.CRITICAL

gate("alert_handler_dispatch",
     handler_ok,
     f"handler_received={len(received)}, severity={received[0].severity.name if received else 'N/A'}")


# ── 7. HFT metrics all exist ─────────────────────────────────

m_hft = TradingMetrics()
# Set values for all hft_* metrics
m_hft.hft_tick_to_signal_latency_ns.labels(symbol_id="1", asset_class="equity").observe(500)
m_hft.hft_signal_to_order_latency_ns.labels(symbol_id="1").observe(200)
m_hft.hft_order_to_fill_latency_ns.labels(symbol_id="1", broker="paper").observe(1000)
m_hft.hft_ticks_processed_total.labels(symbol_id="1").inc(100)
m_hft.hft_orders_submitted_total.labels(symbol_id="1", side="buy", type="MARKET").inc(5)
m_hft.hft_fills_received_total.labels(symbol_id="1", side="buy").inc(4)
m_hft.hft_ring_buffer_occupancy_pct.labels(buffer_name="wal").set(15.0)
m_hft.hft_feature_staleness_ms.labels(symbol_id="1", feature_name="vwap").set(50)
m_hft.hft_feed_connected.labels(symbol_id="1").set(1)
m_hft.hft_portfolio_nav.set(1_000_000)
m_hft.hft_portfolio_pnl_realized.set(5000)
m_hft.hft_portfolio_pnl_unrealized.set(2000)
m_hft.hft_portfolio_drawdown_pct.set(0.02)
m_hft.hft_position_size.labels(symbol_id="1").set(100)
m_hft.hft_position_pnl.labels(symbol_id="1").set(500)
m_hft.hft_vpin.labels(symbol_id="1").set(0.45)
m_hft.hft_regime_posterior.labels(regime_name="LOW_VOL_TRENDING").set(0.75)
m_hft.hft_kill_switch_level.set(0)
m_hft.hft_reconciliation_discrepancy.labels(symbol_id="1").set(0)
m_hft.hft_signal_sharpe_30d.labels(signal_name="momentum").set(1.5)
m_hft.hft_signal_sharpe_60d.labels(signal_name="momentum").set(1.2)
m_hft.hft_signal_weight.labels(signal_name="momentum").set(0.4)
m_hft.hft_feature_kl_divergence.labels(feature_name="vwap").set(0.1)

snap_hft = m_hft.snapshot()
required_metrics = [
    b"hft_tick_to_signal_latency_ns", b"hft_signal_to_order_latency_ns",
    b"hft_order_to_fill_latency_ns", b"hft_ticks_processed_total",
    b"hft_orders_submitted_total", b"hft_fills_received_total",
    b"hft_ring_buffer_occupancy_pct", b"hft_feature_staleness_ms",
    b"hft_feed_connected", b"hft_portfolio_nav", b"hft_portfolio_pnl_realized",
    b"hft_portfolio_pnl_unrealized", b"hft_portfolio_drawdown_pct",
    b"hft_position_size", b"hft_position_pnl", b"hft_vpin",
    b"hft_regime_posterior", b"hft_kill_switch_level",
    b"hft_reconciliation_discrepancy", b"hft_signal_sharpe_30d",
    b"hft_signal_sharpe_60d", b"hft_signal_weight", b"hft_feature_kl_divergence",
]
missing = [m.decode() for m in required_metrics if m not in snap_hft]

gate("all_hft_metrics_exist",
     len(missing) == 0,
     f"checked={len(required_metrics)}, missing={missing[:5] if missing else 'none'}")


# ── 8. HFT alert rules ──────────────────────────────────────

from src.monitoring.alerting import hft_alert_rules

hft_rules = hft_alert_rules()
rule_names = {r.name for r in hft_rules}
expected_rules = {
    "latency_spike_p99", "ring_buffer_occupancy", "hft_reconciliation_discrepancy",
    "hft_portfolio_drawdown_pct", "feed_lost", "feature_staleness_ms",
    "signal_sharpe_30d", "feature_kl_divergence", "kill_switch_level",
}
has_all = expected_rules.issubset(rule_names)

gate("hft_alert_rules_defined",
     has_all and len(hft_rules) >= 10,
     f"rules={len(hft_rules)}, missing={expected_rules - rule_names}")


# ── 9. Signal decay down-weights ────────────────────────────

am_decay = AlertManager()
am_decay.add_rule(AlertRule("signal_sharpe_30d", AlertSeverity.WARNING, 0.0, "lt",
                            cooldown_ns=0))
decay_alerts = am_decay.evaluate("signal_sharpe_30d", -0.5)
decay_fired = len(decay_alerts) == 1

gate("signal_decay_alert_fires",
     decay_fired,
     f"fired={decay_fired}, sharpe=-0.5")


# ── 10. Feature drift detected ──────────────────────────────

am_drift = AlertManager()
am_drift.add_rule(AlertRule("feature_kl_divergence", AlertSeverity.WARNING, 0.5, "gt",
                            cooldown_ns=0))
drift_alerts = am_drift.evaluate("feature_kl_divergence", 0.8)
drift_fired = len(drift_alerts) == 1

gate("feature_drift_alert_fires",
     drift_fired,
     f"fired={drift_fired}, kl=0.8")


# ── Summary ─────────────────────────────────────────────────────

print()
n_pass = sum(1 for _, p, _ in results if p)
n_total = len(results)
tag = "ALL_PASS" if n_pass == n_total else "FAIL"
print(f"Phase 8 validation: {n_pass}/{n_total} — {tag}")
sys.exit(0 if tag == "ALL_PASS" else 1)
