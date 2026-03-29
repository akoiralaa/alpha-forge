"""Phase 8 tests — Metrics, health checks, alerting."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.monitoring.metrics import TradingMetrics
from src.monitoring.health import HealthChecker, HealthStatus
from src.monitoring.alerting import (
    AlertManager,
    AlertRule,
    AlertSeverity,
    default_trading_rules,
)


# ── Metrics Tests ───────────────────────────────────────────────

class TestMetrics:
    def test_gauge_set_and_read(self):
        m = TradingMetrics()
        m.nav.set(1_000_000)
        assert m.nav._value.get() == 1_000_000

    def test_counter_increment(self):
        m = TradingMetrics()
        m.orders_submitted.labels(side="buy", order_type="MARKET").inc()
        m.orders_submitted.labels(side="buy", order_type="MARKET").inc()
        val = m.orders_submitted.labels(side="buy", order_type="MARKET")._value.get()
        assert val == 2.0

    def test_histogram_observe(self):
        m = TradingMetrics()
        m.fill_latency_ms.observe(50)
        m.fill_latency_ms.observe(150)
        # Histogram sum should be 200
        assert m.fill_latency_ms._sum.get() == 200.0

    def test_labeled_gauge(self):
        m = TradingMetrics()
        m.signal_value.labels(signal_name="momentum").set(0.75)
        val = m.signal_value.labels(signal_name="momentum")._value.get()
        assert val == 0.75

    def test_snapshot_produces_bytes(self):
        m = TradingMetrics()
        m.nav.set(500_000)
        m.drawdown_pct.set(0.05)
        output = m.snapshot()
        assert isinstance(output, bytes)
        assert b"trading_nav_dollars" in output
        assert b"trading_drawdown_pct" in output

    def test_isolated_registries(self):
        m1 = TradingMetrics()
        m2 = TradingMetrics()
        m1.nav.set(100)
        m2.nav.set(200)
        assert m1.nav._value.get() == 100
        assert m2.nav._value.get() == 200

    def test_risk_metrics(self):
        m = TradingMetrics()
        m.kill_switch_level.set(2)
        m.vpin.set(0.88)
        m.risk_checks_failed.labels(reason="FAT_FINGER").inc()
        assert m.kill_switch_level._value.get() == 2
        assert m.vpin._value.get() == 0.88

    def test_data_metrics(self):
        m = TradingMetrics()
        m.ticks_processed.inc(1000)
        m.data_gaps.inc()
        assert m.ticks_processed._value.get() == 1000
        assert m.data_gaps._value.get() == 1


# ── Health Check Tests ──────────────────────────────────────────

class TestHealthChecker:
    def test_all_healthy(self):
        hc = HealthChecker()
        hc.register("broker", lambda: (HealthStatus.HEALTHY, "connected"))
        hc.register("data", lambda: (HealthStatus.HEALTHY, "streaming"))
        result = hc.check()
        assert result.is_healthy
        assert len(result.components) == 2

    def test_one_unhealthy(self):
        hc = HealthChecker()
        hc.register("broker", lambda: (HealthStatus.HEALTHY, "ok"))
        hc.register("data", lambda: (HealthStatus.UNHEALTHY, "no data for 30s"))
        result = hc.check()
        assert result.status == HealthStatus.UNHEALTHY
        assert len(result.unhealthy_components) == 1
        assert result.unhealthy_components[0].name == "data"

    def test_degraded_status(self):
        hc = HealthChecker()
        hc.register("broker", lambda: (HealthStatus.DEGRADED, "high latency"))
        hc.register("data", lambda: (HealthStatus.HEALTHY, "ok"))
        result = hc.check()
        assert result.status == HealthStatus.DEGRADED
        assert len(result.degraded_components) == 1

    def test_exception_in_check(self):
        hc = HealthChecker()
        hc.register("broken", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        result = hc.check()
        assert result.status == HealthStatus.UNHEALTHY
        assert "boom" in result.components[0].message

    def test_unregister(self):
        hc = HealthChecker()
        hc.register("a", lambda: (HealthStatus.HEALTHY, ""))
        hc.register("b", lambda: (HealthStatus.HEALTHY, ""))
        hc.unregister("a")
        assert hc.registered_components == ["b"]

    def test_last_check(self):
        hc = HealthChecker()
        hc.register("x", lambda: (HealthStatus.HEALTHY, ""))
        assert hc.last_check is None
        hc.check()
        assert hc.last_check is not None
        assert hc.last_check.is_healthy

    def test_latency_measured(self):
        hc = HealthChecker()
        hc.register("slow", lambda: (HealthStatus.HEALTHY, "ok"))
        result = hc.check()
        assert result.components[0].latency_ms >= 0

    def test_empty_checker_healthy(self):
        hc = HealthChecker()
        result = hc.check()
        assert result.is_healthy
        assert len(result.components) == 0


# ── Alerting Tests ──────────────────────────────────────────────

class TestAlerting:
    def test_alert_fires_gt(self):
        mgr = AlertManager()
        mgr.add_rule(AlertRule("drawdown", AlertSeverity.WARNING, 0.05, "gt"))
        alerts = mgr.evaluate("drawdown", 0.06)
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING

    def test_alert_no_fire_below_threshold(self):
        mgr = AlertManager()
        mgr.add_rule(AlertRule("drawdown", AlertSeverity.WARNING, 0.05, "gt"))
        alerts = mgr.evaluate("drawdown", 0.03)
        assert len(alerts) == 0

    def test_alert_lt(self):
        mgr = AlertManager()
        mgr.add_rule(AlertRule("cash", AlertSeverity.CRITICAL, 100_000, "lt"))
        alerts = mgr.evaluate("cash", 50_000)
        assert len(alerts) == 1

    def test_cooldown(self):
        mgr = AlertManager()
        mgr.add_rule(AlertRule("x", AlertSeverity.WARNING, 0.5, "gt",
                               cooldown_ns=10_000_000_000))  # 10s
        alerts1 = mgr.evaluate("x", 0.6)
        alerts2 = mgr.evaluate("x", 0.7)  # should be suppressed
        assert len(alerts1) == 1
        assert len(alerts2) == 0

    def test_handler_called(self):
        received = []
        mgr = AlertManager()
        mgr.add_rule(AlertRule("metric", AlertSeverity.INFO, 10, "gt"))
        mgr.add_handler(lambda a: received.append(a))
        mgr.evaluate("metric", 15)
        assert len(received) == 1

    def test_evaluate_all(self):
        mgr = AlertManager()
        mgr.add_rule(AlertRule("a", AlertSeverity.WARNING, 0.5, "gt"))
        mgr.add_rule(AlertRule("b", AlertSeverity.CRITICAL, 100, "gt"))
        alerts = mgr.evaluate_all({"a": 0.8, "b": 50, "c": 999})
        assert len(alerts) == 1  # only 'a' fires

    def test_alert_history(self):
        mgr = AlertManager()
        mgr.add_rule(AlertRule("x", AlertSeverity.WARNING, 0, "gt", cooldown_ns=0))
        mgr.evaluate("x", 1)
        mgr.evaluate("x", 2)
        assert len(mgr.alert_history) == 2

    def test_acknowledge(self):
        mgr = AlertManager()
        mgr.add_rule(AlertRule("x", AlertSeverity.WARNING, 0, "gt", cooldown_ns=0))
        mgr.evaluate("x", 1)
        assert len(mgr.unacknowledged) == 1
        mgr.acknowledge(0)
        assert len(mgr.unacknowledged) == 0

    def test_acknowledge_all(self):
        mgr = AlertManager()
        mgr.add_rule(AlertRule("x", AlertSeverity.WARNING, 0, "gt", cooldown_ns=0))
        mgr.evaluate("x", 1)
        mgr.evaluate("x", 2)
        mgr.acknowledge_all()
        assert len(mgr.unacknowledged) == 0

    def test_default_rules(self):
        rules = default_trading_rules()
        assert len(rules) >= 5
        names = [r.name for r in rules]
        assert "drawdown_pct" in names
        assert "vpin" in names

    def test_multiple_rules_same_metric(self):
        mgr = AlertManager()
        mgr.add_rule(AlertRule("dd", AlertSeverity.WARNING, 0.05, "gt"))
        mgr.add_rule(AlertRule("dd", AlertSeverity.CRITICAL, 0.10, "gt"))
        alerts = mgr.evaluate("dd", 0.12)
        assert len(alerts) == 2
        severities = {a.severity for a in alerts}
        assert AlertSeverity.WARNING in severities
        assert AlertSeverity.CRITICAL in severities
