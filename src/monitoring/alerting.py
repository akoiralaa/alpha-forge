"""Alerting system — threshold-based alerts with cooldown and escalation.

Alerts fire when metrics cross thresholds. Each alert has a cooldown
to prevent spam. Severity levels: INFO, WARNING, CRITICAL.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Optional


class AlertSeverity(IntEnum):
    INFO = 0
    WARNING = 1
    CRITICAL = 2


@dataclass
class Alert:
    """A fired alert."""
    name: str
    severity: AlertSeverity
    message: str
    value: float
    threshold: float
    timestamp_ns: int
    acknowledged: bool = False


@dataclass
class AlertRule:
    """Rule that fires when a value crosses a threshold."""
    name: str
    severity: AlertSeverity
    threshold: float
    comparator: str           # "gt", "lt", "gte", "lte"
    cooldown_ns: int = 60_000_000_000  # 60 seconds default
    message_template: str = "{name}: {value:.4f} {comparator} {threshold:.4f}"
    _last_fired_ns: int = 0

    def evaluate(self, value: float, now_ns: int) -> Optional[Alert]:
        """Check if rule fires. Returns Alert or None."""
        if now_ns - self._last_fired_ns < self.cooldown_ns:
            return None

        fired = False
        if self.comparator == "gt" and value > self.threshold:
            fired = True
        elif self.comparator == "lt" and value < self.threshold:
            fired = True
        elif self.comparator == "gte" and value >= self.threshold:
            fired = True
        elif self.comparator == "lte" and value <= self.threshold:
            fired = True

        if not fired:
            return None

        self._last_fired_ns = now_ns
        return Alert(
            name=self.name,
            severity=self.severity,
            message=self.message_template.format(
                name=self.name, value=value,
                comparator=self.comparator, threshold=self.threshold,
            ),
            value=value,
            threshold=self.threshold,
            timestamp_ns=now_ns,
        )


# Type for alert handler callbacks
AlertHandler = Callable[[Alert], None]


class AlertManager:
    """Manages alert rules and dispatches to handlers."""

    def __init__(self):
        self._rules: list[AlertRule] = []
        self._handlers: list[AlertHandler] = []
        self._fired: list[Alert] = []

    def add_rule(self, rule: AlertRule):
        self._rules.append(rule)

    def add_handler(self, handler: AlertHandler):
        """Register a callback that receives fired alerts."""
        self._handlers.append(handler)

    def evaluate(self, metric_name: str, value: float) -> list[Alert]:
        """Evaluate all rules matching metric_name against value."""
        now = time.time_ns()
        fired = []
        for rule in self._rules:
            if rule.name == metric_name:
                alert = rule.evaluate(value, now)
                if alert:
                    fired.append(alert)
                    self._fired.append(alert)
                    for handler in self._handlers:
                        handler(alert)
        return fired

    def evaluate_all(self, metrics: dict[str, float]) -> list[Alert]:
        """Evaluate all rules against a dict of metric values."""
        all_fired = []
        for name, value in metrics.items():
            all_fired.extend(self.evaluate(name, value))
        return all_fired

    @property
    def alert_history(self) -> list[Alert]:
        return list(self._fired)

    @property
    def unacknowledged(self) -> list[Alert]:
        return [a for a in self._fired if not a.acknowledged]

    def acknowledge(self, index: int):
        """Acknowledge alert by index in history."""
        if 0 <= index < len(self._fired):
            self._fired[index].acknowledged = True

    def acknowledge_all(self):
        for a in self._fired:
            a.acknowledged = True


def default_trading_rules() -> list[AlertRule]:
    """Standard alert rules for a trading system."""
    return [
        AlertRule("drawdown_pct", AlertSeverity.WARNING, 0.05, "gt",
                  message_template="Drawdown {value:.1%} exceeds {threshold:.1%}"),
        AlertRule("drawdown_pct", AlertSeverity.CRITICAL, 0.10, "gt",
                  message_template="CRITICAL drawdown {value:.1%} exceeds {threshold:.1%}"),
        AlertRule("vpin", AlertSeverity.WARNING, 0.85, "gt",
                  message_template="VPIN {value:.2f} elevated above {threshold:.2f}"),
        AlertRule("vpin", AlertSeverity.CRITICAL, 0.90, "gt",
                  message_template="VPIN {value:.2f} critical above {threshold:.2f}"),
        AlertRule("gross_leverage", AlertSeverity.WARNING, 2.5, "gt",
                  message_template="Gross leverage {value:.2f}x above {threshold:.2f}x"),
        AlertRule("fill_latency_ms", AlertSeverity.WARNING, 500.0, "gt",
                  message_template="Fill latency {value:.0f}ms exceeds {threshold:.0f}ms"),
        AlertRule("reconciliation_breaks", AlertSeverity.CRITICAL, 0, "gt",
                  message_template="Reconciliation break detected: count={value:.0f}"),
    ]


def hft_alert_rules() -> list[AlertRule]:
    """HFT-specific alert rules (latency spikes, fill quality, etc.)."""
    return [
        # LatencySpikeP99: p99 > 2x p50
        AlertRule("latency_spike_p99", AlertSeverity.WARNING, 2.0, "gt",
                  message_template="P99 latency spike: ratio {value:.2f} > {threshold:.2f}x median"),

        # RingBufferOverload: occupancy > 80%
        AlertRule("ring_buffer_occupancy", AlertSeverity.WARNING, 80.0, "gt",
                  message_template="Ring buffer occupancy {value:.1f}% > {threshold:.1f}%"),

        # ReconciliationDiscrepancy: != 0 (immediate)
        AlertRule("hft_reconciliation_discrepancy", AlertSeverity.CRITICAL, 0, "gt",
                  cooldown_ns=0,
                  message_template="Reconciliation discrepancy: {value:.0f}"),

        # DrawdownLevel1: > 5%
        AlertRule("hft_portfolio_drawdown_pct", AlertSeverity.WARNING, 5.0, "gt",
                  message_template="Drawdown {value:.1f}% exceeds {threshold:.1f}%"),

        # DrawdownLevel2: > 10% (critical)
        AlertRule("hft_portfolio_drawdown_pct", AlertSeverity.CRITICAL, 10.0, "gt",
                  message_template="CRITICAL drawdown {value:.1f}% exceeds {threshold:.1f}%"),

        # FeedLost: feed_connected == 0
        AlertRule("feed_lost", AlertSeverity.CRITICAL, 0.5, "lt",
                  cooldown_ns=2_000_000_000,
                  message_template="Feed lost: connected={value:.0f}"),

        # FeatureStale: staleness > 1000ms
        AlertRule("feature_staleness_ms", AlertSeverity.WARNING, 1000.0, "gt",
                  message_template="Feature stale: {value:.0f}ms > {threshold:.0f}ms"),

        # SignalDecay30d: sharpe < 0
        AlertRule("signal_sharpe_30d", AlertSeverity.WARNING, 0.0, "lt",
                  message_template="Signal 30d Sharpe decayed: {value:.2f} < {threshold:.2f}"),

        # FeatureDrift: KL > 0.5
        AlertRule("feature_kl_divergence", AlertSeverity.WARNING, 0.5, "gt",
                  message_template="Feature drift: KL={value:.3f} > {threshold:.3f}"),

        # KillSwitchActivated: level > 0 (immediate, critical)
        AlertRule("kill_switch_level", AlertSeverity.CRITICAL, 0, "gt",
                  cooldown_ns=0,
                  message_template="Kill switch activated: level={value:.0f}"),
    ]
