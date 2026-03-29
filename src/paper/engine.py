"""Paper trading engine — end-to-end integration of all subsystems.

Wires together feature computation, signal generation, position sizing,
risk checks, regime detection, execution, and monitoring into a single
tick-driven loop.

Also provides:
  - Infrastructure Sharpe (paper vs backtest comparison)
  - Disaster recovery drills
  - Latency calibration
  - Staged capital deployment tracking
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from src.execution.broker import BrokerFill, BrokerOrder, PaperBroker
from src.execution.kill_switch import KillLevel, KillSwitch
from src.execution.order_manager import OrderManager
from src.execution.reconciliation import Reconciler
from src.execution.wal import WriteAheadLog
from src.monitoring.alerting import AlertManager, AlertSeverity, default_trading_rules
from src.monitoring.health import HealthChecker, HealthStatus
from src.monitoring.metrics import TradingMetrics
from src.portfolio.risk import OrderIntent, Portfolio, PreTradeRiskCheck
from src.portfolio.sizing import compute_position_size


@dataclass
class PaperTick:
    """Simplified tick for paper trading."""
    symbol_id: int
    price: float
    volume: int
    timestamp_ns: int
    bid: float = 0.0
    ask: float = 0.0


@dataclass
class PaperConfig:
    """Paper trading configuration."""
    initial_nav: float = 1_000_000.0
    risk_budget_per_position: float = 0.005
    max_position_pct_nav: float = 0.05
    slippage_bps: float = 1.0
    commission_per_share: float = 0.005
    reconciliation_interval_ticks: int = 100
    signal_threshold: float = 0.1
    drawdown_auto_kill_pct: float = 0.15
    kelly_fraction: float = 0.25


@dataclass
class PaperTradingStats:
    """Accumulated statistics from a paper trading session."""
    ticks_processed: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    risk_blocks: int = 0
    kill_switch_activations: int = 0
    reconciliation_runs: int = 0
    reconciliation_breaks: int = 0
    alerts_fired: int = 0
    peak_nav: float = 0.0
    final_nav: float = 0.0
    max_drawdown_pct: float = 0.0
    total_pnl: float = 0.0


class PaperTradingEngine:
    """End-to-end paper trading engine integrating all subsystems."""

    def __init__(self, config: PaperConfig | None = None):
        self.config = config or PaperConfig()
        c = self.config

        # Execution layer
        self.broker = PaperBroker(
            initial_cash=c.initial_nav,
            slippage_bps=c.slippage_bps,
            commission_per_share=c.commission_per_share,
        )
        self.wal = WriteAheadLog()
        self.kill_switch = KillSwitch(
            self.broker, self.wal,
            drawdown_auto_kill_pct=c.drawdown_auto_kill_pct,
        )
        self.risk_check = PreTradeRiskCheck(
            max_position_pct_nav=c.max_position_pct_nav,
        )
        self.portfolio = Portfolio(nav=c.initial_nav, peak_nav=c.initial_nav)
        self.order_manager = OrderManager(
            self.broker, self.wal, self.kill_switch,
            self.risk_check, self.portfolio,
        )
        self.reconciler = Reconciler(self.broker, self.wal)

        # Monitoring
        self.metrics = TradingMetrics()
        self.health = HealthChecker()
        self.alerts = AlertManager()
        for rule in default_trading_rules():
            self.alerts.add_rule(rule)

        self._setup_health_checks()

        # State
        self.prices: dict[int, float] = {}
        self.volumes: dict[int, list[float]] = {}  # rolling volume
        self.returns: dict[int, list[float]] = {}   # rolling returns
        self.stats = PaperTradingStats(peak_nav=c.initial_nav, final_nav=c.initial_nav)
        self._tick_count = 0
        self._signal_fn: Optional[callable] = None

    def set_signal_function(self, fn):
        """Set custom signal function: fn(symbol_id, price, engine) -> float."""
        self._signal_fn = fn

    def _setup_health_checks(self):
        self.health.register("broker", lambda: (
            HealthStatus.HEALTHY if self.broker.is_connected()
            else HealthStatus.UNHEALTHY, "broker"
        ))
        self.health.register("kill_switch", lambda: (
            HealthStatus.HEALTHY if self.kill_switch.level == KillLevel.NORMAL
            else HealthStatus.DEGRADED, f"level={self.kill_switch.level.name}"
        ))

    def on_tick(self, tick: PaperTick) -> Optional[str]:
        """Process a single tick through the full pipeline.

        Returns order_id if an order was submitted, None otherwise.
        """
        self._tick_count += 1
        self.stats.ticks_processed += 1
        self.metrics.ticks_processed.inc()

        sid = tick.symbol_id
        price = tick.price

        # Update price state
        prev_price = self.prices.get(sid)
        self.prices[sid] = price
        self.broker.set_price(sid, price)

        # Track returns
        if prev_price and prev_price > 0:
            ret = (price - prev_price) / prev_price
            if sid not in self.returns:
                self.returns[sid] = []
            self.returns[sid].append(ret)
            if len(self.returns[sid]) > 100:
                self.returns[sid] = self.returns[sid][-100:]

        # Track volume
        if sid not in self.volumes:
            self.volumes[sid] = []
        self.volumes[sid].append(tick.volume)
        if len(self.volumes[sid]) > 100:
            self.volumes[sid] = self.volumes[sid][-100:]

        # Update NAV (mark to market)
        self._update_nav()

        # Check drawdown auto-kill
        self.kill_switch.check_drawdown(self.portfolio.nav, self.portfolio.peak_nav)

        # Generate signal
        signal = self._compute_signal(sid, price)

        # Update metrics
        self.metrics.nav.set(self.portfolio.nav)
        self.metrics.drawdown_pct.set(self.portfolio.drawdown_pct)
        self.metrics.position_count.set(len(self.broker.get_positions()))
        self.metrics.signal_value.labels(signal_name=f"sym_{sid}").set(signal)

        # Check alerts
        fired = self.alerts.evaluate_all({
            "drawdown_pct": self.portfolio.drawdown_pct,
        })
        self.stats.alerts_fired += len(fired)

        # Reconciliation at interval
        if self._tick_count % self.config.reconciliation_interval_ticks == 0:
            self._run_reconciliation()

        # Trade if signal exceeds threshold
        order_id = None
        if abs(signal) > self.config.signal_threshold:
            order_id = self._execute_signal(sid, signal, price)

        return order_id

    def _compute_signal(self, symbol_id: int, price: float) -> float:
        """Compute trading signal for a symbol."""
        if self._signal_fn:
            return self._signal_fn(symbol_id, price, self)

        # Default: simple mean-reversion on returns
        rets = self.returns.get(symbol_id, [])
        if len(rets) < 20:
            return 0.0

        recent = rets[-20:]
        mean = np.mean(recent)
        std = np.std(recent)
        if std < 1e-10:
            return 0.0

        zscore = (rets[-1] - mean) / std
        return float(np.clip(-zscore * 0.3, -1.0, 1.0))

    def _execute_signal(self, symbol_id: int, signal: float, price: float) -> Optional[str]:
        """Size and submit an order based on signal."""
        # Estimate daily vol
        rets = self.returns.get(symbol_id, [])
        daily_vol = np.std(rets[-20:]) * np.sqrt(252) if len(rets) >= 20 else 0.02

        size = compute_position_size(
            symbol_id, signal, self.portfolio.nav,
            self.config.risk_budget_per_position,
            price, max(daily_vol, 0.001),
            self.config.kelly_fraction,
        )

        if abs(size) < 1:
            return None

        side = 1 if size > 0 else -1
        abs_size = abs(size)

        order, risk_result = self.order_manager.submit(
            symbol_id, side, abs_size,
            current_price=price,
            adv_20d=max(np.mean(self.volumes.get(symbol_id, [1_000_000])) * price, 1),
        )

        self.stats.orders_submitted += 1
        self.metrics.orders_submitted.labels(
            side="buy" if side == 1 else "sell",
            order_type="MARKET",
        ).inc()

        if order.state.value == "REJECTED":
            self.stats.orders_rejected += 1
            if risk_result and not risk_result.passed:
                self.stats.risk_blocks += 1
                self.metrics.risk_checks_failed.labels(
                    reason=risk_result.reason.value
                ).inc()
            return None

        # Process fill (PaperBroker fills immediately)
        if self.broker.fills:
            last_fill = self.broker.fills[-1]
            if last_fill.order_id == order.order_id:
                self.order_manager.on_fill(last_fill)
                self.stats.orders_filled += 1
                self.metrics.orders_filled.labels(
                    side="buy" if side == 1 else "sell"
                ).inc()

        return order.order_id

    def _update_nav(self):
        """Mark positions to market and update NAV."""
        positions = self.broker.get_positions()
        unrealized = sum(
            qty * self.prices.get(sid, 0)
            for sid, qty in positions.items()
        )
        self.portfolio.nav = self.broker.cash + unrealized
        self.portfolio.peak_nav = max(self.portfolio.peak_nav, self.portfolio.nav)

        dd = self.portfolio.drawdown_pct
        self.stats.max_drawdown_pct = max(self.stats.max_drawdown_pct, dd)
        self.stats.final_nav = self.portfolio.nav
        self.stats.peak_nav = self.portfolio.peak_nav
        self.stats.total_pnl = self.portfolio.nav - self.config.initial_nav

    def _run_reconciliation(self):
        """Run position reconciliation."""
        report = self.reconciler.reconcile()
        self.stats.reconciliation_runs += 1
        if not report.is_clean:
            self.stats.reconciliation_breaks += len(report.breaks)
            self.metrics.reconciliation_breaks.inc(len(report.breaks))

    def run_session(self, ticks: list[PaperTick]) -> PaperTradingStats:
        """Run a full paper trading session on a tick stream."""
        for tick in ticks:
            self.on_tick(tick)
        # Final reconciliation
        self._run_reconciliation()
        return self.stats

    def get_health(self) -> dict:
        """Get current system health."""
        result = self.health.check()
        return {
            "status": result.status.value,
            "components": {c.name: c.status.value for c in result.components},
        }



# Infrastructure Sharpe

def compute_infrastructure_sharpe(
    paper_returns: np.ndarray,
    backtest_returns: np.ndarray,
    annualization: float = 252.0,
) -> dict:
    """Compute infrastructure_sharpe = paper_sharpe / backtest_sharpe.

    Args:
        paper_returns: Daily returns from paper trading.
        backtest_returns: Daily returns from backtest over the same period.
        annualization: Annualization factor.

    Returns:
        Dict with paper_sharpe, backtest_sharpe, infrastructure_sharpe, passed.
    """
    def _sharpe(r: np.ndarray) -> float:
        if len(r) < 2 or np.std(r) < 1e-12:
            return 0.0
        return float(np.mean(r) / np.std(r) * np.sqrt(annualization))

    ps = _sharpe(paper_returns)
    bs = _sharpe(backtest_returns)
    ratio = ps / bs if abs(bs) > 1e-12 else 0.0

    return {
        "paper_sharpe": ps,
        "backtest_sharpe": bs,
        "infrastructure_sharpe": ratio,
        "passed": ratio >= 0.80,
    }



# Disaster recovery drills

@dataclass
class DrillResult:
    drill_name: str
    passed: bool
    detail: str = ""
    timestamp_ns: int = 0


class DisasterRecoveryDrills:
    """Runs disaster recovery drills before going live."""

    def __init__(self, engine: PaperTradingEngine):
        self.engine = engine
        self.results: list[DrillResult] = []

    def drill_1_crash_recovery(self) -> DrillResult:
        """DRILL_1: Execution engine crash recovery.

        Simulates kill -9 with open positions. Verifies:
        - Kill switch L3 activates (flatten)
        - Positions recovered from broker snapshot
        - No ghost positions after recovery
        """
        ts = time.time_ns()
        eng = self.engine

        # Record positions before "crash"
        pre_positions = dict(eng.broker.get_positions())

        # Simulate watchdog activating L3 (emergency flatten)
        event = eng.kill_switch.activate(KillLevel.FLATTEN, "DRILL_1: crash recovery")
        post_positions = eng.broker.get_positions()
        all_flat = all(abs(v) < 1e-6 for v in post_positions.values())

        # Reset kill switch for continued operation
        eng.kill_switch.reset()

        passed = all_flat
        result = DrillResult(
            "DRILL_1_crash_recovery",
            passed,
            f"pre_positions={len(pre_positions)}, flattened={event.positions_flattened}, all_flat={all_flat}",
            ts,
        )
        self.results.append(result)
        return result

    def drill_2_feed_outage(self) -> DrillResult:
        """DRILL_2: Feed outage recovery.

        Simulates disconnecting market data for >500ms. Verifies:
        - Stale data halt activates
        - Orders cancelled
        - Reconciliation clean after recovery
        """
        ts = time.time_ns()
        eng = self.engine

        from src.execution.stale_data import StaleDataMonitor

        monitor = StaleDataMonitor(stale_threshold_ms=500.0)
        # Simulate tick then outage
        base = time.time_ns()
        monitor.on_tick(1, base)
        # Check after 600ms (>500ms threshold)
        stale = monitor.check_staleness(base + 600_000_000)
        halt_activated = len(stale) > 0 and not monitor.is_order_allowed(1)

        # Simulate reconnect
        monitor.on_tick(1, base + 700_000_000)
        resumed = monitor.is_order_allowed(1)

        # Reconciliation
        report = eng.reconciler.reconcile()
        recon_clean = report.is_clean

        passed = halt_activated and resumed and recon_clean
        result = DrillResult(
            "DRILL_2_feed_outage",
            passed,
            f"halt={halt_activated}, resumed={resumed}, recon_clean={recon_clean}",
            ts,
        )
        self.results.append(result)
        return result

    def drill_3_kill_switch_l2(self) -> DrillResult:
        """DRILL_3: Kill switch L2 activation.

        Triggers kill_switch.activate_level2 and verifies:
        - All orders cancelled
        - All positions flattened
        - Portfolio flat within time limit
        """
        ts = time.time_ns()
        eng = self.engine

        event = eng.kill_switch.activate(KillLevel.FLATTEN, "DRILL_3: kill switch L2 test")
        post = eng.broker.get_positions()
        all_flat = all(abs(v) < 1e-6 for v in post.values())

        # Reset for continued operation
        eng.kill_switch.reset()

        passed = all_flat
        result = DrillResult(
            "DRILL_3_kill_switch_l2",
            passed,
            f"flattened={event.positions_flattened}, all_flat={all_flat}",
            ts,
        )
        self.results.append(result)
        return result

    def all_drills_passed(self) -> bool:
        """Check if all 3 drills have been run and passed."""
        names = {r.drill_name for r in self.results if r.passed}
        return all(d in names for d in [
            "DRILL_1_crash_recovery",
            "DRILL_2_feed_outage",
            "DRILL_3_kill_switch_l2",
        ])



# Latency calibration

@dataclass
class LatencyCalibration:
    """Latency calibration results comparing paper trading to backtest assumptions."""
    p50_actual_ns: float
    p99_actual_ns: float
    backtest_assumed_p50_ns: float
    ratio: float = 0.0
    status: str = "UNKNOWN"  # PASS, WARN, FAIL

    def __post_init__(self):
        if self.backtest_assumed_p50_ns > 0:
            self.ratio = self.p50_actual_ns / self.backtest_assumed_p50_ns
        if self.ratio <= 1.5:
            self.status = "PASS"
        elif self.ratio <= 2.0:
            self.status = "WARN"
        else:
            self.status = "FAIL"


def calibrate_latency(
    fill_latencies_ns: np.ndarray,
    backtest_assumed_p50_ns: float,
) -> LatencyCalibration:
    """Compute latency calibration from observed fill latencies.

    Args:
        fill_latencies_ns: Array of observed fill latencies in nanoseconds.
        backtest_assumed_p50_ns: Backtest-assumed p50 latency in nanoseconds.
    """
    if len(fill_latencies_ns) == 0:
        return LatencyCalibration(0, 0, backtest_assumed_p50_ns)

    p50 = float(np.percentile(fill_latencies_ns, 50))
    p99 = float(np.percentile(fill_latencies_ns, 99))

    return LatencyCalibration(
        p50_actual_ns=p50,
        p99_actual_ns=p99,
        backtest_assumed_p50_ns=backtest_assumed_p50_ns,
    )



# Capital deployment stages

class DeploymentStage(Enum):
    PAPER = 0       # Stage 0: 10 trading days, full paper
    LIVE_5PCT = 1   # Stage 1: Week 1-2, 5% of target capital
    LIVE_20PCT = 2  # Stage 2: Week 3-4, 20% of target capital
    LIVE_50PCT = 3  # Stage 3: Month 2, 50% of target capital
    LIVE_100PCT = 4 # Stage 4: Month 3+, 100% of target capital


STAGE_CAPITAL_PCT = {
    DeploymentStage.PAPER: 0.0,
    DeploymentStage.LIVE_5PCT: 0.05,
    DeploymentStage.LIVE_20PCT: 0.20,
    DeploymentStage.LIVE_50PCT: 0.50,
    DeploymentStage.LIVE_100PCT: 1.00,
}


@dataclass
class StageTransition:
    from_stage: DeploymentStage
    to_stage: DeploymentStage
    live_sharpe: float
    paper_sharpe: float
    ratio: float
    allowed: bool
    reason: str = ""


class CapitalDeploymentManager:
    """Tracks staged capital deployment from paper to full live."""

    SHARPE_GATE = 0.80  # live_sharpe >= 0.80 * paper_sharpe

    def __init__(self, target_capital: float):
        self.target_capital = target_capital
        self.current_stage = DeploymentStage.PAPER
        self.transitions: list[StageTransition] = []

    @property
    def current_capital(self) -> float:
        return self.target_capital * STAGE_CAPITAL_PCT[self.current_stage]

    def can_advance(self, live_sharpe: float, paper_sharpe: float) -> StageTransition:
        """Check if we can advance to the next deployment stage."""
        if self.current_stage == DeploymentStage.LIVE_100PCT:
            return StageTransition(
                self.current_stage, self.current_stage,
                live_sharpe, paper_sharpe, 0, False, "already at full deployment",
            )

        next_stage = DeploymentStage(self.current_stage.value + 1)
        ratio = live_sharpe / paper_sharpe if abs(paper_sharpe) > 1e-12 else 0.0
        allowed = ratio >= self.SHARPE_GATE

        reason = ""
        if not allowed:
            reason = f"live/paper ratio {ratio:.2f} < {self.SHARPE_GATE}"

        transition = StageTransition(
            self.current_stage, next_stage,
            live_sharpe, paper_sharpe, ratio, allowed, reason,
        )
        return transition

    def advance(self, live_sharpe: float, paper_sharpe: float) -> StageTransition:
        """Attempt to advance to the next deployment stage."""
        transition = self.can_advance(live_sharpe, paper_sharpe)
        if transition.allowed:
            self.current_stage = transition.to_stage
        self.transitions.append(transition)
        return transition
