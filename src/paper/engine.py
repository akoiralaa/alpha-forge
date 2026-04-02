
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
from src.portfolio.allocator import CentralRiskAllocator
from src.portfolio.capacity import LiquidityCapacityModel, LiquiditySnapshot
from src.portfolio.risk import OrderIntent, Portfolio, PreTradeRiskCheck
from src.portfolio.sizing import compute_position_size

@dataclass
class PaperTick:
    symbol_id: int
    price: float
    volume: int
    timestamp_ns: int
    bid: float = 0.0
    ask: float = 0.0

@dataclass
class PaperConfig:
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
        self.latest_ticks: dict[int, PaperTick] = {}
        self.stats = PaperTradingStats(peak_nav=c.initial_nav, final_nav=c.initial_nav)
        self._tick_count = 0
        self._signal_fn: Optional[callable] = None
        self._strategy_fns: dict[str, callable] = {}
        self._strategy_allocator: CentralRiskAllocator | None = None
        self._capacity_model: LiquidityCapacityModel | None = None
        self._strategy_signals_by_symbol: dict[int, dict[str, float]] = {}

    def set_signal_function(self, fn):
        self._signal_fn = fn
        self._strategy_fns = {}
        self._strategy_allocator = None
        self._capacity_model = None

    def set_strategy_functions(
        self,
        strategy_fns: dict[str, callable],
        allocator: CentralRiskAllocator | None = None,
        capacity_model: LiquidityCapacityModel | None = None,
    ):
        self._strategy_fns = dict(strategy_fns)
        self._strategy_allocator = allocator
        self._capacity_model = capacity_model or LiquidityCapacityModel()
        self._signal_fn = None

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
        self._tick_count += 1
        self.stats.ticks_processed += 1
        self.metrics.ticks_processed.inc()

        sid = tick.symbol_id
        price = tick.price
        self.latest_ticks[sid] = tick

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
            self._update_strategy_realized_performance(sid, ret)

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
        self.metrics.peak_nav.set(self.portfolio.peak_nav)
        self.metrics.daily_pnl.set(self.stats.total_pnl)
        self.metrics.gross_exposure.set(self.portfolio.gross_exposure)
        self.metrics.net_exposure.set(self.portfolio.net_exposure)
        self.metrics.position_count.set(len(self.broker.get_positions()))
        self.metrics.kill_switch_level.set(int(self.kill_switch.level))
        self.metrics.vpin.set(self.portfolio.current_vpin)
        self.metrics.signal_value.labels(signal_name=f"sym_{sid}").set(signal)
        self._update_strategy_metrics()

        # Check alerts
        fired = self.alerts.evaluate_all({
            "drawdown_pct": self.portfolio.drawdown_pct,
            "gross_leverage": self.portfolio.gross_exposure / max(self.portfolio.nav, 1e-12),
            "reconciliation_breaks": float(self.stats.reconciliation_breaks),
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
        if self._strategy_fns:
            strategy_signals = {
                name: float(np.clip(fn(symbol_id, price, self), -1.0, 1.0))
                for name, fn in self._strategy_fns.items()
            }
            self._strategy_signals_by_symbol[symbol_id] = strategy_signals
            for name, value in strategy_signals.items():
                self.metrics.signal_value.labels(signal_name=f"strategy_{name}").set(value)

            if self._strategy_allocator:
                return self._strategy_allocator.combine_signals(strategy_signals)

            active = [sig for sig in strategy_signals.values() if abs(sig) > 1e-12]
            if not active:
                return 0.0
            return float(np.clip(np.mean(active), -1.0, 1.0))

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
        self._observe_strategy_capacity(symbol_id, abs_size * price)

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

    def _build_liquidity_snapshot(self, symbol_id: int, price: float) -> LiquiditySnapshot:
        tick = self.latest_ticks.get(symbol_id)
        spread_bps = 2.0
        if tick is not None and tick.bid > 0 and tick.ask > 0 and price > 0:
            spread_bps = (tick.ask - tick.bid) / price * 10_000.0
        rets = self.returns.get(symbol_id, [])
        realized_vol_daily = (
            np.std(rets[-20:]) * np.sqrt(252)
            if len(rets) >= 5 else 0.02
        )
        adv_usd = max(np.mean(self.volumes.get(symbol_id, [1_000_000])) * price, 1.0)
        return LiquiditySnapshot(
            symbol_id=symbol_id,
            price=max(price, 1e-8),
            adv_usd=adv_usd,
            spread_bps=spread_bps,
            realized_vol_daily=max(realized_vol_daily, 1e-4),
        )

    def _observe_strategy_capacity(self, symbol_id: int, order_notional_usd: float) -> None:
        if not self._strategy_allocator or not self._capacity_model:
            return
        strategy_signals = self._strategy_signals_by_symbol.get(symbol_id)
        if not strategy_signals:
            return
        weights = self._strategy_allocator.target_weights()
        contributions = {
            name: abs(weights.get(name, 0.0) * signal)
            for name, signal in strategy_signals.items()
            if name in weights and abs(signal) > 1e-12
        }
        total = sum(contributions.values())
        if total <= 1e-12:
            return

        snapshot = self._build_liquidity_snapshot(symbol_id, self.prices.get(symbol_id, 0.0) or 0.0)
        for name, contribution in contributions.items():
            frac = contribution / total
            estimate = self._capacity_model.estimate_order(
                snapshot, order_notional_usd * frac
            )
            self._strategy_allocator.observe_capacity(
                name,
                utilization=estimate.utilization,
                impact_bps=estimate.impact_bps,
            )

    def _update_strategy_realized_performance(self, symbol_id: int, realized_return: float) -> None:
        if not self._strategy_allocator:
            return
        strategy_signals = self._strategy_signals_by_symbol.get(symbol_id)
        if not strategy_signals:
            return
        for name, signal in strategy_signals.items():
            self._strategy_allocator.observe(name, float(signal) * realized_return)

    def _update_strategy_metrics(self) -> None:
        if not self._strategy_allocator:
            return
        for name, snapshot in self._strategy_allocator.snapshot().items():
            self.metrics.strategy_weight.labels(strategy_name=name).set(snapshot["weight"])
            self.metrics.strategy_realized_sharpe.labels(strategy_name=name).set(snapshot["realized_sharpe"])
            self.metrics.strategy_expected_sharpe.labels(strategy_name=name).set(snapshot["expected_sharpe"])
            self.metrics.strategy_performance_gap.labels(strategy_name=name).set(snapshot["performance_gap"])
            self.metrics.strategy_capacity_utilization.labels(strategy_name=name).set(snapshot["capacity_utilization"])
            self.metrics.strategy_avg_impact_bps.labels(strategy_name=name).set(snapshot["avg_impact_bps"])

    def _update_nav(self):
        positions = self.broker.get_positions()
        self.portfolio.positions = {
            sid: qty * self.prices.get(sid, 0.0)
            for sid, qty in positions.items()
            if abs(qty) > 1e-12
        }
        unrealized = sum(
            qty * self.prices.get(sid, 0)
            for sid, qty in positions.items()
        )
        self.portfolio.nav = self.broker.cash + unrealized
        self.portfolio.peak_nav = max(self.portfolio.peak_nav, self.portfolio.nav)
        self.portfolio.daily_pnl_pct = (
            (self.portfolio.nav - self.config.initial_nav) / self.config.initial_nav
            if self.config.initial_nav > 0 else 0.0
        )

        dd = self.portfolio.drawdown_pct
        self.stats.max_drawdown_pct = max(self.stats.max_drawdown_pct, dd)
        self.stats.final_nav = self.portfolio.nav
        self.stats.peak_nav = self.portfolio.peak_nav
        self.stats.total_pnl = self.portfolio.nav - self.config.initial_nav

    def _run_reconciliation(self):
        report = self.reconciler.reconcile()
        self.stats.reconciliation_runs += 1
        if not report.is_clean:
            self.stats.reconciliation_breaks += len(report.breaks)
            self.metrics.reconciliation_breaks.inc(len(report.breaks))

    def run_session(self, ticks: list[PaperTick]) -> PaperTradingStats:
        for tick in ticks:
            self.on_tick(tick)
        # Final reconciliation
        self._run_reconciliation()
        return self.stats

    def get_health(self) -> dict:
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

    def __init__(self, engine: PaperTradingEngine):
        self.engine = engine
        self.results: list[DrillResult] = []

    def drill_1_crash_recovery(self) -> DrillResult:
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
        names = {r.drill_name for r in self.results if r.passed}
        return all(d in names for d in [
            "DRILL_1_crash_recovery",
            "DRILL_2_feed_outage",
            "DRILL_3_kill_switch_l2",
        ])

# Latency calibration

@dataclass
class LatencyCalibration:
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
    live_return: float | None = None
    paper_return: float | None = None
    live_max_drawdown: float | None = None
    infrastructure_sharpe: float | None = None
    reconciliation_breaks: int = 0
    critical_alerts: int = 0
    trading_days: int = 0
    failed_checks: list[str] = field(default_factory=list)

class CapitalDeploymentManager:

    SHARPE_GATE = 0.80  # live_sharpe >= 0.80 * paper_sharpe
    LIVE_SHARPE_GATE = 1.00
    LIVE_RETURN_GATE = 0.15
    MAX_DRAWDOWN_GATE = 0.12
    INFRA_SHARPE_GATE = 0.80
    MIN_TRADING_DAYS = {
        DeploymentStage.LIVE_5PCT: 10,
        DeploymentStage.LIVE_20PCT: 20,
        DeploymentStage.LIVE_50PCT: 40,
        DeploymentStage.LIVE_100PCT: 60,
    }

    def __init__(self, target_capital: float):
        self.target_capital = target_capital
        self.current_stage = DeploymentStage.PAPER
        self.transitions: list[StageTransition] = []

    @property
    def current_capital(self) -> float:
        return self.target_capital * STAGE_CAPITAL_PCT[self.current_stage]

    def can_advance(
        self,
        live_sharpe: float,
        paper_sharpe: float,
        *,
        live_return: float | None = None,
        paper_return: float | None = None,
        live_max_drawdown: float | None = None,
        infrastructure_sharpe: float | None = None,
        reconciliation_breaks: int = 0,
        critical_alerts: int = 0,
        trading_days: int = 0,
        enforce_fund_hurdles: bool = False,
    ) -> StageTransition:
        if self.current_stage == DeploymentStage.LIVE_100PCT:
            return StageTransition(
                self.current_stage, self.current_stage,
                live_sharpe, paper_sharpe, 0, False, "already at full deployment",
            )

        next_stage = DeploymentStage(self.current_stage.value + 1)
        ratio = live_sharpe / paper_sharpe if abs(paper_sharpe) > 1e-12 else 0.0
        failed_checks: list[str] = []

        if ratio < self.SHARPE_GATE:
            failed_checks.append(f"live/paper sharpe ratio {ratio:.2f} < {self.SHARPE_GATE:.2f}")

        if enforce_fund_hurdles:
            required_days = self.MIN_TRADING_DAYS.get(next_stage, 0)
            if trading_days < required_days:
                failed_checks.append(f"trading days {trading_days} < required {required_days}")
            if live_sharpe < self.LIVE_SHARPE_GATE:
                failed_checks.append(f"live sharpe {live_sharpe:.2f} < {self.LIVE_SHARPE_GATE:.2f}")
            if live_return is None:
                failed_checks.append("missing live annualized return")
            elif live_return < self.LIVE_RETURN_GATE:
                failed_checks.append(f"live return {live_return:.2%} < {self.LIVE_RETURN_GATE:.0%}")
            if live_max_drawdown is None:
                failed_checks.append("missing live max drawdown")
            elif live_max_drawdown > self.MAX_DRAWDOWN_GATE:
                failed_checks.append(f"live max drawdown {live_max_drawdown:.2%} > {self.MAX_DRAWDOWN_GATE:.0%}")
            if infrastructure_sharpe is None:
                failed_checks.append("missing infrastructure sharpe")
            elif infrastructure_sharpe < self.INFRA_SHARPE_GATE:
                failed_checks.append(
                    f"infrastructure sharpe {infrastructure_sharpe:.2f} < {self.INFRA_SHARPE_GATE:.2f}"
                )
            if reconciliation_breaks > 0:
                failed_checks.append(f"reconciliation breaks {reconciliation_breaks} > 0")
            if critical_alerts > 0:
                failed_checks.append(f"critical alerts {critical_alerts} > 0")

        allowed = len(failed_checks) == 0
        reason = "; ".join(failed_checks)

        transition = StageTransition(
            self.current_stage, next_stage,
            live_sharpe, paper_sharpe, ratio, allowed, reason,
            live_return=live_return,
            paper_return=paper_return,
            live_max_drawdown=live_max_drawdown,
            infrastructure_sharpe=infrastructure_sharpe,
            reconciliation_breaks=reconciliation_breaks,
            critical_alerts=critical_alerts,
            trading_days=trading_days,
            failed_checks=failed_checks,
        )
        return transition

    def advance(
        self,
        live_sharpe: float,
        paper_sharpe: float,
        **kwargs,
    ) -> StageTransition:
        transition = self.can_advance(live_sharpe, paper_sharpe, **kwargs)
        if transition.allowed:
            self.current_stage = transition.to_stage
        self.transitions.append(transition)
        return transition
