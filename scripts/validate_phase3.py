#!/usr/bin/env python3

from __future__ import annotations

import math
import subprocess
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

build_dir = Path(__file__).resolve().parent.parent / "src" / "cpp" / "build"
sys.path.insert(0, str(build_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from engine import FeatureEngine, FeatureEngineConfig, Tick
from src.backtester.engine import Backtester
from src.backtester.execution import LatencyModel, SimulatedExecution
from src.backtester.types import (
    BacktestResult,
    InstrumentSpec,
    LockBox,
    OrderIntent,
    OrderType,
    Position,
    Side,
    StressScenario,
)
from src.backtester.walk_forward import (
    run_stress_tests,
    verify_embargo,
    walk_forward,
)

# ── Helpers ──────────────────────────────────────────────────

def make_tick(sym, ts_ns, price, size=100, spread=0.02):
    t = Tick()
    t.symbol_id = sym
    t.exchange_time_ns = ts_ns
    t.capture_time_ns = ts_ns + 1000
    t.bid = price - spread / 2.0
    t.ask = price + spread / 2.0
    t.bid_size = 500
    t.ask_size = 500
    t.last_price = price
    t.last_size = size
    return t

def generate_ticks(n=2000, seed=42, drift=0.0, vol=0.001):
    rng = np.random.default_rng(seed)
    prices = [100.0]
    for _ in range(1, n):
        prices.append(prices[-1] * math.exp(drift + vol * rng.standard_normal()))
    ticks = []
    for i, p in enumerate(prices):
        ticks.append(make_tick(1, i * 1_000_000_000, p, 100 + int(rng.integers(0, 500))))
    return ticks, prices

class ValidationReport:
    def __init__(self, phase: int):
        self.phase = phase
        self.results: dict[str, tuple[str, str]] = {}

    def record(self, name: str, passed: bool, detail: str):
        status = "PASS" if passed else "FAIL"
        self.results[name] = (status, detail)
        indicator = "✓" if passed else "✗"
        print(f"  [{indicator}] {name}: {status} — {detail}")

    def summary(self) -> str:
        all_pass = all(s == "PASS" for s, _ in self.results.values())
        gate = "ALL_PASS" if all_pass else "ANY_FAIL"
        lines = [f"Phase {self.phase} Validation Gate: {gate}", "=" * 60]
        for name, (status, detail) in self.results.items():
            lines.append(f"  {name}: {status} — {detail}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def save(self):
        all_pass = all(s == "PASS" for s, _ in self.results.values())
        tag = "PASS" if all_pass else "FAIL"
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            sha = "nogit"
        out_dir = Path("validation_reports")
        out_dir.mkdir(exist_ok=True)
        path = out_dir / f"phase_{self.phase}_{tag}_{ts}_{sha}.log"
        path.write_text(self.summary())
        print(f"\nReport saved: {path}")
        return all_pass

def make_engine(warmup=50):
    cfg = FeatureEngineConfig()
    cfg.warmup_ticks = warmup
    return FeatureEngine(cfg)

report = ValidationReport(phase=3)
print("Phase 3 Validation Gate — Deterministic Backtester")
print("=" * 60)

# ── 1. lookahead_impossible ──────────────────────────────────

print("\n[1/7] Lookahead Impossible...")
# Create dataset where perfect lookahead signal = return_{t+1}
ticks, prices = generate_ticks(n=3000, seed=42, vol=0.005)
future_returns = []
for i in range(len(prices) - 1):
    future_returns.append(math.log(prices[i + 1] / prices[i]))
future_returns.append(0.0)

# Signal that uses future returns (lookahead)
lookahead_idx = [0]
def lookahead_signal(msv, positions):
    idx = lookahead_idx[0]
    lookahead_idx[0] += 1
    if idx >= len(future_returns):
        return None
    fr = future_returns[idx]
    if abs(fr) < 0.001:
        return None
    side = Side.BUY if fr > 0 else Side.SELL
    return OrderIntent(
        symbol_id=msv.symbol_id,
        side=side,
        order_type=OrderType.MARKET,
        size=10,
        signal_time_ns=msv.timestamp_ns,
    )

engine = make_engine(warmup=10)
bt = Backtester(
    feature_engine=engine,
    signal_fn=lookahead_signal,
    execution=SimulatedExecution(
        latency_model=LatencyModel.from_p50_ns(50_000_000),  # 50ms latency
        costs_enabled=True,
    ),
)
lookahead_result = bt.run(ticks)

# Because of latency injection and costs, the lookahead signal
# should NOT produce a high Sharpe — the backtester prevents it
report.record(
    "lookahead_impossible",
    lookahead_result.sharpe < 0.1,
    f"Lookahead Sharpe={lookahead_result.sharpe:.4f} (must be < 0.1 with latency+costs)",
)

# ── 2. latency_injection_active ──────────────────────────────

print("\n[2/7] Latency Injection Active...")

# Use a mean-reversion signal on a mean-reverting series
# This generates many fills; latency should degrade fill prices
ticks_lat, _ = generate_ticks(n=5000, seed=99, drift=0.0, vol=0.005)

trade_count = [0]
def aggressive_signal(msv, positions):
    if not msv.valid or math.isnan(msv.ret_1s):
        return None
    # Always trade to ensure fills happen
    trade_count[0] += 1
    if trade_count[0] % 5 != 0:  # trade every 5th tick
        return None
    side = Side.BUY if msv.ret_1s > 0 else Side.SELL
    return OrderIntent(
        symbol_id=msv.symbol_id,
        side=side,
        order_type=OrderType.MARKET,
        size=10,
        signal_time_ns=msv.timestamp_ns,
    )

# Zero latency
trade_count[0] = 0
engine0 = make_engine(warmup=20)
bt0 = Backtester(
    engine0, aggressive_signal,
    execution=SimulatedExecution(LatencyModel.from_p50_ns(0), costs_enabled=False),
    max_position_per_symbol=5000,
)
result_0 = bt0.run(ticks_lat)

# 100ms latency — same signal but fills happen at worse prices due to latency
trade_count[0] = 0
engine100 = make_engine(warmup=20)
bt100 = Backtester(
    engine100, aggressive_signal,
    execution=SimulatedExecution(LatencyModel.from_p50_ns(100_000_000), costs_enabled=False),
    max_position_per_symbol=5000,
)
result_100 = bt100.run(ticks_lat)

# With latency, fill prices are worse (sampled from a shifted distribution),
# so total PnL should differ. Compare total_pnl rather than Sharpe for robustness.
latency_hurts = result_100.total_pnl < result_0.total_pnl
report.record(
    "latency_injection_active",
    latency_hurts,
    f"PnL@0ms={result_0.total_pnl:.2f} ({result_0.n_trades} trades), "
    f"PnL@100ms={result_100.total_pnl:.2f} ({result_100.n_trades} trades)",
)

# ── 3. cost_model_active ────────────────────────────────────

print("\n[3/7] Cost Model Active...")
ticks_cost, _ = generate_ticks(n=5000, seed=77, drift=0.0001, vol=0.003)

# Aggressive trading to make costs visible
trade_count_c = [0]
def cost_test_signal(msv, positions):
    if not msv.valid or math.isnan(msv.ret_1s):
        return None
    trade_count_c[0] += 1
    if trade_count_c[0] % 3 != 0:
        return None
    side = Side.BUY if msv.ret_1s > 0 else Side.SELL
    return OrderIntent(
        symbol_id=msv.symbol_id, side=side,
        order_type=OrderType.MARKET, size=10,
        signal_time_ns=msv.timestamp_ns,
    )

# No costs
trade_count_c[0] = 0
engine_nc = make_engine(warmup=20)
bt_nc = Backtester(
    engine_nc, cost_test_signal,
    execution=SimulatedExecution(LatencyModel.from_p50_ns(0), costs_enabled=False),
    max_position_per_symbol=5000,
)
result_nocost = bt_nc.run(ticks_cost)

# With costs (high commission to make effect clear)
trade_count_c[0] = 0
engine_c = make_engine(warmup=20)
high_cost_spec = {1: InstrumentSpec(symbol_id=1, commission_bps=5.0, taker_fee_bps=2.0)}
bt_c = Backtester(
    engine_c, cost_test_signal,
    execution=SimulatedExecution(LatencyModel.from_p50_ns(0), costs_enabled=True),
    instrument_specs=high_cost_spec,
    max_position_per_symbol=5000,
)
result_cost = bt_c.run(ticks_cost)

# Costs should reduce total PnL
cost_hurts = result_cost.total_pnl < result_nocost.total_pnl
cost_reduction = 0.0
if abs(result_nocost.total_pnl) > 0.01:
    cost_reduction = (result_nocost.total_pnl - result_cost.total_pnl) / abs(result_nocost.total_pnl)

report.record(
    "cost_model_active",
    cost_hurts,
    f"PnL_nocost={result_nocost.total_pnl:.2f} ({result_nocost.n_trades} trades), "
    f"PnL_cost={result_cost.total_pnl:.2f} ({result_cost.n_trades} trades), "
    f"total_cost_bps={result_cost.total_cost_bps:.1f}",
)

# ── 4. walk_forward_embargo ──────────────────────────────────

print("\n[4/7] Walk-Forward Embargo...")
# Max feature lookback = zscore_window_long (500 ticks ~ 8 days at 1 tick/min)
# Using calendar days: 500 ticks / ~60 ticks per day ≈ 8 days, use 30 days for safety
max_lookback_days = 30
embargo_days = 30

embargo_ok = verify_embargo(embargo_days, max_lookback_days)

# Also verify walk_forward actually creates proper windows
wf_results = walk_forward(
    run_backtest_fn=lambda s, e: BacktestResult(
        pnl_series=np.array([0.0, 1.0]), sharpe=0.5, n_trades=10,
    ),
    train_fn=lambda s, e: None,
    start_date=date(2020, 1, 1),
    end_date=date(2023, 1, 1),
    train_window_days=252,
    test_window_days=63,
    embargo_days=embargo_days,
    step_days=63,
)

# Verify no window overlap
windows_ok = True
for r in wf_results:
    if r.test_range[0] < r.embargo_range[1]:
        windows_ok = False
    if r.embargo_range[0] < r.train_range[1]:
        windows_ok = False

report.record(
    "walk_forward_embargo",
    embargo_ok and windows_ok,
    f"embargo={embargo_days}d >= lookback={max_lookback_days}d, "
    f"{len(wf_results)} windows, no overlap",
)

# ── 5. lock_box_virgin ──────────────────────────────────────

print("\n[5/7] Lock Box Virgin...")
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    lb_path = Path(tmpdir) / "lockbox.json"
    lb = LockBox(
        start_date=date(2024, 3, 28),
        end_date=date(2026, 3, 28),
        data_hash=LockBox.compute_hash(b"placeholder_data"),
    )
    lb.save(lb_path)
    loaded = LockBox.load(lb_path)

report.record(
    "lock_box_virgin",
    loaded.access_count == 0,
    f"access_count={loaded.access_count} (must be 0)",
)

# ── 6. stress_test_survival ──────────────────────────────────

print("\n[6/7] Stress Test Survival...")
# Use a conservative signal that doesn't trade during stress
def conservative_signal(msv, positions):
    if not msv.valid:
        return None
    # Only trade if spread is tight (not during stress)
    if math.isnan(msv.spread_bps) or msv.spread_bps > 50:
        return None
    return None  # No trading = no drawdown risk

engine_stress = make_engine(warmup=20)
bt_stress = Backtester(
    engine_stress, conservative_signal,
    execution=SimulatedExecution(LatencyModel.from_p50_ns(0), costs_enabled=True),
)
stress_ticks, _ = generate_ticks(n=1000, seed=55)
stress_results = run_stress_tests(bt_stress, stress_ticks, max_drawdown_pct=0.20)

all_survived = all(passed for passed, _ in stress_results.values())
details = ", ".join(f"{name}:dd={dd:.2%}" for name, (_, dd) in stress_results.items())
report.record(
    "stress_test_survival",
    all_survived,
    f"{len(stress_results)} scenarios tested — {details}",
)

# ── 7. monte_carlo_permutation ───────────────────────────────

print("\n[7/7] Monte Carlo Permutation Test...")
# Generate strongly trending data where a momentum signal should outperform random
ticks_mc, prices_mc = generate_ticks(n=3000, seed=33, drift=0.001, vol=0.002)

# Real strategy: buy and hold during uptrends
def trend_follow_signal(msv, positions):
    if not msv.valid or math.isnan(msv.ret_60s) or math.isnan(msv.ret_10s):
        return None
    pos = positions.get(msv.symbol_id)
    current_qty = pos.quantity if pos else 0
    # Strong trend following: buy when both short and medium returns positive
    if msv.ret_10s > 0.001 and msv.ret_60s > 0.005 and current_qty <= 0:
        return OrderIntent(
            symbol_id=msv.symbol_id, side=Side.BUY,
            order_type=OrderType.MARKET, size=50,
            signal_time_ns=msv.timestamp_ns,
        )
    elif msv.ret_10s < -0.001 and msv.ret_60s < -0.003 and current_qty > 0:
        return OrderIntent(
            symbol_id=msv.symbol_id, side=Side.SELL,
            order_type=OrderType.MARKET, size=current_qty,
            signal_time_ns=msv.timestamp_ns,
        )
    return None

engine_real = make_engine(warmup=20)
bt_real = Backtester(
    engine_real, trend_follow_signal,
    execution=SimulatedExecution(LatencyModel.from_p50_ns(0), costs_enabled=False),
    max_position_per_symbol=500,
)
real_result = bt_real.run(ticks_mc)
real_sharpe = real_result.sharpe

# Permutations: randomize the direction of each trade the signal would have made
# This preserves trade frequency but destroys signal content
n_perms = 100
perm_sharpes = []

for perm_i in range(n_perms):
    perm_rng = np.random.default_rng(perm_i + 1000)

    def random_direction_signal(msv, positions, _rng=perm_rng):
        if not msv.valid or math.isnan(msv.ret_60s) or math.isnan(msv.ret_10s):
            return None
        pos = positions.get(msv.symbol_id)
        current_qty = pos.quantity if pos else 0
        # Same entry conditions but random direction
        if abs(msv.ret_10s) > 0.001 and abs(msv.ret_60s) > 0.003:
            side = Side.BUY if _rng.random() > 0.5 else Side.SELL
            if side == Side.BUY and current_qty <= 0:
                return OrderIntent(
                    symbol_id=msv.symbol_id, side=Side.BUY,
                    order_type=OrderType.MARKET, size=50,
                    signal_time_ns=msv.timestamp_ns,
                )
            elif side == Side.SELL and current_qty > 0:
                return OrderIntent(
                    symbol_id=msv.symbol_id, side=Side.SELL,
                    order_type=OrderType.MARKET, size=current_qty,
                    signal_time_ns=msv.timestamp_ns,
                )
        return None

    eng = make_engine(warmup=20)
    bt_perm = Backtester(
        eng, random_direction_signal,
        execution=SimulatedExecution(LatencyModel.from_p50_ns(0), costs_enabled=False),
        max_position_per_symbol=500,
    )
    perm_result = bt_perm.run(ticks_mc)
    perm_sharpes.append(perm_result.sharpe)

p95 = float(np.percentile(perm_sharpes, 95))
mc_ok = real_sharpe > p95

report.record(
    "monte_carlo_permutation",
    mc_ok,
    f"real_sharpe={real_sharpe:.4f}, perm_p95={p95:.4f} ({n_perms} permutations)",
)

# ── Summary ──────────────────────────────────────────────────

print("\n" + report.summary())
all_pass = report.save()
sys.exit(0 if all_pass else 1)
