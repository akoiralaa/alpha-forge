
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

results: list[tuple[str, bool, str]] = []

def gate(name: str, passed: bool, detail: str = ""):
    tag = "PASS" if passed else "FAIL"
    results.append((name, passed, detail))
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))

from src.execution.wal import OrderState, WALEntry, WriteAheadLog
from src.execution.broker import BrokerFill, BrokerOrder, PaperBroker
from src.execution.kill_switch import KillLevel, KillSwitch
from src.execution.reconciliation import Reconciler
from src.execution.order_manager import OrderManager
from src.portfolio.risk import Portfolio, PreTradeRiskCheck

# ── 1. WAL durability ──────────────────────────────────────────

import tempfile, os
with tempfile.TemporaryDirectory() as td:
    db = os.path.join(td, "wal.db")
    wal = WriteAheadLog(db)
    wal.append(WALEntry(0, 0, "O1", OrderState.PENDING, 1, 1, "MARKET", 100))
    wal.append(WALEntry(0, 0, "O1", OrderState.FILLED, 1, 1, "MARKET", 100, filled_size=100))
    wal.close()
    # Reopen
    wal2 = WriteAheadLog(db)
    latest = wal2.get_latest_state("O1")
    durable = latest is not None and latest.state == OrderState.FILLED
    replay = wal2.replay()
    replay_ok = "O1" in replay and replay["O1"].state == OrderState.FILLED
    wal2.close()

gate("wal_durability",
     durable and replay_ok,
     f"persisted={durable}, replay={replay_ok}")

# ── 2. Kill switch flattens positions ──────────────────────────

broker = PaperBroker(slippage_bps=0, commission_per_share=0)
broker.set_price(1, 100.0)
broker.set_price(2, 50.0)
broker.submit_order(BrokerOrder("setup1", 1, 1, 200, "MARKET"))
broker.submit_order(BrokerOrder("setup2", 2, -1, 100, "MARKET"))
wal = WriteAheadLog()
ks = KillSwitch(broker, wal)

pre_pos = dict(broker.positions)
event = ks.activate(KillLevel.FLATTEN, "validation test")
post_pos = broker.get_positions()
all_flat = all(abs(v) < 1e-6 for v in post_pos.values())

gate("kill_switch_flatten",
     event.positions_flattened == 2 and all_flat,
     f"flattened={event.positions_flattened}, all_flat={all_flat}, pre={pre_pos}")

# ── 3. Kill switch blocks new orders ──────────────────────────

broker2 = PaperBroker()
broker2.set_price(1, 100.0)
wal2 = WriteAheadLog()
ks2 = KillSwitch(broker2, wal2)
ks2.activate(KillLevel.CANCEL_ONLY, "test block")
blocked = not ks2.is_order_allowed()
ks2.reset()
unblocked = ks2.is_order_allowed()

gate("kill_switch_blocks_orders",
     blocked and unblocked,
     f"blocked={blocked}, after_reset={unblocked}")

# ── 4. Drawdown auto-kill ─────────────────────────────────────

broker3 = PaperBroker()
wal3 = WriteAheadLog()
ks3 = KillSwitch(broker3, wal3, drawdown_auto_kill_pct=0.10)
no_trigger = ks3.check_drawdown(nav=950_000, peak_nav=1_000_000)
trigger = ks3.check_drawdown(nav=850_000, peak_nav=1_000_000)

gate("drawdown_auto_kill",
     no_trigger is None and trigger is not None and trigger.level == KillLevel.FLATTEN,
     f"5%dd=no_trigger, 15%dd=level={trigger.level.name if trigger else 'N/A'}")

# ── 5. Reconciliation detects breaks ─────────────────────────

broker4 = PaperBroker()
wal4 = WriteAheadLog()
# WAL says we have 100 shares, broker says 50
wal4.append(WALEntry(0, 0, "O1", OrderState.FILLED, 1, 1, "MARKET", 100, filled_size=100))
broker4.positions[1] = 50.0
recon = Reconciler(broker4, wal4)
report = recon.reconcile()
has_break = not report.is_clean and len(report.breaks) == 1
break_correct = report.breaks[0].difference == 50.0 if has_break else False

# Clean reconciliation
broker5 = PaperBroker()
wal5 = WriteAheadLog()
wal5.append(WALEntry(0, 0, "O1", OrderState.FILLED, 1, 1, "MARKET", 100, filled_size=100))
broker5.positions[1] = 100.0
recon2 = Reconciler(broker5, wal5)
report2 = recon2.reconcile()

gate("reconciliation_break_detection",
     has_break and break_correct and report2.is_clean,
     f"mismatch_detected={has_break}, diff={50 if break_correct else 'wrong'}, clean={report2.is_clean}")

# ── 6. Order manager end-to-end ───────────────────────────────

broker6 = PaperBroker(slippage_bps=0, commission_per_share=0)
broker6.set_price(1, 100.0)
wal6 = WriteAheadLog()
ks6 = KillSwitch(broker6, wal6)
risk = PreTradeRiskCheck()
port = Portfolio(nav=1_000_000)
mgr = OrderManager(broker6, wal6, ks6, risk, port)

# Normal order succeeds
order1, r1 = mgr.submit(1, 1, 100, current_price=100.0)
normal_ok = order1.state == OrderState.SUBMITTED

# Fat finger blocked
order2, r2 = mgr.submit(1, 1, 200_000, adv_20d=1_000_000)
risk_blocked = order2.state == OrderState.REJECTED and r2 is not None and not r2.passed

# Kill switch blocks
ks6.activate(KillLevel.CANCEL_ONLY, "test")
order3, _ = mgr.submit(1, 1, 100)
kill_blocked = order3.state == OrderState.REJECTED

# WAL has full history
wal_count = wal6.entry_count()

gate("order_manager_e2e",
     normal_ok and risk_blocked and kill_blocked and wal_count >= 4,
     f"normal={normal_ok}, risk_block={risk_blocked}, kill_block={kill_blocked}, wal_entries={wal_count}")

# ── 7. Self-trade prevention ──────────────────────────────────

from src.execution.self_trade import would_self_match

buy_order = BrokerOrder("new1", 1, 1, 100, "LIMIT", limit_price=50.0)
resting_sell = BrokerOrder("rest1", 1, -1, 100, "LIMIT", limit_price=49.0)
resting_buy = BrokerOrder("rest2", 1, 1, 100, "LIMIT", limit_price=48.0)

cross = would_self_match(buy_order, [resting_sell])
no_cross = would_self_match(buy_order, [resting_buy])

gate("self_trade_block",
     cross and not no_cross,
     f"crossing_blocked={cross}, same_side_ok={not no_cross}")

# ── 8. Stale data halt ───────────────────────────────────────

from src.execution.stale_data import StaleDataMonitor

sdm = StaleDataMonitor(stale_threshold_ms=100.0)
base_ns = 1_000_000_000_000
sdm.on_tick(1, base_ns)
sdm.on_tick(2, base_ns)

# Not stale yet
early_stale = sdm.check_staleness(base_ns + 50_000_000)
allowed_before = sdm.is_order_allowed(1)

# Now stale (>100ms)
later_stale = sdm.check_staleness(base_ns + 200_000_000)
blocked_after = not sdm.is_order_allowed(1)

# Tick resumes → no longer stale
sdm.on_tick(1, base_ns + 300_000_000)
resumed = sdm.is_order_allowed(1)

gate("stale_data_halt",
     len(early_stale) == 0 and allowed_before and len(later_stale) == 2 and blocked_after and resumed,
     f"early_stale={len(early_stale)}, blocked={blocked_after}, resumed={resumed}")

# ── 9. Execution mode decision ───────────────────────────────

from src.execution.execution_mode import execution_mode

passive = execution_mode(0.5, 30.0, 10.0, 0.5)
aggressive = execution_mode(0.5, 5.0, 10.0, 0.5)
high_vpin = execution_mode(0.5, 5.0, 10.0, 0.95)
block = execution_mode(0.5, 5.0, 10.0, 0.5, regime_execution_mode="REDUCE_ONLY")
passive_only = execution_mode(0.5, 5.0, 10.0, 0.5, regime_execution_mode="PASSIVE_ONLY")

gate("execution_mode_decision",
     passive == "PASSIVE" and aggressive == "AGGRESSIVE" and high_vpin == "PASSIVE"
     and block == "BLOCK" and passive_only == "PASSIVE",
     f"passive={passive}, aggressive={aggressive}, high_vpin={high_vpin}, block={block}, passive_only={passive_only}")

# ── Summary ─────────────────────────────────────────────────────

print()
n_pass = sum(1 for _, p, _ in results if p)
n_total = len(results)
tag = "ALL_PASS" if n_pass == n_total else "FAIL"
print(f"Phase 7 validation: {n_pass}/{n_total} — {tag}")
sys.exit(0 if tag == "ALL_PASS" else 1)
