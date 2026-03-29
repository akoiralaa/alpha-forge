"""Phase 5 validation gate — Portfolio, risk, sizing, HRP, attribution."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

results: list[tuple[str, bool, str]] = []


def gate(name: str, passed: bool, detail: str = ""):
    tag = "PASS" if passed else "FAIL"
    results.append((name, passed, detail))
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))


# ── 1. Volatility scaling ───────────────────────────────────────

from src.portfolio.sizing import compute_position_size

size_low_vol = compute_position_size(1, 0.5, 1_000_000, 0.005, 100.0, 0.02)
size_high_vol = compute_position_size(1, 0.5, 1_000_000, 0.005, 100.0, 0.04)
ratio = size_high_vol / size_low_vol if size_low_vol != 0 else 0
gate("volatility_scaling",
     abs(ratio - 0.5) < 0.05,
     f"2x vol → size ratio={ratio:.3f} (expect ~0.5)")


# ── 2. HRP cluster balance ──────────────────────────────────────

from src.portfolio.hrp import hrp_weights

rng = np.random.default_rng(42)
n = 200
base = rng.standard_normal(n)
corr_block = np.column_stack([base + rng.standard_normal(n) * 0.1 for _ in range(5)])
indep_block = rng.standard_normal((n, 5))
data = np.column_stack([corr_block, indep_block])
returns = pd.DataFrame(data, columns=range(10))
w = hrp_weights(returns)
cluster_weight = w.iloc[:5].sum()
weights_positive = (w >= 0).all()
weights_sum_one = abs(w.sum() - 1.0) < 1e-6

gate("hrp_cluster_balance",
     cluster_weight < 0.55 and weights_positive and weights_sum_one,
     f"corr cluster weight={cluster_weight:.3f}, all positive={weights_positive}, sum={w.sum():.6f}")


# ── 3. Drawdown circuit breaker ─────────────────────────────────

from src.portfolio.risk import (
    Portfolio, OrderIntent, PreTradeRiskCheck, RiskCheckReason,
    cluster_exposure_check,
)

check = PreTradeRiskCheck()
port_dd1 = Portfolio(nav=900_000, peak_nav=1_000_000)
order = OrderIntent(symbol_id=1, side=1, size=100, current_price=100.0, adv_20d=1_000_000)
r1 = check.check(order, port_dd1)

port_dd2 = Portfolio(nav=800_000, peak_nav=1_000_000)
r2 = check.check(order, port_dd2)

gate("drawdown_circuit_breaker",
     not r1.passed and r1.reason == RiskCheckReason.DRAWDOWN_LEVEL1
     and not r2.passed and r2.reason == RiskCheckReason.DRAWDOWN_LEVEL2,
     f"L1={r1.reason.value}, L2={r2.reason.value}")


# ── 4. VPIN halt ────────────────────────────────────────────────

port_vpin = Portfolio(nav=1_000_000, current_vpin=0.91)
market_o = OrderIntent(symbol_id=1, side=1, size=100, order_type="MARKET",
                       current_price=100.0, adv_20d=1_000_000)
limit_o = OrderIntent(symbol_id=1, side=1, size=100, order_type="LIMIT",
                      limit_price=100.0, current_mid=100.0,
                      current_price=100.0, adv_20d=1_000_000)
r_mkt = check.check(market_o, port_vpin)
r_lmt = check.check(limit_o, port_vpin)

gate("vpin_halt",
     not r_mkt.passed and r_mkt.reason == RiskCheckReason.VPIN_HALT and r_lmt.passed,
     f"market={r_mkt.reason.value}, limit_passed={r_lmt.passed}")


# ── 5. Fat finger block ────────────────────────────────────────

port_ff = Portfolio(nav=1_000_000)
fat_size = OrderIntent(symbol_id=1, side=1, size=110_000, adv_20d=1_000_000,
                       current_price=100.0)
fat_price = OrderIntent(symbol_id=1, side=1, size=100, order_type="LIMIT",
                        limit_price=106.0, current_mid=100.0,
                        current_price=100.0, adv_20d=1_000_000)
r_fs = check.check(fat_size, port_ff)
r_fp = check.check(fat_price, port_ff)

gate("fat_finger_block",
     not r_fs.passed and r_fs.reason == RiskCheckReason.FAT_FINGER_SIZE
     and not r_fp.passed and r_fp.reason == RiskCheckReason.FAT_FINGER_PRICE,
     f"size={r_fs.reason.value}, price={r_fp.reason.value}")


# ── 6. Kelly constraint ────────────────────────────────────────

size_unconstrained = compute_position_size(1, 0.5, 1_000_000, 0.005, 100.0, 0.02)
size_constrained = compute_position_size(1, 0.1, 1_000_000, 0.005, 100.0, 0.02, kelly_fraction=0.25)
vol_scaled_max = int(1_000_000 * 0.005 / (100.0 * 0.02))

gate("kelly_constraint",
     abs(size_constrained) <= vol_scaled_max and size_constrained > 0,
     f"constrained={size_constrained}, vol_max={vol_scaled_max}")


# ── Summary ─────────────────────────────────────────────────────

print()
n_pass = sum(1 for _, p, _ in results if p)
n_total = len(results)
tag = "ALL_PASS" if n_pass == n_total else "FAIL"
print(f"Phase 5 validation: {n_pass}/{n_total} — {tag}")
sys.exit(0 if tag == "ALL_PASS" else 1)
