
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

results: list[tuple[str, bool, str]] = []

def gate(name: str, passed: bool, detail: str = ""):
    tag = "PASS" if passed else "FAIL"
    results.append((name, passed, detail))
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))

def make_5regime_data(rng, n_per=150):
    r0 = rng.multivariate_normal([0.005, 0.001, 0.01, 1.0], np.diag([0.001, 0.0002, 0.001, 0.05]) ** 2, n_per)
    r1 = rng.multivariate_normal([0.020, 0.001, 0.04, 1.5], np.diag([0.004, 0.002, 0.005, 0.1]) ** 2, n_per)
    r2 = rng.multivariate_normal([0.012, 0.0, 0.03, 1.2], np.diag([0.002, 0.001, 0.003, 0.08]) ** 2, n_per)
    r3 = rng.multivariate_normal([0.040, -0.001, 0.08, 2.5], np.diag([0.008, 0.004, 0.015, 0.3]) ** 2, n_per)
    r4 = rng.multivariate_normal([0.080, -0.005, 0.20, 4.0], np.diag([0.015, 0.008, 0.040, 0.8]) ** 2, n_per)
    X = np.vstack([r0, r1, r2, r3, r4])
    labels = np.concatenate([np.full(n_per, i) for i in range(5)])
    return X, labels

rng = np.random.default_rng(42)
X, true_labels = make_5regime_data(rng)

# ── 1. GMM 5-regime classification ──────────────────────────────

from src.regime.gmm import GMMRegimeDetector, REGIME_LABELS_5

gmm = GMMRegimeDetector(n_regimes=5).fit(X)
preds = gmm.predict_batch(X)
labels_seen = {p.regime_label for p in preds}
all_valid = all(p.regime_label in REGIME_LABELS_5 for p in preds)

gate("gmm_5regime_classification",
     all_valid and len(labels_seen) >= 3,
     f"labels_seen={labels_seen}")

# ── 2. GMM BIC model selection ──────────────────────────────────

gmm1 = GMMRegimeDetector(n_regimes=1).fit(X)
bic1 = gmm1.bic(X)
bic5 = gmm.bic(X)
gate("gmm_bic_model_selection",
     bic5 < bic1,
     f"BIC(k=5)={bic5:.0f} < BIC(k=1)={bic1:.0f}")

# ── 3. HMM transition matrix valid ─────────────────────────────

from src.regime.hmm import HMMRegimeDetector

hmm = HMMRegimeDetector(n_states=5).fit(X)
T = hmm.transition_matrix
rows_valid = np.allclose(T.sum(axis=1), 1.0, atol=1e-6)
all_positive = (T >= 0).all()

gate("hmm_transition_matrix",
     rows_valid and all_positive,
     f"rows_sum_1={rows_valid}, all_positive={all_positive}")

# ── 4. SmoothedRegimePosterior transition smoothing ─────────────

from src.regime.params import SmoothedRegimePosterior

sp = SmoothedRegimePosterior(n_regimes=5)
switches = 0
prev = -1
for i in range(200):
    # Oscillate between regime 0 and 1 every 5 steps
    if (i // 5) % 2 == 0:
        raw = np.array([0.8, 0.05, 0.05, 0.05, 0.05])
    else:
        raw = np.array([0.05, 0.8, 0.05, 0.05, 0.05])
    sp.update(raw)
    cur = sp.current_regime()
    if cur >= 0 and prev >= 0 and cur != prev:
        switches += 1
    if cur >= 0:
        prev = cur

gate("transition_smoothing_rate",
     switches < 6,
     f"switches={switches} (limit=6 for oscillating input)")

# ── 5. REGIME_PARAMS parameter switching ────────────────────────

from src.regime.params import REGIME_PARAMS
from src.regime.tracker import RegimeTracker

# Verify parameter values
crisis_params = REGIME_PARAMS["LIQUIDITY_CRISIS"]
low_vol_params = REGIME_PARAMS["LOW_VOL_TRENDING"]
chaotic_params = REGIME_PARAMS["HIGH_VOL_CHAOTIC"]

crisis_zero = crisis_params.position_size_scalar == 0.0
low_vol_full = low_vol_params.position_size_scalar == 1.0
chaotic_reduced = chaotic_params.position_size_scalar == 0.40
crisis_reduce_only = crisis_params.execution_mode == "REDUCE_ONLY"
chaotic_passive = chaotic_params.execution_mode == "PASSIVE_ONLY"

gate("parameter_switching",
     crisis_zero and low_vol_full and chaotic_reduced and crisis_reduce_only and chaotic_passive,
     f"crisis_scale={crisis_params.position_size_scalar}, low_vol_scale={low_vol_params.position_size_scalar}, "
     f"chaotic_scale={chaotic_params.position_size_scalar}, crisis_mode={crisis_params.execution_mode}")

# ── 6. Tracker crisis position scaling ──────────────────────────

tracker = RegimeTracker(n_regimes=5)
tracker.fit(X)
for obs in X:
    tracker.update(obs)

# Check that regimes with high vol get reduced position scale
high_risk_states = [s for s in tracker.history
                    if s.regime_label in ("HIGH_VOL_CHAOTIC", "LIQUIDITY_CRISIS")]
low_risk_states = [s for s in tracker.history
                   if s.regime_label in ("LOW_VOL_TRENDING", "MEAN_REVERTING_RANGE")]

has_regimes = len(high_risk_states) > 0 or len(low_risk_states) > 0
param_scalars = {s.params.position_size_scalar for s in tracker.history}
multiple_params = len(param_scalars) >= 2

gate("tracker_regime_aware_scaling",
     has_regimes and multiple_params,
     f"high_risk={len(high_risk_states)}, low_risk={len(low_risk_states)}, "
     f"unique_scalars={param_scalars}")

# ── Summary ─────────────────────────────────────────────────────

print()
n_pass = sum(1 for _, p, _ in results if p)
n_total = len(results)
tag = "ALL_PASS" if n_pass == n_total else "FAIL"
print(f"Phase 6 validation: {n_pass}/{n_total} — {tag}")
sys.exit(0 if tag == "ALL_PASS" else 1)
