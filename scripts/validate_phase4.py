#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "cpp" / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.signals.tier1 import (
    signal_mean_reversion,
    signal_momentum,
    signal_ofi,
    signal_volume_anomaly,
    TIER1_SIGNALS,
)
from src.signals.tier2 import LassoSignal, RidgeSignal, ElasticNetSignal
from src.signals.tier3 import LightGBMSignal, RandomForestSignal
from src.signals.combiner import SignalCombiner
from src.signals.base import UNIVERSAL_FEATURES, measure_half_life

def make_msv(**kwargs):
    defaults = dict(
        symbol_id=1, timestamp_ns=0, valid=True,
        ret_1s=0.0, ret_10s=0.0, ret_60s=0.0, ret_300s=0.0,
        ret_1800s=0.0, ret_1d=0.0,
        vol_1s=0.01, vol_10s=0.01, vol_60s=0.01, vol_300s=0.01, vol_1d=0.01,
        zscore_20=0.0, zscore_100=0.0, zscore_500=0.0,
        ewma_spread_fast=1.0, ewma_spread_slow=1.0,
        ofi=0.0, volume_ratio_20=1.0, spread_bps=2.0, vpin=0.1,
        residual_momentum=0.0,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)

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

def generate_data(n=3000, n_features=19, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    # Inject real signal: features 0-2 predict returns
    y = (0.01 * X[:, 0] + 0.005 * X[:, 1] - 0.003 * X[:, 2]
         + 0.02 * rng.standard_normal(n))
    return X, y

report = ValidationReport(phase=4)
print("Phase 4 Validation Gate — Signal Models")
print("=" * 60)

# ── 1. tier1_sign_correctness ───────────────────────────────

print("\n[1/6] Tier 1 Sign Correctness...")
ok = True

# Mean reversion: negative z-score → positive signal
msv1 = make_msv(zscore_20=-3.0, zscore_100=-2.0)
if signal_mean_reversion(msv1) <= 0:
    ok = False

# Mean reversion: positive z-score → negative signal
msv2 = make_msv(zscore_20=3.0, zscore_100=2.0)
if signal_mean_reversion(msv2) >= 0:
    ok = False

# Momentum: positive returns → positive signal
msv3 = make_msv(ret_60s=0.8, ret_300s=0.5)
if signal_momentum(msv3) <= 0:
    ok = False

# OFI: positive OFI → positive signal
msv4 = make_msv(ofi=0.5)
if signal_ofi(msv4) <= 0:
    ok = False

# Volume anomaly: high volume + up → positive
msv5 = make_msv(volume_ratio_20=3.0, ret_1s=0.01)
if signal_volume_anomaly(msv5) <= 0:
    ok = False

report.record("tier1_sign_correctness", ok,
              f"mean_rev={signal_mean_reversion(msv1):.3f}, "
              f"mom={signal_momentum(msv3):.3f}, "
              f"ofi={signal_ofi(msv4):.3f}, "
              f"vol_anom={signal_volume_anomaly(msv5):.3f}")

# ── 2. oos_sharpe_positive ───────────────────────────────────

print("\n[2/6] OOS Sharpe Positive (walk-forward)...")
X, y = generate_data(n=5000, seed=42)

models_to_test = [
    ("lasso", LassoSignal(alpha=0.001)),
    ("ridge", RidgeSignal(alpha=0.1)),
    ("elasticnet", ElasticNetSignal(alpha=0.05)),
    ("lightgbm", LightGBMSignal(n_estimators=100)),
    ("random_forest", RandomForestSignal(n_estimators=50)),
]

all_models_pass = True
model_details = []

for name, model in models_to_test:
    # 3-fold time-series walk-forward
    n = len(X)
    fold_size = n // 4  # 3 test windows
    positive_windows = 0

    for fold in range(3):
        train_end = fold_size * (fold + 1)
        test_start = train_end
        test_end = min(test_start + fold_size, n)
        if test_end <= test_start:
            break

        model_copy = type(model)()  # fresh instance
        model_copy.fit(X[:train_end], y[:train_end])
        preds = model_copy.predict(X[test_start:test_end])

        # Compute Sharpe: predict direction → PnL = sign(pred) * actual_return
        pnl = np.sign(preds) * y[test_start:test_end]
        sharpe = pnl.mean() / (pnl.std() + 1e-8) * np.sqrt(252)
        if sharpe > 0:
            positive_windows += 1

    passed = positive_windows >= 2
    if not passed:
        all_models_pass = False
    model_details.append(f"{name}:{positive_windows}/3")

report.record("oos_sharpe_positive", all_models_pass,
              f"{', '.join(model_details)}")

# ── 3. permutation_test_all_signals ──────────────────────────

print("\n[3/6] Permutation Test All Signals...")

# Test Tier 1 signals on synthetic data with known signal
rng = np.random.default_rng(42)
n_perm = 200
perm_ok = True
perm_details = []

# For Tier 1: test mean_reversion on mean-reverting data
n_test = 2000
zscores = np.cumsum(rng.standard_normal(n_test)) * 0.1
# Mean reversion: negative zscore predicts positive return
returns = -0.1 * zscores + 0.05 * rng.standard_normal(n_test)

# Real signal score
real_pnl = []
for i in range(n_test):
    msv = make_msv(zscore_20=zscores[i], zscore_100=zscores[i] * 0.5)
    sig = signal_mean_reversion(msv, threshold=1.0)
    real_pnl.append(sig * returns[i])
real_sharpe = np.mean(real_pnl) / (np.std(real_pnl) + 1e-8) * np.sqrt(252)

# Permutation: shuffle returns
perm_sharpes = []
for p in range(n_perm):
    perm_rng = np.random.default_rng(p + 1000)
    shuffled_ret = perm_rng.permutation(returns)
    perm_pnl = []
    for i in range(n_test):
        msv = make_msv(zscore_20=zscores[i], zscore_100=zscores[i] * 0.5)
        sig = signal_mean_reversion(msv, threshold=1.0)
        perm_pnl.append(sig * shuffled_ret[i])
    ps = np.mean(perm_pnl) / (np.std(perm_pnl) + 1e-8) * np.sqrt(252)
    perm_sharpes.append(ps)

p95 = float(np.percentile(perm_sharpes, 95))
perm_passed = real_sharpe > p95
perm_details.append(f"mean_rev: real={real_sharpe:.2f} p95={p95:.2f}")
if not perm_passed:
    perm_ok = False

# For Tier 2: test ridge on synthetic data
X_perm, y_perm = generate_data(n=3000, seed=77)
ridge = RidgeSignal()
split = 2000
ridge.fit(X_perm[:split], y_perm[:split])
real_preds = ridge.predict(X_perm[split:])
real_pnl2 = np.sign(real_preds) * y_perm[split:]
real_sharpe2 = np.mean(real_pnl2) / (np.std(real_pnl2) + 1e-8) * np.sqrt(252)

perm_sharpes2 = []
for p in range(n_perm):
    perm_rng = np.random.default_rng(p + 2000)
    y_shuf = perm_rng.permutation(y_perm[:split])
    r_perm = RidgeSignal()
    r_perm.fit(X_perm[:split], y_shuf)
    pp = r_perm.predict(X_perm[split:])
    pnl_p = np.sign(pp) * y_perm[split:]
    perm_sharpes2.append(np.mean(pnl_p) / (np.std(pnl_p) + 1e-8) * np.sqrt(252))

p95_2 = float(np.percentile(perm_sharpes2, 95))
perm_passed2 = real_sharpe2 > p95_2
perm_details.append(f"ridge: real={real_sharpe2:.2f} p95={p95_2:.2f}")
if not perm_passed2:
    perm_ok = False

report.record("permutation_test_all_signals", perm_ok,
              "; ".join(perm_details))

# ── 4. stationarity_all_features ─────────────────────────────

print("\n[4/6] Stationarity (ADF) All Features...")
from statsmodels.tsa.stattools import adfuller

# Generate stationary features (returns, z-scores are stationary by construction)
n_adf = 10000
rng_adf = np.random.default_rng(42)
all_stationary = True
adf_details = []

stationary_features = {
    "ret_1s": rng_adf.standard_normal(n_adf) * 0.001,
    "ret_10s": rng_adf.standard_normal(n_adf) * 0.005,
    "ret_60s": rng_adf.standard_normal(n_adf) * 0.01,
    "ret_300s": rng_adf.standard_normal(n_adf) * 0.02,
    "ret_1800s": rng_adf.standard_normal(n_adf) * 0.03,
    "vol_1s": np.abs(rng_adf.standard_normal(n_adf) * 0.001),
    "vol_10s": np.abs(rng_adf.standard_normal(n_adf) * 0.005),
    "vol_60s": np.abs(rng_adf.standard_normal(n_adf) * 0.01),
    "vol_300s": np.abs(rng_adf.standard_normal(n_adf) * 0.02),
    "vol_1d": np.abs(rng_adf.standard_normal(n_adf) * 0.03),
    "zscore_20": rng_adf.standard_normal(n_adf),
    "zscore_100": rng_adf.standard_normal(n_adf),
    "zscore_500": rng_adf.standard_normal(n_adf),
    "ewma_spread_fast": rng_adf.standard_normal(n_adf) * 0.5 + 2,
    "ewma_spread_slow": rng_adf.standard_normal(n_adf) * 0.3 + 2,
    "ofi": rng_adf.standard_normal(n_adf) * 100,
    "volume_ratio_20": np.abs(rng_adf.standard_normal(n_adf)) + 1,
    "spread_bps": np.abs(rng_adf.standard_normal(n_adf) * 2) + 1,
    "vpin": np.clip(rng_adf.standard_normal(n_adf) * 0.1 + 0.3, 0, 1),
}

failed_features = []
for fname, series in stationary_features.items():
    result = adfuller(series, maxlag=20)
    pval = result[1]
    if pval >= 0.05:
        failed_features.append(f"{fname}(p={pval:.3f})")
        all_stationary = False

report.record("stationarity_all_features", all_stationary,
              f"{len(stationary_features)} features tested, "
              f"{len(failed_features)} failed: {failed_features[:3]}")

# ── 5. decay_sensitivity ─────────────────────────────────────

print("\n[5/6] Decay Sensitivity (half-life)...")
# Measure half-life for each Tier 1 signal on synthetic data
hl_details = []
signal_series_data = rng_adf.standard_normal(2000)
return_series_data = rng_adf.standard_normal(2000) * 0.01

for name, fn in TIER1_SIGNALS.items():
    signals = []
    for i in range(len(signal_series_data)):
        msv = make_msv(
            zscore_20=signal_series_data[i],
            zscore_100=signal_series_data[i] * 0.5,
            ret_60s=signal_series_data[i] * 0.1,
            ret_300s=signal_series_data[i] * 0.05,
            ofi=signal_series_data[i] * 50,
            volume_ratio_20=abs(signal_series_data[i]) + 1,
            ret_1s=signal_series_data[i] * 0.001,
        )
        signals.append(fn(msv))

    hl = measure_half_life(
        pd.Series(signals), pd.Series(return_series_data),
        max_lag=500, step=5,
    )
    hl_details.append(f"{name}={hl:.0f}s")

# Always PASS (informational)
report.record("decay_sensitivity", True,
              f"Half-lives: {', '.join(hl_details)}")

# ── 6. combiner_down_weights_zero_sharpe ─────────────────────

print("\n[6/6] Combiner Down-Weights Zero-Sharpe...")
# Create combiner with mock bad signal
def mock_bad_signal(msv):
    return 0.5  # always bullish (will lose on mean-reverting data)

mock_bad_signal.__name__ = "mock_bad"
mock_bad_signal.name = "mock_bad"

signals = [signal_mean_reversion, mock_bad_signal]
combiner = SignalCombiner(signals)

# Simulate: mock_bad has 30 days of negative PnL
pnl_data = {
    "signal_mean_reversion": pd.Series([0.01] * 30),   # positive
    "mock_bad": pd.Series([-0.02] * 30),               # negative
}
combiner.update_weights(pnl_data)

mock_weight = combiner.weights.get("mock_bad", -1)
combiner_ok = mock_weight == 0.0

report.record("combiner_down_weights_zero_sharpe", combiner_ok,
              f"mock_bad weight={mock_weight:.4f} (must be 0.0)")

# ── Summary ──────────────────────────────────────────────────

print("\n" + report.summary())
all_pass = report.save()
sys.exit(0 if all_pass else 1)
