"""Phase 4 tests — Signal models."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src" / "cpp" / "build"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.signals.tier1 import (
    signal_mean_reversion,
    signal_momentum,
    signal_ofi,
    signal_volume_anomaly,
)
from src.signals.tier2 import (
    ARSignal,
    ElasticNetSignal,
    LassoSignal,
    PairsTradingSignal,
    RidgeSignal,
)
from src.signals.tier3 import LightGBMSignal, MLPSignal, RandomForestSignal
from src.signals.combiner import SignalCombiner, lead_lag_impulse
from src.signals.base import measure_half_life, msv_to_features
from tests.phase4.conftest import generate_features_and_returns, make_msv


# ── Tier 1 ───────────────────────────────────────────────────

class TestTier1:
    def test_mean_reversion_negative_zscore_gives_positive(self):
        msv = make_msv(zscore_20=-3.0, zscore_100=-2.0)
        assert signal_mean_reversion(msv) > 0

    def test_mean_reversion_positive_zscore_gives_negative(self):
        msv = make_msv(zscore_20=3.0, zscore_100=2.0)
        assert signal_mean_reversion(msv) < 0

    def test_mean_reversion_neutral(self):
        msv = make_msv(zscore_20=0.5, zscore_100=0.2)
        assert signal_mean_reversion(msv) == 0.0

    def test_momentum_positive(self):
        msv = make_msv(ret_60s=0.5, ret_300s=0.3)
        assert signal_momentum(msv) > 0

    def test_momentum_negative(self):
        msv = make_msv(ret_60s=-0.5, ret_300s=-0.3)
        assert signal_momentum(msv) < 0

    def test_momentum_clipped(self):
        msv = make_msv(ret_60s=100.0, ret_300s=100.0)
        assert signal_momentum(msv) == 1.0

    def test_ofi_positive(self):
        msv = make_msv(ofi=0.5)
        assert signal_ofi(msv) > 0

    def test_ofi_clipped(self):
        msv = make_msv(ofi=10.0)
        assert signal_ofi(msv) == 1.0

    def test_volume_anomaly_below_threshold(self):
        msv = make_msv(volume_ratio_20=1.5, ret_1s=0.01)
        assert signal_volume_anomaly(msv) == 0.0

    def test_volume_anomaly_above_threshold(self):
        msv = make_msv(volume_ratio_20=3.0, ret_1s=0.01)
        assert signal_volume_anomaly(msv) > 0

    def test_all_signals_bounded(self):
        for z in np.linspace(-5, 5, 20):
            msv = make_msv(
                zscore_20=z, zscore_100=z * 0.5,
                ret_60s=z * 0.1, ret_300s=z * 0.05,
                ofi=z * 0.1, volume_ratio_20=abs(z), ret_1s=z * 0.01,
            )
            for fn in [signal_mean_reversion, signal_momentum, signal_ofi, signal_volume_anomaly]:
                s = fn(msv)
                assert -1.0 <= s <= 1.0, f"{fn.__name__} returned {s}"

    def test_nan_handling(self):
        msv = make_msv(zscore_20=float("nan"))
        assert signal_mean_reversion(msv) == 0.0


# ── Tier 2 ───────────────────────────────────────────────────

class TestTier2:
    def test_lasso_fit_predict(self):
        X, y = generate_features_and_returns(n=500)
        m = LassoSignal()
        m.fit(X, y)
        preds = m.predict(X[:10])
        assert len(preds) == 10
        assert m.feature_importance()

    def test_ridge_fit_predict(self):
        X, y = generate_features_and_returns(n=500)
        m = RidgeSignal()
        m.fit(X, y)
        preds = m.predict(X[:10])
        assert len(preds) == 10

    def test_elasticnet_fit_predict(self):
        X, y = generate_features_and_returns(n=500)
        m = ElasticNetSignal()
        m.fit(X, y)
        preds = m.predict(X[:10])
        assert len(preds) == 10

    def test_ar_fit_predict(self):
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.standard_normal(200)) * 0.01
        m = ARSignal(p=5)
        m.fit(np.zeros((len(y), 1)), y)
        # Create lag matrix for prediction
        lags = np.column_stack([y[5 - i - 1:10 - i - 1] for i in range(5)])
        preds = m.predict(lags)
        assert len(preds) == 5

    def test_pairs_trading(self):
        rng = np.random.default_rng(42)
        n = 500
        b = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
        a = 1.5 * b + 10 + rng.standard_normal(n) * 2  # cointegrated
        X = np.column_stack([a, b])
        m = PairsTradingSignal()
        m.fit(X, np.zeros(n))
        preds = m.predict(X[:10])
        assert len(preds) == 10

    def test_predict_before_fit(self):
        m = LassoSignal()
        preds = m.predict(np.zeros((5, 19)))
        assert np.all(preds == 0)

    def test_predict_proba_range(self):
        X, y = generate_features_and_returns(n=500)
        m = RidgeSignal()
        m.fit(X, y)
        proba = m.predict_proba(X[:10])
        assert np.all(proba >= 0) and np.all(proba <= 1)


# ── Tier 3 ───────────────────────────────────────────────────

class TestTier3:
    def test_lightgbm_fit_predict(self):
        X, y = generate_features_and_returns(n=1000)
        m = LightGBMSignal(n_estimators=50)
        m.fit(X, y)
        preds = m.predict(X[:10])
        assert len(preds) == 10
        assert m.feature_importance()

    def test_random_forest_fit_predict(self):
        X, y = generate_features_and_returns(n=500)
        m = RandomForestSignal(n_estimators=10)
        m.fit(X, y)
        preds = m.predict(X[:10])
        assert len(preds) == 10

    def test_mlp_fit_predict(self):
        X, y = generate_features_and_returns(n=500)
        m = MLPSignal(hidden_layers=(32, 16), max_iter=50)
        m.fit(X, y)
        preds = m.predict(X[:10])
        assert len(preds) == 10


# ── Combiner ─────────────────────────────────────────────────

class TestCombiner:
    def test_equal_weights(self):
        signals = [signal_mean_reversion, signal_momentum]
        combiner = SignalCombiner(signals)
        # All weights should be 0.5
        for w in combiner.weights.values():
            assert abs(w - 0.5) < 1e-6

    def test_combine_returns_bounded(self):
        signals = [signal_mean_reversion, signal_momentum, signal_ofi]
        combiner = SignalCombiner(signals)
        msv = make_msv(zscore_20=-3.0, zscore_100=-2.0, ret_60s=0.5, ret_300s=0.3, ofi=0.5)
        score = combiner.combine(msv)
        assert -1.0 <= score <= 1.0

    def test_update_weights_zeros_bad_signals(self):
        signals = [signal_mean_reversion, signal_momentum]
        combiner = SignalCombiner(signals)
        # Keys match function __name__
        pnl = {
            "signal_mean_reversion": pd.Series([-0.01] * 30),  # negative
            "signal_momentum": pd.Series([0.01] * 30),  # positive
        }
        combiner.update_weights(pnl)
        assert combiner.weights["signal_mean_reversion"] == 0.0
        assert combiner.weights["signal_momentum"] > 0.0

    def test_combine_with_fitted_model(self):
        X, y = generate_features_and_returns(n=500)
        ridge = RidgeSignal()
        ridge.fit(X, y)
        signals = [signal_mean_reversion, ridge]
        combiner = SignalCombiner(signals)
        msv = make_msv(zscore_20=-2.5, zscore_100=-1.5)
        score = combiner.combine(msv)
        assert isinstance(score, float)

    def test_invalid_msv_returns_zero(self):
        combiner = SignalCombiner([signal_mean_reversion])
        msv = make_msv(valid=False)
        assert combiner.combine(msv) == 0.0


# ── Lead-lag ─────────────────────────────────────────────────

class TestLeadLag:
    def test_impulse_above_threshold(self):
        imp = lead_lag_impulse(2.0, 5.0, 10.0, threshold=1.5)
        assert imp > 0

    def test_impulse_below_threshold(self):
        imp = lead_lag_impulse(1.0, 5.0, 10.0, threshold=1.5)
        assert imp == 0.0

    def test_impulse_decays(self):
        imp1 = lead_lag_impulse(2.0, 1.0, 10.0)
        imp2 = lead_lag_impulse(2.0, 100.0, 10.0)
        assert imp1 > imp2


# ── Half-life ────────────────────────────────────────────────

class TestHalfLife:
    def test_returns_float(self):
        rng = np.random.default_rng(42)
        signal = pd.Series(rng.standard_normal(1000))
        returns = pd.Series(rng.standard_normal(1000))
        hl = measure_half_life(signal, returns, max_lag=100, step=5)
        assert isinstance(hl, float)
        assert hl > 0


# ── Feature extraction ───────────────────────────────────────

class TestFeatureExtraction:
    def test_msv_to_features(self):
        msv = make_msv()
        f = msv_to_features(msv)
        assert f.shape == (19,)
        assert not np.any(np.isnan(f))
