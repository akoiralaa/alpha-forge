
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.regime.gmm import GMMRegimeDetector, REGIME_LABELS_5
from src.regime.hmm import HMMRegimeDetector
from src.regime.params import (
    REGIME_PARAMS,
    RegimeParams,
    SmoothedRegimePosterior,
    get_regime_params,
)
from src.regime.tracker import RegimeTracker

def _make_5regime_data(rng, n_per=150):
    # LOW_VOL_TRENDING: low vol, positive drift
    r0 = rng.multivariate_normal(
        [0.005, 0.001, 0.01, 1.0],
        np.diag([0.001, 0.0002, 0.001, 0.05]) ** 2, n_per)
    # HIGH_VOL_TRENDING: medium-high vol, positive drift
    r1 = rng.multivariate_normal(
        [0.020, 0.001, 0.04, 1.5],
        np.diag([0.004, 0.002, 0.005, 0.1]) ** 2, n_per)
    # MEAN_REVERTING_RANGE: medium vol, zero drift
    r2 = rng.multivariate_normal(
        [0.012, 0.0, 0.03, 1.2],
        np.diag([0.002, 0.001, 0.003, 0.08]) ** 2, n_per)
    # HIGH_VOL_CHAOTIC: high vol, negative drift
    r3 = rng.multivariate_normal(
        [0.040, -0.001, 0.08, 2.5],
        np.diag([0.008, 0.004, 0.015, 0.3]) ** 2, n_per)
    # LIQUIDITY_CRISIS: very high vol, very negative drift, wide spread
    r4 = rng.multivariate_normal(
        [0.080, -0.005, 0.20, 4.0],
        np.diag([0.015, 0.008, 0.040, 0.8]) ** 2, n_per)

    X = np.vstack([r0, r1, r2, r3, r4])
    labels = np.concatenate([np.full(n_per, i) for i in range(5)])
    return X, labels

def _make_3regime_data(rng, n_per=200):
    low = rng.multivariate_normal(
        [0.005, 0.0001, 0.01, 1.0],
        np.diag([0.001, 0.0001, 0.001, 0.01]) ** 2, n_per)
    normal = rng.multivariate_normal(
        [0.015, 0.0, 0.03, 1.5],
        np.diag([0.003, 0.001, 0.005, 0.1]) ** 2, n_per)
    high = rng.multivariate_normal(
        [0.050, -0.002, 0.10, 3.0],
        np.diag([0.010, 0.005, 0.020, 0.5]) ** 2, n_per)
    X = np.vstack([low, normal, high])
    labels = np.array([0] * n_per + [1] * n_per + [2] * n_per)
    return X, labels

# ── GMM Tests ───────────────────────────────────────────────────

class TestGMM:
    def test_fit_predict_5regime(self):
        rng = np.random.default_rng(42)
        X, _ = _make_5regime_data(rng)
        det = GMMRegimeDetector(n_regimes=5).fit(X)
        result = det.predict(X[0])
        assert 0 <= result.regime_id < 5
        assert result.regime_label in REGIME_LABELS_5

    def test_5regime_labels(self):
        rng = np.random.default_rng(42)
        X, _ = _make_5regime_data(rng)
        det = GMMRegimeDetector(n_regimes=5).fit(X)
        results = det.predict_batch(X)
        labels = {r.regime_label for r in results}
        # Should detect multiple distinct regimes
        assert len(labels) >= 3

    def test_low_vol_detected(self):
        rng = np.random.default_rng(42)
        X, _ = _make_3regime_data(rng)
        det = GMMRegimeDetector(n_regimes=3).fit(X)
        result = det.predict(X[10])
        assert result.regime_label == "low_vol"

    def test_high_vol_detected(self):
        rng = np.random.default_rng(42)
        X, _ = _make_3regime_data(rng)
        det = GMMRegimeDetector(n_regimes=3).fit(X)
        result = det.predict(X[-10])
        assert result.regime_label == "high_vol"

    def test_batch_predict(self):
        rng = np.random.default_rng(42)
        X, _ = _make_3regime_data(rng)
        det = GMMRegimeDetector(n_regimes=3).fit(X)
        results = det.predict_batch(X[:50])
        assert len(results) == 50
        assert all(abs(r.probabilities.sum() - 1.0) < 1e-6 for r in results)

    def test_bic_model_selection(self):
        rng = np.random.default_rng(42)
        X, _ = _make_3regime_data(rng)
        det1 = GMMRegimeDetector(n_regimes=1).fit(X)
        det3 = GMMRegimeDetector(n_regimes=3).fit(X)
        assert det3.bic(X) < det1.bic(X)

    def test_unfitted_raises(self):
        det = GMMRegimeDetector(n_regimes=3)
        with pytest.raises(RuntimeError):
            det.predict(np.zeros(4))

    def test_classification_accuracy_3regime(self):
        rng = np.random.default_rng(42)
        X, true_labels = _make_3regime_data(rng)
        det = GMMRegimeDetector(n_regimes=3).fit(X)
        results = det.predict_batch(X)
        pred = np.array([r.regime_id for r in results])
        accuracy = (pred == true_labels).mean()
        assert accuracy > 0.80

# ── HMM Tests ───────────────────────────────────────────────────

class TestHMM:
    def test_fit_decode(self):
        rng = np.random.default_rng(42)
        X, _ = _make_3regime_data(rng)
        det = HMMRegimeDetector(n_states=3).fit(X)
        states = det.decode(X)
        assert len(states) == len(X)

    def test_transition_matrix_valid(self):
        rng = np.random.default_rng(42)
        X, _ = _make_3regime_data(rng)
        det = HMMRegimeDetector(n_states=3).fit(X)
        T = det.transition_matrix
        np.testing.assert_allclose(T.sum(axis=1), 1.0, atol=1e-6)

    def test_regime_persistence(self):
        rng = np.random.default_rng(42)
        X, _ = _make_3regime_data(rng, n_per=200)
        det = HMMRegimeDetector(n_states=3).fit(X)
        durations = det.regime_persistence()
        assert any(d > 2 for d in durations)

    def test_stationary_distribution(self):
        rng = np.random.default_rng(42)
        X, _ = _make_3regime_data(rng)
        det = HMMRegimeDetector(n_states=3).fit(X)
        pi = det.stationary_distribution
        assert abs(pi.sum() - 1.0) < 1e-6
        assert (pi >= 0).all()

    def test_predict_next_online(self):
        rng = np.random.default_rng(42)
        X, _ = _make_3regime_data(rng)
        det = HMMRegimeDetector(n_states=3).fit(X)
        for i in range(10):
            state = det.predict_next(X[i])
            assert 0 <= state.state_id < 3

    def test_5state_hmm(self):
        rng = np.random.default_rng(42)
        X, _ = _make_5regime_data(rng)
        det = HMMRegimeDetector(n_states=5).fit(X)
        states = det.decode(X)
        assert len(states) == len(X)
        labels = {s.state_label for s in states}
        assert len(labels) >= 2

    def test_unfitted_raises(self):
        det = HMMRegimeDetector(n_states=3)
        with pytest.raises(RuntimeError):
            det.decode(np.zeros((10, 4)))

# ── SmoothedRegimePosterior Tests ────────────────────────────────

class TestSmoothedPosterior:
    def test_smoothing_reduces_oscillation(self):
        sp = SmoothedRegimePosterior(n_regimes=3)
        # Feed alternating posteriors
        switches = 0
        prev = -1
        for i in range(100):
            if i % 2 == 0:
                raw = np.array([0.8, 0.1, 0.1])
            else:
                raw = np.array([0.1, 0.8, 0.1])
            sp.update(raw)
            cur = sp.current_regime()
            if cur >= 0 and prev >= 0 and cur != prev:
                switches += 1
            if cur >= 0:
                prev = cur
        # Smoothing should prevent rapid switching
        assert switches < 10

    def test_confirmation_steps(self):
        sp = SmoothedRegimePosterior(n_regimes=3)
        # Feed consistent regime 0 — need enough steps for BETA=0.10 smoothing
        # to push smoothed[0] above SWITCH_THRESHOLD=0.60, then CONFIRMATION_STEPS=10 more
        for _ in range(50):
            sp.update(np.array([0.95, 0.025, 0.025]))
        # After many consistent steps, should be confirmed
        assert sp.current_regime() == 0

    def test_posterior_sums_to_one(self):
        sp = SmoothedRegimePosterior(n_regimes=5)
        for _ in range(20):
            raw = np.random.dirichlet(np.ones(5))
            smoothed = sp.update(raw)
            assert abs(smoothed.sum() - 1.0) < 1e-6

    def test_reset(self):
        sp = SmoothedRegimePosterior(n_regimes=3)
        for _ in range(20):
            sp.update(np.array([0.9, 0.05, 0.05]))
        sp.reset()
        assert sp.current_regime() == -1
        np.testing.assert_allclose(sp.smoothed, np.ones(3) / 3, atol=1e-6)

# ── REGIME_PARAMS Tests ─────────────────────────────────────────

class TestRegimeParams:
    def test_all_5_regimes_defined(self):
        for label in REGIME_LABELS_5:
            assert label in REGIME_PARAMS

    def test_liquidity_crisis_zero_size(self):
        p = REGIME_PARAMS["LIQUIDITY_CRISIS"]
        assert p.position_size_scalar == 0.0
        assert p.execution_mode == "REDUCE_ONLY"
        assert p.momentum_weight == 0.0

    def test_low_vol_trending_full_size(self):
        p = REGIME_PARAMS["LOW_VOL_TRENDING"]
        assert p.position_size_scalar == 1.0
        assert p.momentum_weight == 0.60

    def test_high_vol_chaotic_reduced(self):
        p = REGIME_PARAMS["HIGH_VOL_CHAOTIC"]
        assert p.position_size_scalar == 0.40
        assert p.execution_mode == "PASSIVE_ONLY"

    def test_get_regime_params_fallback(self):
        p = get_regime_params("UNKNOWN_REGIME")
        assert isinstance(p, RegimeParams)
        assert p.position_size_scalar == 1.0

# ── Tracker Tests ────────────────────────────────────────────────

class TestRegimeTracker:
    def test_fit_and_update_5regime(self):
        rng = np.random.default_rng(42)
        X, _ = _make_5regime_data(rng)
        tracker = RegimeTracker(n_regimes=5)
        tracker.fit(X)
        state = tracker.update(X[0])
        assert state.regime_id >= 0
        assert state.regime_label in REGIME_LABELS_5
        assert isinstance(state.params, RegimeParams)

    def test_params_switch_with_regime(self):
        rng = np.random.default_rng(42)
        X, _ = _make_5regime_data(rng)
        tracker = RegimeTracker(n_regimes=5)
        tracker.fit(X)
        for obs in X:
            tracker.update(obs)
        # Collect unique params seen
        param_scalars = {s.params.position_size_scalar for s in tracker.history}
        # Should see at least 2 different position scalars
        assert len(param_scalars) >= 2

    def test_crisis_gates_signals(self):
        rng = np.random.default_rng(42)
        X, _ = _make_5regime_data(rng)
        tracker = RegimeTracker(n_regimes=5)
        tracker.fit(X)
        for obs in X:
            tracker.update(obs)
        # Check that high-risk regimes reduce scale
        crisis_states = [s for s in tracker.history
                         if s.regime_label in ("HIGH_VOL_CHAOTIC", "LIQUIDITY_CRISIS")]
        if crisis_states:
            avg_scale = np.mean([s.position_scale for s in crisis_states])
            assert avg_scale <= 0.5

    def test_smoothed_posterior_stored(self):
        rng = np.random.default_rng(42)
        X, _ = _make_5regime_data(rng)
        tracker = RegimeTracker(n_regimes=5)
        tracker.fit(X)
        state = tracker.update(X[0])
        assert state.smoothed_posterior is not None
        assert abs(state.smoothed_posterior.sum() - 1.0) < 1e-6

    def test_history_tracking(self):
        rng = np.random.default_rng(42)
        X, _ = _make_5regime_data(rng)
        tracker = RegimeTracker(n_regimes=5)
        tracker.fit(X)
        for obs in X[:20]:
            tracker.update(obs)
        assert len(tracker.history) == 20
        assert tracker.current is not None

    def test_transition_count(self):
        rng = np.random.default_rng(42)
        X, _ = _make_5regime_data(rng, n_per=100)
        tracker = RegimeTracker(n_regimes=5)
        tracker.fit(X)
        for obs in X:
            tracker.update(obs)
        assert tracker.transition_count() < len(X) * 0.5

    def test_regime_durations(self):
        rng = np.random.default_rng(42)
        X, _ = _make_5regime_data(rng)
        tracker = RegimeTracker(n_regimes=5)
        tracker.fit(X)
        for obs in X:
            tracker.update(obs)
        durations = tracker.regime_durations()
        assert sum(durations.values()) == len(X)

    def test_unfitted_raises(self):
        tracker = RegimeTracker(n_regimes=5)
        with pytest.raises(RuntimeError):
            tracker.update(np.zeros(4))

    def test_confidence_range(self):
        rng = np.random.default_rng(42)
        X, _ = _make_5regime_data(rng)
        tracker = RegimeTracker(n_regimes=5)
        tracker.fit(X)
        for obs in X[:50]:
            tracker.update(obs)
        for s in tracker.history:
            assert 0.0 <= s.confidence <= 1.0

    def test_3regime_backward_compat(self):
        rng = np.random.default_rng(42)
        X, _ = _make_3regime_data(rng)
        tracker = RegimeTracker(n_regimes=3)
        tracker.fit(X)
        state = tracker.update(X[0])
        assert state.regime_id >= 0
