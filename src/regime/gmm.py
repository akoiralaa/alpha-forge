"""Gaussian Mixture Model regime detector.

Classifies market observations into regimes (e.g., low-vol, normal, crisis)
based on clustering of feature vectors: [realized_vol, return, spread, volume_ratio].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass
class GMMRegime:
    """Result of a single-step GMM classification."""
    regime_id: int                # 0-based regime label
    regime_label: str             # human-readable label
    probabilities: np.ndarray     # posterior probs per regime
    confidence: float             # max posterior probability


# Canonical ordering: regimes sorted by ascending mean volatility
# Protocol-defined 5-regime labels
REGIME_LABELS_5 = [
    "LOW_VOL_TRENDING",
    "HIGH_VOL_TRENDING",
    "MEAN_REVERTING_RANGE",
    "HIGH_VOL_CHAOTIC",
    "LIQUIDITY_CRISIS",
]

REGIME_LABELS = {
    3: ["low_vol", "normal", "high_vol"],
    4: ["low_vol", "normal", "high_vol", "crisis"],
    5: REGIME_LABELS_5,
}


def _default_labels(n: int) -> list[str]:
    if n in REGIME_LABELS:
        return REGIME_LABELS[n]
    return [f"regime_{i}" for i in range(n)]


class GMMRegimeDetector:
    """Fits a GMM to historical feature observations and classifies new ones."""

    def __init__(
        self,
        n_regimes: int = 3,
        vol_index: int = 0,
        covariance_type: str = "full",
        n_init: int = 10,
        random_state: int = 42,
    ):
        self.n_regimes = n_regimes
        self.vol_index = vol_index
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.random_state = random_state
        self.model: Optional[GaussianMixture] = None
        self.labels: list[str] = _default_labels(n_regimes)
        self._regime_order: Optional[np.ndarray] = None
        self.fitted = False

    def fit(self, X: np.ndarray) -> "GMMRegimeDetector":
        """Fit GMM to observation matrix X (n_samples, n_features).

        After fitting, regimes are re-ordered so regime 0 has the lowest
        mean volatility (column vol_index).
        """
        self.model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        self.model.fit(X)

        # Sort regimes by ascending mean volatility
        vol_means = self.model.means_[:, self.vol_index]
        self._regime_order = np.argsort(vol_means)
        self.fitted = True
        return self

    def predict(self, x: np.ndarray) -> GMMRegime:
        """Classify a single observation vector."""
        if not self.fitted or self.model is None:
            raise RuntimeError("GMMRegimeDetector not fitted")

        x2d = x.reshape(1, -1)
        raw_probs = self.model.predict_proba(x2d)[0]

        # Reorder probabilities to canonical ordering
        probs = raw_probs[self._regime_order]
        regime_raw = np.argmax(probs)
        regime_id = int(regime_raw)

        return GMMRegime(
            regime_id=regime_id,
            regime_label=self.labels[regime_id] if regime_id < len(self.labels) else f"regime_{regime_id}",
            probabilities=probs,
            confidence=float(probs[regime_id]),
        )

    def predict_batch(self, X: np.ndarray) -> list[GMMRegime]:
        """Classify multiple observations."""
        if not self.fitted or self.model is None:
            raise RuntimeError("GMMRegimeDetector not fitted")

        raw_probs = self.model.predict_proba(X)
        reordered = raw_probs[:, self._regime_order]
        results = []
        for i in range(len(X)):
            probs = reordered[i]
            regime_id = int(np.argmax(probs))
            results.append(GMMRegime(
                regime_id=regime_id,
                regime_label=self.labels[regime_id] if regime_id < len(self.labels) else f"regime_{regime_id}",
                probabilities=probs,
                confidence=float(probs[regime_id]),
            ))
        return results

    def bic(self, X: np.ndarray) -> float:
        """Bayesian Information Criterion (lower = better fit with penalty)."""
        if not self.fitted or self.model is None:
            raise RuntimeError("GMMRegimeDetector not fitted")
        return float(self.model.bic(X))

    def aic(self, X: np.ndarray) -> float:
        """Akaike Information Criterion."""
        if not self.fitted or self.model is None:
            raise RuntimeError("GMMRegimeDetector not fitted")
        return float(self.model.aic(X))
