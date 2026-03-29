"""Signal combiner: weighted ensemble with rolling Sharpe rebalancing."""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd

from src.signals.base import SignalModel, msv_to_features


# ── Lead-lag pairs ───────────────────────────────────────────

LEAD_LAG_PAIRS = [
    # (leader_symbol, lagger_symbol, est_lag_ms, threshold_sigmas)
    ("ES", "SPY", 10, 1.5),
    ("QQQ", "AAPL", 100, 1.5),
    ("QQQ", "MSFT", 100, 1.5),
    ("QQQ", "NVDA", 100, 1.5),
    ("CL", "XLE", 300, 1.5),
    ("ZN", "XLU", 1000, 1.5),
    ("EURUSD", "GBPUSD", 30, 1.5),
]


def lead_lag_impulse(
    leader_zscore: float,
    elapsed_ms: float,
    tau_ms: float,
    threshold: float = 1.5,
) -> float:
    """Compute lead-lag impulse with exponential decay.

    Returns signal strength in [-1, 1].
    """
    if abs(leader_zscore) < threshold:
        return 0.0
    decay = math.exp(-elapsed_ms / tau_ms) if tau_ms > 0 else 0.0
    return max(-1.0, min(1.0, leader_zscore * decay))


# ── Signal combiner ─────────────────────────────────────────

class SignalCombiner:
    """Weighted ensemble of signal models with rolling Sharpe rebalancing."""

    def __init__(
        self,
        signals: List,
        lookback_days: int = 30,
    ):
        self.signals = list(signals)
        self.lookback = lookback_days
        # Equal weight initially
        self.weights: Dict[str, float] = {
            self._key(s): 1.0 / len(signals) for s in signals
        }

    @staticmethod
    def _key(signal) -> str:
        if hasattr(signal, "name") and isinstance(signal.name, str):
            return signal.name
        if hasattr(signal, "__name__"):
            return signal.__name__
        return str(id(signal))

    def update_weights(self, recent_pnl_by_signal: Dict[str, pd.Series]):
        """Recompute weights proportional to rolling Sharpe.

        Non-performing signals (Sharpe <= 0) get weight 0.
        """
        sharpes = {}
        for key, pnl in recent_pnl_by_signal.items():
            if len(pnl) < 2:
                sharpes[key] = 0.0
                continue
            std = pnl.std()
            sr = pnl.mean() / (std + 1e-8) * np.sqrt(252)
            sharpes[key] = max(0.0, sr)

        total = sum(sharpes.values()) + 1e-8
        self.weights = {k: sharpes.get(k, 0.0) / total for k in self.weights}

    def combine(self, msv) -> float:
        """Combine all signals for a single MSV.

        Returns aggregated score.
        """
        if not getattr(msv, "valid", False):
            return 0.0

        features = msv_to_features(msv).reshape(1, -1)
        score = 0.0
        for signal in self.signals:
            key = self._key(signal)
            weight = self.weights.get(key, 0.0)
            if weight < 1e-10:
                continue

            if callable(signal) and not hasattr(signal, "predict"):
                # Tier 1 function
                s = signal(msv)
            elif hasattr(signal, "predict"):
                # Tier 2/3 model
                pred = signal.predict(features)
                s = float(pred[0]) if len(pred) > 0 else 0.0
            else:
                s = 0.0

            score += weight * s

        return max(-1.0, min(1.0, score))

    def combine_multi(self, msvs: dict) -> Dict[int, float]:
        """Combine signals for multiple symbols.

        Args:
            msvs: Dict of symbol_id -> MSV.

        Returns:
            Dict of symbol_id -> combined score.
        """
        results = {}
        for symbol_id, msv in msvs.items():
            results[symbol_id] = self.combine(msv)
        return results
