"""Per-regime dynamic parameter sets and smoothed posterior tracker.

Defines regime-conditional parameters for signal weights,
position sizing, stop losses, and execution mode.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RegimeParams:
    """Dynamic parameter set applied when a regime is active."""
    momentum_weight: float = 0.33
    mean_reversion_weight: float = 0.33
    ofi_weight: float = 0.33
    position_size_scalar: float = 1.0
    stop_loss_sigma: float = 2.0
    execution_mode: str = "PASSIVE_PREFERRED"  # PASSIVE_PREFERRED, PASSIVE_ONLY, REDUCE_ONLY


# Default parameter sets per regime
REGIME_PARAMS: dict[str, RegimeParams] = {
    "LOW_VOL_TRENDING": RegimeParams(
        momentum_weight=0.60,
        mean_reversion_weight=0.20,
        ofi_weight=0.20,
        position_size_scalar=1.00,
        stop_loss_sigma=3.00,
        execution_mode="PASSIVE_PREFERRED",
    ),
    "HIGH_VOL_TRENDING": RegimeParams(
        momentum_weight=0.70,
        mean_reversion_weight=0.10,
        ofi_weight=0.20,
        position_size_scalar=0.70,
        stop_loss_sigma=2.00,
        execution_mode="PASSIVE_PREFERRED",
    ),
    "MEAN_REVERTING_RANGE": RegimeParams(
        momentum_weight=0.20,
        mean_reversion_weight=0.60,
        ofi_weight=0.20,
        position_size_scalar=1.00,
        stop_loss_sigma=1.50,
        execution_mode="PASSIVE_PREFERRED",
    ),
    "HIGH_VOL_CHAOTIC": RegimeParams(
        momentum_weight=0.33,
        mean_reversion_weight=0.33,
        ofi_weight=0.33,
        position_size_scalar=0.40,
        stop_loss_sigma=2.00,
        execution_mode="PASSIVE_ONLY",
    ),
    "LIQUIDITY_CRISIS": RegimeParams(
        momentum_weight=0.00,
        mean_reversion_weight=0.00,
        ofi_weight=0.00,
        position_size_scalar=0.00,
        stop_loss_sigma=1.00,
        execution_mode="REDUCE_ONLY",
    ),
}


def get_regime_params(regime_label: str) -> RegimeParams:
    """Get parameters for a regime label. Falls back to defaults."""
    return REGIME_PARAMS.get(regime_label, RegimeParams())


class SmoothedRegimePosterior:
    """Smoothed posterior with confirmation logic.

    Prevents noisy regime oscillation by requiring confirmation steps
    before committing to a regime switch.
    """

    BETA: float = 0.10
    CONFIRMATION_STEPS: int = 10
    SWITCH_THRESHOLD: float = 0.60

    def __init__(self, n_regimes: int = 5):
        self.n_regimes = n_regimes
        self.smoothed = np.ones(n_regimes) / n_regimes
        self.confirmation_counter: dict[int, int] = defaultdict(int)
        self._current_regime: int = -1

    def update(self, raw_posterior: np.ndarray) -> np.ndarray:
        """Apply exponential smoothing to raw posterior."""
        self.smoothed = (
            self.BETA * raw_posterior
            + (1.0 - self.BETA) * self.smoothed
        )
        # Renormalize
        self.smoothed = self.smoothed / (self.smoothed.sum() + 1e-300)

        # Update confirmation counters
        dominant = int(np.argmax(self.smoothed))
        if self.smoothed[dominant] >= self.SWITCH_THRESHOLD:
            self.confirmation_counter[dominant] += 1
            # Reset other counters
            for k in list(self.confirmation_counter.keys()):
                if k != dominant:
                    self.confirmation_counter[k] = 0
        else:
            # No dominant regime — reset all
            for k in list(self.confirmation_counter.keys()):
                self.confirmation_counter[k] = 0

        return self.smoothed

    def current_regime(self) -> int:
        """Return confirmed regime ID, or -1 if uncertain."""
        dominant = int(np.argmax(self.smoothed))
        if (self.smoothed[dominant] >= self.SWITCH_THRESHOLD
                and self.confirmation_counter.get(dominant, 0) >= self.CONFIRMATION_STEPS):
            self._current_regime = dominant
            return dominant
        # Keep previous regime if one was confirmed
        if self._current_regime >= 0:
            return self._current_regime
        return -1

    def reset(self):
        self.smoothed = np.ones(self.n_regimes) / self.n_regimes
        self.confirmation_counter = defaultdict(int)
        self._current_regime = -1
