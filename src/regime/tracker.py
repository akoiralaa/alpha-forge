
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.regime.gmm import GMMRegimeDetector, GMMRegime
from src.regime.hmm import HMMRegimeDetector, HMMState
from src.regime.params import (
    REGIME_PARAMS,
    RegimeParams,
    SmoothedRegimePosterior,
    get_regime_params,
)

@dataclass
class RegimeState:
    regime_id: int
    regime_label: str
    gmm_regime: GMMRegime
    hmm_state: HMMState
    agreement: bool             # GMM and HMM agree on regime
    position_scale: float       # multiplicative scale for position sizing [0, 1]
    signal_gate_open: bool      # whether new signals should be acted on
    confidence: float           # combined confidence
    params: RegimeParams = field(default_factory=RegimeParams)
    smoothed_posterior: Optional[np.ndarray] = None

class RegimeTracker:

    def __init__(
        self,
        n_regimes: int = 5,
        vol_index: int = 0,
        gmm_weight: float = 0.4,
        hmm_weight: float = 0.6,
        disagreement_scale: float = 0.5,
    ):
        self.n_regimes = n_regimes
        self.gmm_weight = gmm_weight
        self.hmm_weight = hmm_weight
        self.disagreement_scale = disagreement_scale

        self.gmm = GMMRegimeDetector(n_regimes=n_regimes, vol_index=vol_index)
        self.hmm = HMMRegimeDetector(n_states=n_regimes, vol_index=vol_index)
        self.smoother = SmoothedRegimePosterior(n_regimes=n_regimes)
        self.fitted = False
        self._history: list[RegimeState] = []

    def fit(self, X: np.ndarray) -> "RegimeTracker":
        self.gmm.fit(X)
        self.hmm.fit(X)
        self.smoother.reset()
        self.fitted = True
        return self

    def update(self, x: np.ndarray) -> RegimeState:
        if not self.fitted:
            raise RuntimeError("RegimeTracker not fitted")

        gmm_result = self.gmm.predict(x)
        hmm_result = self.hmm.predict_next(x)

        agreement = gmm_result.regime_id == hmm_result.state_id

        # Combined confidence: weighted average of posteriors
        combined_probs = (
            self.gmm_weight * gmm_result.probabilities
            + self.hmm_weight * hmm_result.state_probs
        )

        # Apply smoothing (protocol 6.3)
        smoothed = self.smoother.update(combined_probs)
        confirmed_id = self.smoother.current_regime()

        # Use confirmed regime if available, otherwise argmax of smoothed
        if confirmed_id >= 0:
            regime_id = confirmed_id
        else:
            regime_id = int(np.argmax(smoothed))

        confidence = float(smoothed[regime_id])

        # Get regime label
        label = self.gmm.labels[regime_id] if regime_id < len(self.gmm.labels) else f"regime_{regime_id}"

        # Get per-regime parameters (protocol 6.4)
        params = get_regime_params(label)

        # Position scale from REGIME_PARAMS
        base_scale = params.position_size_scalar
        if not agreement:
            base_scale *= self.disagreement_scale
        position_scale = max(0.0, min(1.0, base_scale))

        # Signal gate: closed if position_size_scalar == 0 or execution_mode == REDUCE_ONLY
        signal_gate = params.position_size_scalar > 0 and params.execution_mode != "REDUCE_ONLY"
        if not agreement and label in ("HIGH_VOL_CHAOTIC", "LIQUIDITY_CRISIS"):
            signal_gate = False

        state = RegimeState(
            regime_id=regime_id,
            regime_label=label,
            gmm_regime=gmm_result,
            hmm_state=hmm_result,
            agreement=agreement,
            position_scale=position_scale,
            signal_gate_open=signal_gate,
            confidence=confidence,
            params=params,
            smoothed_posterior=smoothed.copy(),
        )
        self._history.append(state)
        return state

    @property
    def current(self) -> Optional[RegimeState]:
        return self._history[-1] if self._history else None

    @property
    def history(self) -> list[RegimeState]:
        return self._history

    def regime_durations(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self._history:
            counts[s.regime_label] = counts.get(s.regime_label, 0) + 1
        return counts

    def transition_count(self) -> int:
        if len(self._history) < 2:
            return 0
        return sum(
            1 for i in range(1, len(self._history))
            if self._history[i].regime_id != self._history[i - 1].regime_id
        )
