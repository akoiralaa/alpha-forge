
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from hmmlearn.hmm import GaussianHMM

@dataclass
class HMMState:
    state_id: int
    state_label: str
    state_probs: np.ndarray
    confidence: float
    transition_prob: float    # P(switch from previous state to current)

class HMMRegimeDetector:

    def __init__(
        self,
        n_states: int = 3,
        vol_index: int = 0,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42,
    ):
        self.n_states = n_states
        self.vol_index = vol_index
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.model: Optional[GaussianHMM] = None
        self._state_order: Optional[np.ndarray] = None
        self._prev_state: Optional[int] = None
        self._belief: Optional[np.ndarray] = None  # forward belief in canonical order
        self.fitted = False
        self.labels: list[str] = self._default_labels(n_states)

    @staticmethod
    def _default_labels(n: int) -> list[str]:
        from src.regime.gmm import REGIME_LABELS
        return REGIME_LABELS.get(n, [f"state_{i}" for i in range(n)])

    def fit(self, X: np.ndarray) -> "HMMRegimeDetector":
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.model.fit(X)

        # Sort states by ascending mean volatility
        vol_means = self.model.means_[:, self.vol_index]
        self._state_order = np.argsort(vol_means)
        self._prev_state = None
        self._belief = None
        self.fitted = True
        return self

    def decode(self, X: np.ndarray) -> list[HMMState]:
        if not self.fitted or self.model is None:
            raise RuntimeError("HMMRegimeDetector not fitted")

        _, raw_states = self.model.decode(X)
        raw_probs = self.model.predict_proba(X)

        # Build reverse map: raw_state -> canonical_state
        reverse_map = np.zeros(self.n_states, dtype=int)
        for canonical, raw in enumerate(self._state_order):
            reverse_map[raw] = canonical

        # Reorder transition matrix
        transmat = self.model.transmat_[np.ix_(self._state_order, self._state_order)]

        results = []
        for t in range(len(X)):
            canonical = int(reverse_map[raw_states[t]])
            probs = raw_probs[t][self._state_order]

            if t == 0:
                trans_prob = float(probs[canonical])
            else:
                prev_canonical = int(reverse_map[raw_states[t - 1]])
                trans_prob = float(transmat[prev_canonical, canonical])

            results.append(HMMState(
                state_id=canonical,
                state_label=self.labels[canonical],
                state_probs=probs,
                confidence=float(probs[canonical]),
                transition_prob=trans_prob,
            ))

        return results

    def predict_next(self, x: np.ndarray) -> HMMState:
        if not self.fitted or self.model is None:
            raise RuntimeError("HMMRegimeDetector not fitted")

        # Compute emission likelihood for each raw state
        from scipy.stats import multivariate_normal as mvn
        n_states = self.n_states
        log_likelihoods = np.zeros(n_states)
        for k in range(n_states):
            mean = self.model.means_[k]
            cov = self.model.covars_[k]
            log_likelihoods[k] = mvn.logpdf(x, mean=mean, cov=cov, allow_singular=True)

        # Convert to canonical ordering
        transmat = self.model.transmat_[np.ix_(self._state_order, self._state_order)]
        # Reorder log likelihoods to canonical order
        canon_ll = log_likelihoods[self._state_order]

        if self._belief is None:
            # Initialize from stationary distribution
            startprob = self.model.startprob_[self._state_order]
            self._belief = startprob

        # Forward step: predict then update
        predicted = self._belief @ transmat  # predict step
        # Update with observation likelihood
        ll_shifted = canon_ll - canon_ll.max()  # numerical stability
        obs_prob = np.exp(ll_shifted)
        updated = predicted * obs_prob
        updated = updated / (updated.sum() + 1e-300)

        state_id = int(np.argmax(updated))
        self._belief = updated

        if self._prev_state is not None:
            trans_prob = float(transmat[self._prev_state, state_id])
        else:
            trans_prob = float(updated[state_id])

        self._prev_state = state_id

        return HMMState(
            state_id=state_id,
            state_label=self.labels[state_id],
            state_probs=updated,
            confidence=float(updated[state_id]),
            transition_prob=trans_prob,
        )

    @property
    def transition_matrix(self) -> np.ndarray:
        if not self.fitted or self.model is None:
            raise RuntimeError("HMMRegimeDetector not fitted")
        return self.model.transmat_[np.ix_(self._state_order, self._state_order)]

    @property
    def stationary_distribution(self) -> np.ndarray:
        T = self.transition_matrix
        eigenvalues, eigenvectors = np.linalg.eig(T.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        pi = np.abs(pi)  # clip numerical noise
        pi = pi / pi.sum()
        return pi

    def regime_persistence(self) -> np.ndarray:
        T = self.transition_matrix
        return 1.0 / (1.0 - np.diag(T) + 1e-10)

    def log_likelihood(self, X: np.ndarray) -> float:
        if not self.fitted or self.model is None:
            raise RuntimeError("HMMRegimeDetector not fitted")
        return float(self.model.score(X))
