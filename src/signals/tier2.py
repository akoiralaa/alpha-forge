
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import TimeSeriesSplit

from src.signals.base import UNIVERSAL_FEATURES

class LassoSignal:

    name = "lasso"

    def __init__(self, alpha: float = 0.001):
        self.alpha = alpha
        self.model = Lasso(alpha=alpha, max_iter=5000)
        self._feature_names = list(UNIVERSAL_FEATURES)
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.zeros(X.shape[0])
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        return 1.0 / (1.0 + np.exp(-preds * 100))

    def feature_importance(self) -> Dict[str, float]:
        if not self._fitted:
            return {}
        coefs = self.model.coef_
        return {self._feature_names[i]: float(abs(coefs[i]))
                for i in range(min(len(coefs), len(self._feature_names)))}

class RidgeSignal:

    name = "ridge"

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self._feature_names = list(UNIVERSAL_FEATURES)
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.zeros(X.shape[0])
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        return 1.0 / (1.0 + np.exp(-preds * 100))

    def feature_importance(self) -> Dict[str, float]:
        if not self._fitted:
            return {}
        coefs = self.model.coef_
        return {self._feature_names[i]: float(abs(coefs[i]))
                for i in range(min(len(coefs), len(self._feature_names)))}

class ElasticNetSignal:

    name = "elasticnet"

    def __init__(self, alpha: float = 0.05, l1_ratio: float = 0.5):
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
        self._feature_names = list(UNIVERSAL_FEATURES)
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.zeros(X.shape[0])
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        return 1.0 / (1.0 + np.exp(-preds * 100))

    def feature_importance(self) -> Dict[str, float]:
        if not self._fitted:
            return {}
        coefs = self.model.coef_
        return {self._feature_names[i]: float(abs(coefs[i]))
                for i in range(min(len(coefs), len(self._feature_names)))}

class ARSignal:

    name = "ar"

    def __init__(self, p: int = 5):
        self.p = p
        self._coefs: np.ndarray | None = None
        self._intercept = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(y) < self.p + 10:
            return
        # Build lag matrix
        lags = np.column_stack([y[self.p - i - 1: len(y) - i - 1] for i in range(self.p)])
        target = y[self.p:]
        # OLS fit
        X_aug = np.column_stack([np.ones(len(target)), lags])
        try:
            coefs, _, _, _ = np.linalg.lstsq(X_aug, target, rcond=None)
            self._intercept = coefs[0]
            self._coefs = coefs[1:]
        except np.linalg.LinAlgError:
            self._coefs = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._coefs is None:
            return np.zeros(X.shape[0])
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # Use last p columns as lags
        lags = X[:, :self.p] if X.shape[1] >= self.p else X
        return self._intercept + lags @ self._coefs[:lags.shape[1]]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        return 1.0 / (1.0 + np.exp(-preds * 100))

    def feature_importance(self) -> Dict[str, float]:
        if self._coefs is None:
            return {}
        return {f"lag_{i+1}": float(abs(c)) for i, c in enumerate(self._coefs)}

class PairsTradingSignal:

    name = "pairs"

    def __init__(self, entry_z: float = 2.0, exit_z: float = 0.5):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self._hedge_ratio = 1.0
        self._spread_mean = 0.0
        self._spread_std = 1.0
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.shape[1] < 2 or len(X) < 20:
            return
        a, b = X[:, 0], X[:, 1]
        # OLS: a = beta * b + alpha + epsilon
        X_ols = np.column_stack([b, np.ones(len(b))])
        coefs, _, _, _ = np.linalg.lstsq(X_ols, a, rcond=None)
        self._hedge_ratio = coefs[0]
        spread = a - self._hedge_ratio * b
        self._spread_mean = float(np.mean(spread))
        self._spread_std = float(np.std(spread))
        if self._spread_std < 1e-10:
            self._spread_std = 1.0
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.zeros(X.shape[0])
        a, b = X[:, 0], X[:, 1]
        spread = a - self._hedge_ratio * b
        z = (spread - self._spread_mean) / self._spread_std
        # Invert: negative z-score → buy signal (spread too low → long A)
        return -z

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        return 1.0 / (1.0 + np.exp(-preds))

    def feature_importance(self) -> Dict[str, float]:
        return {"hedge_ratio": abs(self._hedge_ratio),
                "spread_mean": abs(self._spread_mean),
                "spread_std": self._spread_std}

TIER2_MODELS = {
    "lasso": LassoSignal,
    "ridge": RidgeSignal,
    "elasticnet": ElasticNetSignal,
    "ar": ARSignal,
    "pairs": PairsTradingSignal,
}
