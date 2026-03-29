
from __future__ import annotations

from typing import Dict

import numpy as np

from src.signals.base import UNIVERSAL_FEATURES

class LightGBMSignal:

    name = "lightgbm"

    def __init__(self, **kwargs):
        import lightgbm as lgb

        defaults = {
            "objective": "regression",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "min_data_in_leaf": 100,
            "n_estimators": 500,
            "verbose": -1,
        }
        defaults.update(kwargs)
        self._early_stopping = defaults.pop("early_stopping_rounds", 50)
        self.model = lgb.LGBMRegressor(**defaults)
        self._feature_names = list(UNIVERSAL_FEATURES)
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Time-series split for early stopping
        n = len(X)
        split = int(n * 0.8)
        if split < 10 or n - split < 5:
            self.model.set_params(n_estimators=50)
            self.model.fit(X, y)
        else:
            X_tr, X_val = X[:split], X[split:]
            y_tr, y_val = y[:split], y[split:]
            self.model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    __import__("lightgbm").early_stopping(
                        self._early_stopping, verbose=False
                    ),
                ],
            )
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
        imp = self.model.feature_importances_
        return {self._feature_names[i]: float(imp[i])
                for i in range(min(len(imp), len(self._feature_names)))}

class RandomForestSignal:

    name = "random_forest"

    def __init__(self, **kwargs):
        from sklearn.ensemble import RandomForestRegressor

        defaults = {
            "n_estimators": 200,
            "max_depth": 5,
            "max_features": "sqrt",
            "min_samples_leaf": 50,
            "n_jobs": -1,
            "random_state": 42,
        }
        defaults.update(kwargs)
        self.model = RandomForestRegressor(**defaults)
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
        imp = self.model.feature_importances_
        return {self._feature_names[i]: float(imp[i])
                for i in range(min(len(imp), len(self._feature_names)))}

class MLPSignal:

    name = "mlp"

    def __init__(self, hidden_layers: tuple = (64, 32), dropout: float = 0.3,
                 max_iter: int = 500, **kwargs):
        from sklearn.neural_network import MLPRegressor

        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
            **kwargs,
        )
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
        # MLP doesn't have direct feature importance; use input weight norms
        if not self._fitted or not hasattr(self.model, "coefs_"):
            return {}
        first_layer = self.model.coefs_[0]  # shape (n_features, hidden_size)
        importance = np.abs(first_layer).sum(axis=1)
        return {self._feature_names[i]: float(importance[i])
                for i in range(min(len(importance), len(self._feature_names)))}

TIER3_MODELS = {
    "lightgbm": LightGBMSignal,
    "random_forest": RandomForestSignal,
    "mlp": MLPSignal,
}
