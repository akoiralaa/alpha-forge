
from __future__ import annotations

from typing import Dict, Protocol, runtime_checkable

import numpy as np
import pandas as pd

@runtime_checkable
class SignalModel(Protocol):

    name: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...

    def feature_importance(self) -> Dict[str, float]: ...

# Standard feature names from MSV for model input
UNIVERSAL_FEATURES = [
    "ret_1s", "ret_10s", "ret_60s", "ret_300s", "ret_1800s",
    "vol_1s", "vol_10s", "vol_60s", "vol_300s", "vol_1d",
    "zscore_20", "zscore_100", "zscore_500",
    "ewma_spread_fast", "ewma_spread_slow",
    "ofi", "volume_ratio_20", "spread_bps", "vpin",
]

def msv_to_features(msv) -> np.ndarray:
    vals = []
    for name in UNIVERSAL_FEATURES:
        v = getattr(msv, name, np.nan)
        vals.append(v if not np.isnan(v) else 0.0)
    return np.array(vals, dtype=np.float64)

def msv_to_feature_dict(msv) -> dict:
    d = {}
    for name in UNIVERSAL_FEATURES:
        v = getattr(msv, name, np.nan)
        d[name] = v if not np.isnan(v) else 0.0
    return d

def measure_half_life(
    signal_series: pd.Series,
    return_series: pd.Series,
    max_lag: int = 3600,
    step: int = 10,
) -> float:
    lags = list(range(1, max_lag, step))
    correlations = []
    for lag in lags:
        shifted = return_series.shift(-lag)
        valid = signal_series.notna() & shifted.notna()
        if valid.sum() < 10:
            correlations.append(0.0)
            continue
        corr = signal_series[valid].corr(shifted[valid])
        correlations.append(corr if not np.isnan(corr) else 0.0)

    if not correlations or max(correlations) <= 0:
        return float(max_lag)

    peak_corr = max(correlations)
    half_corr = peak_corr / 2.0
    for i, corr in enumerate(correlations):
        if corr <= half_corr and i > correlations.index(peak_corr):
            return float(lags[i])
    return float(max_lag)
