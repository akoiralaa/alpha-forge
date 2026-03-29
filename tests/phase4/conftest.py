"""Phase 4 test fixtures."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

build_dir = Path(__file__).resolve().parent.parent.parent / "src" / "cpp" / "build"
sys.path.insert(0, str(build_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def make_msv(**kwargs):
    """Create a mock MSV with given field values, NaN for the rest."""
    defaults = dict(
        symbol_id=1, timestamp_ns=0, valid=True,
        ret_1s=0.0, ret_10s=0.0, ret_60s=0.0, ret_300s=0.0,
        ret_1800s=0.0, ret_1d=0.0,
        vol_1s=0.01, vol_10s=0.01, vol_60s=0.01, vol_300s=0.01, vol_1d=0.01,
        zscore_20=0.0, zscore_100=0.0, zscore_500=0.0,
        ewma_spread_fast=1.0, ewma_spread_slow=1.0,
        ofi=0.0, volume_ratio_20=1.0, spread_bps=2.0, vpin=0.1,
        residual_momentum=0.0,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def generate_features_and_returns(n=2000, n_features=19, seed=42):
    """Generate synthetic features X and forward returns y with signal content."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    # y has weak linear relationship with first 3 features + noise
    y = 0.01 * X[:, 0] + 0.005 * X[:, 1] - 0.003 * X[:, 2] + 0.02 * rng.standard_normal(n)
    return X, y
