
from __future__ import annotations

import math

import numpy as np

def compute_position_size(
    symbol_id: int,
    signal_score: float,
    portfolio_nav: float,
    per_position_risk_budget: float = 0.005,
    current_price: float = 100.0,
    daily_vol: float = 0.02,
    kelly_fraction: float = 0.25,
) -> int:
    if abs(signal_score) < 1e-10 or current_price <= 0 or daily_vol <= 0:
        return 0

    target_dollar_risk = portfolio_nav * per_position_risk_budget
    vol_scaled_shares = target_dollar_risk / (current_price * daily_vol)

    # Kelly: f* = signal / variance, with fractional scaling
    kelly_max_shares = (
        abs(signal_score) * kelly_fraction
        * portfolio_nav / (current_price * daily_vol ** 2 + 1e-10)
    )

    raw_shares = min(vol_scaled_shares, kelly_max_shares) * np.sign(signal_score)
    return int(raw_shares)
