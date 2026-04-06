from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio.sizing import build_fractional_kelly_overlay


def _synthetic_signal_and_returns(n: int = 260) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    a = np.sin(np.linspace(0, 14, n))
    b = np.cos(np.linspace(0, 11, n))
    c = -(0.6 * a + 0.4 * b)
    signal = pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)
    return signal, pd.DataFrame(index=idx, columns=["A", "B", "C"], dtype=float)


def test_fractional_kelly_overlay_increases_when_edge_positive():
    signal, returns = _synthetic_signal_and_returns()
    for col in returns.columns:
        shifted = signal[col].shift(1).fillna(0.0)
        returns[col] = 0.0018 * np.sign(shifted)

    scale, diag = build_fractional_kelly_overlay(
        signal,
        returns,
        lookback=90,
        min_obs=35,
        kelly_fraction=0.25,
        min_scale=0.70,
        max_scale=1.30,
    )

    assert diag["enabled"] == 1.0
    assert float(scale.mean()) > 1.0
    assert diag["kelly_avg"] > 0.0


def test_fractional_kelly_overlay_decreases_when_edge_negative():
    signal, returns = _synthetic_signal_and_returns()
    for col in returns.columns:
        shifted = signal[col].shift(1).fillna(0.0)
        returns[col] = -0.0018 * np.sign(shifted)

    scale, diag = build_fractional_kelly_overlay(
        signal,
        returns,
        lookback=90,
        min_obs=35,
        kelly_fraction=0.25,
        min_scale=0.70,
        max_scale=1.30,
    )

    assert float(scale.mean()) < 1.0
    assert diag["kelly_avg"] < 0.0


def test_sentiment_sensitivity_shifts_scale_up():
    signal, returns = _synthetic_signal_and_returns()
    for col in returns.columns:
        shifted = signal[col].shift(1).fillna(0.0)
        returns[col] = 0.0014 * np.sign(shifted)
    idx = signal.index
    sentiment = pd.Series(0.6, index=idx, dtype=float)

    base_scale, _ = build_fractional_kelly_overlay(
        signal,
        returns,
        lookback=80,
        min_obs=30,
        kelly_fraction=0.20,
        sentiment_sensitivity=0.0,
    )
    sentiment_scale, _ = build_fractional_kelly_overlay(
        signal,
        returns,
        lookback=80,
        min_obs=30,
        kelly_fraction=0.20,
        sentiment_series=sentiment,
        sentiment_sensitivity=0.25,
    )

    assert float(sentiment_scale.mean()) > float(base_scale.mean())
