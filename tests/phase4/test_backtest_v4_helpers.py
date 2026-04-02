from __future__ import annotations

import numpy as np
import pandas as pd

import backtest_v4 as v4


def _bars(end: str, periods: int) -> pd.DataFrame:
    dates = pd.date_range(end=end, periods=periods, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "date": dates,
            "close": np.linspace(100.0, 110.0, len(dates)),
            "volume": np.linspace(1_000, 2_000, len(dates)),
        }
    )


def test_filter_histories_for_backtest_drops_short_and_stale_names():
    universe = [
        ("AAPL", "EQUITY"),
        ("SHORT", "EQUITY"),
        ("STALE", "EQUITY"),
        ("CL", "COMMODITY"),
    ]
    all_data = {
        "AAPL": _bars("2026-04-01", 900),
        "SHORT": _bars("2026-04-01", 120),
        "STALE": _bars("2025-11-01", 900),
        "CL": _bars("2026-04-01", 400),
    }

    filtered, dropped, latest = v4.filter_histories_for_backtest(all_data, universe)

    assert latest is not None
    assert set(filtered) == {"AAPL", "CL"}
    assert {(sym, reason) for sym, _, reason, _, _ in dropped} == {
        ("SHORT", "short_history"),
        ("STALE", "stale_history"),
    }


def test_build_strategy_evidence_multipliers_penalizes_weak_sleeves():
    idx = pd.date_range("2024-01-01", periods=220, freq="D")
    strategy_returns = pd.DataFrame(
        {
            "good": np.tile([0.010, -0.002], 110),
            "bad": np.tile([-0.010, 0.002], 110),
        },
        index=idx,
    )

    multipliers = v4.build_strategy_evidence_multipliers(strategy_returns, idx)

    assert multipliers["good"].iloc[-1] > multipliers["bad"].iloc[-1]
    assert multipliers["bad"].iloc[-1] < 0.5


def test_default_core_strategy_weights_sum_to_one():
    weights = v4.default_core_strategy_weights()
    assert set(weights) == {"momentum", "quality", "carry", "sector_rot", "high_52w"}
    assert sum(weights.values()) == 1.0


def test_build_high_52w_kill_switch_disables_in_crisis_and_weak_perf():
    idx = pd.date_range("2025-01-01", periods=180, freq="D")
    regime = pd.DataFrame(
        {
            "regime_label": ["LOW_VOL_TRENDING"] * 150 + ["LIQUIDITY_CRISIS"] * 30,
            "confidence": [0.8] * 180,
            "signal_gate": [True] * 180,
        },
        index=idx,
    )
    strategy_returns = pd.DataFrame(
        {
            "high_52w": np.r_[np.full(120, -0.008), np.full(60, -0.006)],
        },
        index=idx,
    )

    gate = v4.build_high_52w_kill_switch(idx, regime, strategy_returns)
    assert gate.iloc[-1] == 0.0
    assert gate.iloc[130] <= 0.25


def test_build_strategy_signal_weights_for_capacity_limits_and_normalizes():
    signal_row = pd.Series({"A": 0.8, "B": -0.4, "C": 0.2, "D": 0.01})
    weights = v4.build_strategy_signal_weights_for_capacity(signal_row, gross_weight=0.60, top_k=3)

    assert set(weights.keys()) == {"A", "B", "C"}
    gross = sum(abs(v) for v in weights.values())
    assert np.isclose(gross, 0.60, atol=1e-9)
