from __future__ import annotations

import numpy as np
import pandas as pd

import backtest_v4 as v4
from backtest_v8 import (
    apply_entry_exit_hysteresis,
    build_macro_overlay_scaler,
    build_regime_router_v10,
    build_sleeve_governance_multipliers,
    build_synthetic_option_overlay,
    build_yearly_risk_budget_scaler,
    compute_dynamic_execution_cost,
    load_macro_panel,
    sanitize_cross_asset_returns,
)


def test_sanitize_cross_asset_returns_clips_unrealistic_futures_moves():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    returns = pd.DataFrame(
        {
            "ES": [0.01, 0.02, 0.60, -0.70, 0.03],
            "EURUSD": [0.001, -0.002, 0.004, -0.003, 0.002],
        },
        index=idx,
    )
    v4.ASSET_TYPES["ES"] = "FUTURE"
    v4.ASSET_TYPES["EURUSD"] = "FX"

    cleaned = sanitize_cross_asset_returns(returns, ["ES", "EURUSD"])
    assert cleaned["ES"].iloc[2] == 0.0
    assert cleaned["ES"].iloc[3] == 0.0
    assert np.isclose(cleaned["ES"].iloc[0], 0.01)
    assert np.isclose(cleaned["EURUSD"].iloc[2], 0.004)


def test_option_overlay_stays_off_in_calm_regime():
    idx = pd.date_range("2022-01-01", periods=40, freq="D")
    spy_ret = pd.Series(0.0005, index=idx)
    env = pd.Series(1.15, index=idx)
    breadth = pd.Series(0.70, index=idx)

    option_ret, notionals, turnover, short_cov = build_synthetic_option_overlay(
        spy_ret=spy_ret,
        env=env,
        breadth=breadth,
        regime_info=None,
        roll_days=10,
        max_notional=0.05,
        strike_daily=0.02,
        short_strike_daily=0.07,
        payout_mult=1.0,
        theta_bps_daily=2.0,
        short_credit_bps_daily=0.9,
        activation_score=0.38,
        severe_score=0.78,
    )
    assert float(notionals.max()) == 0.0
    assert float(option_ret.abs().sum()) == 0.0
    assert float(turnover.abs().sum()) == 0.0
    assert float(short_cov.abs().sum()) == 0.0


def test_option_overlay_activates_in_crisis_regime():
    idx = pd.date_range("2022-01-01", periods=40, freq="D")
    spy_ret = pd.Series([0.0] * 20 + [-0.04] * 20, index=idx)
    env = pd.Series(0.55, index=idx)
    breadth = pd.Series(0.30, index=idx)
    regime_info = pd.DataFrame(
        {
            "regime_label": ["LIQUIDITY_CRISIS"] * len(idx),
            "confidence": [0.9] * len(idx),
            "position_scale": [0.55] * len(idx),
        },
        index=idx,
    )

    option_ret, notionals, turnover, short_cov = build_synthetic_option_overlay(
        spy_ret=spy_ret,
        env=env,
        breadth=breadth,
        regime_info=regime_info,
        roll_days=10,
        max_notional=0.05,
        strike_daily=0.02,
        short_strike_daily=0.07,
        payout_mult=1.0,
        theta_bps_daily=2.0,
        short_credit_bps_daily=0.9,
        activation_score=0.38,
        severe_score=0.78,
    )
    assert float(notionals.max()) > 0.0
    assert float(option_ret.sum()) > 0.0
    assert float(turnover.sum()) > 0.0
    assert float(short_cov.max()) > 0.0


def test_sleeve_governance_cooldown_kills_weak_sleeve():
    idx = pd.date_range("2021-01-01", periods=120, freq="D")
    symbols = ["AAA", "BBB"]
    returns = pd.DataFrame(
        {
            "AAA": [-0.03] * 80 + [0.01] * 40,
            "BBB": [0.002] * 120,
        },
        index=idx,
    )
    weak = pd.DataFrame({"AAA": [0.40] * len(idx), "BBB": [0.0] * len(idx)}, index=idx)
    strong = pd.DataFrame({"AAA": [0.0] * len(idx), "BBB": [0.30] * len(idx)}, index=idx)
    multipliers, diag = build_sleeve_governance_multipliers(
        sleeve_weights={"weak": weak, "strong": strong},
        returns=returns[symbols],
        expected_sharpes={"weak": 0.20, "strong": 0.10},
        window=40,
        min_obs=20,
        kill_sharpe=-0.05,
        kill_dd=0.08,
        cooldown_days=10,
        min_mult=0.30,
        max_mult=1.20,
        smooth_days=1,
    )
    assert float(diag["weak"]["kill_events"]) >= 1
    assert float(multipliers["weak"].min()) <= 0.31
    assert float(multipliers["strong"].mean()) > float(multipliers["weak"].mean())


def test_dynamic_execution_cost_penalizes_low_liquidity_regime():
    idx = pd.date_range("2022-01-01", periods=80, freq="D")
    cols = ["A"]
    weights = pd.DataFrame({"A": [0.0] + [0.20, -0.20] * 39 + [0.20]}, index=idx)
    prices = pd.DataFrame({"A": np.linspace(100, 104, len(idx))}, index=idx)
    returns = pd.DataFrame({"A": [0.003] * 40 + [0.08] * 40}, index=idx)
    volumes = pd.DataFrame({"A": [2_000_000] * 40 + [50_000] * 40}, index=idx)
    static_cost = (weights.diff().abs().fillna(0.0)["A"] * (3.0 / 10000.0)).sum()
    dynamic_cost, diag = compute_dynamic_execution_cost(
        weights=weights,
        prices=prices,
        returns=returns,
        volumes=volumes,
        bps_by_symbol={"A": 3.0},
        vol_lookback=10,
        adv_lookback=10,
    )
    assert float(dynamic_cost.sum()) > float(static_cost)
    assert float(diag["avg_cost_scale"]) > 1.0


def test_macro_overlay_returns_neutral_when_cache_missing():
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    scale, diag = build_macro_overlay_scaler(
        index=idx,
        cache_file="/tmp/does_not_exist_macro_cache.parquet",
        max_de_risk=0.12,
        min_scale=0.85,
        smooth_days=3,
        z_window=20,
    )
    assert np.allclose(scale.values, 1.0)
    assert str(diag.get("status")) == "no_macro_cache"


def test_macro_overlay_de_risks_in_deteriorating_macro_regime(tmp_path):
    idx = pd.date_range("2020-01-01", periods=220, freq="D")
    macro = pd.DataFrame(
        {
            "date": idx,
            "T10Y2Y": np.linspace(1.2, -0.8, len(idx)),
            "UNRATE": np.linspace(3.5, 8.0, len(idx)),
            "FEDFUNDS": np.linspace(0.5, 4.0, len(idx)),
        }
    )
    macro_file = tmp_path / "fred_daily.parquet"
    macro.to_parquet(macro_file, index=False)
    scale, diag = build_macro_overlay_scaler(
        index=idx,
        cache_file=str(macro_file),
        max_de_risk=0.20,
        min_scale=0.80,
        smooth_days=5,
        z_window=60,
    )
    assert str(diag.get("status")) == "ok"
    assert float(scale.min()) >= 0.80 - 1e-9
    assert float(scale.max()) <= 1.00 + 1e-9
    assert float(scale.iloc[-1]) < float(scale.iloc[40])


def test_macro_panel_aligns_on_calendar_dates_for_tz_aware_index(tmp_path):
    macro_idx = pd.date_range("2020-01-01", periods=10, freq="D")
    macro = pd.DataFrame(
        {
            "date": macro_idx,
            "T10Y2Y": np.linspace(1.0, 0.0, len(macro_idx)),
            "UNRATE": np.linspace(3.0, 4.0, len(macro_idx)),
            "FEDFUNDS": np.linspace(0.5, 1.0, len(macro_idx)),
        }
    )
    macro_file = tmp_path / "fred_daily.parquet"
    macro.to_parquet(macro_file, index=False)

    # Simulate price index carrying UTC timestamps with non-midnight times.
    eval_idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2020-01-03 04:00:00+00:00"),
            pd.Timestamp("2020-01-06 04:00:00+00:00"),
            pd.Timestamp("2020-01-07 04:00:00+00:00"),
        ]
    )
    panel = load_macro_panel(str(macro_file), eval_idx)
    assert panel.shape[0] == len(eval_idx)
    assert panel.notna().all().all()


def test_regime_router_v10_flags_crash_in_deteriorating_tape(tmp_path):
    idx = pd.date_range("2020-01-01", periods=180, freq="D")
    prices = pd.DataFrame({"SPY": np.concatenate([np.linspace(300, 330, 90), np.linspace(330, 250, 90)])}, index=idx)
    env = pd.Series(np.concatenate([np.full(90, 1.1), np.full(90, 0.6)]), index=idx)
    breadth = pd.Series(np.concatenate([np.full(90, 0.65), np.full(90, 0.30)]), index=idx)
    macro = pd.DataFrame(
        {
            "date": idx,
            "T10Y2Y": np.concatenate([np.full(90, 0.8), np.full(90, -0.5)]),
            "UNRATE": np.linspace(3.5, 7.0, len(idx)),
            "FEDFUNDS": np.linspace(1.0, 4.0, len(idx)),
        }
    )
    macro_file = tmp_path / "fred_daily.parquet"
    macro.to_parquet(macro_file, index=False)
    router, diag = build_regime_router_v10(
        index=idx,
        prices=prices,
        env=env,
        breadth=breadth,
        macro_cache_file=str(macro_file),
    )
    assert diag.get("status") == "ok"
    assert "crash_prob" in router
    assert float((router["state"] == "crash").sum()) > 0
    assert float(router["crash_prob"].iloc[-1]) > float(router["crash_prob"].iloc[30])


def test_whipsaw_hysteresis_reduces_turnover_on_chop():
    idx = pd.date_range("2021-01-01", periods=40, freq="D")
    raw = pd.DataFrame(
        {
            "A": [0.02 if i % 2 == 0 else -0.02 for i in range(len(idx))],
            "B": [0.0 if i % 3 else 0.015 for i in range(len(idx))],
        },
        index=idx,
    )
    filtered, diag = apply_entry_exit_hysteresis(
        raw,
        entry_days=2,
        exit_days=2,
        entry_abs=0.01,
        exit_abs=0.005,
    )
    raw_turn = float(raw.diff().abs().sum(axis=1).mean())
    filt_turn = float(filtered.diff().abs().sum(axis=1).mean())
    assert filt_turn <= raw_turn
    assert diag["turnover_out"] <= diag["turnover_in"]


def test_yearly_budget_scaler_de_risks_when_vol_explodes():
    idx = pd.date_range("2022-01-01", periods=260, freq="D")
    weights = pd.DataFrame({"SPY": 0.9}, index=idx)
    returns = pd.DataFrame(
        {
            "SPY": np.concatenate([np.full(120, 0.001), np.full(140, 0.03)]),
        },
        index=idx,
    )
    scaler, diag = build_yearly_risk_budget_scaler(
        weights=weights,
        returns=returns,
        annual_vol_budget=0.18,
        annual_dd_budget=0.18,
        min_scale=0.70,
        max_scale=1.10,
        smooth_days=3,
    )
    assert diag.get("status") == "ok"
    assert float(scaler.min()) >= 0.70 - 1e-9
    assert float(scaler.max()) <= 1.10 + 1e-9
    assert float(scaler.iloc[-1]) < float(scaler.iloc[40])
