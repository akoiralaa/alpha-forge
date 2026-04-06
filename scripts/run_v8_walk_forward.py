#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml


_KEY_TO_CLASS = {
    "sector_etfs": "ETF",
    "equities": "EQUITY",
    "equity_index_futures": "FUTURE",
    "commodity_futures": "COMMODITY",
    "fixed_income_futures": "BOND",
    "fx_pairs": "FX",
    "vix_futures": "VOLATILITY",
}


def _cached_config_path(config_path: str, cache_dir: str) -> tuple[str, list[str]]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    instruments = cfg.get("instruments", {})
    dropped: list[str] = []
    for key, asset_class in _KEY_TO_CLASS.items():
        symbols = list(instruments.get(key, []) or [])
        keep = []
        for sym in symbols:
            cache_file = os.path.expanduser(os.path.join(cache_dir, f"{sym}_{asset_class}.parquet"))
            if os.path.exists(cache_file):
                keep.append(sym)
            else:
                dropped.append(f"{sym}_{asset_class}")
        instruments[key] = keep
    cfg["instruments"] = instruments
    if not dropped:
        return config_path, dropped
    tmp = tempfile.NamedTemporaryFile(prefix="v8_cached_universe_", suffix=".yaml", delete=False)
    tmp.close()
    with open(tmp.name, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return tmp.name, dropped


def _infer_history_bounds(config_path: str, cache_dir: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    instruments = cfg.get("instruments", {})

    candidates = []
    if "SPY" in set(instruments.get("sector_etfs", []) or []):
        candidates.append(("SPY", "ETF"))
    for key, asset_class in _KEY_TO_CLASS.items():
        for sym in instruments.get(key, []) or []:
            candidates.append((sym, asset_class))

    for sym, asset_class in candidates:
        cache_file = os.path.expanduser(os.path.join(cache_dir, f"{sym}_{asset_class}.parquet"))
        if not os.path.exists(cache_file):
            continue
        df = pd.read_parquet(cache_file, columns=["date"])
        if df.empty:
            continue
        start = pd.Timestamp(df["date"].min()).normalize()
        end = pd.Timestamp(df["date"].max()).normalize()
        return start, end
    raise RuntimeError("Could not infer history bounds from cache files.")


def _build_folds(
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_train_years: int,
    test_years: int,
    step_years: int,
) -> list[dict[str, str]]:
    folds = []
    test_start = start + pd.DateOffset(years=min_train_years)
    i = 1
    while True:
        test_end = (test_start + pd.DateOffset(years=test_years)) - pd.Timedelta(days=1)
        if test_end > end:
            break
        folds.append(
            {
                "fold_id": i,
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
            }
        )
        test_start = test_start + pd.DateOffset(years=step_years)
        i += 1
    return folds


def _read_best_params_from_sweep(path: str) -> dict[str, float]:
    defaults = {
        "force_event_weight": 0.05,
        "overlay_min_signal": 0.06,
        "etf_gross": 0.20,
        "futures_gross": 0.20,
        "fx_gross": 0.12,
        "option_max_notional": 0.04,
        "option_always_on_min_notional": 0.0,
        "total_gross_cap": 2.0,
        "gov_min_mult": 0.35,
        "exec_liq_cost_mult": 0.70,
        "target_vol": 0.15,
        "risk_floor": 0.80,
        "risk_ceiling": 1.12,
        "risk_smooth_days": 7,
        "enable_kelly_sentiment_overlay": 0.0,
        "enable_regime_risk_scaler": 0.0,
        "enable_regime_router_v10": 0.0,
        "enable_whipsaw_control_v10": 0.0,
        "enable_yearly_risk_budget_v10": 0.0,
        "enable_adaptive_v10_layers": 0.0,
        "adaptive_v10_crash_prob_trigger": 0.45,
        "adaptive_v10_soft_gate": 0.0,
        "enable_state_param_bank_v10": 0.0,
        "state_risk_on_mult": 1.00,
        "state_risk_off_mult": 0.92,
        "state_crash_mult": 0.72,
        "state_hedge_risk_on_mult": 0.90,
        "state_hedge_risk_off_mult": 1.05,
        "state_hedge_crash_mult": 1.30,
        "state_gross_risk_on_mult": 1.00,
        "state_gross_risk_off_mult": 0.92,
        "state_gross_crash_mult": 0.74,
        "state_option_risk_on_mult": 0.92,
        "state_option_risk_off_mult": 1.08,
        "state_option_crash_mult": 1.25,
        "enable_weak_sleeve_hard_demote": 0.0,
        "weak_demote_sharpe": -0.10,
        "weak_recover_sharpe": 0.15,
        "weak_demote_confirm_days": 21,
        "weak_recover_confirm_days": 42,
        "weak_demote_mult": 0.0,
        "enable_intraday_cache_sleeve": 0.0,
        "intraday_gross": 0.08,
        "intraday_max_pos": 0.03,
        "intraday_min_signal": 0.08,
        "intraday_rebal_freq": 5,
        "intraday_allow_shorts": 0.0,
        "event_min_source_quality": 0.0,
        "enable_macro_overlay": 0.0,
        "router_vol_window": 21,
        "router_trend_window": 200,
        "router_crash_threshold": 0.62,
        "router_risk_off_threshold": 0.50,
        "router_smooth_days": 3,
        "yearly_vol_budget": 0.18,
        "yearly_dd_budget": 0.18,
        "yearly_budget_min_scale": 0.75,
        "yearly_budget_max_scale": 1.10,
        "yearly_budget_smooth_days": 5,
        "macro_max_de_risk": 0.12,
        "macro_min_scale": 0.85,
        "macro_smooth_days": 5,
        "option_short_strike_daily": 0.07,
        "option_short_credit_bps_daily": 0.9,
        "option_activation_score": 0.38,
        "option_severe_score": 0.78,
    }
    if not os.path.exists(path):
        return defaults
    df = pd.read_csv(path)
    if df.empty:
        return defaults
    if "constraint_ok" in df.columns and df["constraint_ok"].astype(bool).any():
        row = df[df["constraint_ok"].astype(bool)].iloc[0]
    else:
        row = df.iloc[0]
    out = defaults.copy()
    for k in list(out.keys()):
        if k in row and pd.notna(row[k]):
            out[k] = float(row[k])
    return out


def _read_params_from_file(path: str) -> dict[str, float]:
    defaults = {
        "force_event_weight": 0.05,
        "overlay_min_signal": 0.06,
        "etf_gross": 0.20,
        "futures_gross": 0.20,
        "fx_gross": 0.12,
        "option_max_notional": 0.04,
        "option_always_on_min_notional": 0.0,
        "total_gross_cap": 2.0,
        "gov_min_mult": 0.35,
        "exec_liq_cost_mult": 0.70,
        "target_vol": 0.15,
        "risk_floor": 0.80,
        "risk_ceiling": 1.12,
        "risk_smooth_days": 7,
        "enable_kelly_sentiment_overlay": 0.0,
        "enable_regime_risk_scaler": 0.0,
        "enable_regime_router_v10": 0.0,
        "enable_whipsaw_control_v10": 0.0,
        "enable_yearly_risk_budget_v10": 0.0,
        "enable_adaptive_v10_layers": 0.0,
        "adaptive_v10_crash_prob_trigger": 0.45,
        "adaptive_v10_soft_gate": 0.0,
        "enable_state_param_bank_v10": 0.0,
        "state_risk_on_mult": 1.00,
        "state_risk_off_mult": 0.92,
        "state_crash_mult": 0.72,
        "state_hedge_risk_on_mult": 0.90,
        "state_hedge_risk_off_mult": 1.05,
        "state_hedge_crash_mult": 1.30,
        "state_gross_risk_on_mult": 1.00,
        "state_gross_risk_off_mult": 0.92,
        "state_gross_crash_mult": 0.74,
        "state_option_risk_on_mult": 0.92,
        "state_option_risk_off_mult": 1.08,
        "state_option_crash_mult": 1.25,
        "enable_weak_sleeve_hard_demote": 0.0,
        "weak_demote_sharpe": -0.10,
        "weak_recover_sharpe": 0.15,
        "weak_demote_confirm_days": 21,
        "weak_recover_confirm_days": 42,
        "weak_demote_mult": 0.0,
        "enable_intraday_cache_sleeve": 0.0,
        "intraday_gross": 0.08,
        "intraday_max_pos": 0.03,
        "intraday_min_signal": 0.08,
        "intraday_rebal_freq": 5,
        "intraday_allow_shorts": 0.0,
        "event_min_source_quality": 0.0,
        "enable_macro_overlay": 0.0,
        "router_vol_window": 21,
        "router_trend_window": 200,
        "router_crash_threshold": 0.62,
        "router_risk_off_threshold": 0.50,
        "router_smooth_days": 3,
        "yearly_vol_budget": 0.18,
        "yearly_dd_budget": 0.18,
        "yearly_budget_min_scale": 0.75,
        "yearly_budget_max_scale": 1.10,
        "yearly_budget_smooth_days": 5,
        "macro_max_de_risk": 0.12,
        "macro_min_scale": 0.85,
        "macro_smooth_days": 5,
        "option_short_strike_daily": 0.07,
        "option_short_credit_bps_daily": 0.9,
        "option_activation_score": 0.38,
        "option_severe_score": 0.78,
    }
    if not path or not os.path.exists(path):
        return defaults
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    params = payload.get("params", payload)
    out = defaults.copy()
    for k in list(out.keys()):
        if k in params and params[k] is not None:
            out[k] = float(params[k])
    return out


def run() -> int:
    p = argparse.ArgumentParser(description="Strict no-lookahead walk-forward runner for v8.")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--backtest-script", default="backtest_v8.py")
    p.add_argument("--config", default="config/sp500_universe.yaml")
    p.add_argument("--cache-dir", default="~/.one_brain_fund/cache/bars")
    p.add_argument("--params-file", default="config/v8_production_locked.yaml")
    p.add_argument("--sweep-csv", default="data/reports/v8_constraint_sweep.csv")
    p.add_argument("--prefer-sweep", action="store_true")
    p.add_argument("--output-csv", default="data/reports/v8_walk_forward.csv")
    p.add_argument("--min-train-years", type=int, default=5)
    p.add_argument("--test-years", type=int, default=1)
    p.add_argument("--step-years", type=int, default=1)
    p.add_argument("--max-folds", type=int, default=0, help="0 means all folds.")
    p.add_argument("--target-cagr", type=float, default=0.15)
    p.add_argument("--target-sharpe", type=float, default=1.0)
    p.add_argument("--max-dd", type=float, default=0.30)
    p.add_argument(
        "--min-lp-cagr-alpha-vs-spy",
        type=float,
        default=0.0,
        help="Minimum LP CAGR alpha vs SPY required for fold pass.",
    )
    p.add_argument(
        "--min-lp-beat-spy-ratio",
        type=float,
        default=0.55,
        help="Minimum LP yearly beat ratio vs SPY required for fold pass.",
    )
    p.add_argument(
        "--disable-spy-outperformance-gate",
        action="store_true",
        help="Disable LP-vs-SPY pass gates (keeps legacy pass logic).",
    )
    args = p.parse_args()

    run_config, dropped = _cached_config_path(args.config, args.cache_dir)
    if dropped:
        print(
            f"Walk-forward cache-complete mode: dropped {len(dropped)} symbols "
            f"({', '.join(dropped[:20])}{' ...' if len(dropped) > 20 else ''})."
        )

    start, end = _infer_history_bounds(run_config, args.cache_dir)
    folds = _build_folds(start, end, args.min_train_years, args.test_years, args.step_years)
    if args.max_folds > 0:
        folds = folds[: args.max_folds]
    if not folds:
        raise RuntimeError("No folds generated. Lower min-train-years or test-years.")

    if args.prefer_sweep:
        params = _read_best_params_from_sweep(args.sweep_csv)
        source = f"sweep:{args.sweep_csv}"
    else:
        params = _read_params_from_file(args.params_file)
        source = f"locked:{args.params_file}" if os.path.exists(args.params_file) else f"default(no params file)"
    print(f"Using v8 params for walk-forward ({source}):", params)
    Path(os.path.dirname(args.output_csv)).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str | bool]] = []
    for fold in folds:
        with tempfile.NamedTemporaryFile(prefix="v8_wf_metrics_", suffix=".json", delete=False) as tf:
            metrics_path = tf.name
        cmd = [
            args.python,
            args.backtest_script,
            "--config", run_config,
            "--cache-complete-only",
            "--enforce-no-lookahead",
            "--end-date", fold["test_end"],
            "--eval-start", fold["test_start"],
            "--eval-end", fold["test_end"],
            "--force-event-weight", str(params["force_event_weight"]),
            "--overlay-min-signal", str(params["overlay_min_signal"]),
            "--target-vol", str(params["target_vol"]),
            "--etf-gross", str(params["etf_gross"]),
            "--futures-gross", str(params["futures_gross"]),
            "--fx-gross", str(params["fx_gross"]),
            "--option-max-notional", str(params["option_max_notional"]),
            "--option-always-on-min-notional", str(params["option_always_on_min_notional"]),
            "--option-short-strike-daily", str(params["option_short_strike_daily"]),
            "--option-short-credit-bps-daily", str(params["option_short_credit_bps_daily"]),
            "--option-activation-score", str(params["option_activation_score"]),
            "--option-severe-score", str(params["option_severe_score"]),
            "--total-gross-cap", str(params["total_gross_cap"]),
            "--gov-min-mult", str(params["gov_min_mult"]),
            "--exec-liq-cost-mult", str(params["exec_liq_cost_mult"]),
            "--risk-floor", str(params["risk_floor"]),
            "--risk-ceiling", str(params["risk_ceiling"]),
            "--risk-smooth-days", str(int(params["risk_smooth_days"])),
            "--router-vol-window", str(int(params["router_vol_window"])),
            "--router-trend-window", str(int(params["router_trend_window"])),
            "--router-crash-threshold", str(params["router_crash_threshold"]),
            "--router-risk-off-threshold", str(params["router_risk_off_threshold"]),
            "--router-smooth-days", str(int(params["router_smooth_days"])),
            "--yearly-vol-budget", str(params["yearly_vol_budget"]),
            "--yearly-dd-budget", str(params["yearly_dd_budget"]),
            "--yearly-budget-min-scale", str(params["yearly_budget_min_scale"]),
            "--yearly-budget-max-scale", str(params["yearly_budget_max_scale"]),
            "--yearly-budget-smooth-days", str(int(params["yearly_budget_smooth_days"])),
            "--adaptive-v10-crash-prob-trigger", str(params["adaptive_v10_crash_prob_trigger"]),
            "--state-risk-on-mult", str(params["state_risk_on_mult"]),
            "--state-risk-off-mult", str(params["state_risk_off_mult"]),
            "--state-crash-mult", str(params["state_crash_mult"]),
            "--state-hedge-risk-on-mult", str(params["state_hedge_risk_on_mult"]),
            "--state-hedge-risk-off-mult", str(params["state_hedge_risk_off_mult"]),
            "--state-hedge-crash-mult", str(params["state_hedge_crash_mult"]),
            "--state-gross-risk-on-mult", str(params["state_gross_risk_on_mult"]),
            "--state-gross-risk-off-mult", str(params["state_gross_risk_off_mult"]),
            "--state-gross-crash-mult", str(params["state_gross_crash_mult"]),
            "--state-option-risk-on-mult", str(params["state_option_risk_on_mult"]),
            "--state-option-risk-off-mult", str(params["state_option_risk_off_mult"]),
            "--state-option-crash-mult", str(params["state_option_crash_mult"]),
            "--weak-demote-sharpe", str(params["weak_demote_sharpe"]),
            "--weak-recover-sharpe", str(params["weak_recover_sharpe"]),
            "--weak-demote-confirm-days", str(int(params["weak_demote_confirm_days"])),
            "--weak-recover-confirm-days", str(int(params["weak_recover_confirm_days"])),
            "--weak-demote-mult", str(params["weak_demote_mult"]),
            "--intraday-gross", str(params["intraday_gross"]),
            "--intraday-max-pos", str(params["intraday_max_pos"]),
            "--intraday-min-signal", str(params["intraday_min_signal"]),
            "--intraday-rebal-freq", str(int(params["intraday_rebal_freq"])),
            "--event-min-source-quality", str(params["event_min_source_quality"]),
            "--macro-max-de-risk", str(params["macro_max_de_risk"]),
            "--macro-min-scale", str(params["macro_min_scale"]),
            "--macro-smooth-days", str(int(params["macro_smooth_days"])),
            "--metrics-json", metrics_path,
        ]
        if float(params.get("enable_kelly_sentiment_overlay", 0.0)) > 0:
            cmd.append("--enable-kelly-sentiment-overlay")
        if float(params.get("enable_regime_risk_scaler", 0.0)) > 0:
            cmd.append("--enable-regime-risk-scaler")
        if float(params.get("enable_regime_router_v10", 0.0)) > 0:
            cmd.append("--enable-regime-router-v10")
        if float(params.get("enable_whipsaw_control_v10", 0.0)) > 0:
            cmd.append("--enable-whipsaw-control-v10")
        if float(params.get("enable_yearly_risk_budget_v10", 0.0)) > 0:
            cmd.append("--enable-yearly-risk-budget-v10")
        if float(params.get("enable_adaptive_v10_layers", 0.0)) > 0:
            cmd.append("--enable-adaptive-v10-layers")
        if float(params.get("adaptive_v10_soft_gate", 0.0)) > 0:
            cmd.append("--adaptive-v10-soft-gate")
        if float(params.get("enable_state_param_bank_v10", 0.0)) > 0:
            cmd.append("--enable-state-param-bank-v10")
        if float(params.get("enable_weak_sleeve_hard_demote", 0.0)) > 0:
            cmd.append("--enable-weak-sleeve-hard-demote")
        if float(params.get("enable_intraday_cache_sleeve", 0.0)) > 0:
            cmd.append("--enable-intraday-cache-sleeve")
        if float(params.get("intraday_allow_shorts", 0.0)) > 0:
            cmd.append("--intraday-allow-shorts")
        if float(params.get("enable_macro_overlay", 0.0)) > 0:
            cmd.append("--enable-macro-overlay")
        print(
            f"[fold {fold['fold_id']:02d}/{len(folds)}] "
            f"test={fold['test_start']}..{fold['test_end']}",
            flush=True,
        )
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            rows.append(
                {
                    **fold,
                    "status": "failed",
                    "error": (proc.stderr or proc.stdout)[-4000:],
                }
            )
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        lp_cagr = float(metrics.get("lp_net_cagr", 0.0))
        lp_sharpe = float(metrics.get("lp_net_sharpe", 0.0))
        lp_dd = abs(float(metrics.get("lp_net_max_dd", 0.0)))
        lp_alpha_vs_spy = float(metrics.get("lp_cagr_alpha_vs_spy", lp_cagr - float(metrics.get("spy_cagr", 0.0))))
        lp_beat_spy_ratio = float(metrics.get("lp_years_beating_spy_ratio", 0.0))
        pass_fold = lp_cagr >= args.target_cagr and lp_sharpe >= args.target_sharpe and lp_dd <= args.max_dd
        if not args.disable_spy_outperformance_gate:
            pass_fold = (
                pass_fold
                and lp_alpha_vs_spy >= args.min_lp_cagr_alpha_vs_spy
                and lp_beat_spy_ratio >= args.min_lp_beat_spy_ratio
            )
        rows.append(
            {
                **fold,
                **metrics,
                "status": "ok",
                "lp_cagr_alpha_vs_spy": lp_alpha_vs_spy,
                "lp_years_beating_spy_ratio": lp_beat_spy_ratio,
                "wf_pass": bool(pass_fold),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    ok = df[df["status"] == "ok"] if "status" in df.columns else pd.DataFrame()
    print()
    print(f"Wrote {len(df)} rows -> {args.output_csv}")
    if not ok.empty:
        summary = {
            "folds_ok": int(len(ok)),
            "mean_lp_net_cagr": float(ok["lp_net_cagr"].mean()),
            "mean_lp_net_sharpe": float(ok["lp_net_sharpe"].mean()),
            "median_lp_net_max_dd": float(ok["lp_net_max_dd"].median()),
            "mean_lp_cagr_alpha_vs_spy": float(ok["lp_cagr_alpha_vs_spy"].mean()) if "lp_cagr_alpha_vs_spy" in ok else 0.0,
            "mean_lp_years_beating_spy_ratio": float(ok["lp_years_beating_spy_ratio"].mean()) if "lp_years_beating_spy_ratio" in ok else 0.0,
            "pass_count": int(ok["wf_pass"].sum()) if "wf_pass" in ok else 0,
        }
        print("Walk-forward summary:", summary)
        cols = [
            "fold_id",
            "test_start",
            "test_end",
            "lp_net_cagr",
            "lp_net_sharpe",
            "lp_net_max_dd",
            "lp_cagr_alpha_vs_spy",
            "lp_years_beating_spy_ratio",
            "turnover_x_per_year",
            "years_ge_both_targets",
            "wf_pass",
        ]
        cols = [c for c in cols if c in ok.columns]
        print(ok[cols].to_string(index=False))
    if run_config != args.config and os.path.exists(run_config):
        os.remove(run_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
