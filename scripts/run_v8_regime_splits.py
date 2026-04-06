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
        s = pd.Timestamp(df["date"].min()).normalize()
        e = pd.Timestamp(df["date"].max()).normalize()
        if s.tzinfo is not None:
            s = s.tz_convert("UTC").tz_localize(None)
        if e.tzinfo is not None:
            e = e.tz_convert("UTC").tz_localize(None)
        return (
            s,
            e,
        )
    raise RuntimeError("Could not infer history bounds from cache.")


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


def _read_targets_from_file(path: str) -> dict[str, float]:
    defaults = {
        "lp_target_return": 0.15,
        "lp_target_sharpe": 1.0,
        "lp_max_dd": 0.30,
    }
    if not path or not os.path.exists(path):
        return defaults
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    targets = payload.get("targets", {})
    out = defaults.copy()
    for k in list(out.keys()):
        if k in targets and targets[k] is not None:
            out[k] = float(targets[k])
    return out


def _clamp_range(
    start: pd.Timestamp,
    end: pd.Timestamp,
    hist_start: pd.Timestamp,
    hist_end: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    s = max(start, hist_start)
    e = min(end, hist_end)
    if s > e:
        return None
    return s, e


def _naive_ts(ts_like: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts_like).normalize()
    if ts.tzinfo is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts


def run() -> int:
    p = argparse.ArgumentParser(description="Run strict no-lookahead regime-split validation for v8.")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--backtest-script", default="backtest_v8.py")
    p.add_argument("--config", default="config/sp500_universe.yaml")
    p.add_argument("--cache-dir", default="~/.one_brain_fund/cache/bars")
    p.add_argument("--params-file", default="config/v8_production_locked.yaml")
    p.add_argument("--output-csv", default="data/reports/v8_regime_splits.csv")
    p.add_argument("--train-years", type=int, default=5)
    p.add_argument("--recent-years", type=int, default=2)
    p.add_argument("--warmup-days", type=int, default=180)
    p.add_argument("--min-rows", type=int, default=360)
    args = p.parse_args()

    run_config, dropped = _cached_config_path(args.config, args.cache_dir)
    if dropped:
        print(
            f"Regime split cache-complete mode: dropped {len(dropped)} symbols "
            f"({', '.join(dropped[:20])}{' ...' if len(dropped) > 20 else ''})."
        )
    hist_start, hist_end = _infer_history_bounds(run_config, args.cache_dir)
    params = _read_params_from_file(args.params_file)
    targets = _read_targets_from_file(args.params_file)
    print(f"Using locked params ({args.params_file}): {params}")
    print(f"Using locked targets: {targets}")

    recent_start = hist_end - pd.DateOffset(years=max(1, int(args.recent_years))) + pd.Timedelta(days=1)
    splits = [
        ("pre_2008", _naive_ts("2007-01-01"), _naive_ts("2007-12-31")),
        ("crisis_2008", _naive_ts("2008-01-01"), _naive_ts("2008-12-31")),
        ("decade_2010s", _naive_ts("2010-01-01"), _naive_ts("2019-12-31")),
        ("regime_2020_plus", _naive_ts("2020-01-01"), _naive_ts(hist_end)),
        ("recent", _naive_ts(recent_start), _naive_ts(hist_end)),
    ]

    Path(os.path.dirname(args.output_csv)).mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float | bool]] = []
    for split_name, test_start, test_end in splits:
        clamped = _clamp_range(test_start, test_end, hist_start, hist_end)
        if clamped is None:
            rows.append(
                {
                    "split": split_name,
                    "status": "skipped",
                    "reason": "outside_history",
                }
            )
            continue
        test_start, test_end = clamped
        train_start = max(hist_start, test_start - pd.DateOffset(years=max(1, int(args.train_years))))
        train_end = test_start - pd.Timedelta(days=1)
        if train_end <= train_start:
            rows.append(
                {
                    "split": split_name,
                    "status": "skipped",
                    "reason": "insufficient_train_window",
                }
            )
            continue

        with tempfile.NamedTemporaryFile(prefix="v8_regime_metrics_", suffix=".json", delete=False) as tf:
            metrics_path = tf.name
        cmd = [
            args.python,
            args.backtest_script,
            "--config", run_config,
            "--cache-complete-only",
            "--enforce-no-lookahead",
            "--warmup-days", str(args.warmup_days),
            "--min-rows", str(args.min_rows),
            "--start-date", str(train_start.date()),
            "--end-date", str(test_end.date()),
            "--eval-start", str(test_start.date()),
            "--eval-end", str(test_end.date()),
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
        if float(params.get("enable_macro_overlay", 0.0)) > 0:
            cmd.append("--enable-macro-overlay")
        print(
            f"[split {split_name}] train={train_start.date()}..{train_end.date()} "
            f"test={test_start.date()}..{test_end.date()}",
            flush=True,
        )
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            rows.append(
                {
                    "split": split_name,
                    "status": "failed",
                    "train_start": str(train_start.date()),
                    "train_end": str(train_end.date()),
                    "test_start": str(test_start.date()),
                    "test_end": str(test_end.date()),
                    "error": (proc.stderr or proc.stdout)[-4000:],
                }
            )
            continue

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        lp_cagr = float(metrics.get("lp_net_cagr", 0.0))
        lp_sharpe = float(metrics.get("lp_net_sharpe", 0.0))
        lp_dd = abs(float(metrics.get("lp_net_max_dd", 0.0)))
        pass_split = (
            lp_cagr >= float(targets["lp_target_return"])
            and lp_sharpe >= float(targets["lp_target_sharpe"])
            and lp_dd <= float(targets["lp_max_dd"])
        )
        rows.append(
            {
                "split": split_name,
                "status": "ok",
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                **metrics,
                "split_pass": bool(pass_split),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print()
    print(f"Wrote {len(df)} rows -> {args.output_csv}")
    if "status" in df.columns:
        print(df["status"].value_counts(dropna=False).to_string())
    ok = df[df.get("status", "") == "ok"] if "status" in df.columns else pd.DataFrame()
    if not ok.empty:
        cols = [
            "split",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "lp_net_cagr",
            "lp_net_sharpe",
            "lp_net_max_dd",
            "turnover_x_per_year",
            "split_pass",
        ]
        cols = [c for c in cols if c in ok.columns]
        print(ok[cols].to_string(index=False))

    if run_config != args.config and os.path.exists(run_config):
        os.remove(run_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
