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
    tmp = tempfile.NamedTemporaryFile(prefix="v7_cached_universe_", suffix=".yaml", delete=False)
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


def _read_params_from_file(path: str) -> dict[str, float]:
    defaults = {
        "target_vol": 0.15,
        "target_gross": 1.5,
        "rebal_freq": 15,
        "max_pos": 0.06,
        "n_long": 50,
        "n_short": 25,
        "force_event_weight": 0.03,
        "sentiment_weight": 0.20,
        "revision_flow_weight": 0.18,
        "revision_flow_half_life_days": 18.0,
        "preemptive_de_risk": 0.0,
        "crisis_hedge_max": 0.0,
        "crisis_hedge_strength": 0.75,
        "crisis_beta_floor": 0.15,
        "hedge_lookback": 63,
        "enable_kelly_sentiment_overlay": 0.0,
        "kelly_lookback": 126,
        "kelly_min_obs": 42,
        "kelly_fraction": 0.08,
        "kelly_max_abs": 1.0,
        "kelly_min_scale": 0.90,
        "kelly_max_scale": 1.12,
        "kelly_sentiment_sensitivity": 0.08,
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
    p = argparse.ArgumentParser(description="Strict no-lookahead walk-forward runner for v7.")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--backtest-script", default="backtest_v7.py")
    p.add_argument("--config", default="config/sp500_universe.yaml")
    p.add_argument("--cache-dir", default="~/.alphaforge/cache/bars")
    p.add_argument("--params-file", default="")
    p.add_argument("--output-csv", default="data/reports/v7_walk_forward.csv")
    p.add_argument("--min-train-years", type=int, default=5)
    p.add_argument("--test-years", type=int, default=1)
    p.add_argument("--step-years", type=int, default=1)
    p.add_argument("--max-folds", type=int, default=0, help="0 means all folds.")
    p.add_argument("--warmup-days", type=int, default=300)
    p.add_argument("--min-rows", type=int, default=250)
    p.add_argument("--target-cagr", type=float, default=0.15)
    p.add_argument("--target-sharpe", type=float, default=1.0)
    p.add_argument("--max-dd", type=float, default=0.30)
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

    params = _read_params_from_file(args.params_file)
    source = f"params:{args.params_file}" if args.params_file and os.path.exists(args.params_file) else "defaults"
    print(f"Using v7 params for walk-forward ({source}):", params)
    Path(os.path.dirname(args.output_csv)).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str | bool]] = []
    for fold in folds:
        with tempfile.NamedTemporaryFile(prefix="v7_wf_metrics_", suffix=".json", delete=False) as tf:
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
            "--warmup-days", str(args.warmup_days),
            "--min-rows", str(args.min_rows),
            "--target-vol", str(params["target_vol"]),
            "--target-gross", str(params["target_gross"]),
            "--rebal-freq", str(int(params["rebal_freq"])),
            "--max-pos", str(params["max_pos"]),
            "--n-long", str(int(params["n_long"])),
            "--n-short", str(int(params["n_short"])),
            "--force-event-weight", str(params["force_event_weight"]),
            "--sentiment-weight", str(params["sentiment_weight"]),
            "--revision-flow-weight", str(params["revision_flow_weight"]),
            "--revision-flow-half-life-days", str(params["revision_flow_half_life_days"]),
            "--preemptive-de-risk", str(params["preemptive_de_risk"]),
            "--crisis-hedge-max", str(params["crisis_hedge_max"]),
            "--crisis-hedge-strength", str(params["crisis_hedge_strength"]),
            "--crisis-beta-floor", str(params["crisis_beta_floor"]),
            "--hedge-lookback", str(int(params["hedge_lookback"])),
            "--kelly-lookback", str(int(params["kelly_lookback"])),
            "--kelly-min-obs", str(int(params["kelly_min_obs"])),
            "--kelly-fraction", str(params["kelly_fraction"]),
            "--kelly-max-abs", str(params["kelly_max_abs"]),
            "--kelly-min-scale", str(params["kelly_min_scale"]),
            "--kelly-max-scale", str(params["kelly_max_scale"]),
            "--kelly-sentiment-sensitivity", str(params["kelly_sentiment_sensitivity"]),
            "--metrics-json", metrics_path,
        ]
        if float(params.get("enable_kelly_sentiment_overlay", 0.0)) > 0:
            cmd.append("--enable-kelly-sentiment-overlay")
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
        pass_fold = lp_cagr >= args.target_cagr and lp_sharpe >= args.target_sharpe and lp_dd <= args.max_dd
        rows.append(
            {
                **fold,
                **metrics,
                "status": "ok",
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
            "pass_count": int(ok["wf_pass"].sum()) if "wf_pass" in ok else 0,
        }
        print("Walk-forward summary:", summary)
    if run_config != args.config and os.path.exists(run_config):
        os.remove(run_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
