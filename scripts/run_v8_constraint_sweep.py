#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml


def _build_candidates(max_runs: int) -> list[dict[str, float]]:
    grid = {
        "force_event_weight": [0.03, 0.05, 0.07],
        "overlay_min_signal": [0.06, 0.08],
        "target_vol": [0.14, 0.15, 0.17],
        "etf_gross": [0.15, 0.20],
        "futures_gross": [0.15, 0.20],
        "fx_gross": [0.08, 0.12],
        "option_max_notional": [0.02, 0.04],
        "option_always_on_min_notional": [0.0, 0.004],
        "option_short_strike_daily": [0.06, 0.08],
        "option_activation_score": [0.34, 0.42],
        "option_severe_score": [0.72, 0.82],
        "total_gross_cap": [1.8, 2.0],
        "enable_kelly_sentiment_overlay": [0.0, 1.0],
        "enable_regime_risk_scaler": [0.0, 1.0],
        "enable_regime_router_v10": [1.0],
        "enable_whipsaw_control_v10": [1.0],
        "enable_yearly_risk_budget_v10": [1.0],
        "enable_adaptive_v10_layers": [1.0],
        "adaptive_v10_crash_prob_trigger": [0.42, 0.50],
        "adaptive_v10_soft_gate": [0.0, 1.0],
        "enable_state_param_bank_v10": [1.0],
        "state_risk_on_mult": [1.00, 1.03],
        "state_risk_off_mult": [0.90, 0.95],
        "state_crash_mult": [0.68, 0.76],
        "state_hedge_risk_on_mult": [0.85, 0.95],
        "state_hedge_risk_off_mult": [1.00, 1.10],
        "state_hedge_crash_mult": [1.20, 1.35],
        "state_gross_risk_on_mult": [1.00, 1.04],
        "state_gross_risk_off_mult": [0.88, 0.95],
        "state_gross_crash_mult": [0.68, 0.78],
        "state_option_risk_on_mult": [0.88, 0.96],
        "state_option_risk_off_mult": [1.02, 1.12],
        "state_option_crash_mult": [1.18, 1.32],
        "enable_weak_sleeve_hard_demote": [1.0],
        "weak_demote_sharpe": [-0.12, -0.06],
        "weak_recover_sharpe": [0.10, 0.20],
        "weak_demote_confirm_days": [14.0, 21.0],
        "weak_recover_confirm_days": [30.0, 42.0],
        "weak_demote_mult": [0.0, 0.1],
        "enable_intraday_cache_sleeve": [1.0],
        "intraday_gross": [0.04, 0.08],
        "intraday_max_pos": [0.02, 0.03],
        "intraday_min_signal": [0.06, 0.10],
        "intraday_rebal_freq": [3.0, 5.0],
        "intraday_allow_shorts": [0.0, 1.0],
        "event_min_source_quality": [0.70, 0.85],
        "risk_floor": [0.75, 0.82],
        "risk_ceiling": [1.08, 1.16],
        "yearly_vol_budget": [0.17, 0.20],
        "yearly_dd_budget": [0.16, 0.20],
        "yearly_budget_min_scale": [0.72, 0.80],
        "yearly_budget_max_scale": [1.06, 1.12],
        "gov_min_mult": [0.30, 0.35],
        "gov_soft_dd": [0.08, 0.10],
        "gov_hard_dd": [0.18, 0.22],
        "exec_liq_cost_mult": [0.60, 0.80],
    }
    keys = list(grid.keys())
    total_space = 1
    for key in keys:
        total_space *= len(grid[key])

    if total_space <= max_runs:
        rows: list[dict[str, float]] = []
        for vals in itertools.product(*(grid[k] for k in keys)):
            rows.append({k: v for k, v in zip(keys, vals, strict=False)})
        return rows

    rng = random.Random(42)
    seen: set[tuple[float, ...]] = set()
    rows: list[dict[str, float]] = []

    # deterministic anchors for stable low/high edge coverage
    anchors = [
        tuple(grid[k][0] for k in keys),
        tuple(grid[k][-1] for k in keys),
    ]
    for vals in anchors:
        if len(rows) >= max_runs:
            break
        if vals in seen:
            continue
        seen.add(vals)
        rows.append({k: v for k, v in zip(keys, vals, strict=False)})

    while len(rows) < max_runs:
        vals = tuple(rng.choice(grid[k]) for k in keys)
        if vals in seen:
            continue
        seen.add(vals)
        rows.append({k: v for k, v in zip(keys, vals, strict=False)})
    return rows


def _cached_config_path(config_path: str, cache_dir: str) -> tuple[str, list[str]]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    instruments = cfg.get("instruments", {})
    key_to_class = {
        "sector_etfs": "ETF",
        "equities": "EQUITY",
        "equity_index_futures": "FUTURE",
        "commodity_futures": "COMMODITY",
        "fixed_income_futures": "BOND",
        "fx_pairs": "FX",
        "vix_futures": "VOLATILITY",
    }
    dropped: list[str] = []
    for key, asset_class in key_to_class.items():
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


def _score_row(
    row: dict[str, float],
    max_dd_cap: float,
    max_turnover_cap: float,
    min_gross_both_ratio: float,
    gross_target_return: float,
    gross_target_sharpe: float,
    min_lp_cagr_alpha_vs_spy: float,
    min_lp_beat_spy_ratio: float,
    min_gross_beat_spy_ratio: float,
) -> tuple[float, float, bool]:
    dd = abs(float(row.get("lp_net_max_dd", 0.0)))
    turnover = float(row.get("turnover_x_per_year", 0.0))
    lp_cagr = float(row.get("lp_net_cagr", 0.0))
    lp_sharpe = float(row.get("lp_net_sharpe", 0.0))
    spy_cagr = float(row.get("spy_cagr", 0.0))
    lp_cagr_alpha_vs_spy = float(row.get("lp_cagr_alpha_vs_spy", lp_cagr - spy_cagr))
    lp_beat_spy_ratio = float(row.get("lp_years_beating_spy_ratio", 0.0))
    gross_beat_spy_ratio = float(row.get("gross_years_beating_spy_ratio", 0.0))
    lp_both = float(row.get("years_ge_both_targets", 0))
    gross_both = float(row.get("years_ge_gross_both_targets", 0))
    gross_sharpe_years = float(row.get("years_ge_gross_sharpe_target", 0))
    total_years = max(float(row.get("total_years", 0)), 1.0)
    gross_both_ratio = gross_both / total_years
    underwater_days = float(row.get("max_underwater_days_lp", 0.0))
    min_gross_ret = float(row.get("min_gross_year_return", -1.0))
    min_gross_sharpe = float(row.get("min_gross_year_sharpe", -5.0))

    dd_penalty = max(dd - max_dd_cap, 0.0) * 1.8
    turn_penalty = max(turnover - max_turnover_cap, 0.0) * 0.010
    duration_penalty = underwater_days * 0.0008
    min_ret_gap_penalty = max(gross_target_return - min_gross_ret, 0.0) * 0.85
    min_sharpe_gap_penalty = max(gross_target_sharpe - min_gross_sharpe, 0.0) * 0.22
    lp_alpha_gap_penalty = max(min_lp_cagr_alpha_vs_spy - lp_cagr_alpha_vs_spy, 0.0) * 3.5
    lp_beat_ratio_gap_penalty = max(min_lp_beat_spy_ratio - lp_beat_spy_ratio, 0.0) * 2.0
    gross_beat_ratio_gap_penalty = max(min_gross_beat_spy_ratio - gross_beat_spy_ratio, 0.0) * 1.5
    objective_secondary = (
        lp_cagr
        + 0.10 * lp_sharpe
        + 0.20 * gross_both_ratio
        + 0.60 * lp_cagr_alpha_vs_spy
        + 0.20 * lp_beat_spy_ratio
        + 0.10 * gross_beat_spy_ratio
        + 0.0015 * gross_sharpe_years
        + 0.0005 * lp_both
        - dd_penalty
        - turn_penalty
        - duration_penalty
    )
    objective_primary = (
        10.0 * min_ret_gap_penalty
        + 2.5 * min_sharpe_gap_penalty
        + 4.0 * max(min_gross_both_ratio - gross_both_ratio, 0.0)
        + 5.0 * lp_alpha_gap_penalty
        + 2.5 * lp_beat_ratio_gap_penalty
        + 1.5 * gross_beat_ratio_gap_penalty
    )
    constraint_ok = (
        dd <= max_dd_cap
        and turnover <= max_turnover_cap
        and gross_both_ratio >= min_gross_both_ratio
        and lp_cagr_alpha_vs_spy >= min_lp_cagr_alpha_vs_spy
        and lp_beat_spy_ratio >= min_lp_beat_spy_ratio
        and gross_beat_spy_ratio >= min_gross_beat_spy_ratio
    )
    return objective_primary, objective_secondary, constraint_ok


def run() -> int:
    p = argparse.ArgumentParser(description="Constrained v8 sweep for live-deployable settings.")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--backtest-script", default="backtest_v8.py")
    p.add_argument("--config", default="config/sp500_universe.yaml")
    p.add_argument("--max-runs", type=int, default=24)
    p.add_argument("--max-dd-cap", type=float, default=0.30)
    p.add_argument("--max-turnover-cap", type=float, default=20.0)
    p.add_argument(
        "--min-gross-both-ratio",
        type=float,
        default=1.0,
        help="Minimum ratio of years meeting gross return+Sharpe hurdles.",
    )
    p.add_argument("--output-csv", default="data/reports/v8_constraint_sweep.csv")
    p.add_argument("--cache-dir", default="~/.alphaforge/cache/bars")
    p.add_argument("--allow-missing-cache", action="store_true")
    p.add_argument("--gross-target-return", type=float, default=0.15)
    p.add_argument("--gross-target-sharpe", type=float, default=1.0)
    p.add_argument(
        "--min-lp-cagr-alpha-vs-spy",
        type=float,
        default=0.0,
        help="Minimum LP CAGR alpha versus SPY required by constraints.",
    )
    p.add_argument(
        "--min-lp-beat-spy-ratio",
        type=float,
        default=0.55,
        help="Minimum LP yearly beat ratio versus SPY required by constraints.",
    )
    p.add_argument(
        "--min-gross-beat-spy-ratio",
        type=float,
        default=0.50,
        help="Minimum gross yearly beat ratio versus SPY required by constraints.",
    )
    p.add_argument("--enable-macro-overlay", action="store_true")
    p.add_argument("--macro-max-de-risk", type=float, default=0.12)
    p.add_argument("--macro-min-scale", type=float, default=0.85)
    p.add_argument("--macro-smooth-days", type=int, default=5)
    p.add_argument("--start-date", default=None)
    p.add_argument("--eval-start", default=None)
    p.add_argument("--eval-end", default=None)
    p.add_argument("--warmup-days", type=int, default=300)
    p.add_argument("--min-rows", type=int, default=700)
    args = p.parse_args()

    candidates = _build_candidates(max_runs=max(1, args.max_runs))
    run_config = args.config
    dropped_keys: list[str] = []
    if not args.allow_missing_cache:
        run_config, dropped_keys = _cached_config_path(args.config, args.cache_dir)
        if dropped_keys:
            print(
                f"Using cache-complete config for sweep (dropped {len(dropped_keys)} missing symbols).",
                flush=True,
            )
            print("Dropped keys:", ", ".join(dropped_keys), flush=True)
    Path(os.path.dirname(args.output_csv)).mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []

    for i, cfg in enumerate(candidates, start=1):
        with tempfile.NamedTemporaryFile(prefix="v8_metrics_", suffix=".json", delete=False) as tf:
            metrics_path = tf.name
        cmd = [
            args.python,
            args.backtest_script,
            "--config", run_config,
            "--cache-complete-only",
            "--enforce-no-lookahead",
            "--warmup-days", str(args.warmup_days),
            "--min-rows", str(args.min_rows),
            "--force-event-weight", str(cfg["force_event_weight"]),
            "--overlay-min-signal", str(cfg["overlay_min_signal"]),
            "--target-vol", str(cfg["target_vol"]),
            "--etf-gross", str(cfg["etf_gross"]),
            "--futures-gross", str(cfg["futures_gross"]),
            "--fx-gross", str(cfg["fx_gross"]),
            "--option-max-notional", str(cfg["option_max_notional"]),
            "--option-always-on-min-notional", str(cfg["option_always_on_min_notional"]),
            "--option-short-strike-daily", str(cfg["option_short_strike_daily"]),
            "--option-activation-score", str(cfg["option_activation_score"]),
            "--option-severe-score", str(cfg["option_severe_score"]),
            "--total-gross-cap", str(cfg["total_gross_cap"]),
            "--gov-min-mult", str(cfg["gov_min_mult"]),
            "--gov-soft-dd", str(cfg["gov_soft_dd"]),
            "--gov-hard-dd", str(cfg["gov_hard_dd"]),
            "--exec-liq-cost-mult", str(cfg["exec_liq_cost_mult"]),
            "--risk-floor", str(cfg["risk_floor"]),
            "--risk-ceiling", str(cfg["risk_ceiling"]),
            "--yearly-vol-budget", str(cfg["yearly_vol_budget"]),
            "--yearly-dd-budget", str(cfg["yearly_dd_budget"]),
            "--yearly-budget-min-scale", str(cfg["yearly_budget_min_scale"]),
            "--yearly-budget-max-scale", str(cfg["yearly_budget_max_scale"]),
            "--adaptive-v10-crash-prob-trigger", str(cfg["adaptive_v10_crash_prob_trigger"]),
            "--state-risk-on-mult", str(cfg["state_risk_on_mult"]),
            "--state-risk-off-mult", str(cfg["state_risk_off_mult"]),
            "--state-crash-mult", str(cfg["state_crash_mult"]),
            "--state-hedge-risk-on-mult", str(cfg["state_hedge_risk_on_mult"]),
            "--state-hedge-risk-off-mult", str(cfg["state_hedge_risk_off_mult"]),
            "--state-hedge-crash-mult", str(cfg["state_hedge_crash_mult"]),
            "--state-gross-risk-on-mult", str(cfg["state_gross_risk_on_mult"]),
            "--state-gross-risk-off-mult", str(cfg["state_gross_risk_off_mult"]),
            "--state-gross-crash-mult", str(cfg["state_gross_crash_mult"]),
            "--state-option-risk-on-mult", str(cfg["state_option_risk_on_mult"]),
            "--state-option-risk-off-mult", str(cfg["state_option_risk_off_mult"]),
            "--state-option-crash-mult", str(cfg["state_option_crash_mult"]),
            "--weak-demote-sharpe", str(cfg["weak_demote_sharpe"]),
            "--weak-recover-sharpe", str(cfg["weak_recover_sharpe"]),
            "--weak-demote-confirm-days", str(int(cfg["weak_demote_confirm_days"])),
            "--weak-recover-confirm-days", str(int(cfg["weak_recover_confirm_days"])),
            "--weak-demote-mult", str(cfg["weak_demote_mult"]),
            "--intraday-gross", str(cfg["intraday_gross"]),
            "--intraday-max-pos", str(cfg["intraday_max_pos"]),
            "--intraday-min-signal", str(cfg["intraday_min_signal"]),
            "--intraday-rebal-freq", str(int(cfg["intraday_rebal_freq"])),
            "--event-min-source-quality", str(cfg["event_min_source_quality"]),
            "--macro-max-de-risk", str(args.macro_max_de_risk),
            "--macro-min-scale", str(args.macro_min_scale),
            "--macro-smooth-days", str(args.macro_smooth_days),
            "--gross-target-return", str(args.gross_target_return),
            "--gross-target-sharpe", str(args.gross_target_sharpe),
            "--metrics-json", metrics_path,
        ]
        if float(cfg.get("enable_kelly_sentiment_overlay", 0.0)) > 0:
            cmd.append("--enable-kelly-sentiment-overlay")
        if float(cfg.get("enable_regime_risk_scaler", 0.0)) > 0:
            cmd.append("--enable-regime-risk-scaler")
        if float(cfg.get("enable_regime_router_v10", 0.0)) > 0:
            cmd.append("--enable-regime-router-v10")
        if float(cfg.get("enable_whipsaw_control_v10", 0.0)) > 0:
            cmd.append("--enable-whipsaw-control-v10")
        if float(cfg.get("enable_yearly_risk_budget_v10", 0.0)) > 0:
            cmd.append("--enable-yearly-risk-budget-v10")
        if float(cfg.get("enable_adaptive_v10_layers", 0.0)) > 0:
            cmd.append("--enable-adaptive-v10-layers")
        if float(cfg.get("adaptive_v10_soft_gate", 0.0)) > 0:
            cmd.append("--adaptive-v10-soft-gate")
        if float(cfg.get("enable_state_param_bank_v10", 0.0)) > 0:
            cmd.append("--enable-state-param-bank-v10")
        if float(cfg.get("enable_weak_sleeve_hard_demote", 0.0)) > 0:
            cmd.append("--enable-weak-sleeve-hard-demote")
        if float(cfg.get("enable_intraday_cache_sleeve", 0.0)) > 0:
            cmd.append("--enable-intraday-cache-sleeve")
        if float(cfg.get("intraday_allow_shorts", 0.0)) > 0:
            cmd.append("--intraday-allow-shorts")
        if bool(args.enable_macro_overlay):
            cmd.append("--enable-macro-overlay")
        if args.start_date:
            cmd += ["--start-date", str(args.start_date)]
        if args.eval_start:
            cmd += ["--eval-start", str(args.eval_start)]
        if args.eval_end:
            cmd += ["--eval-end", str(args.eval_end)]
        print(f"[{i:02d}/{len(candidates)}] running:", " ".join(cmd[2:]), flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            rows.append(
                {
                    **cfg,
                    "status": "failed",
                    "dropped_cache_keys": len(dropped_keys),
                    "error": (proc.stderr or proc.stdout)[-4000:],
                }
            )
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        row = {
            **cfg,
            **metrics,
            "status": "ok",
            "dropped_cache_keys": len(dropped_keys),
        }
        objective_primary, objective_secondary, constraint_ok = _score_row(
            row,
            max_dd_cap=float(args.max_dd_cap),
            max_turnover_cap=float(args.max_turnover_cap),
            min_gross_both_ratio=float(args.min_gross_both_ratio),
            gross_target_return=float(args.gross_target_return),
            gross_target_sharpe=float(args.gross_target_sharpe),
            min_lp_cagr_alpha_vs_spy=float(args.min_lp_cagr_alpha_vs_spy),
            min_lp_beat_spy_ratio=float(args.min_lp_beat_spy_ratio),
            min_gross_beat_spy_ratio=float(args.min_gross_beat_spy_ratio),
        )
        row["constraint_ok"] = bool(constraint_ok)
        row["objective_primary"] = float(objective_primary)
        row["objective_secondary"] = float(objective_secondary)
        total_years = max(float(row.get("total_years", 0)), 1.0)
        row["gross_both_ratio"] = float(row.get("years_ge_gross_both_targets", 0.0)) / total_years
        row["lp_cagr_alpha_vs_spy"] = float(row.get("lp_cagr_alpha_vs_spy", row.get("lp_net_cagr", 0.0) - row.get("spy_cagr", 0.0)))
        row["lp_years_beating_spy_ratio"] = float(row.get("lp_years_beating_spy_ratio", 0.0))
        row["gross_years_beating_spy_ratio"] = float(row.get("gross_years_beating_spy_ratio", 0.0))
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        sort_cols = ["constraint_ok", "objective_primary", "objective_secondary"]
        df = df.sort_values(by=sort_cols, ascending=[False, True, False]).reset_index(drop=True)
    df.to_csv(args.output_csv, index=False)
    print()
    print(f"Wrote {len(df)} rows -> {args.output_csv}")
    if not df.empty:
        show_cols = [
            "constraint_ok", "objective_primary", "objective_secondary", "lp_net_cagr", "lp_net_sharpe", "lp_net_max_dd",
            "turnover_x_per_year", "final_nav", "years_ge_both_targets", "min_gross_year_return",
            "min_gross_year_sharpe", "years_ge_gross_both_targets", "gross_both_ratio",
            "spy_cagr", "lp_cagr_alpha_vs_spy", "lp_years_beating_spy_ratio", "gross_years_beating_spy_ratio",
            "max_underwater_days_lp", "force_event_weight", "overlay_min_signal", "target_vol",
            "etf_gross", "futures_gross", "fx_gross", "option_max_notional", "option_always_on_min_notional",
            "option_short_strike_daily", "option_activation_score", "option_severe_score",
            "total_gross_cap", "enable_kelly_sentiment_overlay", "enable_regime_risk_scaler",
            "enable_regime_router_v10", "enable_whipsaw_control_v10", "enable_yearly_risk_budget_v10",
            "enable_adaptive_v10_layers", "adaptive_v10_crash_prob_trigger", "adaptive_v10_soft_gate",
            "enable_state_param_bank_v10", "state_risk_on_mult", "state_risk_off_mult", "state_crash_mult",
            "state_gross_risk_on_mult", "state_gross_risk_off_mult", "state_gross_crash_mult",
            "state_option_risk_on_mult", "state_option_risk_off_mult", "state_option_crash_mult",
            "enable_weak_sleeve_hard_demote", "weak_demote_sharpe", "weak_recover_sharpe",
            "weak_demote_confirm_days", "weak_recover_confirm_days", "weak_demote_mult",
            "enable_intraday_cache_sleeve", "intraday_gross", "intraday_max_pos",
            "intraday_min_signal", "intraday_rebal_freq", "intraday_allow_shorts", "event_min_source_quality",
            "risk_floor", "risk_ceiling", "yearly_vol_budget", "yearly_dd_budget", "yearly_budget_min_scale", "yearly_budget_max_scale",
            "gov_min_mult", "gov_soft_dd", "gov_hard_dd",
            "exec_liq_cost_mult",
        ]
        show_cols = [c for c in show_cols if c in df.columns]
        print(df[show_cols].head(10).to_string(index=False))
    if run_config != args.config and os.path.exists(run_config):
        os.remove(run_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
