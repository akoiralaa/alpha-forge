#!/usr/bin/env python3
"""
Monte Carlo validation for AlphaForge v8.2.

Method: stationary block bootstrap on realized daily returns.
  - Blocks of 21 trading days preserve autocorrelation and vol clustering.
  - Sampling is from the walk-forward (no-lookahead) return series.
  - 10,000 simulations × full sample length (and 1-year forward projections).

Usage:
    # Generate returns first:
    ./.venv/bin/python backtest_v8.py --force-event-weight 0.05 \
        --output-returns data/reports/v8_daily_returns.csv

    # Then run Monte Carlo:
    ./.venv/bin/python scripts/run_monte_carlo.py \
        --returns data/reports/v8_daily_returns.csv \
        --n-sims 10000 --horizon-years 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


def block_bootstrap(
    returns: np.ndarray,
    n_sims: int,
    horizon_days: int,
    block_size: int = 21,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Stationary block bootstrap. Returns (n_sims, horizon_days) array of daily returns.
    Preserves vol clustering and autocorrelation within each block.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(returns)
    n_blocks = (horizon_days + block_size - 1) // block_size
    paths = np.empty((n_sims, n_blocks * block_size), dtype=np.float64)
    for i in range(n_sims):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        for b, s in enumerate(starts):
            paths[i, b * block_size : (b + 1) * block_size] = returns[s : s + block_size]
    return paths[:, :horizon_days]


def run_monte_carlo(
    daily_returns: pd.Series,
    n_sims: int = 10_000,
    horizon_years: int = 1,
    block_size: int = 21,
    nav_start: float = 10_000_000,
    seed: int = 42,
) -> dict:
    r = daily_returns.dropna().values
    horizon_days = int(horizon_years * 252)
    rng = np.random.default_rng(seed)

    paths = block_bootstrap(r, n_sims, horizon_days, block_size=block_size, rng=rng)
    terminal = (1 + paths).prod(axis=1) - 1  # total return over horizon

    # Max drawdown per path
    cum = np.cumprod(1 + paths, axis=1)
    rolling_max = np.maximum.accumulate(cum, axis=1)
    dd_series = cum / rolling_max - 1
    max_dd = dd_series.min(axis=1)

    # Annualised return
    cagr = (1 + terminal) ** (1.0 / horizon_years) - 1

    results = {
        "n_sims": n_sims,
        "horizon_years": horizon_years,
        "block_size_days": block_size,
        "return_series_days": len(r),
        "return_series_start": str(daily_returns.dropna().index[0].date()),
        "return_series_end": str(daily_returns.dropna().index[-1].date()),
        "terminal_return": {
            "median": float(np.median(terminal)),
            "p5": float(np.percentile(terminal, 5)),
            "p25": float(np.percentile(terminal, 25)),
            "p75": float(np.percentile(terminal, 75)),
            "p95": float(np.percentile(terminal, 95)),
        },
        "annualised_return": {
            "median": float(np.median(cagr)),
            "p5": float(np.percentile(cagr, 5)),
            "p95": float(np.percentile(cagr, 95)),
        },
        "max_drawdown": {
            "median": float(np.median(max_dd)),
            "p5": float(np.percentile(max_dd, 5)),
            "p25": float(np.percentile(max_dd, 25)),
        },
        "prob_of_loss": float((terminal < 0).mean()),
        "prob_gt_10pct": float((terminal > 0.10).mean()),
        "prob_gt_20pct": float((terminal > 0.20).mean()),
        "prob_gt_50pct": float((terminal > 0.50).mean()),
        "prob_dd_gt_20pct": float((max_dd < -0.20).mean()),
        "prob_dd_gt_30pct": float((max_dd < -0.30).mean()),
    }
    return results


def print_results(r: dict) -> None:
    h = r["horizon_years"]
    print(f"\n{'='*60}")
    print(f"  Monte Carlo — {r['n_sims']:,} simulations × {h}-year forward")
    print(f"  Block size: {r['block_size_days']}d  |  "
          f"Source: {r['return_series_start']} → {r['return_series_end']}  "
          f"({r['return_series_days']} days)")
    print(f"{'='*60}")
    t = r["terminal_return"]
    a = r["annualised_return"]
    d = r["max_drawdown"]
    print(f"  {'Metric':<30}  {'Value':>10}")
    print(f"  {'-'*42}")
    print(f"  {'Median total return':<30}  {t['median']:>+9.1%}")
    print(f"  {'5th percentile total':<30}  {t['p5']:>+9.1%}")
    print(f"  {'95th percentile total':<30}  {t['p95']:>+9.1%}")
    print(f"  {'Median annualised CAGR':<30}  {a['median']:>+9.1%}")
    print(f"  {'5th pct CAGR':<30}  {a['p5']:>+9.1%}")
    print(f"  {'95th pct CAGR':<30}  {a['p95']:>+9.1%}")
    print(f"  {'Median max drawdown':<30}  {d['median']:>+9.1%}")
    print(f"  {'5th pct max drawdown':<30}  {d['p5']:>+9.1%}")
    print(f"  {'P(loss)':<30}  {r['prob_of_loss']:>9.1%}")
    print(f"  {'P(return > 10%)':<30}  {r['prob_gt_10pct']:>9.1%}")
    print(f"  {'P(return > 20%)':<30}  {r['prob_gt_20pct']:>9.1%}")
    print(f"  {'P(max DD > 20%)':<30}  {r['prob_dd_gt_20pct']:>9.1%}")
    print(f"  {'P(max DD > 30%)':<30}  {r['prob_dd_gt_30pct']:>9.1%}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--returns", required=True,
                   help="CSV with 'date' index and 'gross_ret' column (from backtest_v8.py --output-returns)")
    p.add_argument("--col", default="gross_ret", help="Column to use from returns CSV")
    p.add_argument("--n-sims", type=int, default=10_000)
    p.add_argument("--horizon-years", type=int, default=1)
    p.add_argument("--block-size", type=int, default=21)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-json", default=None)
    args = p.parse_args()

    if not os.path.exists(args.returns):
        print(f"ERROR: returns file not found: {args.returns}", file=sys.stderr)
        print("Run: ./.venv/bin/python backtest_v8.py --force-event-weight 0.05 "
              "--output-returns data/reports/v8_daily_returns.csv", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.returns, index_col="date", parse_dates=True)
    if args.col not in df.columns:
        print(f"ERROR: column '{args.col}' not in {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    series = df[args.col]
    results = run_monte_carlo(
        series,
        n_sims=args.n_sims,
        horizon_years=args.horizon_years,
        block_size=args.block_size,
        seed=args.seed,
    )
    print_results(results)

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_json}")
