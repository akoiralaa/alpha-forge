#!/usr/bin/env python3
"""
AlphaForge — Fama-French Factor Attribution
=============================================
Regresses daily portfolio returns against Fama-French 5 factors + Momentum
to decompose P&L into:

    Beta P&L    — did you just ride the market up?
    Size P&L    — are you secretly long small-caps?
    Value P&L   — are you secretly long cheap stocks?
    Profit P&L  — are you actually capturing a quality premium?
    Invest P&L  — conservative-minus-aggressive investment factor
    Momentum P&L— are you just riding trend?
    Pure Alpha  — what's left after stripping everything above

The "trader's answer" to: "If AlphaForge is long AAPL and MSFT, are you
actually longing Quality or just Mega-cap Tech?"

If your alpha loading is 0.9 correlated with Mkt-RF, a PM will say
you're a "beta-masking" trader. The target is:
  - Low Mkt-RF beta (< 0.3)
  - Statistically significant alpha (t-stat > 2.0)
  - Positive MOM loading (momentum strategy should load here)

Data source: Ken French Data Library (free, no API key)
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

Usage
─────
    ./.venv/bin/python scripts/run_factor_attribution.py \\
        --returns data/reports/v8_daily_returns.csv \\
        --output-csv data/reports/factor_attribution.csv
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(os.path.expanduser("~/.alphaforge/cache/factors"))
FF5_URL  = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOM_URL  = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"


def _fetch_ff_csv(url: str, cache_name: str) -> pd.DataFrame:
    """Download and cache a Ken French daily CSV zip."""
    cache_path = CACHE_DIR / cache_name
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    print(f"  Fetching {url} ...", flush=True)
    import requests
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        csv_name = [n for n in z.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
        with z.open(csv_name) as f:
            raw = f.read().decode("latin-1")

    # Ken French CSVs have a text header before the data — find the data block
    lines = raw.split("\n")
    start = next(i for i, l in enumerate(lines) if l.strip().startswith("2") or
                 l.strip().startswith("1"))
    # Find where the annual/monthly block starts (end of daily data)
    end = len(lines)
    for i in range(start, len(lines)):
        line = lines[i].strip()
        if line == "" and i > start + 100:
            end = i
            break

    data_str = "\n".join(lines[start:end])
    df = pd.read_csv(io.StringIO(data_str), header=None)
    df.columns = ["date"] + [f"f{i}" for i in range(len(df.columns) - 1)]

    # Parse date (YYYYMMDD format)
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y%m%d",
                                errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")
    df = df.apply(pd.to_numeric, errors="coerce").dropna() / 100  # convert % to decimal

    df.to_parquet(cache_path)
    return df


def load_ff5_factors() -> pd.DataFrame:
    """Load FF5 factors: Mkt-RF, SMB, HML, RMW, CMA, RF."""
    df = _fetch_ff_csv(FF5_URL, "ff5_daily.parquet")
    df.columns = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    return df


def load_momentum_factor() -> pd.Series:
    """Load MOM factor (UMD)."""
    df = _fetch_ff_csv(MOM_URL, "ff_mom_daily.parquet")
    df.columns = ["MOM"]
    return df["MOM"]


def run_attribution(
    returns: pd.Series,
    ff5: pd.DataFrame,
    mom: pd.Series,
    window_years: float | None = None,
) -> dict:
    """
    OLS regression of excess portfolio returns on FF5 + MOM.
    Returns full attribution with t-stats and R².
    """
    from scipy import stats

    # Align all series
    combined = pd.DataFrame({
        "port_ret": returns,
        "Mkt-RF": ff5["Mkt-RF"],
        "SMB":    ff5["SMB"],
        "HML":    ff5["HML"],
        "RMW":    ff5["RMW"],
        "CMA":    ff5["CMA"],
        "RF":     ff5["RF"],
        "MOM":    mom,
    }).dropna()

    if window_years:
        cutoff = combined.index[-1] - pd.DateOffset(years=window_years)
        combined = combined[combined.index >= cutoff]

    # Excess return = portfolio - risk-free
    y = combined["port_ret"] - combined["RF"]
    X_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
    X = combined[X_cols].values
    X_with_const = np.column_stack([np.ones(len(X)), X])

    # OLS
    beta, residuals, _, _ = np.linalg.lstsq(X_with_const, y.values, rcond=None)
    y_hat = X_with_const @ beta
    resid = y.values - y_hat
    n, k = X_with_const.shape
    sigma2 = np.dot(resid, resid) / (n - k)
    XtX_inv = np.linalg.pinv(X_with_const.T @ X_with_const)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    t_stats = beta / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))

    ss_res = np.dot(resid, resid)
    ss_tot = np.dot(y.values - y.values.mean(), y.values - y.values.mean())
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Annual alpha
    alpha_daily = float(beta[0])
    alpha_annual = (1 + alpha_daily) ** 252 - 1
    alpha_t = float(t_stats[0])

    factor_names = ["Alpha"] + X_cols
    loadings = {name: float(b) for name, b in zip(factor_names, beta)}
    t_stats_d = {name: float(t) for name, t in zip(factor_names, t_stats)}
    p_vals_d = {name: float(p) for name, p in zip(factor_names, p_values)}

    # P&L attribution: annualised contribution of each factor
    factor_means = {col: float(combined[col].mean() * 252) for col in X_cols}
    pnl_attribution = {
        col: loadings[col] * factor_means[col]
        for col in X_cols
    }
    pnl_attribution["Alpha"] = alpha_annual

    return {
        "n_days": int(n),
        "start": str(combined.index[0].date()),
        "end": str(combined.index[-1].date()),
        "alpha_daily": alpha_daily,
        "alpha_annual": alpha_annual,
        "alpha_t_stat": alpha_t,
        "alpha_significant": abs(alpha_t) > 2.0,
        "r_squared": float(r_squared),
        "loadings": loadings,
        "t_stats": t_stats_d,
        "p_values": p_vals_d,
        "pnl_attribution_annual": pnl_attribution,
    }


def print_attribution(result: dict) -> None:
    print(f"\n{'='*70}")
    print(f"  Fama-French 5-Factor + Momentum Attribution")
    print(f"  Period: {result['start']} → {result['end']}  ({result['n_days']} days)")
    print(f"  R²: {result['r_squared']:.3f}   "
          f"({'model explains most of variance' if result['r_squared'] > 0.5 else 'low — significant unexplained component'})")
    print(f"{'='*70}")

    alpha_sig = "✓ SIGNIFICANT" if result["alpha_significant"] else "✗ not significant"
    print(f"\n  Annual Alpha: {result['alpha_annual']:>+8.2%}  "
          f"(t={result['alpha_t_stat']:+.2f})  {alpha_sig}")

    print(f"\n  {'Factor':<10}  {'Loading':>8}  {'t-stat':>7}  "
          f"{'Annual P&L':>11}  {'Verdict'}")
    print(f"  {'-'*62}")

    factor_verdicts = {
        "Mkt-RF": lambda b: "HIGH BETA — mostly market return" if abs(b) > 0.5 else
                            "✓ low beta" if abs(b) < 0.2 else "moderate beta",
        "SMB":    lambda b: "long small-cap tilt" if b > 0.2 else
                            "short small-cap" if b < -0.2 else "✓ size-neutral",
        "HML":    lambda b: "value tilt" if b > 0.2 else
                            "growth tilt" if b < -0.2 else "✓ style-neutral",
        "RMW":    lambda b: "✓ quality premium captured" if b > 0.1 else
                            "low-quality bias" if b < -0.1 else "neutral",
        "CMA":    lambda b: "conservative investment" if b > 0.1 else "neutral",
        "MOM":    lambda b: "✓ momentum loading (expected)" if b > 0.1 else
                            "momentum-short" if b < -0.1 else "neutral",
    }

    loadings = result["loadings"]
    t_stats = result["t_stats"]
    pnl = result["pnl_attribution_annual"]

    for factor in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]:
        b = loadings[factor]
        t = t_stats[factor]
        p = pnl[factor]
        verdict = factor_verdicts[factor](b)
        sig = "*" if abs(t) > 2.0 else " "
        print(f"  {factor:<10}  {b:>+8.3f}  {t:>+7.2f}{sig}  {p:>+10.2%}  {verdict}")

    print(f"\n  {'Pure Alpha':<10}  {'':>8}  {'':>8}  "
          f"{pnl['Alpha']:>+10.2%}  (residual after all factors stripped)")

    # Summary verdict
    print(f"\n  {'─'*66}")
    mkt_loading = abs(loadings["Mkt-RF"])
    if mkt_loading > 0.5:
        print(f"  ⚠ HIGH BETA WARNING: Mkt-RF loading {loadings['Mkt-RF']:+.2f}. "
              f"Strategy may be beta-masking as alpha.")
    else:
        print(f"  ✓ Low market beta ({loadings['Mkt-RF']:+.2f}) — "
              f"returns are not simply a levered market position.")

    if result["alpha_significant"]:
        print(f"  ✓ Statistically significant pure alpha: "
              f"{result['alpha_annual']:+.2%}/yr (t={result['alpha_t_stat']:+.2f})")
    else:
        print(f"  ✗ Alpha not statistically significant at 5% level "
              f"(t={result['alpha_t_stat']:+.2f}). Returns may be fully explained by factors.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fama-French factor attribution")
    p.add_argument("--returns", default="data/reports/v8_daily_returns.csv")
    p.add_argument("--col", default="gross_ret")
    p.add_argument("--window-years", type=float, default=None,
                   help="Use only the last N years (default: full history)")
    p.add_argument("--output-csv", default="data/reports/factor_attribution.csv")
    args = p.parse_args()

    if not os.path.exists(args.returns):
        print(f"ERROR: {args.returns} not found.", file=sys.stderr)
        sys.exit(1)

    print("Loading factors from Ken French Data Library...", flush=True)
    ff5 = load_ff5_factors()
    mom = load_momentum_factor()

    df_ret = pd.read_csv(args.returns, index_col="date", parse_dates=True)
    df_ret.index = pd.to_datetime(df_ret.index, utc=True)
    ff5.index = pd.to_datetime(ff5.index, utc=True)
    mom.index = pd.to_datetime(mom.index, utc=True)

    returns = df_ret[args.col].dropna()
    print(f"Running attribution on {len(returns)} days...", flush=True)

    result = run_attribution(returns, ff5, mom, window_years=args.window_years)
    print_attribution(result)

    # Save flat CSV
    rows = []
    for factor in ["Alpha", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]:
        rows.append({
            "factor": factor,
            "loading": result["loadings"].get(factor, None),
            "t_stat": result["t_stats"].get(factor, None),
            "p_value": result["p_values"].get(factor, None),
            "annual_pnl_contribution": result["pnl_attribution_annual"].get(factor, None),
        })
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")
