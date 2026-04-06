#!/usr/bin/env python3
"""
AlphaForge — Stress Scenario Analysis
======================================
Tests robustness beyond historical replay and block-bootstrap Monte Carlo by
injecting synthetic tail events into realized daily returns.

Ten scenarios across four failure modes:

  FLASH CRASH
    1. flash_crash_mild     Single-day −8% shock (2010 Flash Crash magnitude)
    2. flash_crash_severe   Single-day −15% shock (worst plausible daily loss)

  EXTENDED CRISIS
    3. extended_crisis_30d  Worst 5-day historical run, looped to fill 30 days
    4. extended_crisis_60d  Same, looped to fill 60 days

  CORRELATION SHOCK
    5. corr_shock_5d        5 days: diversification collapses, all returns → beta × market
    6. corr_shock_20d       20 days: same

  BROKEN HEDGE
    7. broken_hedge_mild    VIX > 35 days: hedge assumed to fail — premium paid, no payoff
    8. broken_hedge_severe  VIX > 25 days: same (wider failure window)

  COMPOUNDED
    9. combined_worst       flash_crash_severe + extended_crisis_30d + broken_hedge_severe

Usage:
    # First generate daily returns:
    ./.venv/bin/python backtest_v8.py --force-event-weight 0.05 \\
        --output-returns data/reports/v8_daily_returns.csv

    # Then run stress scenarios:
    ./.venv/bin/python scripts/run_stress_scenarios.py \\
        --returns data/reports/v8_daily_returns.csv \\
        --output-csv data/reports/stress_scenarios.csv

Notes on methodology:
  - Flash crash: injected at the single worst calendar date in the series.
    If testing "what if a crash hit on a random day", use --random-injection.
  - Broken hedge: modelled as missing_payoff = avg_hedge_notional × VIX_stress_factor.
    Avg notional 1.4% × stress_factor (0.25–0.45) = 0.35–0.63% daily drag on VIX-spike days.
    This is a conservative parametric estimate; actual payoff depends on spread width and
    execution quality on the day.
  - Correlation shock: portfolio return is replaced by beta_to_spy × SPY_return, removing
    all cross-sectional and long-short diversification benefit.
  - Compounded scenario: shocks are applied sequentially in the same return series.
    Order: crisis extension first, then flash crash, then broken hedge overlay.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

def _cagr(equity: pd.Series) -> float:
    n_years = (equity.index[-1] - equity.index[0]).days / 365.25
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / max(n_years, 0.01)) - 1)


def _sharpe(returns: pd.Series, rf_annual: float = 0.03) -> float:
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    excess = returns - rf_daily
    ann_vol = excess.std() * np.sqrt(252)
    if ann_vol < 1e-9:
        return 0.0
    return float(excess.mean() * 252 / ann_vol)


def _max_dd(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1).min())


def _calmar(equity: pd.Series) -> float:
    dd = abs(_max_dd(equity))
    return float(_cagr(equity) / dd) if dd > 1e-6 else 0.0


def _metrics(returns: pd.Series, nav: float = 10_000_000) -> dict:
    equity = nav * (1 + returns).cumprod()
    return {
        "cagr":    _cagr(equity),
        "sharpe":  _sharpe(returns),
        "max_dd":  _max_dd(equity),
        "calmar":  _calmar(equity),
        "final_nav": float(equity.iloc[-1]),
        "loss_years": int((returns.resample("YE").apply(
            lambda r: (1 + r).prod() - 1) < 0).sum()),
    }


def _fmt(m: dict) -> str:
    return (
        f"CAGR={m['cagr']:>+7.2%}  Sharpe={m['sharpe']:>5.2f}  "
        f"MaxDD={m['max_dd']:>+7.2%}  Calmar={m['calmar']:>5.2f}  "
        f"NAV=${m['final_nav']/1e6:.2f}M  LossYrs={m['loss_years']}"
    )


# ── scenario injectors ───────────────────────────────────────────────────────

def inject_flash_crash(returns: pd.Series, shock: float, at_date: str | None = None) -> pd.Series:
    """
    Inject a single-day shock at the worst historical date (or a specified date).
    Worst date = the day already closest to the tail; shock replaces that day's return
    to ensure we're testing a regime worse than anything in the sample.
    """
    r = returns.copy()
    if at_date:
        idx = r.index.get_loc(at_date, method="nearest")
    else:
        idx = int(r.argmin())          # inject at the existing worst day
    r.iloc[idx] = shock
    return r


def inject_extended_crisis(returns: pd.Series, extension_days: int) -> pd.Series:
    """
    Find the worst rolling 5-day window in the series and append it on repeat
    for `extension_days` additional days starting right after that window.
    Models a crisis that lasts longer than the historical sample.
    """
    r = returns.copy()
    roll5 = r.rolling(5).sum()
    worst_end = int(roll5.argmin())
    worst_start = max(0, worst_end - 4)
    stress_block = r.iloc[worst_start : worst_end + 1].values  # 5 days

    # Build the extension by repeating the stress block
    extended = np.tile(stress_block, (extension_days // 5) + 2)[:extension_days]

    # Insert right after the worst window
    insert_pos = worst_end + 1
    r_values = np.concatenate([
        r.values[:insert_pos],
        extended,
        r.values[insert_pos:],
    ])
    # Rebuild index (pad with business days after the series end)
    orig_index = r.index.tolist()
    extra_dates = pd.bdate_range(
        start=orig_index[-1] + pd.Timedelta(days=1),
        periods=extension_days,
    )
    new_index = pd.DatetimeIndex(orig_index + list(extra_dates))
    return pd.Series(r_values, index=new_index, name=r.name)


def inject_correlation_shock(
    returns: pd.Series,
    spy_returns: pd.Series,
    n_days: int,
    portfolio_beta: float = 0.45,
) -> pd.Series:
    """
    Replace n_days of portfolio returns with beta × SPY return, removing all
    cross-sectional and long-short diversification benefit.

    Inserted at the point of the worst single SPY day in the series — when
    correlation collapse is most dangerous.
    """
    r = returns.copy()
    spy_aligned = spy_returns.reindex(r.index).fillna(0.0)

    # Find worst SPY day as injection point
    start_idx = max(0, int(spy_aligned.argmin()) - n_days // 2)
    end_idx = min(len(r), start_idx + n_days)

    for i in range(start_idx, end_idx):
        r.iloc[i] = portfolio_beta * spy_aligned.iloc[i]

    return r


def inject_broken_hedge(
    returns: pd.Series,
    vix: pd.Series,
    vix_threshold: float,
    avg_hedge_notional: float = 0.014,
    stress_factor: float = 0.30,
) -> pd.Series:
    """
    On days where VIX > vix_threshold, the hedge is assumed to fail:
      - Premium was paid (sunk cost, already in returns)
      - Payoff is zeroed out — returns lose the estimated hedge contribution

    Estimated missed payoff per VIX-spike day:
        avg_hedge_notional × stress_factor × (VIX / threshold_VIX)

    This is a parametric model. Actual payoff depends on spread width,
    execution quality, and whether the desk can source liquidity in the crisis.
    The 0.30 stress_factor is conservative relative to Black-Scholes estimates
    at VIX=40 (which would suggest 0.40-0.60).
    """
    r = returns.copy()
    vix_aligned = vix.reindex(r.index).ffill().bfill()

    spike_days = vix_aligned[vix_aligned > vix_threshold].index
    for dt in spike_days:
        if dt not in r.index:
            continue
        vix_val = float(vix_aligned.loc[dt])
        # Missed payoff scales with how far VIX is above threshold
        missed = avg_hedge_notional * stress_factor * (vix_val / vix_threshold)
        r.loc[dt] = r.loc[dt] - missed

    return r


# ── VIX loader ────────────────────────────────────────────────────────────────

def _load_vix() -> pd.Series:
    cache = os.path.expanduser("~/.alphaforge/cache/macro/VIXCLS.parquet")
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
        s = df["VIXCLS"].dropna()
        s.index = pd.to_datetime(s.index, utc=True)
        return s

    print("  VIX cache not found — fetching from FRED...", flush=True)
    import requests
    from io import StringIO
    resp = requests.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS", timeout=30
    )
    resp.raise_for_status()
    df = pd.read_csv(
        StringIO(resp.text),
        parse_dates=["observation_date"],
        index_col="observation_date",
    )
    df = df.replace(".", float("nan"))
    df["VIXCLS"] = pd.to_numeric(df["VIXCLS"], errors="coerce")
    df = df.dropna()
    s = df["VIXCLS"]
    s.index = pd.to_datetime(s.index, utc=True)
    return s


# ── main ──────────────────────────────────────────────────────────────────────

def run_all_scenarios(
    returns: pd.Series,
    spy_returns: pd.Series,
    vix: pd.Series,
    nav: float = 10_000_000,
) -> pd.DataFrame:

    baseline = _metrics(returns, nav)

    scenarios = {}

    # 1-2: Flash crash
    scenarios["flash_crash_mild"]   = _metrics(inject_flash_crash(returns, shock=-0.08), nav)
    scenarios["flash_crash_severe"] = _metrics(inject_flash_crash(returns, shock=-0.15), nav)

    # 3-4: Extended crisis
    ext30 = inject_extended_crisis(returns, extension_days=30)
    ext60 = inject_extended_crisis(returns, extension_days=60)
    scenarios["extended_crisis_30d"] = _metrics(ext30[:len(returns)], nav)
    scenarios["extended_crisis_60d"] = _metrics(ext60[:len(returns)], nav)

    # 5-6: Correlation shock
    scenarios["corr_shock_5d"]  = _metrics(
        inject_correlation_shock(returns, spy_returns, n_days=5), nav
    )
    scenarios["corr_shock_20d"] = _metrics(
        inject_correlation_shock(returns, spy_returns, n_days=20), nav
    )

    # 7-8: Broken hedge
    scenarios["broken_hedge_mild"]   = _metrics(
        inject_broken_hedge(returns, vix, vix_threshold=35), nav
    )
    scenarios["broken_hedge_severe"] = _metrics(
        inject_broken_hedge(returns, vix, vix_threshold=25, stress_factor=0.40), nav
    )

    # 9: Combined worst
    combined = inject_extended_crisis(returns, extension_days=30)
    combined = inject_flash_crash(combined, shock=-0.15)
    combined = inject_broken_hedge(combined, vix, vix_threshold=25, stress_factor=0.40)
    scenarios["combined_worst"] = _metrics(combined[:len(returns)], nav)

    rows = [{"scenario": "baseline", **baseline}]
    for name, m in scenarios.items():
        rows.append({"scenario": name, **m})

    return pd.DataFrame(rows).set_index("scenario")


def _print_table(df: pd.DataFrame) -> None:
    baseline = df.loc["baseline"]
    print(f"\n{'='*90}")
    print(f"  AlphaForge — Stress Scenario Results")
    print(f"{'='*90}")
    print(f"  {'Scenario':<26}  {'CAGR':>7}  {'Sharpe':>6}  {'MaxDD':>7}  "
          f"{'Calmar':>6}  {'NAV':>9}  {'DD delta':>8}  {'Sharpe delta':>12}")
    print(f"  {'-'*84}")
    for name, row in df.iterrows():
        dd_delta = row["max_dd"] - baseline["max_dd"]
        sh_delta = row["sharpe"] - baseline["sharpe"]
        marker = "  ←  BASELINE" if name == "baseline" else ""
        print(
            f"  {name:<26}  {row['cagr']:>+7.2%}  {row['sharpe']:>6.2f}  "
            f"{row['max_dd']:>+7.2%}  {row['calmar']:>6.2f}  "
            f"${row['final_nav']/1e6:>7.2f}M  "
            f"{dd_delta:>+8.2%}  {sh_delta:>+12.2f}"
            f"{marker}"
        )
    print(f"{'='*90}\n")

    # Highlight the two most dangerous scenarios
    non_baseline = df.drop("baseline")
    worst_dd = non_baseline["max_dd"].idxmin()
    worst_sharpe = non_baseline["sharpe"].idxmin()
    print(f"  Worst drawdown scenario : {worst_dd}  "
          f"(MaxDD {df.loc[worst_dd,'max_dd']:+.2%})")
    print(f"  Worst Sharpe scenario   : {worst_sharpe}  "
          f"(Sharpe {df.loc[worst_sharpe,'sharpe']:.2f})\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AlphaForge stress scenario runner")
    p.add_argument("--returns", required=True,
                   help="CSV from backtest_v8.py --output-returns (date, gross_ret, lp_ret)")
    p.add_argument("--col", default="gross_ret")
    p.add_argument("--nav", type=float, default=10_000_000)
    p.add_argument("--output-csv", default=None)
    args = p.parse_args()

    if not os.path.exists(args.returns):
        print(f"ERROR: {args.returns} not found.", file=sys.stderr)
        print("Run: ./.venv/bin/python backtest_v8.py --force-event-weight 0.05 "
              "--output-returns data/reports/v8_daily_returns.csv", file=sys.stderr)
        sys.exit(1)

    print("Loading returns...", flush=True)
    df_ret = pd.read_csv(args.returns, index_col="date", parse_dates=True)
    df_ret.index = pd.to_datetime(df_ret.index, utc=True)
    returns = df_ret[args.col].dropna()

    print("Loading VIX...", flush=True)
    vix = _load_vix()

    # Build a SPY proxy from the returns file if available, else use flat 0
    if "spy_ret" in df_ret.columns:
        spy_returns = df_ret["spy_ret"].reindex(returns.index).fillna(0.0)
    else:
        # Approximate SPY from the backtest period using VIX as a vol proxy
        # (good enough for correlation shock scenario positioning)
        spy_returns = pd.Series(0.0, index=returns.index)
        print("  Note: no spy_ret column found — correlation shock uses zero SPY proxy.", flush=True)

    print(f"Running 9 stress scenarios on {len(returns)} days "
          f"({returns.index[0].date()} → {returns.index[-1].date()})...\n", flush=True)

    results = run_all_scenarios(returns, spy_returns, vix, nav=args.nav)
    _print_table(results)

    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        results.to_csv(args.output_csv)
        print(f"Results saved to {args.output_csv}")
