#!/usr/bin/env python3
"""
One Brain Fund v5 — Capacity-Aware Dynamic Allocation

What is new versus v4:
  - Central allocator tracks realized vs expected performance by strategy
  - Capacity model penalizes crowded / high-impact sleeves
  - Strategy weights adapt through time instead of staying mostly static

What is NOT in this first v5 backtest:
  - NLP / FinBERT / Hugging Face event alpha
  - true intraday/HFT alpha (the local "tick" store is daily-snapshot-like)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

import backtest_v4 as v4
from src.data.ingest.data_manager import build_data_manager_from_env
from src.portfolio.allocator import CentralRiskAllocator, StrategyExpectation
from src.portfolio.capacity import LiquidityCapacityModel, LiquiditySnapshot


V5_STRATEGIES = ("momentum", "quality", "carry", "sector_rot", "high_52w")

V5_EXPECTATIONS = {
    "momentum": StrategyExpectation(
        "momentum", expected_return_annual=0.13, expected_vol_annual=0.18,
        base_weight=0.28, min_weight=0.05, max_weight=0.42,
    ),
    "quality": StrategyExpectation(
        "quality", expected_return_annual=0.10, expected_vol_annual=0.12,
        base_weight=0.24, min_weight=0.05, max_weight=0.38,
    ),
    "carry": StrategyExpectation(
        "carry", expected_return_annual=0.08, expected_vol_annual=0.10,
        base_weight=0.16, min_weight=0.03, max_weight=0.28,
    ),
    "sector_rot": StrategyExpectation(
        "sector_rot", expected_return_annual=0.09, expected_vol_annual=0.12,
        base_weight=0.12, min_weight=0.03, max_weight=0.24,
    ),
    "high_52w": StrategyExpectation(
        "high_52w", expected_return_annual=0.11, expected_vol_annual=0.14,
        base_weight=0.20, min_weight=0.04, max_weight=0.34,
    ),
}


def build_spread_matrix(index, symbols):
    spread_map = {}
    for sym in symbols:
        atype = v4.ASSET_TYPES.get(sym, "EQUITY")
        if atype == "FX":
            spread_map[sym] = 1.5
        elif atype in {"ETF", "BOND", "FUTURE", "COMMODITY", "VOLATILITY"}:
            spread_map[sym] = 2.0
        else:
            spread_map[sym] = 5.0
    return pd.DataFrame({sym: spread_map[sym] for sym in symbols}, index=index)


def build_daily_liquidity_inputs(prices, volumes, symbols):
    dollar_volume = (prices[symbols] * volumes[symbols]).replace([np.inf, -np.inf], np.nan)
    adv_usd = dollar_volume.rolling(20, min_periods=5).mean().bfill()
    realized_vol = prices[symbols].pct_change().rolling(20, min_periods=10).std().bfill() * np.sqrt(252)
    spread_bps = build_spread_matrix(prices.index, symbols)
    return adv_usd, realized_vol, spread_bps


def safe_scalar(value, default):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def build_strategy_sleeve(
    signal,
    prices,
    returns,
    env,
    breadth,
    equity_syms,
    target_gross,
    rebal_freq,
    max_pos,
    n_long,
    n_short,
    regime_info=None,
    blend_rate=0.88,
    min_signal=0.03,
):
    inst_vol = returns[equity_syms].rolling(20, min_periods=10).std() * np.sqrt(252)
    inst_vol = inst_vol.bfill().clip(lower=0.04)

    sym_sector = {sym: v4.SECTOR_MAP.get(sym, "Other") for sym in equity_syms}
    sector_stocks = defaultdict(list)
    for sym in equity_syms:
        sector_stocks[sym_sector[sym]].append(sym)

    weights = pd.DataFrame(0.0, index=prices.index, columns=equity_syms)
    target_w = pd.Series(0.0, index=equity_syms)
    live_w = pd.Series(0.0, index=equity_syms)

    running_nav = 1.0
    peak_nav = 1.0

    for i in range(280, len(prices)):
        if i > 280:
            day_ret = float((live_w * returns[equity_syms].iloc[i]).sum())
            running_nav *= (1 + day_ret)
            peak_nav = max(peak_nav, running_nav)

        current_dd = (running_nav / peak_nav) - 1 if peak_nav > 0 else 0.0
        dd_scale = 1.0
        if current_dd < -0.25:
            dd_scale = 0.55
        elif current_dd < -0.15:
            dd_scale = 0.80

        regime_allows_shorts = False
        if regime_info is not None:
            regime_row = regime_info.iloc[i]
            regime_allows_shorts = bool(regime_row.get("allow_shorts", False))

        if i % rebal_freq != 0 and i > 280:
            live_w = target_w * dd_scale
            weights.iloc[i] = live_w
            continue

        row = signal.iloc[i].dropna()
        row = row[row.abs() > min_signal]
        if len(row) < 20:
            live_w = target_w * dd_scale
            weights.iloc[i] = live_w
            continue

        e = float(env.iloc[i])
        b = float(breadth.iloc[i]) if i < len(breadth) else 0.5
        bearish = e < 0.80 or b < 0.42
        shorts_active = regime_allows_shorts and bearish

        long_picks, short_picks = [], []
        for _, syms in sector_stocks.items():
            sector_sigs = row.reindex(syms).dropna().sort_values(ascending=False)
            if len(sector_sigs) < 2:
                continue

            n_l = max(int(len(sector_sigs) * 0.25), 1)
            for sym in sector_sigs.head(n_l).index:
                if sector_sigs[sym] > min_signal:
                    long_picks.append((sym, float(sector_sigs[sym])))

            if shorts_active:
                n_s = max(int(len(sector_sigs) * 0.12), 1)
                for sym in sector_sigs.tail(n_s).index:
                    if sector_sigs[sym] < -max(min_signal, 0.12):
                        short_picks.append((sym, float(sector_sigs[sym])))

        long_picks.sort(key=lambda x: x[1], reverse=True)
        short_picks.sort(key=lambda x: x[1])
        long_picks = long_picks[:n_long]
        short_picks = short_picks[:n_short]

        w = pd.Series(0.0, index=equity_syms)
        breadth_mult = float(np.clip(b * 1.4, 0.60, 1.15))
        if shorts_active:
            long_budget = target_gross * min(e, 1.10) * 0.78 * breadth_mult
            short_budget = target_gross * 0.18 * max(0.25, 1 - breadth_mult / 1.15)
        else:
            long_budget = target_gross * min(e, 1.30) * breadth_mult
            short_budget = 0.0

        long_budget *= dd_scale
        short_budget *= dd_scale

        if long_picks:
            inv = {
                sym: 1.0 / max(float(inst_vol[sym].iloc[i]), 0.05)
                for sym, _ in long_picks
            }
            total_inv = sum(inv.values())
            for sym, sig_val in long_picks:
                raw_w = long_budget * inv[sym] / max(total_inv, 1e-12)
                tilt = 0.75 + 0.50 * min(abs(sig_val), 1.0)
                w[sym] = float(np.clip(raw_w * tilt, 0.0, max_pos))

        if short_picks and short_budget > 0:
            inv = {
                sym: 1.0 / max(float(inst_vol[sym].iloc[i]), 0.05)
                for sym, _ in short_picks
            }
            total_inv = sum(inv.values())
            for sym, _ in short_picks:
                raw_w = -short_budget * inv[sym] / max(total_inv, 1e-12)
                w[sym] = float(np.clip(raw_w, -max_pos * 0.45, 0.0))

        prev = target_w.copy()
        delta = w - prev
        small = delta.abs() < 0.0025
        w[small] = prev[small]

        keep = pd.Series(blend_rate, index=equity_syms)
        entering = (prev.abs() < 1e-12) & (w.abs() > 1e-12)
        exiting = (prev.abs() > 1e-12) & (w.abs() < 1e-12)
        flipping = (
            (prev.abs() > 1e-12)
            & (w.abs() > 1e-12)
            & (np.sign(prev) != np.sign(w))
        )
        keep[entering] = min(blend_rate, 0.70)
        keep[exiting] = 0.50
        keep[flipping] = 0.30
        w = (1 - keep) * w + keep * prev
        w[(w.abs() < 0.0015) & exiting] = 0.0

        target_w = w.clip(lower=-max_pos * 0.45, upper=max_pos)
        live_w = target_w * dd_scale
        weights.iloc[i] = live_w

    return weights


def compute_strategy_sleeves(strategies, prices, returns, env, breadth, equity_syms, args, regime_info):
    sleeves = {}
    for name in V5_STRATEGIES:
        print(f"  Building sleeve: {name}...", end=" ", flush=True)
        t0 = time.time()
        sleeves[name] = build_strategy_sleeve(
            strategies[name], prices, returns, env, breadth, equity_syms,
            target_gross=args.target_gross,
            rebal_freq=args.rebal_freq,
            max_pos=args.max_pos,
            n_long=args.n_long,
            n_short=args.n_short,
            regime_info=regime_info,
        )
        print(f"{time.time()-t0:.1f}s")
    return sleeves


def build_v5_portfolio(sleeves, prices, returns, volumes, equity_syms, args, regime_info):
    adv_usd, realized_vol, spread_bps = build_daily_liquidity_inputs(prices, volumes, equity_syms)
    allocator = CentralRiskAllocator(
        [V5_EXPECTATIONS[name] for name in V5_STRATEGIES],
        exploration_floor=args.exploration_floor,
        capacity_soft_limit=args.capacity_soft_limit,
        min_observations=max(args.rebal_freq, 20),
    )
    capacity_model = LiquidityCapacityModel(impact_k=args.impact_k)

    strat_turnover = {name: sleeves[name].diff().abs().sum(axis=1).fillna(0.0) for name in V5_STRATEGIES}
    strat_cost = {
        name: strat_turnover[name] * (args.slippage + 1.0) / 10000.0
        for name in V5_STRATEGIES
    }
    strat_ret = {
        name: (sleeves[name].shift(1) * returns[equity_syms]).sum(axis=1).fillna(0.0) - strat_cost[name]
        for name in V5_STRATEGIES
    }

    combined = pd.DataFrame(0.0, index=prices.index, columns=equity_syms)
    strategy_weight_history = pd.DataFrame(0.0, index=prices.index, columns=V5_STRATEGIES)
    capacity_history = pd.DataFrame(0.0, index=prices.index, columns=V5_STRATEGIES)

    nav = args.nav
    warmup = 300
    for i in range(warmup, len(prices)):
        if i > warmup:
            for name in V5_STRATEGIES:
                allocator.observe(name, float(strat_ret[name].iloc[i - 1]))

        if i % args.rebal_freq == 0:
            for name in V5_STRATEGIES:
                row = sleeves[name].iloc[i]
                active = row[row.abs() > 1e-8]
                if active.empty:
                    continue
                snapshots = {
                    sym: LiquiditySnapshot(
                        symbol_id=sym,
                        price=safe_scalar(prices[sym].iloc[i], 0.0),
                        adv_usd=max(safe_scalar(adv_usd[sym].iloc[i], 0.0), 1.0),
                        spread_bps=safe_scalar(spread_bps[sym].iloc[i], 5.0),
                        realized_vol_daily=max(safe_scalar(realized_vol[sym].iloc[i], 0.0), 1e-4),
                    )
                    for sym in active.index
                }
                est = capacity_model.estimate_strategy_capacity(
                    name,
                    active.to_dict(),
                    snapshots,
                    nav_usd=nav,
                    turnover=max(float(strat_turnover[name].iloc[i]), 1.0 / max(args.rebal_freq, 1)),
                )
                allocator.observe_capacity(
                    name,
                    utilization=est.utilization,
                    capacity_nav_limit=est.nav_capacity_usd,
                    impact_bps=est.weighted_impact_bps,
                )
                capacity_history.iloc[i, capacity_history.columns.get_loc(name)] = est.utilization

        strat_w = allocator.target_weights()
        strategy_weight_history.iloc[i] = pd.Series(strat_w)

        row = pd.Series(0.0, index=equity_syms)
        for name, w in strat_w.items():
            row = row.add(sleeves[name].iloc[i] * w, fill_value=0.0)

        gross = row.abs().sum()
        if gross > 1e-8:
            row *= args.target_gross / gross
        combined.iloc[i] = row.clip(lower=-args.max_pos, upper=args.max_pos)

        if i > warmup:
            day_ret = float((combined.iloc[i - 1] * returns[equity_syms].iloc[i]).sum())
            nav *= (1 + day_ret)

    port_ret = (combined.shift(1) * returns[equity_syms]).sum(axis=1)
    port_rvol = port_ret.rolling(20, min_periods=10).std() * np.sqrt(252)
    final_weights = combined.copy()
    for i in range(320, len(prices)):
        rv = port_rvol.iloc[i]
        if np.isnan(rv) or rv < 0.01:
            continue
        scale = float(np.clip(args.target_vol / rv, 0.35, 1.85))
        final_weights.iloc[i] = combined.iloc[i] * scale

    return final_weights, strategy_weight_history, capacity_history, strat_ret


def run(args):
    universe = v4.load_universe_static(args.config)
    print(f"\nUniverse: {len(universe)} instruments\n")

    cache_hits = sum(
        1 for sym, at in universe
        if not args.no_cache and os.path.exists(os.path.join(v4.CACHE_DIR, f"{sym}_{at}.parquet"))
    )
    all_cached = cache_hits == len(universe) and not args.no_cache

    dm = build_data_manager_from_env()
    if all_cached:
        print(f"All {cache_hits} instruments cached — skipping provider connection\n")
    else:
        print("Connecting data providers...")
        results = dm.connect_all()
        connected = [k for k, v in results.items() if v]
        print(f"  Connected: {', '.join(connected) if connected else 'none'}")
        print()

    print("=" * 70)
    print("  FETCHING HISTORICAL DATA")
    print("=" * 70)
    all_data = {}
    for idx, (sym, atype) in enumerate(universe, 1):
        print(f"  [{idx:3d}/{len(universe)}] {sym:8s} ({atype:10s}) ", end="", flush=True)
        df = v4.fetch_daily_bars(dm, sym, atype, use_cache=not args.no_cache)
        if df.empty:
            print("-- no data")
            continue
        d0 = df["date"].iloc[0].strftime("%Y-%m-%d")
        d1 = df["date"].iloc[-1].strftime("%Y-%m-%d")
        print(f"-- {len(df):,} days ({d0} to {d1})")
        all_data[sym] = df
    dm.disconnect_all()

    prices, volumes, symbols = v4.build_price_matrix(all_data)
    if prices is None:
        sys.exit(1)

    equity_syms = [s for s in symbols if v4.ASSET_TYPES.get(s) == "EQUITY"]
    returns = prices.pct_change().fillna(0)
    mkt_ret = returns["SPY"] if "SPY" in returns.columns else returns.iloc[:, 0]

    print("\n" + "=" * 70)
    print("  COMPUTING V5 STRATEGY SIGNALS")
    print("=" * 70)
    strategies = {
        "momentum": v4.strategy_momentum(prices, returns, equity_syms, mkt_ret),
        "quality": v4.strategy_quality(prices, returns, equity_syms, mkt_ret),
        "carry": v4.strategy_carry(prices, returns, equity_syms),
        "sector_rot": v4.strategy_sector_rotation(prices, returns, equity_syms),
        "high_52w": v4.strategy_52w_high(prices, equity_syms),
    }
    print()

    env = v4.compute_environment(prices)
    breadth = v4.compute_breadth(prices, equity_syms)
    regime_info = v4.compute_regime_states(prices, equity_syms)

    print("=" * 70)
    print("  BUILDING V5 STRATEGY SLEEVES")
    print("=" * 70)
    sleeves = compute_strategy_sleeves(
        strategies, prices, returns, env, breadth, equity_syms, args, regime_info
    )
    print()

    print("=" * 70)
    print("  WALK-FORWARD BACKTEST v5 (allocator + capacity)")
    print("=" * 70)
    print(f"  NAV: ${args.nav:,.0f} | Vol target: {args.target_vol:.0%} | Rebal: {args.rebal_freq}d")
    print(f"  Gross target: {args.target_gross:.1f}x | Long: {args.n_long} | Short: {args.n_short}")
    print(f"  Exploration floor: {args.exploration_floor:.0%} | Capacity soft limit: {args.capacity_soft_limit:.0%}\n")

    t0 = time.time()
    weights, strategy_weight_history, capacity_history, strat_ret = build_v5_portfolio(
        sleeves, prices, returns, volumes, equity_syms, args, regime_info
    )
    print(f"  Portfolio construction: {time.time()-t0:.1f}s\n")

    warmup = 300
    port_ret = (weights.shift(1) * returns[equity_syms]).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    tx_cost = turnover * (args.slippage + 1.0) / 10000
    net_ret = (port_ret - tx_cost).iloc[warmup:]
    weights_post = weights.iloc[warmup:]
    turnover_post = turnover.iloc[warmup:]

    equity_curve = args.nav * (1 + net_ret).cumprod()
    lp_ret = v4.apply_hedge_fund_fees(net_ret, args.mgmt_fee, args.perf_fee)
    lp_equity_curve = args.nav * (1 + lp_ret).cumprod()
    dates = equity_curve.index
    n_years = len(net_ret) / 252

    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / max(n_years, 0.01)) - 1
    s = v4.sharpe(net_ret)
    so = v4.sortino(net_ret)
    dd = v4.max_dd(equity_curve)
    lp_cagr = (lp_equity_curve.iloc[-1] / lp_equity_curve.iloc[0]) ** (1 / max(n_years, 0.01)) - 1
    lp_s = v4.sharpe(lp_ret)
    lp_dd = v4.max_dd(lp_equity_curve)
    avg_gross = weights_post.abs().sum(axis=1).mean()
    avg_turn = turnover_post.mean() * 252
    tx_bps = tx_cost.iloc[warmup:].mean() * 252 * 10000 if len(tx_cost) > warmup else 0

    spy_ret = prices["SPY"].pct_change().fillna(0).iloc[warmup:]
    spy_eq = args.nav * (1 + spy_ret).cumprod()
    spy_cagr = (spy_eq.iloc[-1] / spy_eq.iloc[0]) ** (1 / max(n_years, 0.01)) - 1
    spy_sharpe = v4.sharpe(spy_ret)
    spy_dd = v4.max_dd(spy_eq)

    print("  ── Aggregate ──")
    print(f"  Gross CAGR:    {cagr:+.2%}   (SPY: {spy_cagr:+.2%})")
    print(f"  Gross Sharpe:  {s:.2f}      (SPY: {spy_sharpe:.2f})")
    print(f"  Gross Sortino: {so:.2f}")
    print(f"  Gross Max DD:  {dd:.2%}   (SPY: {spy_dd:.2%})")
    print(f"  LP Net CAGR:   {lp_cagr:+.2%}   (after {args.mgmt_fee:.0%}/{args.perf_fee:.0%})")
    print(f"  LP Net Sharpe: {lp_s:.2f}")
    print(f"  LP Net Max DD: {lp_dd:.2%}")
    print(f"  Final NAV:     ${equity_curve.iloc[-1]:,.0f}  (SPY: ${spy_eq.iloc[-1]:,.0f})")
    print(f"  LP NAV:        ${lp_equity_curve.iloc[-1]:,.0f}")
    print(f"  Avg gross:     {avg_gross:.2f}x")
    print(f"  Turnover:      {avg_turn:.0f}x/yr")
    print(f"  Tx costs:      {tx_bps:.0f} bps/yr")
    print()

    avg_strategy_weights = strategy_weight_history.iloc[warmup:].mean().sort_values(ascending=False)
    avg_capacity = capacity_history.iloc[warmup:].replace(0, np.nan).mean().fillna(0).sort_values(ascending=False)
    print("  ── Strategy Weight Mix ──")
    for name, weight in avg_strategy_weights.items():
        cap = avg_capacity.get(name, 0.0)
        realized = v4.sharpe(strat_ret[name].iloc[warmup:])
        print(f"  {name:<12s} avg_w={weight:>5.1%} | sleeve_sharpe={realized:>4.2f} | cap_util={cap:>5.2f}")
    print()

    print("  ── Year-by-Year: LP Net vs SPY ──")
    print(f"  {'Year':<6} {'LP Ret':>8} {'SPY':>8} {'Alpha':>8} {'LP Shp':>8} {'Hit':>4}")
    print("  " + "-" * 46)
    hit_both = 0
    hit_return = 0
    hit_sharpe = 0
    years = sorted(set(d.year if hasattr(d, 'year') else d.date().year for d in dates))
    for year in years:
        mask = pd.Series([d.year if hasattr(d, 'year') else d.date().year for d in dates], index=lp_ret.index) == year
        yr = lp_ret[mask]
        yr_eq = lp_equity_curve[mask]
        if len(yr) < 5:
            continue
        yr_ret = (yr_eq.iloc[-1] / yr_eq.iloc[0]) - 1
        yr_s = v4.sharpe(yr)

        spy_mask = pd.Series([d.year if hasattr(d, 'year') else d.date().year for d in spy_ret.index], index=spy_ret.index) == year
        spy_yr = spy_ret[spy_mask]
        spy_eq_yr = spy_eq[spy_mask]
        spy_yr_ret = (spy_eq_yr.iloc[-1] / spy_eq_yr.iloc[0]) - 1 if len(spy_eq_yr) > 1 else 0
        alpha = yr_ret - spy_yr_ret
        meets_return = yr_ret >= args.lp_target_return
        meets_sharpe = yr_s >= args.lp_target_sharpe
        if meets_return:
            hit_return += 1
        if meets_sharpe:
            hit_sharpe += 1
        if meets_return and meets_sharpe:
            hit_both += 1
        print(f"  {year:<6} {yr_ret:>+7.2%} {spy_yr_ret:>+7.2%} {alpha:>+7.2%} {yr_s:>7.2f} {'Y' if meets_return and meets_sharpe else 'N':>4}")
    print("  " + "-" * 46)
    print(f"  Years >= {args.lp_target_return:.0%} net: {hit_return}/{len(years)}")
    print(f"  Years >= {args.lp_target_sharpe:.1f} Sharpe: {hit_sharpe}/{len(years)}")
    print(f"  Years hitting both hurdles: {hit_both}/{len(years)}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="One Brain Fund v5 — allocator + capacity")
    p.add_argument("--config", default="config/sp500_universe.yaml")
    p.add_argument("--env", default=".env")
    p.add_argument("--nav", type=float, default=10_000_000)
    p.add_argument("--target-vol", type=float, default=0.15)
    p.add_argument("--target-gross", type=float, default=1.5)
    p.add_argument("--slippage", type=float, default=2.0)
    p.add_argument("--rebal-freq", type=int, default=15)
    p.add_argument("--max-pos", type=float, default=0.06)
    p.add_argument("--n-long", type=int, default=45)
    p.add_argument("--n-short", type=int, default=18)
    p.add_argument("--mgmt-fee", type=float, default=0.02)
    p.add_argument("--perf-fee", type=float, default=0.20)
    p.add_argument("--lp-target-return", type=float, default=0.15)
    p.add_argument("--lp-target-sharpe", type=float, default=1.0)
    p.add_argument("--exploration-floor", type=float, default=0.15)
    p.add_argument("--capacity-soft-limit", type=float, default=0.70)
    p.add_argument("--impact-k", type=float, default=0.5)
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    if os.path.exists(args.env):
        from dotenv import load_dotenv
        load_dotenv(args.env)

    run(args)
