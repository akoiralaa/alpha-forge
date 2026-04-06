#!/usr/bin/env python3
"""
AlphaForge v7 — v4 alpha core plus point-in-time event/sentiment alpha.

This version does not pretend sparse data is rich data. The event sleeve weight
scales with actual coverage so the backtest stays honest when fundamentals/news
coverage is thin.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

import backtest_v4 as v4
from src.signals.event_alpha import EventAlphaConfig, build_event_alpha_signal


def zero_signal_like(prices: pd.DataFrame, equity_syms: list[str]) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=prices.index, columns=equity_syms)


def build_v7_strategy_weights(event_weight: float) -> dict[str, float]:
    weights = v4.default_core_strategy_weights()
    event_weight = float(np.clip(event_weight, 0.0, 0.18))
    remaining = event_weight
    for name, floor in [("high_52w", 0.03), ("carry", 0.02), ("sector_rot", 0.06), ("momentum", 0.18)]:
        available = max(weights[name] - floor, 0.0)
        take = min(available, remaining)
        weights[name] -= take
        remaining -= take
        if remaining <= 1e-12:
            break
    if remaining > 1e-12:
        scale = (sum(weights.values()) - remaining) / max(sum(weights.values()), 1e-12)
        weights = {name: max(weight * scale, 0.0) for name, weight in weights.items()}
    weights.update({
        "mean_reversion": 0.0,
        "bab": 0.0,
        "earnings_drift": event_weight,
    })
    return weights


def run(args):
    universe = v4.load_universe_static(args.config)
    print(f"\nUniverse: {len(universe)} instruments\n")

    cache_hits = sum(
        1
        for sym, at in universe
        if not args.no_cache and os.path.exists(os.path.join(v4.CACHE_DIR, f"{sym}_{at}.parquet"))
    )
    all_cached = cache_hits == len(universe) and not args.no_cache

    dm = v4.build_data_manager_from_env()
    if all_cached:
        print(f"All {cache_hits} instruments cached — skipping provider connection\n")
    else:
        print("Connecting data providers...")
        results = dm.connect_all()
        connected = [k for k, val in results.items() if val]
        print(f"  Connected: {', '.join(connected) if connected else 'none'}")
        if cache_hits:
            print(f"  Cached: {cache_hits}/{len(universe)}")
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

    all_data, dropped_histories, latest_date = v4.filter_histories_for_backtest(all_data, universe)
    if dropped_histories:
        reason_counts = pd.Series([reason for _, _, reason, _, _ in dropped_histories]).value_counts()
        print(
            "\nDropped for history hygiene: "
            + ", ".join(f"{reason}={count}" for reason, count in reason_counts.items())
        )
        for sym, atype, reason, n_bars, end_date in dropped_histories[:12]:
            print(f"  - {sym:8s} ({atype:10s}) {reason:14s} rows={n_bars:4d} end={end_date}")
        if len(dropped_histories) > 12:
            print(f"  ... {len(dropped_histories) - 12} more\n")

    total_bars = sum(len(df) for df in all_data.values())
    latest_str = latest_date.date().isoformat() if latest_date is not None else "n/a"
    print(f"\nTotal: {len(all_data)} instruments, {total_bars:,} bars | Latest: {latest_str}\n")

    prices, volumes, symbols = v4.build_price_matrix(all_data)
    if prices is None:
        sys.exit(1)
    print(f"  Matrix: {prices.shape[0]} x {prices.shape[1]} instruments\n")

    equity_syms = [s for s in symbols if v4.ASSET_TYPES.get(s) == "EQUITY"]
    print(f"  Tradeable equities: {len(equity_syms)}")
    print(f"  Sectors: {len(set(v4.SECTOR_MAP.get(s, 'Other') for s in equity_syms))}\n")

    returns = prices.pct_change().fillna(0)
    mkt_ret = returns["SPY"] if "SPY" in returns.columns else returns.iloc[:, 0]

    print("=" * 70)
    print("  COMPUTING v7 SIGNALS")
    print("=" * 70)
    t_total = time.time()

    strategies = {
        "momentum": v4.strategy_momentum(prices, returns, equity_syms, mkt_ret),
        "mean_reversion": v4.strategy_mean_reversion(prices, returns, equity_syms, mkt_ret),
        "quality": v4.strategy_quality(prices, returns, equity_syms, mkt_ret),
        "bab": zero_signal_like(prices, equity_syms),
        "carry": v4.strategy_carry(prices, returns, equity_syms),
        "sector_rot": v4.strategy_sector_rotation(prices, returns, equity_syms),
        "earnings_drift": zero_signal_like(prices, equity_syms),
        "high_52w": v4.strategy_52w_high(prices, equity_syms),
    }

    event_build = build_event_alpha_signal(
        prices,
        equity_syms,
        fundamentals_path=args.fundamentals_db,
        symbol_master_path=args.symbol_master_db,
        event_store_path=args.events_db,
        config=EventAlphaConfig(
            event_target_weight=args.event_target_weight,
            sentiment_weight=args.sentiment_weight,
        ),
    )
    strategies["earnings_drift"] = event_build.composite
    event_weight = (
        float(args.force_event_weight)
        if args.force_event_weight is not None
        else event_build.suggested_weight
    )
    strategy_weights = build_v7_strategy_weights(event_weight)

    print(f"\n  Total signal computation: {time.time()-t_total:.1f}s")
    print(f"  Event alpha coverage: fundamentals={event_build.fundamental_coverage} | "
          f"estimates={event_build.estimate_coverage} | revisions={event_build.revision_coverage} | "
          f"sentiment={event_build.sentiment_coverage} | events={event_build.total_events}")
    print(f"  Event sentiment source quality: {event_build.sentiment_quality_scale:.2f}")
    print(f"  Event alpha quality scale: {event_build.data_quality_scale:.2f}")
    print(f"  Event alpha live weight: {strategy_weights['earnings_drift']:.1%}\n")

    env = v4.compute_environment(prices)
    breadth = v4.compute_breadth(prices, equity_syms)
    regime_info = v4.compute_regime_states(prices, equity_syms)

    print("=" * 70)
    print("  WALK-FORWARD BACKTEST v7 (v4 core + event alpha)")
    print("=" * 70)
    print(f"  NAV: ${args.nav:,.0f} | Vol target: {args.target_vol:.0%} | "
          f"Rebal: {args.rebal_freq}d | Max pos: {args.max_pos:.0%}")
    print(f"  Target gross: {args.target_gross:.1f}x | Long: {args.n_long} | Short: {args.n_short}")
    print(f"  Strategy weights: momentum={strategy_weights['momentum']:.0%}, "
          f"quality={strategy_weights['quality']:.0%}, carry={strategy_weights['carry']:.0%}, "
          f"sector_rot={strategy_weights['sector_rot']:.0%}, high_52w={strategy_weights['high_52w']:.0%}, "
          f"event={strategy_weights['earnings_drift']:.0%}\n")

    t0 = time.time()
    weights, portfolio_diag = v4.build_multi_strategy_portfolio(
        strategies,
        prices,
        returns,
        env,
        breadth,
        equity_syms,
        args.target_vol,
        args.rebal_freq,
        args.max_pos,
        args.n_long,
        args.n_short,
        args.target_gross,
        regime_info=regime_info,
        hedge_symbol="SPY",
        crisis_hedge_max=args.crisis_hedge_max,
        crisis_hedge_strength=args.crisis_hedge_strength,
        crisis_beta_floor=args.crisis_beta_floor,
        preemptive_de_risk=args.preemptive_de_risk,
        hedge_lookback=args.hedge_lookback,
        strategy_weights=strategy_weights,
        volumes=volumes,
        nav_usd=args.nav,
        use_dynamic_allocator=not args.disable_dynamic_allocator,
        use_capacity_constraints=not args.disable_capacity_constraints,
        capacity_impact_k=args.capacity_impact_k,
        capacity_participation_limit=args.capacity_participation_limit,
        capacity_impact_limit_bps=args.capacity_impact_limit_bps,
        capacity_max_spread_bps=args.capacity_max_spread_bps,
    )
    weights = weights.fillna(0)
    print(f"  Portfolio construction: {time.time()-t0:.1f}s\n")
    if portfolio_diag.get("allocator_enabled"):
        print(
            f"  Dynamic allocator: on | Capacity updates: {portfolio_diag.get('allocator_capacity_updates', 0)} "
            f"| Entropy: {portfolio_diag.get('allocator_weight_entropy', float('nan')):.2f}"
        )
    else:
        print("  Dynamic allocator: off")
    print(
        f"  52w sleeve active: {portfolio_diag.get('high_52w_active_pct', 1.0):.1%} | "
        f"Capacity clamps: {portfolio_diag.get('capacity_clamp_events', 0)} "
        f"(max util seen: {portfolio_diag.get('max_capacity_utilization_seen', 0.0):.2f}x)\n"
    )

    warmup = 300
    trade_cols = list(weights.columns)
    port_ret = (weights.shift(1) * returns[trade_cols]).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    tx_cost = turnover * (args.slippage + 1.0) / 10000
    net_ret = (port_ret - tx_cost).iloc[warmup:]
    weights_post = weights.iloc[warmup:]
    turnover_post = turnover.iloc[warmup:]

    equity_curve = args.nav * (1 + net_ret).cumprod()
    lp_ret = v4.apply_hedge_fund_fees(net_ret, args.mgmt_fee, args.perf_fee)
    lp_equity_curve = args.nav * (1 + lp_ret).cumprod()
    n_years = v4.elapsed_years_from_index(equity_curve.index)

    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / max(n_years, 0.01)) - 1
    s = v4.sharpe(net_ret)
    dd = v4.max_dd(equity_curve)
    lp_cagr = (lp_equity_curve.iloc[-1] / lp_equity_curve.iloc[0]) ** (1 / max(n_years, 0.01)) - 1
    lp_s = v4.sharpe(lp_ret)
    lp_dd = v4.max_dd(lp_equity_curve)
    avg_gross = weights_post.abs().sum(axis=1).mean()
    avg_turn = turnover_post.mean() * 252
    tx_bps = tx_cost.iloc[warmup:].mean() * 252 * 10000 if len(tx_cost) > warmup else 0.0

    spy_ret = prices["SPY"].pct_change().fillna(0).iloc[warmup:]
    spy_ret = spy_ret.reindex(net_ret.index).fillna(0.0)
    spy_eq = args.nav * (1 + spy_ret).cumprod()
    spy_cagr = (spy_eq.iloc[-1] / spy_eq.iloc[0]) ** (1 / max(n_years, 0.01)) - 1
    spy_sharpe = v4.sharpe(spy_ret)
    spy_dd = v4.max_dd(spy_eq)

    print("  ── Aggregate ──")
    print(f"  Gross CAGR:    {cagr:+.2%}   (SPY: {spy_cagr:+.2%})")
    print(f"  Gross Sharpe:  {s:.2f}      (SPY: {spy_sharpe:.2f})")
    print(f"  Gross Max DD:  {dd:.2%}   (SPY: {spy_dd:.2%})")
    print(f"  LP Net CAGR:   {lp_cagr:+.2%}   (after {args.mgmt_fee:.0%}/{args.perf_fee:.0%})")
    print(f"  LP Net Sharpe: {lp_s:.2f}")
    print(f"  LP Net Max DD: {lp_dd:.2%}")
    print(f"  Final NAV:     ${equity_curve.iloc[-1]:,.0f}  (SPY: ${spy_eq.iloc[-1]:,.0f})")
    print(f"  LP NAV:        ${lp_equity_curve.iloc[-1]:,.0f}")
    print(f"  Avg gross:     {avg_gross:.2f}x")
    print(f"  Turnover:      {avg_turn:.0f}x/yr")
    print(f"  Tx costs:      {tx_bps:.0f} bps/yr")
    print(f"  Sample window: {equity_curve.index[0].date()} -> {equity_curve.index[-1].date()} ({n_years:.2f} years)")
    print()

    dates = lp_ret.index
    total_years = len(set(d.year for d in dates))
    hit_return = hit_sharpe = hit_both = 0
    for year in sorted(set(d.year for d in dates)):
        yr = lp_ret[[d.year == year for d in lp_ret.index]]
        yr_eq = lp_equity_curve[[d.year == year for d in lp_equity_curve.index]]
        if len(yr) < 5:
            continue
        yr_ret = (yr_eq.iloc[-1] / yr_eq.iloc[0]) - 1
        yr_s = v4.sharpe(yr)
        meets_return = yr_ret >= args.lp_target_return
        meets_sharpe = yr_s >= args.lp_target_sharpe
        if meets_return:
            hit_return += 1
        if meets_sharpe:
            hit_sharpe += 1
        if meets_return and meets_sharpe:
            hit_both += 1

    print("  ── Annual Hurdles ──")
    print(f"  Years >= {args.lp_target_return:.0%} net: {hit_return}/{total_years}")
    print(f"  Years >= {args.lp_target_sharpe:.1f} Sharpe: {hit_sharpe}/{total_years}")
    print(f"  Years hitting both hurdles: {hit_both}/{total_years}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AlphaForge v7 — v4 core + event alpha")
    p.add_argument("--config", default="config/sp500_universe.yaml")
    p.add_argument("--env", default=".env")
    p.add_argument("--nav", type=float, default=10_000_000)
    p.add_argument("--target-vol", type=float, default=0.15)
    p.add_argument("--target-gross", type=float, default=1.5)
    p.add_argument("--slippage", type=float, default=2.0)
    p.add_argument("--rebal-freq", type=int, default=15)
    p.add_argument("--max-pos", type=float, default=0.06)
    p.add_argument("--n-long", type=int, default=50)
    p.add_argument("--n-short", type=int, default=25)
    p.add_argument("--event-target-weight", type=float, default=0.14)
    p.add_argument("--force-event-weight", type=float, default=None)
    p.add_argument("--sentiment-weight", type=float, default=0.20)
    p.add_argument("--fundamentals-db", default="data/fundamentals.db")
    p.add_argument("--symbol-master-db", default="data/symbol_master.db")
    p.add_argument("--events-db", default="data/events.db")
    p.add_argument("--preemptive-de-risk", type=float, default=0.0)
    p.add_argument("--crisis-hedge-max", type=float, default=0.0)
    p.add_argument("--crisis-hedge-strength", type=float, default=0.75)
    p.add_argument("--crisis-beta-floor", type=float, default=0.15)
    p.add_argument("--hedge-lookback", type=int, default=63)
    p.add_argument("--mgmt-fee", type=float, default=0.02)
    p.add_argument("--perf-fee", type=float, default=0.20)
    p.add_argument("--lp-target-return", type=float, default=0.15)
    p.add_argument("--lp-target-sharpe", type=float, default=1.0)
    p.add_argument("--disable-dynamic-allocator", action="store_true")
    p.add_argument("--disable-capacity-constraints", action="store_true")
    p.add_argument("--capacity-impact-k", type=float, default=0.45)
    p.add_argument("--capacity-participation-limit", type=float, default=0.05)
    p.add_argument("--capacity-impact-limit-bps", type=float, default=15.0)
    p.add_argument("--capacity-max-spread-bps", type=float, default=14.0)
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    if os.path.exists(args.env):
        from dotenv import load_dotenv

        load_dotenv(args.env)

    run(args)
