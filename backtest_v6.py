#!/usr/bin/env python3
"""
AlphaForge v6 — v4 alpha core with capacity clipping and sparse PIT events.

Design goals:
  - keep the stronger v4 multi-factor engine as the return core
  - use only mild allocator tilts, not full v5-style rotation
  - hard-clip crowded sleeves and oversized rebalance deltas
  - add a small fundamental event sleeve only where PIT data exists

This is a more investable evolution, not a claim that we suddenly solved
Medallion-scale alpha with the current data.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import backtest_v4 as v4
import backtest_v5 as v5
from src.data.fundamentals import FundamentalsStore
from src.data.ingest.data_manager import build_data_manager_from_env
from src.data.symbol_master import SymbolMaster
from src.portfolio.allocator import CentralRiskAllocator, StrategyExpectation
from src.portfolio.capacity import LiquidityCapacityModel, LiquiditySnapshot


V6_STRATEGIES = ("momentum", "quality", "carry", "sector_rot", "high_52w", "pit_event")

V6_EXPECTATIONS = {
    "momentum": StrategyExpectation(
        "momentum", expected_return_annual=0.13, expected_vol_annual=0.18,
        base_weight=0.25, min_weight=0.15, max_weight=0.32,
    ),
    "quality": StrategyExpectation(
        "quality", expected_return_annual=0.10, expected_vol_annual=0.12,
        base_weight=0.22, min_weight=0.12, max_weight=0.30,
    ),
    "carry": StrategyExpectation(
        "carry", expected_return_annual=0.08, expected_vol_annual=0.10,
        base_weight=0.16, min_weight=0.08, max_weight=0.22,
    ),
    "sector_rot": StrategyExpectation(
        "sector_rot", expected_return_annual=0.09, expected_vol_annual=0.12,
        base_weight=0.09, min_weight=0.03, max_weight=0.14,
    ),
    "high_52w": StrategyExpectation(
        "high_52w", expected_return_annual=0.11, expected_vol_annual=0.14,
        base_weight=0.23, min_weight=0.12, max_weight=0.30,
    ),
    "pit_event": StrategyExpectation(
        "pit_event", expected_return_annual=0.06, expected_vol_annual=0.15,
        base_weight=0.05, min_weight=0.00, max_weight=0.08,
    ),
}

REGIME_TILTS = {
    "LOW_VOL_TRENDING": {
        "momentum": 1.12,
        "quality": 0.97,
        "carry": 0.95,
        "sector_rot": 1.00,
        "high_52w": 1.15,
        "pit_event": 1.00,
    },
    "MEAN_REVERTING_RANGE": {
        "momentum": 0.92,
        "quality": 1.04,
        "carry": 1.00,
        "sector_rot": 1.10,
        "high_52w": 0.88,
        "pit_event": 1.00,
    },
    "HIGH_VOL_TRENDING": {
        "momentum": 0.98,
        "quality": 1.08,
        "carry": 1.05,
        "sector_rot": 1.00,
        "high_52w": 0.92,
        "pit_event": 1.08,
    },
    "HIGH_VOL_CHAOTIC": {
        "momentum": 0.82,
        "quality": 1.18,
        "carry": 1.08,
        "sector_rot": 0.92,
        "high_52w": 0.76,
        "pit_event": 1.12,
    },
    "LIQUIDITY_CRISIS": {
        "momentum": 0.74,
        "quality": 1.20,
        "carry": 1.12,
        "sector_rot": 0.95,
        "high_52w": 0.66,
        "pit_event": 1.15,
    },
}


def build_pit_event_signal(
    prices: pd.DataFrame,
    equity_syms: list[str],
    fundamentals_path: str = "data/fundamentals.db",
    symbol_master_path: str = "data/symbol_master.db",
    decay_days: int = 42,
) -> tuple[pd.DataFrame, int]:
    """
    Build a sparse post-event signal from real PIT earnings surprises.

    The local database is thin, so we keep this sleeve small and honest:
    if there is only one supported name, it becomes a small orthogonal overlay,
    not a major return engine.
    """
    event_signal = pd.DataFrame(0.0, index=prices.index, columns=equity_syms)
    if not (os.path.exists(fundamentals_path) and os.path.exists(symbol_master_path)):
        return event_signal, 0

    fund = FundamentalsStore(fundamentals_path)
    sm = SymbolMaster(symbol_master_path)
    supported = set()
    try:
        fund_df = fund.to_dataframe()
        if fund_df.empty:
            return event_signal, 0

        metrics = {"EPS", "ANALYST_EST_EPS"}
        fund_df = fund_df[fund_df["metric_name"].isin(metrics)].copy()
        if fund_df.empty:
            return event_signal, 0

        for canonical_id, grp in fund_df.groupby("canonical_id"):
            grp = grp.sort_values("published_at_ns")
            actuals = grp[grp["metric_name"] == "EPS"]
            ests = grp[grp["metric_name"] == "ANALYST_EST_EPS"].sort_values("published_at_ns")
            if actuals.empty or ests.empty:
                continue

            for actual in actuals.itertuples():
                surprise = fund.get_earnings_surprise(canonical_id, int(actual.published_at_ns))
                if surprise is None or not np.isfinite(surprise):
                    continue

                ticker = sm.get_ticker_at(canonical_id, int(actual.published_at_ns))
                if ticker not in event_signal.columns:
                    continue

                supported.add(ticker)
                est_hist = ests[ests["published_at_ns"] <= actual.published_at_ns]
                revision = 0.0
                if len(est_hist) >= 2:
                    prev_est = float(est_hist.iloc[-2]["value"])
                    last_est = float(est_hist.iloc[-1]["value"])
                    if abs(prev_est) > 1e-9:
                        revision = (last_est - prev_est) / abs(prev_est)

                strength = float(np.clip(3.5 * surprise + 1.5 * revision, -1.5, 1.5))
                start_dt = pd.to_datetime(int(actual.published_at_ns), unit="ns", utc=True)
                if prices.index.tz is None:
                    start_dt = start_dt.tz_localize(None)
                else:
                    start_dt = start_dt.tz_convert(prices.index.tz)
                start_idx = prices.index.searchsorted(start_dt, side="left")
                if start_idx >= len(prices.index):
                    continue

                for offset in range(decay_days):
                    j = start_idx + offset
                    if j >= len(prices.index):
                        break
                    decay = np.exp(-offset / 10.0)
                    event_signal.iat[j, event_signal.columns.get_loc(ticker)] += strength * decay
    finally:
        sm.close()
        fund.close()

    ranked = v4.cross_sectional_rank(event_signal)
    return ranked.fillna(0.0), len(supported)


def compute_ic_boosts(strategies, returns, equity_syms, start=300, step=20):
    fwd_5 = returns[equity_syms].rolling(5).sum().shift(-5)
    boosts = pd.DataFrame(1.0, index=returns.index, columns=list(strategies))
    for name, sig in strategies.items():
        for i in range(start, len(returns), step):
            ics = []
            for t in range(max(0, i - 63), i, 5):
                f_row = sig.iloc[t].reindex(equity_syms).values
                r_row = fwd_5.iloc[t].values
                valid = ~(np.isnan(f_row) | np.isnan(r_row))
                if valid.sum() <= 20:
                    continue
                ic = np.corrcoef(rankdata(f_row[valid]), rankdata(r_row[valid]))[0, 1]
                if np.isfinite(ic):
                    ics.append(ic)
            mult = 1.0
            if ics:
                mult = float(np.clip(1.0 + np.mean(ics) * 8.0, 0.80, 1.20))
            boosts.iloc[i:min(i + step, len(boosts)), boosts.columns.get_loc(name)] = mult
    return boosts


def build_snapshots(prices, adv_usd, realized_vol, spread_bps, i, symbols):
    snapshots = {}
    for sym in symbols:
        if sym not in prices.columns:
            continue
        snapshots[sym] = LiquiditySnapshot(
            symbol_id=sym,
            price=v5.safe_scalar(prices[sym].iloc[i], 0.0),
            adv_usd=max(v5.safe_scalar(adv_usd[sym].iloc[i], 0.0), 1.0),
            spread_bps=v5.safe_scalar(spread_bps[sym].iloc[i], 5.0),
            realized_vol_daily=max(v5.safe_scalar(realized_vol[sym].iloc[i], 0.0), 1e-4),
        )
    return snapshots


def clip_rebalance_delta(prev_w, new_w, nav_usd, snapshots, capacity_model):
    clipped = new_w.copy()
    clip_count = 0
    for sym, target in new_w.items():
        delta = float(target - prev_w.get(sym, 0.0))
        if abs(delta) < 1e-12:
            continue
        snap = snapshots.get(sym)
        if snap is None:
            continue
        desired_notional = abs(delta) * nav_usd
        max_order, _ = capacity_model.max_order_notional_usd(snap)
        if not np.isfinite(max_order) or max_order <= 0 or desired_notional <= max_order:
            continue
        scale = max_order / max(desired_notional, 1e-12)
        clipped[sym] = prev_w.get(sym, 0.0) + delta * scale
        clip_count += 1
    return clipped, clip_count


def build_v6_portfolio(strategies, sleeves, prices, returns, volumes, equity_syms, args, regime_info, env, breadth, ic_boosts):
    adv_usd, realized_vol, spread_bps = v5.build_daily_liquidity_inputs(prices, volumes, equity_syms)
    capacity_model = LiquidityCapacityModel(impact_k=args.impact_k)
    allocator = CentralRiskAllocator(
        [V6_EXPECTATIONS[name] for name in V6_STRATEGIES],
        exploration_floor=args.exploration_floor,
        capacity_soft_limit=args.capacity_soft_limit,
        min_observations=max(args.rebal_freq * 2, 40),
    )

    strat_turnover = {name: sleeves[name].diff().abs().sum(axis=1).fillna(0.0) for name in V6_STRATEGIES}
    strat_cost = {
        name: strat_turnover[name] * (args.slippage + 1.0) / 10000.0
        for name in V6_STRATEGIES
    }
    strat_ret = {
        name: (sleeves[name].shift(1) * returns[equity_syms]).sum(axis=1).fillna(0.0) - strat_cost[name]
        for name in V6_STRATEGIES
    }

    inst_vol = returns[equity_syms].rolling(20, min_periods=10).std() * np.sqrt(252)
    inst_vol = inst_vol.bfill().clip(lower=0.02)

    sym_sector = {sym: v4.SECTOR_MAP.get(sym, "Other") for sym in equity_syms}
    sector_stocks = defaultdict(list)
    for sym in equity_syms:
        sector_stocks[sym_sector[sym]].append(sym)

    weights = pd.DataFrame(0.0, index=prices.index, columns=equity_syms)
    strategy_weight_history = pd.DataFrame(0.0, index=prices.index, columns=V6_STRATEGIES)
    strategy_cap_history = pd.DataFrame(0.0, index=prices.index, columns=V6_STRATEGIES)
    portfolio_capacity_scale = pd.Series(1.0, index=prices.index)
    symbol_clip_counts = pd.Series(0.0, index=prices.index)

    target_w = pd.Series(0.0, index=equity_syms)
    live_w = pd.Series(0.0, index=equity_syms)
    current_strat_weights = pd.Series(
        {name: V6_EXPECTATIONS[name].base_weight for name in V6_STRATEGIES}, dtype=float
    )
    current_strat_weights /= current_strat_weights.sum()
    current_cap_util = pd.Series(0.0, index=V6_STRATEGIES)
    current_port_cap_scale = 1.0

    running_nav = 1.0
    peak_nav = 1.0
    nav_usd = args.nav

    warmup = 300
    for i in range(warmup, len(prices)):
        if i > warmup:
            day_ret = float((live_w * returns[equity_syms].iloc[i]).sum())
            running_nav *= (1 + day_ret)
            peak_nav = max(peak_nav, running_nav)
            nav_usd *= (1 + day_ret)
            for name in V6_STRATEGIES:
                allocator.observe(name, float(strat_ret[name].iloc[i - 1]))

        current_dd = (running_nav / peak_nav) - 1 if peak_nav > 0 else 0.0
        dd_scale = 1.0
        if current_dd < -0.30:
            dd_scale = 0.45
        elif current_dd < -0.22:
            dd_scale = 0.70
        elif current_dd < -0.12:
            dd_scale = 0.85

        regime_allows_shorts = False
        regime_position_scale = 1.0
        if regime_info is not None:
            regime_row = regime_info.iloc[i]
            regime_allows_shorts = bool(regime_row.get("allow_shorts", False))
            regime_position_scale = float(regime_row.get("position_scale", 1.0))

        overlay_scale = dd_scale * float(np.clip(regime_position_scale, 0.45, 1.20))
        overlay_scale *= current_port_cap_scale
        overlay_scale = float(np.clip(overlay_scale, 0.30, 1.10))

        if i % args.rebal_freq == 0:
            base_w = pd.Series({name: V6_EXPECTATIONS[name].base_weight for name in V6_STRATEGIES}, dtype=float)
            if regime_info is not None:
                label = regime_info.iloc[i]["regime_label"]
                confidence = float(np.clip(regime_info.iloc[i]["confidence"], 0.0, 1.0))
                tilt_map = REGIME_TILTS.get(label, {})
                intensity = 0.40 + 0.60 * confidence
                for name in V6_STRATEGIES:
                    mult = tilt_map.get(name, 1.0)
                    base_w[name] *= 1.0 + (mult - 1.0) * intensity
            base_w /= max(base_w.sum(), 1e-12)

            alloc_target = pd.Series(allocator.target_weights()).reindex(V6_STRATEGIES).fillna(base_w)
            mixed_w = (1.0 - args.allocator_blend) * base_w + args.allocator_blend * alloc_target

            cap_scales = {}
            for name in V6_STRATEGIES:
                active = sleeves[name].iloc[i]
                active = active[active.abs() > 1e-8]
                if active.empty:
                    current_cap_util[name] = 0.0
                    cap_scales[name] = 0.0
                    continue

                snapshots = build_snapshots(
                    prices, adv_usd, realized_vol, spread_bps, i, list(active.index)
                )
                est = capacity_model.estimate_strategy_capacity(
                    name,
                    active.to_dict(),
                    snapshots,
                    nav_usd=nav_usd,
                    turnover=max(float(strat_turnover[name].iloc[i]), 1.0 / max(args.rebal_freq, 1)),
                )
                allocator.observe_capacity(
                    name,
                    utilization=est.utilization,
                    capacity_nav_limit=est.nav_capacity_usd,
                    impact_bps=est.weighted_impact_bps,
                )
                current_cap_util[name] = est.utilization if np.isfinite(est.utilization) else 0.0
                cap_scales[name] = float(
                    np.clip(
                        args.hard_capacity_util / max(est.utilization, 1e-6),
                        0.0,
                        1.0,
                    )
                ) if est.utilization > args.hard_capacity_util else 1.0

            score_mult = ic_boosts.iloc[i].reindex(V6_STRATEGIES).fillna(1.0).clip(lower=0.85, upper=1.15)
            strat_w = mixed_w * pd.Series(cap_scales).reindex(V6_STRATEGIES).fillna(1.0) * score_mult
            if strat_w.sum() <= 0:
                strat_w = base_w.copy()
            strat_w /= max(strat_w.sum(), 1e-12)
            current_strat_weights = strat_w

            current_port_cap_scale = float(
                np.clip(
                    np.average(
                        np.array([cap_scales.get(name, 1.0) for name in V6_STRATEGIES]),
                        weights=np.array([mixed_w.get(name, 0.0) for name in V6_STRATEGIES]),
                    ),
                    0.45,
                    1.0,
                )
            )
            overlay_scale = dd_scale * float(np.clip(regime_position_scale, 0.45, 1.20)) * current_port_cap_scale
            overlay_scale = float(np.clip(overlay_scale, 0.30, 1.10))

            composite_row = pd.Series(0.0, index=equity_syms)
            for name in V6_STRATEGIES:
                composite_row = composite_row.add(
                    strategies[name].iloc[i].reindex(equity_syms).fillna(0.0) * current_strat_weights[name],
                    fill_value=0.0,
                )

            valid = composite_row.dropna()
            valid = valid[valid.abs() > 0.005]
            if len(valid) >= 20:
                e = float(env.iloc[i])
                b = float(breadth.iloc[i]) if i < len(breadth) else 0.5
                bearish = e < 0.75 or b < 0.40
                shorts_active = regime_allows_shorts and bearish

                long_picks, short_picks = [], []
                for _, syms in sector_stocks.items():
                    sector_sigs = valid.reindex(syms).dropna()
                    if len(sector_sigs) < 2:
                        continue
                    sector_sigs = sector_sigs.sort_values(ascending=False)
                    n_l = max(int(len(sector_sigs) * 0.30), 1)
                    for sym in sector_sigs.head(n_l).index:
                        if sector_sigs[sym] > 0.05:
                            long_picks.append((sym, float(sector_sigs[sym])))
                    if shorts_active:
                        n_s = max(int(len(sector_sigs) * 0.15), 1)
                        for sym in sector_sigs.tail(n_s).index:
                            if sector_sigs[sym] < -0.22:
                                short_picks.append((sym, float(sector_sigs[sym])))

                long_picks.sort(key=lambda x: x[1], reverse=True)
                short_picks.sort(key=lambda x: x[1])
                long_picks = long_picks[:args.n_long]
                short_picks = short_picks[:args.n_short]

                candidate = pd.Series(0.0, index=equity_syms)
                breadth_mult = float(np.clip(b * 1.5, 0.55, 1.20))
                if shorts_active:
                    long_budget = args.target_gross * min(e, 1.05) * 0.72 * breadth_mult
                    short_budget = args.target_gross * 0.22 * max(0.20, 1 - breadth_mult / 1.20)
                else:
                    long_budget = args.target_gross * min(e, 1.35) * breadth_mult
                    short_budget = 0.0

                long_budget *= overlay_scale
                short_budget *= overlay_scale

                if long_picks:
                    inv = {sym: 1.0 / max(float(inst_vol[sym].iloc[i]), 0.05) for sym, _ in long_picks}
                    total_inv = sum(inv.values())
                    for sym, sig_val in long_picks:
                        raw_w = long_budget * inv[sym] / max(total_inv, 1e-12)
                        tilt = 0.70 + 0.60 * min(abs(sig_val), 1.0)
                        candidate[sym] = float(np.clip(raw_w * tilt, 0.0, args.max_pos))

                if short_picks and short_budget > 0:
                    inv = {sym: 1.0 / max(float(inst_vol[sym].iloc[i]), 0.05) for sym, _ in short_picks}
                    total_inv = sum(inv.values())
                    for sym, _ in short_picks:
                        raw_w = -short_budget * inv[sym] / max(total_inv, 1e-12)
                        candidate[sym] = float(np.clip(raw_w, -args.max_pos * 0.45, 0.0))

                prev_target = target_w.copy()
                delta = candidate - prev_target
                small = delta.abs() < 0.003
                candidate[small] = prev_target[small]

                keep = pd.Series(0.90, index=equity_syms)
                entering = (prev_target.abs() < 1e-12) & (candidate.abs() > 1e-12)
                exiting = (prev_target.abs() > 1e-12) & (candidate.abs() < 1e-12)
                flipping = (
                    (prev_target.abs() > 1e-12)
                    & (candidate.abs() > 1e-12)
                    & (np.sign(prev_target) != np.sign(candidate))
                )
                keep[entering] = 0.72
                keep[exiting] = 0.50
                keep[flipping] = 0.30
                candidate = (1 - keep) * candidate + keep * prev_target
                candidate[(candidate.abs() < 0.002) & exiting] = 0.0

                active_mask = (candidate.abs() > 1e-8) | (prev_target.abs() > 1e-8)
                active_symbols = [sym for sym, flag in active_mask.items() if flag]
                snapshots = build_snapshots(
                    prices, adv_usd, realized_vol, spread_bps, i, active_symbols
                )
                candidate, clip_count = clip_rebalance_delta(
                    prev_target, candidate, nav_usd, snapshots, capacity_model
                )
                symbol_clip_counts.iloc[i] = clip_count

                target_w = candidate.clip(lower=-args.max_pos * 0.45, upper=args.max_pos)

        live_w = target_w * overlay_scale if overlay_scale < 1.0 else target_w.copy()
        weights.iloc[i] = live_w
        strategy_weight_history.iloc[i] = current_strat_weights
        strategy_cap_history.iloc[i] = current_cap_util
        portfolio_capacity_scale.iloc[i] = current_port_cap_scale

    port_ret = (weights.shift(1) * returns[equity_syms]).sum(axis=1)
    port_rvol = port_ret.rolling(20, min_periods=10).std() * np.sqrt(252)
    final_weights = weights.copy()
    for i in range(320, len(prices)):
        rv = port_rvol.iloc[i]
        if np.isnan(rv) or rv < 0.01:
            continue
        scale = float(np.clip(args.target_vol / rv, 0.30, 2.0))
        final_weights.iloc[i] = weights.iloc[i] * scale

    return (
        final_weights,
        strategy_weight_history.ffill().fillna(0.0),
        strategy_cap_history.ffill().fillna(0.0),
        portfolio_capacity_scale.ffill().fillna(1.0),
        symbol_clip_counts,
        strat_ret,
    )


def compute_v6_strategy_sleeves(strategies, prices, returns, env, breadth, equity_syms, args, regime_info):
    sleeves = {}
    for name in V6_STRATEGIES:
        print(f"  Building sleeve: {name}...", end=" ", flush=True)
        t0 = time.time()
        sleeves[name] = v5.build_strategy_sleeve(
            strategies[name],
            prices,
            returns,
            env,
            breadth,
            equity_syms,
            target_gross=args.target_gross,
            rebal_freq=args.rebal_freq,
            max_pos=args.max_pos,
            n_long=args.n_long,
            n_short=args.n_short,
            regime_info=regime_info,
        )
        print(f"{time.time()-t0:.1f}s")
    return sleeves


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

    total_bars = sum(len(df) for df in all_data.values())
    print(f"\nTotal: {len(all_data)} instruments, {total_bars:,} bars\n")

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
    print("  COMPUTING v6 STRATEGY SIGNALS")
    print("=" * 70)
    t_total = time.time()
    strategies = {}
    strategies["momentum"] = v4.strategy_momentum(prices, returns, equity_syms, mkt_ret)
    strategies["quality"] = v4.strategy_quality(prices, returns, equity_syms, mkt_ret)
    strategies["carry"] = v4.strategy_carry(prices, returns, equity_syms)
    strategies["sector_rot"] = v4.strategy_sector_rotation(prices, returns, equity_syms)
    strategies["high_52w"] = v4.strategy_52w_high(prices, equity_syms)
    strategies["pit_event"], event_coverage = build_pit_event_signal(
        prices, equity_syms, args.fundamentals_db, args.symbol_master_db
    )
    print(f"  [S9] PIT event sleeve... {event_coverage} symbols with real EPS/estimate coverage")
    print(f"\n  Total signal computation: {time.time()-t_total:.1f}s\n")

    env = v4.compute_environment(prices)
    breadth = v4.compute_breadth(prices, equity_syms)
    regime_info = v4.compute_regime_states(prices, equity_syms)
    ic_boosts = compute_ic_boosts(strategies, returns, equity_syms)

    print("=" * 70)
    print("  BUILDING v6 STRATEGY SLEEVES")
    print("=" * 70)
    sleeves = compute_v6_strategy_sleeves(strategies, prices, returns, env, breadth, equity_syms, args, regime_info)
    print()

    print("=" * 70)
    print("  WALK-FORWARD BACKTEST v6 (v4 core + allocator + capacity clipping)")
    print("=" * 70)
    print(f"  NAV: ${args.nav:,.0f} | Vol target: {args.target_vol:.0%} | Rebal: {args.rebal_freq}d")
    print(f"  Gross target: {args.target_gross:.1f}x | Long: {args.n_long} | Short: {args.n_short}")
    print(f"  Allocator blend: {args.allocator_blend:.0%} | Hard cap util: {args.hard_capacity_util:.0%}\n")

    t0 = time.time()
    weights, strat_w_hist, strat_cap_hist, port_cap_scale, symbol_clips, strat_ret = build_v6_portfolio(
        strategies, sleeves, prices, returns, volumes, equity_syms, args, regime_info, env, breadth, ic_boosts
    )
    weights = weights.fillna(0.0)
    print(f"  Portfolio construction: {time.time()-t0:.1f}s\n")

    warmup = 300
    port_ret = (weights.shift(1) * returns[equity_syms]).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    tx_cost = turnover * (args.slippage + 1.0) / 10000.0
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
    tx_bps = tx_cost.iloc[warmup:].mean() * 252 * 10000 if len(tx_cost) > warmup else 0.0

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
    print(f"  Avg port cap scale: {port_cap_scale.iloc[warmup:].mean():.2f}")
    print(f"  Avg symbol clips / rebalance day: {symbol_clips.iloc[warmup:].replace(0, np.nan).mean():.2f}")
    print()

    print("  ── Strategy Weight Mix ──")
    avg_strat_w = strat_w_hist.iloc[warmup:].mean()
    avg_strat_cap = strat_cap_hist.iloc[warmup:].mean()
    for name in avg_strat_w.sort_values(ascending=False).index:
        sleeve_s = v4.sharpe(strat_ret[name].iloc[warmup:])
        print(f"  {name:11s} avg_w={avg_strat_w[name]:>5.1%} | sleeve_sharpe={sleeve_s:.2f} | cap_util={avg_strat_cap[name]:.2f}")
    print()

    print("  ── Year-by-Year: LP Net vs SPY ──")
    print(f"  {'Year':<6} {'LP Ret':>8} {'SPY':>8} {'Alpha':>8} {'LP Shp':>8} {'Hit':>4}")
    print("  " + "-" * 46)
    hit_return = hit_sharpe = hit_both = 0
    for year in sorted(set(d.year if hasattr(d, 'year') else d.date().year for d in dates)):
        mask = pd.Series([d.year if hasattr(d, 'year') else d.date().year for d in dates], index=lp_ret.index) == year
        yr = lp_ret[mask]
        yr_eq = lp_equity_curve[mask]
        if len(yr) < 5:
            continue
        yr_ret = (yr_eq.iloc[-1] / yr_eq.iloc[0]) - 1
        yr_s = v4.sharpe(yr)

        spy_mask = pd.Series([d.year if hasattr(d, 'year') else d.date().year for d in spy_ret.index], index=spy_ret.index) == year
        spy_yr = spy_ret[spy_mask]
        spy_yr_eq = spy_eq[spy_mask]
        spy_yr_ret = (spy_yr_eq.iloc[-1] / spy_yr_eq.iloc[0]) - 1 if len(spy_yr_eq) > 1 else 0.0

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
    total_years = len(set(d.year if hasattr(d, 'year') else d.date().year for d in dates))
    print("  " + "-" * 46)
    print(f"  Years >= {args.lp_target_return:.0%} net: {hit_return}/{total_years}")
    print(f"  Years >= {args.lp_target_sharpe:.1f} Sharpe: {hit_sharpe}/{total_years}")
    print(f"  Years hitting both hurdles: {hit_both}/{total_years}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AlphaForge v6 — Investable overlay on v4")
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
    p.add_argument("--exploration-floor", type=float, default=0.05)
    p.add_argument("--capacity-soft-limit", type=float, default=0.85)
    p.add_argument("--hard-capacity-util", type=float, default=0.95)
    p.add_argument("--allocator-blend", type=float, default=0.18)
    p.add_argument("--impact-k", type=float, default=0.5)
    p.add_argument("--fundamentals-db", default="data/fundamentals.db")
    p.add_argument("--symbol-master-db", default="data/symbol_master.db")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    if os.path.exists(args.env):
        from dotenv import load_dotenv
        load_dotenv(args.env)

    run(args)
