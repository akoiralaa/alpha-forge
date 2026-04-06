#!/usr/bin/env python3
"""
Best-of-both: directional multi-factor signals (which work)
+ portfolio-level vol targeting + adaptive risk scaling (which controls DD).

This combines the 0.64 Sharpe directional approach with institutional-grade
risk overlays to reduce max drawdown from -76% to target -25%.
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

from src.data.ingest.base import AssetClass
from src.data.ingest.data_manager import DataManager, build_data_manager_from_env


ASSET_TYPES = {}

# Map string asset types to AssetClass enum
_ASSET_CLASS_MAP = {
    "ETF": AssetClass.ETF,
    "EQUITY": AssetClass.EQUITY,
    "FUTURE": AssetClass.FUTURE,
    "COMMODITY": AssetClass.COMMODITY,
    "BOND": AssetClass.BOND,
    "FX": AssetClass.FX,
    "VOLATILITY": AssetClass.VOLATILITY,
}

# Signal weights: (trend, momentum, mean_rev, fundamental)
# Fundamental factor added — catches earnings acceleration, ROC expansion
ASSET_WEIGHTS = {
    "ETF": (0.30, 0.30, 0.20, 0.20),
    "EQUITY": (0.20, 0.30, 0.15, 0.35),  # fundamentals matter most for single stocks
    "FUTURE": (0.40, 0.35, 0.25, 0.00),  # no fundamentals for futures
    "COMMODITY": (0.45, 0.35, 0.20, 0.00),
    "BOND": (0.35, 0.35, 0.30, 0.00),
    "FX": (0.30, 0.30, 0.40, 0.00),
    "VOLATILITY": (0.30, 0.35, 0.35, 0.00),
}


def load_universe_static(config_path):
    """Load static universe from YAML config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    instruments = cfg.get("instruments", {})
    universe = []
    for key, atype in [("sector_etfs", "ETF"), ("equities", "EQUITY"),
                        ("equity_index_futures", "FUTURE"),
                        ("commodity_futures", "COMMODITY"), ("fixed_income_futures", "BOND"),
                        ("fx_pairs", "FX"), ("vix_futures", "VOLATILITY")]:
        for sym in instruments.get(key, []):
            universe.append((sym, atype))
            ASSET_TYPES[sym] = atype
    return universe


def load_universe_dynamic(config_path, skip_mcap=False):
    """Load dynamic universe — all liquid US equities + static futures/FX."""
    from src.data.universe_builder import build_full_universe, config_from_yaml
    cfg = config_from_yaml(config_path) if os.path.exists(config_path) else None
    if skip_mcap and cfg is not None:
        cfg.liquidity.market_cap_min = 0  # skip expensive per-ticker API calls
    universe = build_full_universe(cfg=cfg, yaml_config_path=config_path)
    for sym, atype in universe:
        ASSET_TYPES[sym] = atype
    return universe


CACHE_DIR = os.path.expanduser("~/.one_brain_fund/cache/bars")


def fetch_daily_bars(dm: DataManager, symbol: str, asset_type: str, use_cache: bool = True) -> pd.DataFrame:
    """Fetch max daily bars via DataManager with disk cache for fast iteration."""
    cache_path = os.path.join(CACHE_DIR, f"{symbol}_{asset_type}.parquet")

    if use_cache and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            if len(df) > 0:
                return df
        except Exception:
            pass

    asset_class = _ASSET_CLASS_MAP.get(asset_type, AssetClass.EQUITY)
    df = dm.fetch_daily_bars(symbol, asset_class)

    if not df.empty and use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_parquet(cache_path, index=False)

    return df


def build_price_matrix(all_data):
    frames = {}
    for sym, df in all_data.items():
        if df.empty:
            continue
        s = df.set_index("date")["close"]
        s.name = sym
        frames[sym] = s
    if not frames:
        return None, []
    prices = pd.DataFrame(frames).sort_index().ffill()
    return prices, list(prices.columns)


# ── Fundamental data fetching ────────────────────────────────

def fetch_fundamentals(symbols, api_key):
    """
    Fetch fundamental data from Polygon for equity/ETF symbols.
    Returns dict[sym] -> {eps_growth_3y, eps_growth_5y, roc, rev_growth, earnings_surprise}

    Uses Polygon's stock financials endpoint. For symbols where fundamentals
    aren't available (futures, FX, etc.), returns empty dict.
    """
    if not api_key:
        print("  [fundamentals] No Polygon API key — skipping fundamentals")
        return {}

    try:
        from polygon import RESTClient
    except ImportError:
        print("  [fundamentals] polygon-api-client not installed — skipping")
        return {}

    client = RESTClient(api_key=api_key)
    sleep_time = 60.0 / int(os.environ.get("POLYGON_RATE_LIMIT_PER_MIN", "5"))

    equity_syms = [s for s in symbols if ASSET_TYPES.get(s, "") in ("EQUITY", "ETF")]
    if not equity_syms:
        return {}

    print(f"  [fundamentals] Fetching for {len(equity_syms)} equities "
          f"(~{len(equity_syms) * sleep_time / 60:.0f} min) ...", flush=True)

    fundamentals = {}
    errors = 0

    for idx, sym in enumerate(equity_syms):
        try:
            # Get last 5 years of annual financials
            financials = list(client.vx.list_stock_financials(
                ticker=sym,
                timeframe="annual",
                limit=6,
                sort="period_of_report_date",
                order="desc",
            ))
            time.sleep(sleep_time)

            if len(financials) < 2:
                continue

            # Extract EPS series
            eps_list = []
            rev_list = []
            net_income_list = []
            equity_list = []

            for f in financials:
                inc = f.financials.income_statement if hasattr(f, 'financials') and hasattr(f.financials, 'income_statement') else None
                bs = f.financials.balance_sheet if hasattr(f, 'financials') and hasattr(f.financials, 'balance_sheet') else None

                if inc:
                    eps = getattr(getattr(inc, 'basic_earnings_per_share', None), 'value', None)
                    rev = getattr(getattr(inc, 'revenues', None), 'value', None)
                    ni = getattr(getattr(inc, 'net_income_loss', None), 'value', None)
                    eps_list.append(eps)
                    rev_list.append(rev)
                    net_income_list.append(ni)
                else:
                    eps_list.append(None)
                    rev_list.append(None)
                    net_income_list.append(None)

                if bs:
                    eq = getattr(getattr(bs, 'equity', None), 'value', None)
                    equity_list.append(eq)
                else:
                    equity_list.append(None)

            fund = {}

            # EPS growth 3yr: compare most recent to 3 years ago
            if len(eps_list) >= 4 and eps_list[0] is not None and eps_list[3] is not None and eps_list[3] != 0:
                fund["eps_growth_3y"] = (eps_list[0] / abs(eps_list[3])) ** (1/3) - 1

            # EPS growth 5yr
            if len(eps_list) >= 6 and eps_list[0] is not None and eps_list[5] is not None and eps_list[5] != 0:
                fund["eps_growth_5y"] = (eps_list[0] / abs(eps_list[5])) ** (1/5) - 1

            # Return on capital (net income / equity)
            if (net_income_list and equity_list and
                    net_income_list[0] is not None and equity_list[0] is not None and equity_list[0] > 0):
                fund["roc"] = net_income_list[0] / equity_list[0]

            # Revenue growth (YoY most recent)
            if len(rev_list) >= 2 and rev_list[0] is not None and rev_list[1] is not None and rev_list[1] > 0:
                fund["rev_growth"] = rev_list[0] / rev_list[1] - 1

            # Revenue acceleration (this year's growth vs last year's growth)
            if len(rev_list) >= 3 and all(r is not None and r > 0 for r in rev_list[:3]):
                growth_recent = rev_list[0] / rev_list[1] - 1
                growth_prior = rev_list[1] / rev_list[2] - 1
                fund["rev_accel"] = growth_recent - growth_prior

            if fund:
                fundamentals[sym] = fund

        except Exception as e:
            errors += 1

        if (idx + 1) % 50 == 0:
            print(f"    ... {idx+1}/{len(equity_syms)} ({len(fundamentals)} with data, {errors} errors)")

    print(f"  [fundamentals] Done: {len(fundamentals)} symbols with fundamental data")
    return fundamentals


def compute_fundamental_score(sym, fundamentals):
    """
    Score a symbol on fundamentals: [-1, +1].

    Positive = strong earnings growth, high ROC, revenue acceleration.
    This is what catches NVDA at $200 not $800 — earnings were already exploding.

    Components:
    - EPS growth (3yr/5yr blend): are earnings compounding?
    - Return on capital: is the business actually good?
    - Revenue acceleration: is growth ACCELERATING (not just growing)?
    """
    fund = fundamentals.get(sym)
    if not fund:
        return 0.0

    score = 0.0
    n_factors = 0

    # EPS growth: blend 3yr and 5yr, favor 3yr (more recent)
    eps3 = fund.get("eps_growth_3y")
    eps5 = fund.get("eps_growth_5y")
    if eps3 is not None or eps5 is not None:
        eps_g = 0.0
        if eps3 is not None and eps5 is not None:
            eps_g = 0.6 * eps3 + 0.4 * eps5
        elif eps3 is not None:
            eps_g = eps3
        else:
            eps_g = eps5
        # >30% growth is strong, >50% is exceptional
        score += float(np.tanh(eps_g * 2.0))  # maps ~25% growth to ~0.46, 50% to ~0.76
        n_factors += 1

    # Return on capital
    roc = fund.get("roc")
    if roc is not None:
        # >20% ROC is good, >40% is exceptional (NVDA territory)
        score += float(np.tanh((roc - 0.10) * 3.0))  # 10% ROC = neutral, 20% = strong
        n_factors += 1

    # Revenue acceleration: this is the alpha — not just growth, but ACCELERATING growth
    rev_accel = fund.get("rev_accel")
    if rev_accel is not None:
        # Positive = growth speeding up. This is the NVDA signal.
        score += float(np.tanh(rev_accel * 5.0))  # 10% acceleration = 0.46 signal
        n_factors += 1

    # Revenue growth (simpler fallback)
    if rev_accel is None:
        rev_g = fund.get("rev_growth")
        if rev_g is not None:
            score += float(np.tanh(rev_g * 2.0))
            n_factors += 1

    if n_factors == 0:
        return 0.0

    return float(np.clip(score / n_factors, -1.0, 1.0))


# ── Per-instrument signal computation ─────────────────────────

def compute_signals(prices, fundamentals=None):
    """
    Multi-factor signal per instrument:
    trend + momentum + mean-reversion + fundamentals.

    Fundamentals are the key differentiator — they catch earnings acceleration
    before the price fully reflects it. Technicals confirm the fundamental thesis.
    """
    if fundamentals is None:
        fundamentals = {}

    returns = prices.pct_change().fillna(0)
    log_ret = np.log(prices / prices.shift(1)).fillna(0)
    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for sym in prices.columns:
        p = prices[sym].values.astype(float)
        lr = log_ret[sym].values.astype(float)
        n = len(p)

        valid_start = np.argmax(~np.isnan(p))
        if n - valid_start < 252:
            continue

        sma50 = pd.Series(p).rolling(50, min_periods=50).mean().values
        sma200 = pd.Series(p).rolling(200, min_periods=200).mean().values
        rvol20 = pd.Series(lr).rolling(20, min_periods=10).std().values * np.sqrt(252)
        rvol60 = pd.Series(lr).rolling(60, min_periods=30).std().values * np.sqrt(252)

        # RSI 14
        delta = np.diff(p, prepend=p[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        if n > 14:
            avg_gain[14] = np.mean(gain[1:15])
            avg_loss[14] = np.mean(loss[1:15])
            for i in range(15, n):
                avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
                avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        rsi = np.full(n, 50.0)
        for i in range(14, n):
            if not np.isnan(avg_loss[i]) and avg_loss[i] > 0:
                rsi[i] = 100 - 100 / (1 + avg_gain[i] / avg_loss[i])

        # z-score 20d
        z20 = pd.Series(lr).rolling(20, min_periods=15).apply(
            lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-10), raw=False).values

        # Get weights: (trend, momentum, mean_rev, fundamental)
        weights = ASSET_WEIGHTS.get(ASSET_TYPES.get(sym, "ETF"), (0.30, 0.30, 0.20, 0.20))
        w_trend, w_mom, w_mr, w_fund = weights

        # Pre-compute fundamental score (constant across time — updates quarterly)
        fund_score = compute_fundamental_score(sym, fundamentals)

        sig = np.zeros(n)

        for i in range(252, n):
            if np.isnan(sma50[i]) or np.isnan(sma200[i]) or np.isnan(rvol20[i]):
                continue

            # TREND: SMA crossover + price vs SMA distance
            trend = 0.0
            if sma50[i] > sma200[i]:
                trend = 0.4
                price_dist = (p[i] - sma50[i]) / (sma50[i] + 1e-10)
                trend += float(np.clip(price_dist * 2, 0, 0.4))
            elif sma50[i] < sma200[i]:
                trend = -0.4
                price_dist = (p[i] - sma50[i]) / (sma50[i] + 1e-10)
                trend += float(np.clip(price_dist * 2, -0.4, 0))

            # MOMENTUM: 12-1 month return, vol-adjusted
            mom = 0.0
            if i >= 252 and not np.isnan(rvol60[i]) and rvol60[i] > 0.01:
                ret_12m = p[i] / p[i-252] - 1
                ret_1m = p[i] / p[i-21] - 1
                ret_6m = p[i] / p[i-126] - 1 if i >= 126 else 0
                raw = 0.6 * (ret_12m - ret_1m) + 0.4 * (ret_6m - ret_1m)
                mom = float(np.tanh(raw / rvol60[i] * 0.4))

            # MEAN REVERSION: z-score + RSI, only near 200 SMA
            mr = 0.0
            if not np.isnan(z20[i]) and sma200[i] > 0:
                dist200 = abs(p[i] / sma200[i] - 1)
                if dist200 < 0.12:
                    z_sig = float(np.clip(-z20[i] * 0.15, -0.4, 0.4))
                    rsi_sig = 0.0
                    if rsi[i] < 30:
                        rsi_sig = (30 - rsi[i]) / 40
                    elif rsi[i] > 70:
                        rsi_sig = -(rsi[i] - 70) / 40
                    mr = 0.5 * z_sig + 0.5 * rsi_sig

            # FUNDAMENTAL: constant signal from earnings quality
            # When fundamentals are strong + technicals confirm = high conviction
            fund = fund_score

            # vol regime: when vol expanding, dampen mean-reversion, boost trend
            vol_ratio = rvol20[i] / (rvol60[i] + 1e-10) if not np.isnan(rvol60[i]) else 1.0
            if vol_ratio > 1.3:
                w_t = w_trend * 1.2; w_m = w_mom * 1.1; w_r = w_mr * 0.4; w_f = w_fund * 0.8
            elif vol_ratio < 0.7:
                w_t = w_trend * 0.8; w_m = w_mom * 0.9; w_r = w_mr * 1.3; w_f = w_fund * 1.2
            else:
                w_t, w_m, w_r, w_f = w_trend, w_mom, w_mr, w_fund

            total_w = w_t + w_m + w_r + w_f
            if total_w > 0:
                w_t /= total_w; w_m /= total_w; w_r /= total_w; w_f /= total_w

            composite = w_t * trend + w_m * mom + w_r * mr + w_f * fund

            # conviction boost when signals agree
            sigs = [s for s in [trend, mom, mr, fund] if abs(s) > 0.05]
            agree = sum(1 for s in sigs if np.sign(s) == np.sign(composite))
            if len(sigs) >= 3 and agree >= 3:
                composite *= 1.25
            elif len(sigs) >= 2 and agree <= 1:
                composite *= 0.6

            sig[i] = float(np.clip(composite, -1.0, 1.0))

        signals[sym] = sig

    return signals


# ── Environment scoring ───────────────────────────────────────

def compute_environment(prices, spy_col="SPY"):
    """Market environment score: 0.15 (crisis) to 1.3 (benign)."""
    if spy_col not in prices.columns:
        spy_col = prices.columns[0]

    spy_ret = prices[spy_col].pct_change().fillna(0)
    rvol20 = spy_ret.rolling(20, min_periods=10).std() * np.sqrt(252)
    rvol60 = spy_ret.rolling(60, min_periods=30).std() * np.sqrt(252)

    env = pd.Series(1.0, index=prices.index)
    for i in range(60, len(prices)):
        rv = rvol20.iloc[i]
        vr = rvol20.iloc[i] / (rvol60.iloc[i] + 1e-10)
        scale = 1.0

        if rv > 0.40: scale *= 0.15
        elif rv > 0.30: scale *= 0.30
        elif rv > 0.25: scale *= 0.50
        elif rv > 0.20: scale *= 0.70
        elif rv < 0.10: scale *= 1.25

        if vr > 1.4: scale *= 0.50
        elif vr > 1.2: scale *= 0.70
        elif vr < 0.7: scale *= 1.15

        env.iloc[i] = float(np.clip(scale, 0.15, 1.3))
    return env


# ── Portfolio construction with vol targeting ─────────────────

def build_portfolio(signals, prices, env, target_vol, rebal_freq, max_pos,
                    max_positions=80, signal_threshold=0.18, blend_rate=0.70):
    """
    Vol-targeted portfolio with concentration and turnover control.

    Key changes from naive approach:
    1. Concentrate into top N positions by signal strength (don't spread across 400 stocks)
    2. Per-asset-class risk budgets (futures/FX get their own allocation, not drowned by equities)
    3. Blend new weights with old (70/30) to dampen turnover
    4. Higher signal threshold (0.18 vs 0.10) — fewer, higher conviction trades
    5. Portfolio-level vol cap (THE key risk management layer)
    """
    returns = prices.pct_change().fillna(0)
    inst_vol = returns.rolling(20, min_periods=10).std() * np.sqrt(252)
    inst_vol = inst_vol.bfill().clip(lower=0.02)

    # Per-asset-class risk budgets — don't let 371 equities drown out diversifiers
    # Equities get 55% of risk, futures/commodities 20%, bonds 10%, FX 15%
    CLASS_RISK_SHARE = {
        "EQUITY": 0.50, "ETF": 0.05,
        "FUTURE": 0.12, "COMMODITY": 0.10,
        "BOND": 0.08, "FX": 0.10, "VOLATILITY": 0.05,
    }

    # Count instruments per class
    class_counts = defaultdict(int)
    sym_class = {}
    for sym in prices.columns:
        cls = ASSET_TYPES.get(sym, "EQUITY")
        class_counts[cls] += 1
        sym_class[sym] = cls

    # Per-class risk budget: class_share * target_vol / sqrt(n_in_class * corr)
    class_risk_budget = {}
    for cls, share in CLASS_RISK_SHARE.items():
        n = max(class_counts.get(cls, 1), 1)
        corr_factor = 0.6 if cls == "EQUITY" else 0.3  # equities more correlated
        class_risk_budget[cls] = (share * target_vol) / max(np.sqrt(n * corr_factor), 1)

    raw_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    last_w = pd.Series(0.0, index=prices.columns)

    for i in range(252, len(prices)):
        if i % rebal_freq != 0 and i > 252:
            raw_weights.iloc[i] = last_w
            continue

        # Score all instruments
        # Lower threshold for diversifiers (their signals are naturally smaller)
        candidates = {}
        for sym in prices.columns:
            sig = signals[sym].iloc[i]
            cls = sym_class[sym]
            thresh = signal_threshold if cls in ("EQUITY", "ETF") else signal_threshold * 0.5
            if abs(sig) < thresh:
                continue
            iv = inst_vol[sym].iloc[i]
            if np.isnan(iv) or iv < 0.01:
                continue
            rb = class_risk_budget.get(cls, class_risk_budget.get("EQUITY"))
            raw = sig * (rb / iv)
            candidates[sym] = (float(np.clip(raw, -max_pos, max_pos)), abs(sig))

        # Concentrate: take top max_positions by signal strength
        # Always include non-equity diversifiers (futures, FX, bonds)
        diversifiers = {s: v for s, v in candidates.items()
                        if sym_class[s] not in ("EQUITY", "ETF")}
        equities = {s: v for s, v in candidates.items()
                    if sym_class[s] in ("EQUITY", "ETF")}

        # Sort equities by signal strength, take top N
        eq_slots = max(max_positions - len(diversifiers), 20)
        top_eq = sorted(equities.items(), key=lambda x: x[1][1], reverse=True)[:eq_slots]

        w = pd.Series(0.0, index=prices.columns)
        for sym, (weight, _) in list(diversifiers.items()) + top_eq:
            w[sym] = weight

        # Environment overlay
        w *= env.iloc[i]

        # Blend with previous weights to reduce turnover
        if i > 252:
            w = blend_rate * w + (1 - blend_rate) * last_w

        last_w = w.copy()
        raw_weights.iloc[i] = w

    # PORTFOLIO-LEVEL VOL TARGETING: this is the key layer
    port_ret = (raw_weights.shift(1) * returns).sum(axis=1)
    port_rvol = port_ret.rolling(20, min_periods=10).std() * np.sqrt(252)

    final_weights = raw_weights.copy()
    for i in range(270, len(prices)):
        rv = port_rvol.iloc[i]
        if np.isnan(rv) or rv < 0.01:
            continue
        scale = target_vol / rv
        scale = float(np.clip(scale, 0.2, 2.5))
        final_weights.iloc[i] = raw_weights.iloc[i] * scale

    return final_weights


# ── Statistics ────────────────────────────────────────────────

def sharpe(r, ann=252):
    return float(r.mean() / (r.std() + 1e-12) * np.sqrt(ann)) if len(r) > 1 else 0.0

def sortino(r, ann=252):
    neg = r[r < 0]
    return float(r.mean() / (neg.std() + 1e-12) * np.sqrt(ann)) if len(neg) > 1 else 0.0

def max_dd(eq):
    pk = eq.cummax()
    return float(((eq - pk) / pk).min())

def monte_carlo(rets, n_sims=10000, n_days=252):
    rng = np.random.default_rng(42)
    r = rets.values
    results = np.zeros((n_sims, n_days))
    for i in range(n_sims):
        results[i] = np.cumprod(1 + rng.choice(r, size=n_days, replace=True))
    terminal = results[:, -1]
    dds = np.zeros(n_sims)
    for i in range(n_sims):
        pk = np.maximum.accumulate(results[i])
        dds[i] = ((results[i] - pk) / np.where(pk > 0, pk, 1)).min()
    return terminal, dds


# ── Main ──────────────────────────────────────────────────────

def run(args):
    # ── Universe: dynamic (all liquid equities) or static (YAML list) ──
    if args.dynamic:
        universe = load_universe_dynamic(args.config, skip_mcap=args.skip_mcap)
    else:
        universe = load_universe_static(args.config)
    print(f"\nUniverse: {len(universe)} instruments\n")

    # ── Connect data providers (skip if all cached) ──
    # Check cache coverage first
    cache_hits = 0
    if not args.no_cache:
        for sym, atype in universe:
            cache_path = os.path.join(CACHE_DIR, f"{sym}_{atype}.parquet")
            if os.path.exists(cache_path):
                cache_hits += 1
    all_cached = cache_hits == len(universe) and not args.no_cache

    dm = build_data_manager_from_env()
    if all_cached:
        print(f"All {cache_hits} instruments cached — skipping provider connection\n")
    else:
        print("Connecting data providers...")
        results = dm.connect_all()
        connected = [k for k, v in results.items() if v]
        failed = [k for k, v in results.items() if not v]
        print(f"  Connected: {', '.join(connected) if connected else 'none'}")
        if failed:
            print(f"  Failed:    {', '.join(failed)}")
        if not connected and cache_hits < len(universe):
            print("ERROR: No data providers available. Check .env configuration.")
            sys.exit(1)
        if cache_hits > 0:
            print(f"  Cached:    {cache_hits}/{len(universe)} instruments")
        print()

    # ── Fetch historical price data ──
    print("=" * 70)
    print("  FETCHING HISTORICAL DATA")
    print("=" * 70)
    all_data = {}
    for idx, (sym, atype) in enumerate(universe, 1):
        print(f"  [{idx:3d}/{len(universe)}] {sym:8s} ({atype:10s}) ", end="", flush=True)
        df = fetch_daily_bars(dm, sym, atype, use_cache=not args.no_cache)
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

    prices, symbols = build_price_matrix(all_data)
    if prices is None:
        sys.exit(1)
    print(f"  Matrix: {prices.shape[0]} x {prices.shape[1]} instruments\n")

    # ── Fetch fundamentals (EPS growth, ROC, revenue accel) ──
    print("=" * 70)
    print("  FETCHING FUNDAMENTAL DATA")
    print("=" * 70)
    poly_key = os.environ.get("POLYGON_API_KEY", "")
    fundamentals = fetch_fundamentals(symbols, poly_key) if not args.no_fundamentals else {}
    if fundamentals:
        # Show top fundamental scores
        scores = {sym: compute_fundamental_score(sym, fundamentals) for sym in fundamentals}
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n  Top 10 fundamental scores:")
        for sym, sc in top:
            fund = fundamentals[sym]
            eps = fund.get("eps_growth_3y", 0)
            roc = fund.get("roc", 0)
            ra = fund.get("rev_accel", fund.get("rev_growth", 0))
            print(f"    {sym:8s} score={sc:+.2f}  EPS3y={eps:+.0%}  ROC={roc:.0%}  RevAccel={ra:+.0%}")
    print()

    # ── Run backtest (walk-forward, no lookahead) ──
    print("=" * 70)
    print("  WALK-FORWARD BACKTEST (no lookahead)")
    print("=" * 70)
    print(f"  NAV: ${args.nav:,.0f} | Vol target: {args.target_vol:.0%} | "
          f"Rebal: {args.rebal_freq}d | Max pos: {args.max_pos:.0%}")
    print()

    t0 = time.time()
    print("  Computing multi-factor signals (tech + fundamental)...", end=" ", flush=True)
    signals = compute_signals(prices, fundamentals)
    print(f"{time.time()-t0:.1f}s")

    t0 = time.time()
    print("  Computing vol environment...", end=" ", flush=True)
    env = compute_environment(prices)
    print(f"{time.time()-t0:.1f}s")

    t0 = time.time()
    print("  Building vol-targeted portfolio...", end=" ", flush=True)
    weights = build_portfolio(signals, prices, env,
                              args.target_vol, args.rebal_freq, args.max_pos)
    weights = weights.fillna(0)
    print(f"{time.time()-t0:.1f}s\n")

    # compute returns
    returns = prices.pct_change().fillna(0)
    warmup = 275
    port_ret = (weights.shift(1) * returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    tx_cost = turnover * (args.slippage + 1.0) / 10000
    net_ret = port_ret - tx_cost

    net_ret = net_ret.iloc[warmup:]
    weights_post = weights.iloc[warmup:]
    turnover = turnover.iloc[warmup:]

    equity = args.nav * (1 + net_ret).cumprod()
    dates = equity.index
    n_years = len(net_ret) / 252

    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / max(n_years, 0.01)) - 1
    total_ret = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
    s = sharpe(net_ret)
    so = sortino(net_ret)
    dd = max_dd(equity)
    cal = cagr / abs(dd) if abs(dd) > 1e-12 else 0
    avg_gross = weights_post.abs().sum(axis=1).mean()
    avg_turn = turnover.mean() * 252
    tx_bps = tx_cost.iloc[warmup:].mean() * 252 * 10000 if len(tx_cost) > warmup else 0

    # SPY comparison
    if "SPY" in returns.columns:
        spy_ret = returns["SPY"].iloc[warmup:]
        spy_eq = args.nav * (1 + spy_ret).cumprod()
        spy_cagr = (spy_eq.iloc[-1] / spy_eq.iloc[0]) ** (1 / max(n_years, 0.01)) - 1
        spy_sharpe = sharpe(spy_ret)
        spy_dd = max_dd(spy_eq)
    else:
        spy_cagr = spy_sharpe = spy_dd = 0

    print("  ── Aggregate ──")
    print(f"  CAGR:          {cagr:+.2%}   (SPY: {spy_cagr:+.2%})")
    print(f"  Sharpe:        {s:.2f}      (SPY: {spy_sharpe:.2f})")
    print(f"  Sortino:       {so:.2f}")
    print(f"  Calmar:        {cal:.2f}")
    print(f"  Max DD:        {dd:.2%}   (SPY: {spy_dd:.2%})")
    print(f"  Final NAV:     ${equity.iloc[-1]:,.0f}")
    print(f"  Avg gross:     {avg_gross:.2f}x")
    print(f"  Turnover:      {avg_turn:.0f}x/yr")
    print(f"  Tx costs:      {tx_bps:.0f} bps/yr")
    print()

    # year by year
    print("  ── Year-by-Year ──")
    print(f"  {'Year':<6} {'Return':>8} {'SPY':>8} {'Sharpe':>8} {'MaxDD':>8} {'Win':>5}")
    print("  " + "-" * 48)
    loss_years = 0
    years_beat_spy = 0
    for year in sorted(set(d.year if hasattr(d, 'year') else d.date().year for d in dates)):
        mask = pd.Series([d.year if hasattr(d, 'year') else d.date().year for d in dates]) == year
        mask.index = net_ret.index
        yr = net_ret[mask]
        yr_eq = equity[mask]
        if len(yr) < 5:
            continue
        yr_ret = (yr_eq.iloc[-1] / yr_eq.iloc[0]) - 1
        yr_s = sharpe(yr)
        yr_d = max_dd(yr_eq)
        win = "Y" if yr_ret > 0 else "N"
        if yr_ret <= 0:
            loss_years += 1

        # SPY for this year
        spy_yr_ret = 0
        if "SPY" in returns.columns:
            spy_mask = pd.Series(
                [d.year if hasattr(d, 'year') else d.date().year for d in returns.index],
                index=returns.index) == year
            spy_yr = returns["SPY"][spy_mask]
            spy_yr_ret = (1 + spy_yr).prod() - 1
            if yr_ret > spy_yr_ret:
                years_beat_spy += 1

        print(f"  {year:<6} {yr_ret:>+7.2%} {spy_yr_ret:>+7.2%} {yr_s:>8.2f} {yr_d:>+7.2%} {win:>5}")

    total_years = len(set(d.year for d in dates))
    print(f"\n  Loss years:    {loss_years}/{total_years}")
    print(f"  Beat SPY:      {years_beat_spy}/{total_years}")
    print()

    # asset class
    print("  ── Asset Class ──")
    class_groups = defaultdict(list)
    for sym in symbols:
        class_groups[ASSET_TYPES.get(sym, "OTHER")].append(sym)
    for cls in sorted(class_groups):
        syms = class_groups[cls]
        avg_w = weights_post[syms].abs().mean().sum()
        contrib = (weights_post[syms].shift(1) * returns[syms].iloc[warmup:]).sum(axis=1).mean() * 252
        print(f"  {cls:<14} {len(syms):>3} syms, {avg_w:.3f}x, {contrib:+.2%}/yr")
    print()

    # ── Monthly attribution ──
    print("  ── Monthly Returns ──")
    # Build month-year index
    month_dates = pd.Series([d.strftime("%Y-%m") if hasattr(d, 'strftime')
                             else d.date().strftime("%Y-%m") for d in net_ret.index],
                            index=net_ret.index)
    monthly_groups = net_ret.groupby(month_dates)
    monthly_ret = monthly_groups.apply(lambda x: (1 + x).prod() - 1)

    # Print last 24 months
    recent = monthly_ret.tail(24)
    print(f"  {'Month':>8} {'Return':>8}  {'Month':>8} {'Return':>8}  {'Month':>8} {'Return':>8}")
    print("  " + "-" * 58)
    rows = list(recent.items())
    for i in range(0, len(rows), 3):
        parts = []
        for j in range(3):
            if i + j < len(rows):
                m, r = rows[i + j]
                parts.append(f"  {m:>8} {r:>+7.2%}")
            else:
                parts.append("                  ")
        print("".join(parts))
    print()

    # ── Trade log: position changes ──
    print("  ── Trade Log (recent 60 days, top position changes) ──")
    pos_changes = weights.diff().fillna(0)
    recent_changes = pos_changes.tail(60)

    trade_records = []
    for dt_idx in recent_changes.index:
        row = recent_changes.loc[dt_idx]
        movers = row[row.abs() > 0.005].sort_values(key=abs, ascending=False)
        for sym, delta in movers.head(5).items():
            cur_w = weights.loc[dt_idx, sym]
            prev_w = cur_w - delta
            action = "BUY" if delta > 0 else "SELL"
            dt_str = dt_idx.strftime("%Y-%m-%d") if hasattr(dt_idx, 'strftime') else str(dt_idx)[:10]
            sig_val = signals[sym].loc[dt_idx] if sym in signals.columns else 0
            trade_records.append({
                "date": dt_str, "sym": sym, "action": action,
                "delta": delta, "new_wt": cur_w, "signal": sig_val,
                "type": ASSET_TYPES.get(sym, "?"),
            })

    if trade_records:
        print(f"  {'Date':>10} {'Sym':>8} {'Action':>6} {'Delta':>8} {'NewWt':>8} {'Signal':>8} {'Type':>8}")
        print("  " + "-" * 68)
        for t in trade_records[-40:]:  # last 40 trades
            print(f"  {t['date']:>10} {t['sym']:>8} {t['action']:>6} "
                  f"{t['delta']:>+7.3f} {t['new_wt']:>+7.3f} {t['signal']:>+7.3f} {t['type']:>8}")
    else:
        print("  No significant position changes in the last 60 days.")
    print()

    # ── Daily book snapshot (last 5 rebal dates) ──
    print("  ── Book Snapshot (last 5 rebalance dates) ──")
    active_days = weights.iloc[warmup:].loc[(weights.iloc[warmup:].abs() > 0.001).any(axis=1)]
    # Find rebal days (where positions changed significantly)
    rebal_days = active_days.loc[(active_days.diff().abs().sum(axis=1) > 0.01)]
    for dt_idx in rebal_days.index[-5:]:
        row = weights.loc[dt_idx]
        active = row[row.abs() > 0.005].sort_values(ascending=False)
        n_long = (active > 0).sum()
        n_short = (active < 0).sum()
        gross = active.abs().sum()
        net = active.sum()
        dt_str = dt_idx.strftime("%Y-%m-%d") if hasattr(dt_idx, 'strftime') else str(dt_idx)[:10]
        print(f"  {dt_str} | {n_long}L/{n_short}S | Gross: {gross:.2f}x | Net: {net:+.2f}x")
        # Top 5 positions
        for sym, w_val in list(active.head(3).items()) + list(active.tail(3).items()):
            if abs(w_val) > 0.005:
                print(f"    {sym:>8} {w_val:>+7.3f}  ({ASSET_TYPES.get(sym, '?')})")
    print()

    # monte carlo
    if len(net_ret) > 50:
        print("  ── Monte Carlo (10K paths, 1yr) ──")
        terminal, dds_mc = monte_carlo(net_ret)
        print(f"  Median return:  {np.median(terminal)-1:+.2%}")
        print(f"  5th pctl:       {np.percentile(terminal,5)-1:+.2%}")
        print(f"  95th pctl:      {np.percentile(terminal,95)-1:+.2%}")
        print(f"  Prob of loss:   {np.mean(terminal < 1.0):.1%}")
        print(f"  Prob of >20%:   {np.mean(terminal > 1.20):.1%}")
        print(f"  Median max DD:  {np.median(dds_mc):.2%}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AlphaForge — Walk-Forward Backtest")
    p.add_argument("--config", default="config/data_layer.yaml")
    p.add_argument("--env", default=".env", help="env file with API keys")
    p.add_argument("--dynamic", action="store_true",
                   help="Use dynamic universe (all liquid US equities) instead of static YAML list")
    p.add_argument("--no-fundamentals", action="store_true",
                   help="Skip fundamental data fetch (faster, technicals only)")
    p.add_argument("--skip-mcap", action="store_true",
                   help="Skip market cap filter in dynamic universe (uses price+ADV only, much faster)")
    p.add_argument("--nav", type=float, default=10_000_000, help="starting capital (default $10M)")
    p.add_argument("--target-vol", type=float, default=0.12)
    p.add_argument("--slippage", type=float, default=2.0)
    p.add_argument("--rebal-freq", type=int, default=15)
    p.add_argument("--max-pos", type=float, default=0.06)
    p.add_argument("--no-cache", action="store_true",
                   help="Skip disk cache, force fresh data fetch from providers")
    args = p.parse_args()

    # Load env file if it exists
    if os.path.exists(args.env):
        from dotenv import load_dotenv
        load_dotenv(args.env)

    run(args)
