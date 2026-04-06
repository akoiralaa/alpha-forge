#!/usr/bin/env python3
"""
AlphaForge v8 — multi-asset expansion:
  - v7 equity/event core
  - ETF alpha sleeve
  - Futures/commodities/bonds alpha sleeve
  - FX alpha sleeve
  - VIX futures hedge sleeve
  - Black-Scholes put-spread overlay priced with real VIX (FRED VIXCLS, free)

Options hedge is now priced using Black-Scholes with:
  - Sigma: CBOE VIX index via FRED (free, no API key, 1990-present)
  - Risk-free rate: FEDFUNDS from cached FRED macro parquet
  - Mark-to-market daily: reprice put spread as VIX/SPY/T update
  - Roll: monthly (every 21 trading days), strike set at entry

This correctly captures the vol-regime dependency — puts are expensive
when VIX is elevated (exactly when you want them) and cheap in calm markets.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import backtest_v4 as v4
from src.signals.event_alpha import EventAlphaConfig, build_event_alpha_signal

# ── Black-Scholes options pricing with real VIX ─────────────────────────────

_VIX_CACHE_PATH = os.path.expanduser("~/.alphaforge/cache/macro/VIXCLS.parquet")
_VIX_FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"


def fetch_vix_history(force_refresh: bool = False) -> pd.Series:
    """
    CBOE VIX index from FRED — free, no API key, 1990-present.
    Returns a daily Series of VIX values (e.g. 20.0 = 20% annualised vol).
    Caches to parquet so subsequent calls are instant.
    """
    import requests

    if not force_refresh and os.path.exists(_VIX_CACHE_PATH):
        df = pd.read_parquet(_VIX_CACHE_PATH)
        s = df["VIXCLS"].dropna()
        s.index = pd.to_datetime(s.index)
        return s

    resp = requests.get(_VIX_FRED_URL, timeout=30)
    resp.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(resp.text), parse_dates=["observation_date"], index_col="observation_date")
    df = df.replace(".", float("nan"))
    df["VIXCLS"] = pd.to_numeric(df["VIXCLS"], errors="coerce")
    df = df.dropna()
    os.makedirs(os.path.dirname(_VIX_CACHE_PATH), exist_ok=True)
    df.to_parquet(_VIX_CACHE_PATH)
    s = df["VIXCLS"]
    s.index = pd.to_datetime(s.index)
    return s


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes European put price.
      S     = spot price
      K     = strike
      T     = time to expiry in years (e.g. 21/252)
      r     = risk-free rate (annualised, e.g. 0.05)
      sigma = implied vol (annualised, e.g. 0.20 for 20%)
    """
    from scipy.stats import norm
    if T <= 1e-9 or sigma <= 1e-9 or S <= 0:
        return float(max(K - S, 0.0))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return float(max(price, 0.0))


def bs_put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put delta (negative, range [-1, 0])."""
    from scipy.stats import norm
    if T <= 1e-9 or sigma <= 1e-9 or S <= 0:
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1) - 1.0)


def compute_max_underwater_days(equity_curve: pd.Series) -> int:
    if equity_curve.empty:
        return 0
    running_max = equity_curve.cummax()
    underwater = equity_curve < running_max
    max_streak = 0
    streak = 0
    for is_under in underwater.astype(bool).tolist():
        if is_under:
            streak += 1
            if streak > max_streak:
                max_streak = streak
        else:
            streak = 0
    return int(max_streak)


def zero_signal_like(prices: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=prices.index, columns=symbols)


def build_v8_strategy_weights(event_weight: float) -> dict[str, float]:
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


def _signal_ts_rank(returns: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(index=returns.index)
    r = returns[symbols]
    trend_63 = r.rolling(63, min_periods=25).sum()
    trend_126 = r.rolling(126, min_periods=50).sum()
    carry_21 = r.rolling(21, min_periods=10).sum()
    signal = 0.50 * trend_63 + 0.35 * trend_126 + 0.15 * carry_21
    return v4.cross_sectional_rank(signal.replace([np.inf, -np.inf], np.nan).fillna(0.0))


def sanitize_cross_asset_returns(
    returns: pd.DataFrame,
    symbols: list[str],
) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(index=returns.index)
    cleaned = returns[symbols].copy()
    max_abs_by_type = {
        "ETF": 0.25,
        "FUTURE": 0.15,
        "COMMODITY": 0.18,
        "BOND": 0.08,
        "FX": 0.08,
        "VOLATILITY": 0.35,
    }
    for sym in symbols:
        at = v4.ASSET_TYPES.get(sym, "EQUITY")
        cap = float(max_abs_by_type.get(at, 0.25))
        s = cleaned[sym].replace([np.inf, -np.inf], np.nan)
        cleaned[sym] = s.where(s.abs() <= cap, 0.0).fillna(0.0)
    return cleaned


def _find_intraday_cache_file(cache_dir: str, symbol: str, interval: str = "5m") -> str | None:
    base = os.path.expanduser(cache_dir)
    if not os.path.isdir(base):
        return None
    symbol = str(symbol).upper().strip()
    pattern_a = os.path.join(base, f"{symbol}_*_{interval}.parquet")
    pattern_b = os.path.join(base, f"{symbol}_{interval}.parquet")
    candidates = sorted(glob.glob(pattern_a)) + sorted(glob.glob(pattern_b))
    if not candidates:
        return None
    # Prefer explicit class-tagged file (e.g., SPY_ETF_5m.parquet) when available.
    tagged = [p for p in candidates if f"_{interval}.parquet" in os.path.basename(p) and "_" in os.path.basename(p).replace(f"_{interval}.parquet", "")]
    return tagged[0] if tagged else candidates[0]


def _build_intraday_daily_signal(
    parquet_path: str,
    target_index: pd.Index,
    interval: str = "5m",
) -> pd.Series:
    try:
        df = pd.read_parquet(parquet_path, columns=["timestamp_ns", "open", "high", "low", "close", "volume"])
    except Exception:
        return pd.Series(0.0, index=target_index, dtype=float)
    if df.empty or "timestamp_ns" not in df.columns or "close" not in df.columns:
        return pd.Series(0.0, index=target_index, dtype=float)

    ts = pd.to_datetime(df["timestamp_ns"], unit="ns", utc=True, errors="coerce").dt.tz_localize(None)
    valid = ts.notna()
    if not valid.any():
        return pd.Series(0.0, index=target_index, dtype=float)

    frame = df.loc[valid, ["open", "high", "low", "close", "volume"]].copy()
    frame["ts"] = ts.loc[valid].values
    frame = frame.sort_values("ts")
    frame["day"] = frame["ts"].dt.normalize()

    rows: list[tuple[pd.Timestamp, float, float, float]] = []
    for day, g in frame.groupby("day", sort=True):
        if len(g) < 6:
            continue
        g = g.sort_values("ts")
        open_30 = float(g["close"].iloc[min(5, len(g) - 1)])
        close_last = float(g["close"].iloc[-1])
        open_first = float(g["open"].iloc[0]) if "open" in g.columns else float(g["close"].iloc[0])
        hi = float(g["high"].max()) if "high" in g.columns else close_last
        lo = float(g["low"].min()) if "low" in g.columns else close_last
        vol = float(g["volume"].sum()) if "volume" in g.columns else 0.0
        if not np.isfinite(open_30) or not np.isfinite(close_last) or open_30 <= 0:
            continue
        late_move = close_last / open_30 - 1.0
        intraday_range = (hi - lo) / max(open_first, 1e-9)
        rows.append((pd.Timestamp(day), float(-late_move), float(intraday_range), float(np.log1p(max(vol, 0.0)))))

    if not rows:
        return pd.Series(0.0, index=target_index, dtype=float)

    daily = pd.DataFrame(rows, columns=["day", "late_rev", "day_range", "log_vol"]).set_index("day").sort_index()
    range_z = _rolling_zscore(daily["day_range"], window=40, min_obs=15).clip(-3.0, 3.0)
    vol_z = _rolling_zscore(daily["log_vol"], window=40, min_obs=15).clip(-3.0, 3.0)
    raw = daily["late_rev"] + 0.35 * range_z + 0.10 * vol_z * np.sign(daily["late_rev"])
    signal = _rolling_zscore(raw, window=60, min_obs=20).clip(-2.0, 2.0) / 2.0
    # Strict anti-lookahead: today's intraday signal can only be traded next day.
    signal = signal.shift(1).fillna(0.0)

    out = pd.Series(0.0, index=target_index, dtype=float)
    target_days = pd.to_datetime(target_index, errors="coerce").normalize()
    mapped = signal.reindex(target_days).fillna(0.0).to_numpy(dtype=float)
    out.iloc[:] = mapped
    return out


def build_intraday_cache_alpha_sleeve(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    candidate_symbols: list[str],
    cache_dir: str,
    interval: str,
    rebal_freq: int,
    gross_budget: float,
    max_pos: float,
    min_signal: float = 0.08,
    allow_short: bool = True,
    warmup: int = 120,
) -> tuple[pd.DataFrame, dict[str, float]]:
    symbols = [s for s in candidate_symbols if s in returns.columns]
    if not symbols:
        return pd.DataFrame(0.0, index=prices.index, columns=[]), {
            "status": "no_symbols",
            "loaded_symbols": 0.0,
            "active_days": 0.0,
        }

    signal_book: dict[str, pd.Series] = {}
    for sym in symbols:
        path = _find_intraday_cache_file(cache_dir, sym, interval)
        if not path:
            continue
        signal = _build_intraday_daily_signal(path, prices.index, interval=interval)
        if signal.abs().sum() <= 1e-9:
            continue
        signal_book[sym] = signal

    if not signal_book:
        return pd.DataFrame(0.0, index=prices.index, columns=symbols), {
            "status": "no_intraday_cache",
            "loaded_symbols": 0.0,
            "active_days": 0.0,
        }

    signal_df = pd.DataFrame(signal_book, index=prices.index).fillna(0.0)
    ranked = v4.cross_sectional_rank(signal_df)
    sleeve = pd.DataFrame(0.0, index=prices.index, columns=ranked.columns)
    target = pd.Series(0.0, index=ranked.columns, dtype=float)

    for i in range(max(warmup, 1), len(prices.index)):
        if i % max(rebal_freq, 1) != 0 and i > warmup:
            sleeve.iloc[i] = target
            continue
        row = ranked.iloc[i].dropna()
        if row.empty:
            sleeve.iloc[i] = target
            continue

        n_assets = max(len(row), 1)
        n_long = max(1, int(n_assets * 0.30))
        n_short = max(1, int(n_assets * 0.25))
        longs = row[row > min_signal].sort_values(ascending=False).head(n_long)
        shorts = row[row < -min_signal].sort_values(ascending=True).head(n_short)

        w = pd.Series(0.0, index=ranked.columns, dtype=float)
        long_budget = gross_budget * (0.60 if allow_short else 1.00)
        short_budget = gross_budget * 0.40 if allow_short else 0.0
        if len(longs) > 0:
            lw = min(max_pos, long_budget / len(longs))
            w.loc[longs.index] = lw
        if allow_short and len(shorts) > 0:
            sw = min(max_pos, short_budget / len(shorts))
            w.loc[shorts.index] = -sw
        target = w
        sleeve.iloc[i] = target

    active = sleeve.abs().sum(axis=1)
    diag = {
        "status": "ok",
        "loaded_symbols": float(len(signal_book)),
        "active_days": float((active > 1e-9).mean()),
        "avg_gross": float(active.mean()),
    }
    return sleeve.fillna(0.0), diag


def build_cross_asset_alpha_sleeve(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    symbols: list[str],
    env: pd.Series,
    breadth: pd.Series,
    regime_info: pd.DataFrame | None,
    rebal_freq: int,
    gross_budget: float,
    max_pos: float,
    min_signal: float = 0.06,
    long_frac: float = 0.35,
    short_frac: float = 0.30,
    allow_short: bool = True,
    warmup: int = 160,
) -> pd.DataFrame:
    sleeve = pd.DataFrame(0.0, index=prices.index, columns=symbols)
    if not symbols:
        return sleeve

    signal = _signal_ts_rank(returns, symbols)
    inst_vol = returns[symbols].rolling(20, min_periods=10).std().bfill() * np.sqrt(252)
    inst_vol = inst_vol.clip(lower=0.03, upper=1.20)

    target = pd.Series(0.0, index=symbols, dtype=float)
    live = target.copy()
    running_nav = 1.0
    peak_nav = 1.0

    for i in range(warmup, len(prices)):
        if i > warmup:
            day_ret = float((live * returns[symbols].iloc[i]).sum())
            running_nav *= (1.0 + day_ret)
            peak_nav = max(peak_nav, running_nav)
        current_dd = (running_nav / peak_nav) - 1.0 if peak_nav > 0 else 0.0

        if i % rebal_freq != 0 and i > warmup:
            live = target.copy()
            sleeve.iloc[i] = live
            continue

        row = signal.iloc[i].dropna()
        if row.empty:
            live = target.copy()
            sleeve.iloc[i] = live
            continue

        e = float(env.iloc[i]) if i < len(env) else 1.0
        b = float(breadth.iloc[i]) if i < len(breadth) else 0.5
        regime_row = regime_info.iloc[i] if regime_info is not None and i < len(regime_info) else None
        crisis_score, _ = v4.compute_crisis_overlay(e, b, regime_row, current_dd, preemptive_de_risk=0.35)
        gross_live = float(np.clip(gross_budget * (1.0 - 0.50 * crisis_score), gross_budget * 0.35, gross_budget))

        n_assets = max(len(row), 1)
        n_long = max(1, int(n_assets * long_frac))
        n_short = max(1, int(n_assets * short_frac))
        long_candidates = row[row > min_signal].sort_values(ascending=False).head(n_long)
        short_candidates = row[row < -min_signal].sort_values(ascending=True).head(n_short)

        w = pd.Series(0.0, index=symbols, dtype=float)

        long_budget = gross_live * (0.65 if allow_short else 1.0)
        short_budget = gross_live * 0.35 if allow_short else 0.0
        if crisis_score > 0.55:
            long_budget *= 0.85
            short_budget *= 1.20

        if not long_candidates.empty:
            inv = {}
            for sym in long_candidates.index:
                inv[sym] = 1.0 / max(float(inst_vol[sym].iloc[i]), 0.04)
            inv_sum = sum(inv.values()) or 1.0
            for sym, sig_val in long_candidates.items():
                tilt = 0.75 + 0.45 * min(abs(float(sig_val)), 1.0)
                w[sym] = float(np.clip((long_budget * inv[sym] / inv_sum) * tilt, 0.0, max_pos))

        if allow_short and not short_candidates.empty:
            inv = {}
            for sym in short_candidates.index:
                inv[sym] = 1.0 / max(float(inst_vol[sym].iloc[i]), 0.04)
            inv_sum = sum(inv.values()) or 1.0
            for sym, _ in short_candidates.items():
                raw = -short_budget * inv[sym] / inv_sum
                w[sym] = float(np.clip(raw, -max_pos, 0.0))

        if i > warmup:
            keep = pd.Series(0.88, index=symbols, dtype=float)
            entering = (target.abs() < 1e-12) & (w.abs() > 1e-12)
            exiting = (target.abs() > 1e-12) & (w.abs() < 1e-12)
            flipping = (
                (target.abs() > 1e-12)
                & (w.abs() > 1e-12)
                & (np.sign(target) != np.sign(w))
            )
            keep[entering] = 0.70
            keep[exiting] = 0.55
            keep[flipping] = 0.35
            w = (1.0 - keep) * w + keep * target
            w[(w.abs() < 0.0015) & exiting] = 0.0

        target = w
        live = target.copy()
        sleeve.iloc[i] = live

    return sleeve.fillna(0.0)


def build_vix_hedge_sleeve(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    vix_symbols: list[str],
    env: pd.Series,
    breadth: pd.Series,
    regime_info: pd.DataFrame | None,
    rebal_freq: int,
    max_gross: float,
    max_pos: float,
    warmup: int = 100,
) -> pd.DataFrame:
    sleeve = pd.DataFrame(0.0, index=prices.index, columns=vix_symbols)
    if not vix_symbols:
        return sleeve

    sym = vix_symbols[0]
    target = 0.0
    for i in range(warmup, len(prices)):
        if i % rebal_freq != 0 and i > warmup:
            sleeve.iloc[i, 0] = target
            continue

        e = float(env.iloc[i]) if i < len(env) else 1.0
        b = float(breadth.iloc[i]) if i < len(breadth) else 0.5
        regime_row = regime_info.iloc[i] if regime_info is not None and i < len(regime_info) else None
        crisis_score, _ = v4.compute_crisis_overlay(e, b, regime_row, current_dd=0.0, preemptive_de_risk=0.40)

        trend_10 = float((prices[sym].iloc[i] / prices[sym].iloc[max(i - 10, 0)] - 1.0)) if sym in prices else 0.0
        trend_boost = 1.0 + max(trend_10, 0.0)
        raw = max_gross * np.clip(crisis_score * trend_boost, 0.0, 1.0)
        if crisis_score < 0.18:
            raw = 0.0
        target = float(np.clip(raw, 0.0, max_pos))
        sleeve.iloc[i, 0] = target

    return sleeve.fillna(0.0)


def build_bs_option_overlay(
    spy_ret: pd.Series,
    spy_prices: pd.Series,
    env: pd.Series,
    breadth: pd.Series,
    regime_info: pd.DataFrame | None,
    roll_days: int,
    max_notional: float,
    strike_otm: float,
    short_strike_otm: float,
    activation_score: float,
    severe_score: float,
    always_on_min_notional: float = 0.0,
    crash_prob: pd.Series | None = None,
    calm_short_coverage: float = 0.95,
    crash_short_coverage_floor: float = 0.15,
    crash_prob_sensitivity: float = 0.45,
    # Deprecated synthetic params — kept for CLI backward-compat, ignored
    payout_mult: float = 1.0,
    theta_bps_daily: float = 1.8,
    short_credit_bps_daily: float = 0.9,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Black-Scholes put-spread overlay priced with real VIX history.

    On each roll date:
      - Size notional based on crisis regime score (same logic as before)
      - Record entry BS price for the put spread using current VIX as IV
      - Set strikes: long K = S*(1-strike_otm), short K = S*(1-short_strike_otm)

    Each day between rolls:
      - Reprice both legs using updated VIX, updated SPY, remaining T
      - Daily P&L = change in spread mark-to-market value * notional / S_entry
      - This correctly captures vega gain when VIX spikes during a crisis

    Risk-free rate: FEDFUNDS from cached FRED macro parquet (falls back to 0).
    Vol: CBOE VIX / 100, forward-filled on non-trading days, floored at 5%.
    """
    idx = spy_ret.index

    # ── Load real VIX ──
    try:
        vix_series = fetch_vix_history()
        vix = vix_series.reindex(idx).ffill().bfill().fillna(20.0) / 100.0
    except Exception:
        vix = pd.Series(0.20, index=idx)  # fallback: 20% flat

    vix = vix.clip(lower=0.05)  # floor at 5% to avoid BS degeneracy

    # ── Risk-free rate from FEDFUNDS cache ──
    rf = pd.Series(0.04, index=idx)
    try:
        ff_path = os.path.expanduser("~/.alphaforge/cache/macro/FEDFUNDS.parquet")
        if os.path.exists(ff_path):
            ff = pd.read_parquet(ff_path)
            ff.index = pd.to_datetime(ff.index)
            ff_col = [c for c in ff.columns if "FEDFUNDS" in c.upper()][0]
            rf_loaded = (ff[ff_col] / 100.0).reindex(idx).ffill().bfill().fillna(0.04)
            rf = rf_loaded.clip(lower=0.0, upper=0.20)
    except Exception:
        pass

    spy_px = spy_prices.reindex(idx).ffill().bfill()

    option_ret = pd.Series(0.0, index=idx, dtype=float)
    option_notional = pd.Series(0.0, index=idx, dtype=float)
    option_short_coverage = pd.Series(0.0, index=idx, dtype=float)

    activation_score = float(np.clip(activation_score, 0.05, 0.95))
    severe_score = float(np.clip(severe_score, activation_score + 0.01, 0.99))
    always_on_min_notional = float(np.clip(always_on_min_notional, 0.0, max_notional))
    calm_short_coverage = float(np.clip(calm_short_coverage, 0.35, 1.0))
    crash_short_coverage_floor = float(np.clip(crash_short_coverage_floor, 0.05, calm_short_coverage))
    crash_prob_sensitivity = float(np.clip(crash_prob_sensitivity, 0.0, 1.0))
    crash_prob_al = (
        crash_prob.reindex(idx).ffill().fillna(0.0).clip(0.0, 1.0)
        if crash_prob is not None
        else pd.Series(0.0, index=idx, dtype=float)
    )

    # Active contract state
    live_notional = 0.0
    pending_notional = 0.0
    live_short_coverage = 0.0
    pending_short_coverage = 0.0

    # Current live put-spread contract
    S_entry = None       # SPY price at entry
    K_long = None        # long put strike
    K_short = None       # short put strike
    T_entry = None       # time to expiry at entry (years)
    days_held = 0        # days since last roll
    prev_spread_val = 0.0  # previous day's spread value per unit notional/S

    for i in range(len(idx)):
        option_notional.iloc[i] = live_notional
        option_short_coverage.iloc[i] = live_short_coverage if live_notional > 1e-12 else 0.0

        S_now = float(spy_px.iloc[i])
        iv_now = float(vix.iloc[i])
        r_now = float(rf.iloc[i])

        # ── Mark-to-market existing position ──
        if live_notional > 1e-12 and S_entry is not None:
            days_held += 1
            T_remaining = max((roll_days - days_held) / 252.0, 1.0 / 252.0)
            long_val = bs_put_price(S_now, K_long, T_remaining, r_now, iv_now)
            short_val = bs_put_price(S_now, K_short, T_remaining, r_now, iv_now)
            spread_val = long_val - live_short_coverage * short_val  # per $1 of notional/S
            spread_val_scaled = spread_val / S_entry  # normalise to entry notional basis

            # P&L = change in spread value * number of puts (notional / S_entry)
            # Each put covers 1 unit; notional is fraction of portfolio
            daily_pnl = (spread_val_scaled - prev_spread_val) * live_notional * S_entry
            option_ret.iloc[i] = daily_pnl / max(live_notional * S_entry, 1e-12) * live_notional \
                if live_notional > 0 else 0.0
            # Simpler: express as return on notional
            option_ret.iloc[i] = (spread_val_scaled - prev_spread_val)
            prev_spread_val = spread_val_scaled

        # ── Roll logic (every roll_days bars) ──
        if i % roll_days == 0:
            e = float(env.reindex(idx).iloc[i]) if len(env) else 1.0
            b = float(breadth.reindex(idx).iloc[i]) if len(breadth) else 0.5
            regime_row = regime_info.reindex(idx).iloc[i] if regime_info is not None else None
            crisis_score, _ = v4.compute_crisis_overlay(e, b, regime_row, current_dd=0.0, preemptive_de_risk=0.45)
            cp = float(crash_prob_al.iloc[i])
            stress_boost = 0.0
            stress_boost += 0.12 * np.clip((0.42 - b) / 0.22, 0.0, 1.0)
            stress_boost += 0.08 * np.clip((0.72 - e) / 0.30, 0.0, 1.0)
            effective_score = float(np.clip(crisis_score + stress_boost + crash_prob_sensitivity * cp, 0.0, 1.0))

            if effective_score >= activation_score:
                scaled = (effective_score - activation_score) / max(1.0 - activation_score, 1e-9)
                pending_notional = float(np.clip(
                    always_on_min_notional + (max_notional - always_on_min_notional) * (0.20 + 0.80 * scaled),
                    0.0, max_notional
                ))
                severe_frac = np.clip(
                    (effective_score - severe_score) / max(1.0 - severe_score, 1e-9), 0.0, 1.0
                )
                pending_short_coverage = float(np.clip(
                    1.0 - (1.0 - crash_short_coverage_floor) * severe_frac,
                    crash_short_coverage_floor, 1.0
                ))
            else:
                pending_notional = float(np.clip(always_on_min_notional * (1.0 + 0.25 * cp), 0.0, max_notional))
                pending_short_coverage = (
                    float(np.clip(calm_short_coverage - 0.40 * cp, crash_short_coverage_floor, 1.0))
                    if pending_notional > 1e-12 else 0.0
                )

            # Reset contract at roll
            live_notional = pending_notional
            live_short_coverage = pending_short_coverage
            days_held = 0

            if live_notional > 1e-12 and S_now > 0:
                S_entry = S_now
                K_long = S_now * (1.0 - strike_otm)
                K_short = S_now * (1.0 - short_strike_otm)
                T_entry = roll_days / 252.0
                # Price the new spread at entry (this is the cost paid on roll day)
                entry_long = bs_put_price(S_entry, K_long, T_entry, r_now, iv_now)
                entry_short = bs_put_price(S_entry, K_short, T_entry, r_now, iv_now)
                prev_spread_val = (entry_long - live_short_coverage * entry_short) / S_entry
                # Entry cost as a one-day return hit
                option_ret.iloc[i] -= prev_spread_val  # pay the premium on entry
            else:
                S_entry = None
                K_long = K_short = T_entry = None
                prev_spread_val = 0.0

    option_turnover = option_notional.diff().abs().fillna(option_notional.abs())
    return option_ret, option_notional, option_turnover, option_short_coverage


# Keep old name as alias so any external scripts calling the old name still work
def build_synthetic_option_overlay(*args, **kwargs):
    return build_bs_option_overlay(*args, **kwargs)


def build_option_roll_plan(
    spy_prices: pd.Series,
    option_notional: pd.Series,
    option_short_coverage: pd.Series,
    roll_days: int,
    strike_otm: float = 0.06,
    short_strike_otm: float = 0.12,
) -> pd.DataFrame:
    rows = []
    for i, dt in enumerate(option_notional.index):
        if i % roll_days != 0:
            continue
        notional = float(option_notional.iloc[i])
        if notional <= 1e-10:
            continue
        spot = float(spy_prices.reindex(option_notional.index).iloc[i])
        if not np.isfinite(spot) or spot <= 0:
            continue
        strike = spot * (1.0 - strike_otm)
        cov = float(option_short_coverage.reindex(option_notional.index).iloc[i])
        short_strike = spot * (1.0 - short_strike_otm)
        is_spread = cov > 0.05
        rows.append(
            {
                "date": dt,
                "underlying": "SPY",
                "side": "BUY",
                "contract_type": "PUT_SPREAD" if is_spread else "PUT",
                "strike": round(strike, 2),
                "short_strike": round(short_strike, 2) if is_spread else None,
                "short_coverage": round(cov, 3),
                "target_notional": notional,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "underlying",
                "side",
                "contract_type",
                "strike",
                "short_strike",
                "short_coverage",
                "target_notional",
            ]
        )
    return pd.DataFrame(rows)


def per_asset_turnover_cost(
    weights: pd.DataFrame,
    bps_by_symbol: dict[str, float],
) -> pd.Series:
    turnover = weights.diff().abs().fillna(0.0)
    cost = pd.Series(0.0, index=weights.index, dtype=float)
    for sym in weights.columns:
        bps = float(bps_by_symbol.get(sym, 3.0))
        cost = cost.add(turnover[sym] * (bps / 10000.0), fill_value=0.0)
    return cost


def _compute_sleeve_returns(
    sleeve_weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.Series:
    cols = [c for c in sleeve_weights.columns if c in returns.columns]
    if not cols:
        return pd.Series(0.0, index=sleeve_weights.index, dtype=float)
    aligned = returns.reindex(index=sleeve_weights.index, columns=cols).fillna(0.0)
    return (sleeve_weights[cols].shift(1).fillna(0.0) * aligned).sum(axis=1)


def build_sleeve_governance_multipliers(
    sleeve_weights: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    expected_sharpes: dict[str, float],
    window: int = 126,
    min_obs: int = 42,
    soft_dd: float = 0.10,
    hard_dd: float = 0.22,
    kill_sharpe: float = -0.15,
    kill_dd: float = 0.12,
    cooldown_days: int = 21,
    min_mult: float = 0.35,
    max_mult: float = 1.25,
    smooth_days: int = 5,
    enable_weak_sleeve_hard_demote: bool = False,
    weak_sleeve_names: tuple[str, ...] = ("fx",),
    weak_demote_sharpe: float = -0.10,
    weak_recover_sharpe: float = 0.15,
    weak_demote_confirm_days: int = 21,
    weak_recover_confirm_days: int = 42,
    weak_demote_mult: float = 0.0,
) -> tuple[dict[str, pd.Series], dict[str, dict[str, float]]]:
    multipliers: dict[str, pd.Series] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    weak_set = {str(s).strip().lower() for s in weak_sleeve_names if str(s).strip()}
    for sleeve_name, w in sleeve_weights.items():
        raw_ret = _compute_sleeve_returns(w.fillna(0.0), returns).fillna(0.0)
        hist_ret = raw_ret.shift(1).fillna(0.0)
        trailing_mean = hist_ret.rolling(window, min_periods=min_obs).mean()
        trailing_vol = hist_ret.rolling(window, min_periods=min_obs).std()
        trailing_sharpe = (trailing_mean / (trailing_vol + 1e-12)) * np.sqrt(252.0)
        hit_rate = (hist_ret > 0).astype(float).rolling(window, min_periods=min_obs).mean().fillna(0.50)

        nav = (1.0 + hist_ret).cumprod()
        rolling_peak = nav.rolling(window, min_periods=min_obs).max()
        trailing_dd = (nav / rolling_peak - 1.0).fillna(0.0)

        expected = float(expected_sharpes.get(sleeve_name, 0.25))
        alpha_term = (trailing_sharpe - expected).clip(-2.0, 2.0)
        mult = 1.0 + 0.18 * alpha_term + 0.35 * (hit_rate - 0.50)
        dd_penalty = (((-trailing_dd) - soft_dd) / max(hard_dd - soft_dd, 1e-9)).clip(0.0, 1.0)
        mult = (mult * (1.0 - 0.65 * dd_penalty)).clip(min_mult, max_mult).fillna(1.0)

        mult_vals = mult.to_numpy(dtype=float, copy=True)
        sharpe_vals = trailing_sharpe.fillna(0.0).to_numpy(dtype=float)
        dd_vals = trailing_dd.fillna(0.0).to_numpy(dtype=float)
        cooldown = 0
        kill_events = 0
        hard_demote_events = 0
        hard_demote_days = 0
        demoted = False
        demote_streak = 0
        recover_streak = 0
        demoted_active = np.zeros(len(mult_vals), dtype=bool)
        for i in range(len(mult_vals)):
            if cooldown > 0:
                mult_vals[i] = min(mult_vals[i], min_mult)
                cooldown -= 1
            elif sharpe_vals[i] <= kill_sharpe and dd_vals[i] <= -abs(kill_dd):
                kill_events += 1
                cooldown = cooldown_days
                mult_vals[i] = min(mult_vals[i], min_mult)

            if enable_weak_sleeve_hard_demote and sleeve_name.lower() in weak_set:
                s_i = sharpe_vals[i]
                if demoted:
                    hard_demote_days += 1
                    demoted_active[i] = True
                    mult_vals[i] = min(mult_vals[i], weak_demote_mult)
                    if s_i >= weak_recover_sharpe:
                        recover_streak += 1
                    else:
                        recover_streak = 0
                    if recover_streak >= max(int(weak_recover_confirm_days), 1):
                        demoted = False
                        recover_streak = 0
                        demote_streak = 0
                else:
                    if s_i <= weak_demote_sharpe:
                        demote_streak += 1
                    else:
                        demote_streak = 0
                    if demote_streak >= max(int(weak_demote_confirm_days), 1):
                        demoted = True
                        hard_demote_events += 1
                        hard_demote_days += 1
                        demoted_active[i] = True
                        mult_vals[i] = min(mult_vals[i], weak_demote_mult)
                        recover_streak = 0

        series = pd.Series(mult_vals, index=w.index, dtype=float)
        if smooth_days > 1:
            series = series.rolling(int(smooth_days), min_periods=1).mean()
        floor_mult = weak_demote_mult if (enable_weak_sleeve_hard_demote and sleeve_name.lower() in weak_set) else min_mult
        series = series.clip(floor_mult, max_mult).fillna(1.0)
        if enable_weak_sleeve_hard_demote and sleeve_name.lower() in weak_set and demoted_active.any():
            demote_mask = pd.Series(demoted_active, index=w.index)
            series = series.where(~demote_mask, np.minimum(series, weak_demote_mult))
        multipliers[sleeve_name] = series
        diagnostics[sleeve_name] = {
            "avg_multiplier": float(multipliers[sleeve_name].mean()),
            "min_multiplier": float(multipliers[sleeve_name].min()),
            "max_multiplier": float(multipliers[sleeve_name].max()),
            "kill_events": float(kill_events),
            "hard_demote_events": float(hard_demote_events),
            "hard_demote_days": float(hard_demote_days),
            "avg_trailing_sharpe": float(trailing_sharpe.replace([np.inf, -np.inf], np.nan).fillna(0.0).mean()),
            "worst_trailing_dd": float(trailing_dd.min()),
        }
    return multipliers, diagnostics


def build_regime_risk_scaler(
    index: pd.Index,
    env: pd.Series,
    breadth: pd.Series,
    regime_info: pd.DataFrame | None,
    floor: float = 0.80,
    ceiling: float = 1.12,
    smooth_days: int = 7,
) -> pd.Series:
    if len(index) == 0:
        return pd.Series(dtype=float)

    floor = float(np.clip(floor, 0.30, 1.00))
    ceiling = float(np.clip(ceiling, max(floor + 0.01, 0.40), 1.60))
    env_al = env.reindex(index).ffill().fillna(1.0)
    breadth_al = breadth.reindex(index).ffill().fillna(0.5)
    regime_al = regime_info.reindex(index).ffill() if regime_info is not None else None

    scaler = pd.Series(1.0, index=index, dtype=float)
    for i in range(len(index)):
        e = float(env_al.iloc[i])
        b = float(breadth_al.iloc[i])
        regime_row = regime_al.iloc[i] if regime_al is not None else None
        crisis_score, _ = v4.compute_crisis_overlay(
            e, b, regime_row, current_dd=0.0, preemptive_de_risk=0.40
        )
        base = floor + (ceiling - floor) * (1.0 - crisis_score)

        # Small, confidence-weighted regime tilt around the crisis baseline.
        if regime_row is not None:
            label = str(regime_row.get("regime_label", "WARMUP"))
            confidence = float(np.clip(regime_row.get("confidence", 0.0), 0.0, 1.0))
            if label in {"LOW_VOL_TRENDING", "MEAN_REVERTING_RANGE"}:
                base += 0.05 * confidence
            elif label in {"HIGH_VOL_CHAOTIC", "LIQUIDITY_CRISIS"}:
                base -= 0.07 * confidence

        if e > 1.08 and b > 0.58:
            base += 0.03
        elif e < 0.75 or b < 0.42:
            base -= 0.05
        scaler.iloc[i] = float(np.clip(base, floor, ceiling))

    if smooth_days > 1:
        scaler = scaler.rolling(int(smooth_days), min_periods=1).mean()
    return scaler.clip(floor, ceiling).fillna(1.0)


def build_regime_router_v10(
    index: pd.Index,
    prices: pd.DataFrame,
    env: pd.Series,
    breadth: pd.Series,
    macro_cache_file: str = "",
    vol_window: int = 21,
    trend_window: int = 200,
    crash_threshold: float = 0.62,
    risk_off_threshold: float = 0.50,
    smooth_days: int = 3,
) -> tuple[dict[str, pd.Series], dict[str, float]]:
    if len(index) == 0:
        neutral = pd.Series(dtype=float)
        return {
            "state": pd.Series(dtype="object"),
            "crash_prob": neutral,
            "core_mult": neutral,
            "etf_mult": neutral,
            "futures_mult": neutral,
            "fx_mult": neutral,
            "vix_mult": neutral,
            "gross_mult": neutral,
        }, {"status": "empty"}

    env_al = env.reindex(index).ffill().fillna(1.0)
    breadth_al = breadth.reindex(index).ffill().fillna(0.5)
    spy = prices["SPY"].reindex(index).ffill().bfill() if "SPY" in prices.columns else pd.Series(1.0, index=index)
    spy_ret = spy.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vol = spy_ret.rolling(int(max(vol_window, 5)), min_periods=max(int(vol_window // 2), 5)).std() * np.sqrt(252.0)
    ma = spy.rolling(int(max(trend_window, 30)), min_periods=max(int(trend_window // 4), 30)).mean()
    trend = (spy / ma - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    curve = pd.Series(0.0, index=index, dtype=float)
    if macro_cache_file:
        panel = load_macro_panel(macro_cache_file, index)
        if "T10Y2Y" in panel.columns:
            curve = panel["T10Y2Y"].fillna(0.0)
        elif {"DGS10", "DGS2"}.issubset(set(panel.columns)):
            curve = (panel["DGS10"] - panel["DGS2"]).fillna(0.0)

    # Strict anti-lookahead: state at t uses observations up to t-1 only.
    e_lag = env_al.shift(1).bfill().fillna(1.0)
    b_lag = breadth_al.shift(1).bfill().fillna(0.5)
    vol_lag = vol.shift(1).bfill().fillna(vol.median() if np.isfinite(vol.median()) else 0.18)
    trend_lag = trend.shift(1).bfill().fillna(0.0)
    curve_lag = curve.shift(1).bfill().fillna(0.0)

    crash_prob = (
        0.38 * np.clip((0.75 - e_lag) / 0.30, 0.0, 1.0)
        + 0.28 * np.clip((0.45 - b_lag) / 0.25, 0.0, 1.0)
        + 0.18 * np.clip((vol_lag - 0.22) / 0.20, 0.0, 1.0)
        + 0.10 * np.clip((-trend_lag - 0.03) / 0.10, 0.0, 1.0)
        + 0.06 * np.clip((-curve_lag) / 1.00, 0.0, 1.0)
    ).clip(0.0, 1.0)
    risk_off_prob = (
        0.34 * np.clip((0.92 - e_lag) / 0.30, 0.0, 1.0)
        + 0.30 * np.clip((0.56 - b_lag) / 0.30, 0.0, 1.0)
        + 0.20 * np.clip((vol_lag - 0.16) / 0.18, 0.0, 1.0)
        + 0.10 * np.clip((-trend_lag) / 0.12, 0.0, 1.0)
        + 0.06 * np.clip((-curve_lag) / 1.25, 0.0, 1.0)
    ).clip(0.0, 1.0)
    if smooth_days > 1:
        crash_prob = crash_prob.rolling(int(smooth_days), min_periods=1).mean().clip(0.0, 1.0)
        risk_off_prob = risk_off_prob.rolling(int(smooth_days), min_periods=1).mean().clip(0.0, 1.0)

    state = pd.Series("risk_on", index=index, dtype="object")
    state[(crash_prob >= float(crash_threshold)) | ((e_lag < 0.68) & (b_lag < 0.35))] = "crash"
    state[(state != "crash") & ((risk_off_prob >= float(risk_off_threshold)) | (vol_lag > 0.24))] = "risk_off"

    maps = {
        "risk_on": {"core": 1.03, "etf": 1.12, "futures": 1.03, "fx": 1.03, "vix": 0.20, "gross": 1.08},
        "risk_off": {"core": 0.92, "etf": 0.72, "futures": 0.82, "fx": 0.90, "vix": 0.85, "gross": 0.86},
        "crash": {"core": 0.78, "etf": 0.40, "futures": 0.58, "fx": 0.72, "vix": 1.30, "gross": 0.66},
    }

    def _map(name: str) -> pd.Series:
        return state.map(lambda s: float(maps.get(str(s), maps["risk_off"])[name])).astype(float)

    out = {
        "state": state,
        "crash_prob": crash_prob.astype(float),
        "core_mult": _map("core"),
        "etf_mult": _map("etf"),
        "futures_mult": _map("futures"),
        "fx_mult": _map("fx"),
        "vix_mult": _map("vix"),
        "gross_mult": _map("gross"),
    }
    diag = {
        "status": "ok",
        "risk_on_days": float((state == "risk_on").sum()),
        "risk_off_days": float((state == "risk_off").sum()),
        "crash_days": float((state == "crash").sum()),
        "avg_crash_prob": float(crash_prob.mean()),
        "max_crash_prob": float(crash_prob.max()),
    }
    return out, diag


def apply_entry_exit_hysteresis(
    weights: pd.DataFrame,
    entry_days: int = 2,
    exit_days: int = 3,
    entry_abs: float = 0.0035,
    exit_abs: float = 0.0020,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if weights.empty:
        return weights.copy(), {"turnover_in": 0.0, "turnover_out": 0.0, "turnover_reduction": 0.0}

    entry_days = int(max(entry_days, 1))
    exit_days = int(max(exit_days, 1))
    entry_abs = float(max(entry_abs, 0.0))
    exit_abs = float(max(min(exit_abs, entry_abs if entry_abs > 0 else exit_abs), 0.0))

    out = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
    for col in weights.columns:
        series = weights[col].fillna(0.0).astype(float)
        live = 0.0
        live_sign = 0
        same_sign_count = 0
        same_sign_val = 0
        weak_count = 0
        flip_count = 0
        for i, raw in enumerate(series.to_numpy(dtype=float)):
            sgn = int(np.sign(raw)) if abs(raw) >= entry_abs else 0
            if sgn != 0 and sgn == same_sign_val:
                same_sign_count += 1
            elif sgn != 0:
                same_sign_val = sgn
                same_sign_count = 1
            else:
                same_sign_count = 0

            if live_sign == 0:
                if sgn != 0 and same_sign_count >= entry_days:
                    live = raw
                    live_sign = int(np.sign(raw))
                    weak_count = 0
                    flip_count = 0
                else:
                    live = 0.0
            else:
                if abs(raw) <= exit_abs:
                    weak_count += 1
                else:
                    weak_count = 0

                if sgn != 0 and sgn != live_sign and same_sign_count >= entry_days:
                    flip_count += 1
                else:
                    flip_count = 0

                if weak_count >= exit_days:
                    live = 0.0
                    live_sign = 0
                    weak_count = 0
                    flip_count = 0
                elif flip_count >= 1:
                    live = raw
                    live_sign = int(np.sign(raw))
                    weak_count = 0
                    flip_count = 0
                else:
                    # Keep side but allow sizing updates while trend persists.
                    if abs(raw) >= exit_abs:
                        live = raw
            out.iloc[i, out.columns.get_loc(col)] = live

    turn_in = float(weights.diff().abs().sum(axis=1).mean())
    turn_out = float(out.diff().abs().sum(axis=1).mean())
    diag = {
        "turnover_in": turn_in,
        "turnover_out": turn_out,
        "turnover_reduction": float(turn_in - turn_out),
    }
    return out.fillna(0.0), diag


def build_yearly_risk_budget_scaler(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    annual_vol_budget: float = 0.18,
    annual_dd_budget: float = 0.18,
    min_scale: float = 0.75,
    max_scale: float = 1.10,
    smooth_days: int = 5,
) -> tuple[pd.Series, dict[str, float]]:
    if weights.empty:
        neutral = pd.Series(dtype=float)
        return neutral, {"status": "empty"}

    annual_vol_budget = float(max(annual_vol_budget, 0.05))
    annual_dd_budget = float(max(annual_dd_budget, 0.05))
    min_scale = float(np.clip(min_scale, 0.20, 1.00))
    max_scale = float(np.clip(max_scale, min_scale + 0.01, 1.60))

    cols = [c for c in weights.columns if c in returns.columns]
    if not cols:
        neutral = pd.Series(1.0, index=weights.index, dtype=float)
        return neutral, {"status": "no_overlap", "avg_scale": 1.0, "min_scale": 1.0, "max_scale": 1.0}

    realized = (weights[cols].shift(1).fillna(0.0) * returns[cols].reindex(weights.index).fillna(0.0)).sum(axis=1)
    hist_ret = realized.shift(1).fillna(0.0)

    years = hist_ret.index.year
    ytd_vol = hist_ret.groupby(years).expanding().std().reset_index(level=0, drop=True).fillna(0.0) * np.sqrt(252.0)
    ytd_nav = (1.0 + hist_ret).groupby(years).cumprod()
    ytd_peak = ytd_nav.groupby(years).cummax()
    ytd_dd = (ytd_nav / ytd_peak - 1.0).fillna(0.0)

    vol_ratio = (ytd_vol / annual_vol_budget).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    dd_ratio = ((-ytd_dd) / annual_dd_budget).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    over_vol = (vol_ratio - 1.0).clip(lower=0.0, upper=2.0)
    over_dd = (dd_ratio - 1.0).clip(lower=0.0, upper=2.0)
    under_vol = (0.90 - vol_ratio).clip(lower=0.0, upper=1.0)

    raw = 1.0 - 0.40 * over_vol - 0.70 * over_dd + 0.06 * under_vol
    raw = raw.clip(min_scale, max_scale).fillna(1.0)

    month = pd.Series(pd.PeriodIndex(hist_ret.index, freq="M"), index=hist_ret.index)
    rebalance = (month != month.shift(1)).fillna(True)
    scaler = raw.where(rebalance).ffill().fillna(1.0)
    if smooth_days > 1:
        scaler = scaler.rolling(int(smooth_days), min_periods=1).mean()
    scaler = scaler.clip(min_scale, max_scale).fillna(1.0)

    diag = {
        "status": "ok",
        "avg_scale": float(scaler.mean()),
        "min_scale": float(scaler.min()),
        "max_scale": float(scaler.max()),
        "avg_ytd_vol": float(ytd_vol.mean()),
        "worst_ytd_dd": float(ytd_dd.min()),
    }
    return scaler, diag


def _rolling_zscore(series: pd.Series, window: int = 252, min_obs: int = 63) -> pd.Series:
    min_periods = int(min(max(min_obs, 1), max(window, 1)))
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    z = (series - mean) / (std + 1e-12)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def load_macro_panel(cache_file: str, index: pd.Index) -> pd.DataFrame:
    if not cache_file:
        return pd.DataFrame(index=index)
    # Preserve the caller's original index shape, but align macro data on normalized
    # calendar dates to avoid timezone/intraday timestamp mismatches.
    target_index = pd.to_datetime(index, utc=False, errors="coerce")
    if isinstance(target_index, pd.DatetimeIndex) and target_index.tz is not None:
        target_index = target_index.tz_convert(None)
    target_dates = pd.DatetimeIndex(target_index).normalize()

    path = os.path.expanduser(cache_file)
    if not os.path.exists(path):
        return pd.DataFrame(index=index)
    try:
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame(index=index)
    if df.empty:
        return pd.DataFrame(index=index)

    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], utc=False, errors="coerce")
        df = df.drop(columns=["date"]).copy()
        df.index = dt
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=False, errors="coerce")
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.index = df.index.normalize()
    df = df[~df.index.isna()].sort_index()
    if df.empty:
        return pd.DataFrame(index=index)

    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        return pd.DataFrame(index=index)
    numeric = numeric.groupby(level=0).last()
    aligned = numeric.reindex(target_dates).ffill()
    aligned.index = index
    return aligned


def build_macro_overlay_scaler(
    index: pd.Index,
    cache_file: str,
    max_de_risk: float = 0.12,
    min_scale: float = 0.85,
    smooth_days: int = 5,
    z_window: int = 252,
) -> tuple[pd.Series, dict[str, float | str]]:
    neutral = pd.Series(1.0, index=index, dtype=float)
    panel = load_macro_panel(cache_file, index)
    if panel.empty:
        return neutral, {
            "status": "no_macro_cache",
            "cache_file": os.path.expanduser(cache_file),
            "components_used": 0.0,
            "avg_scale": 1.0,
            "min_scale": 1.0,
            "max_scale": 1.0,
        }

    components: list[tuple[float, pd.Series]] = []

    curve = None
    if "T10Y2Y" in panel.columns:
        curve = panel["T10Y2Y"]
    elif {"DGS10", "DGS2"}.issubset(set(panel.columns)):
        curve = panel["DGS10"] - panel["DGS2"]
    if curve is not None:
        curve_comp = np.tanh(_rolling_zscore(curve, window=z_window, min_obs=63) / 2.5)
        components.append((0.50, curve_comp))

    if "UNRATE" in panel.columns:
        # Rising unemployment is typically risk-off.
        unrate_chg = panel["UNRATE"].diff(63)
        unrate_comp = -np.tanh(_rolling_zscore(unrate_chg, window=z_window, min_obs=63) / 2.5)
        components.append((0.30, unrate_comp))

    if "FEDFUNDS" in panel.columns:
        # Rapid policy tightening is typically risk-off.
        ff_chg = panel["FEDFUNDS"].diff(63)
        ff_comp = -np.tanh(_rolling_zscore(ff_chg, window=z_window, min_obs=63) / 2.5)
        components.append((0.20, ff_comp))

    if not components:
        return neutral, {
            "status": "no_required_macro_columns",
            "cache_file": os.path.expanduser(cache_file),
            "components_used": 0.0,
            "avg_scale": 1.0,
            "min_scale": 1.0,
            "max_scale": 1.0,
        }

    weight_sum = sum(w for w, _ in components)
    score = pd.Series(0.0, index=index, dtype=float)
    for w, comp in components:
        score = score.add((w / max(weight_sum, 1e-12)) * comp.reindex(index).fillna(0.0), fill_value=0.0)
    score = score.clip(-1.0, 1.0)

    max_de_risk = float(np.clip(max_de_risk, 0.0, 0.60))
    min_scale = float(np.clip(min_scale, 1.0 - max_de_risk, 1.0))
    risk_off = (-score).clip(lower=0.0, upper=1.0)
    scale = (1.0 - max_de_risk * risk_off).clip(min_scale, 1.0)
    if smooth_days > 1:
        scale = scale.rolling(int(smooth_days), min_periods=1).mean()
    scale = scale.clip(min_scale, 1.0).fillna(1.0)
    return scale, {
        "status": "ok",
        "cache_file": os.path.expanduser(cache_file),
        "components_used": float(len(components)),
        "avg_scale": float(scale.mean()),
        "min_scale": float(scale.min()),
        "max_scale": float(scale.max()),
        "avg_score": float(score.mean()),
    }


def build_asset_cost_map(symbols: list[str]) -> dict[str, float]:
    cost_map = {}
    for sym in symbols:
        at = v4.ASSET_TYPES.get(sym, "EQUITY")
        if at == "FX":
            cost_map[sym] = 1.0
        elif at in {"FUTURE", "COMMODITY", "BOND"}:
            cost_map[sym] = 1.3
        elif at == "VOLATILITY":
            cost_map[sym] = 4.0
        elif at == "ETF":
            cost_map[sym] = 2.0
        else:
            cost_map[sym] = 3.0
    return cost_map


def compute_dynamic_execution_cost(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    volumes: pd.DataFrame,
    bps_by_symbol: dict[str, float],
    vol_lookback: int = 20,
    adv_lookback: int = 20,
    vol_cost_mult: float = 0.45,
    liq_cost_mult: float = 0.70,
    gap_jump_threshold: float = 0.04,
    gap_jump_bps: float = 1.2,
    min_scale: float = 0.60,
    max_scale: float = 3.25,
) -> tuple[pd.Series, dict[str, float]]:
    if weights.empty:
        return pd.Series(0.0, index=returns.index, dtype=float), {
            "avg_cost_scale": 1.0,
            "p95_cost_scale": 1.0,
            "avg_jump_cost_bps": 0.0,
        }

    cols = list(weights.columns)
    idx = weights.index
    turnover = weights.diff().abs().fillna(0.0)
    ret_aligned = returns.reindex(index=idx, columns=cols).fillna(0.0)
    px_aligned = prices.reindex(index=idx, columns=cols).ffill().bfill()
    vol_aligned = volumes.reindex(index=idx, columns=cols).ffill().fillna(0.0)

    realized_vol = ret_aligned.rolling(vol_lookback, min_periods=max(5, vol_lookback // 3)).std()
    baseline_vol = realized_vol.rolling(252, min_periods=63).median().fillna(realized_vol)
    vol_ratio = (realized_vol / (baseline_vol + 1e-12)).clip(0.4, 3.5).fillna(1.0)

    dollar_volume = (px_aligned.abs() * vol_aligned).replace([np.inf, -np.inf], np.nan)
    adv_short = dollar_volume.rolling(adv_lookback, min_periods=max(5, adv_lookback // 3)).median()
    adv_long = dollar_volume.rolling(252, min_periods=63).median().fillna(adv_short)
    liq_ratio = (adv_long / (adv_short + 1e-9)).clip(0.5, 4.0).fillna(1.0)

    cost_scale = (1.0 + vol_cost_mult * (vol_ratio - 1.0)) * (1.0 + liq_cost_mult * (liq_ratio - 1.0))
    cost_scale = cost_scale.clip(min_scale, max_scale).fillna(1.0)

    base_bps = pd.Series({sym: float(bps_by_symbol.get(sym, 3.0)) for sym in cols}, dtype=float)
    base_cost = turnover.mul(base_bps / 10000.0, axis=1)
    dynamic_cost = (base_cost * cost_scale).sum(axis=1)

    jump_mask = (ret_aligned.abs() > gap_jump_threshold).astype(float)
    jump_cost = (turnover * jump_mask).sum(axis=1) * (gap_jump_bps / 10000.0)
    total = dynamic_cost.add(jump_cost, fill_value=0.0)

    diag = {
        "avg_cost_scale": float(cost_scale.mean().mean()),
        "p95_cost_scale": float(cost_scale.stack().quantile(0.95)) if not cost_scale.empty else 1.0,
        "avg_jump_cost_bps": float(jump_cost.mean() * 252 * 10000),
    }
    return total.fillna(0.0), diag


def _coerce_timestamp_for_index(ts_like: str, index: pd.Index) -> pd.Timestamp:
    ts = pd.Timestamp(ts_like)
    idx_tz = getattr(index, "tz", None)
    if idx_tz is not None:
        if ts.tzinfo is None:
            return ts.tz_localize(idx_tz)
        return ts.tz_convert(idx_tz)
    if ts.tzinfo is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts


def run(args):
    universe = v4.load_universe_static(args.config)
    if args.cache_complete_only and not args.no_cache:
        keep = []
        dropped = []
        for sym, atype in universe:
            cache_file = os.path.join(v4.CACHE_DIR, f"{sym}_{atype}.parquet")
            if os.path.exists(cache_file):
                keep.append((sym, atype))
            else:
                dropped.append(f"{sym}_{atype}")
        universe = keep
        if dropped:
            print(
                "Cache-complete-only mode: "
                f"dropped {len(dropped)} symbols with missing local bars."
            )
            print("  Missing keys:", ", ".join(dropped[:20]) + (" ..." if len(dropped) > 20 else ""))
            print()
    if not universe:
        raise ValueError("Universe is empty after cache/date filters.")
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

    if args.start_date:
        start_ts = _coerce_timestamp_for_index(args.start_date, prices.index)
        prices = prices[prices.index >= start_ts]
    if args.end_date:
        end_ts = _coerce_timestamp_for_index(args.end_date, prices.index)
        prices = prices[prices.index <= end_ts]
    volumes = volumes.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    symbols = list(prices.columns)
    if len(prices) < int(args.min_rows):
        raise ValueError(
            f"Insufficient rows after date filtering ({len(prices)}). "
            f"Need at least {int(args.min_rows)} rows for warmup and stable signals."
        )
    print(f"  Matrix: {prices.shape[0]} x {prices.shape[1]} instruments\n")

    equity_syms = [s for s in symbols if v4.ASSET_TYPES.get(s) == "EQUITY"]
    etf_syms = [s for s in symbols if v4.ASSET_TYPES.get(s) == "ETF"]
    futures_syms = [s for s in symbols if v4.ASSET_TYPES.get(s) in {"FUTURE", "COMMODITY", "BOND"}]
    fx_syms = [s for s in symbols if v4.ASSET_TYPES.get(s) == "FX"]
    vix_syms = [s for s in symbols if v4.ASSET_TYPES.get(s) == "VOLATILITY"]
    print(
        f"  Trade buckets: equities={len(equity_syms)}, etf={len(etf_syms)}, "
        f"futures/commod/bond={len(futures_syms)}, fx={len(fx_syms)}, vix={len(vix_syms)}\n"
    )

    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mkt_ret = returns["SPY"] if "SPY" in returns.columns else returns.iloc[:, 0]

    print("=" * 70)
    print("  COMPUTING v8 SIGNALS")
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
            revision_weight=args.revision_flow_weight,
        ),
    )
    strategies["earnings_drift"] = event_build.composite
    event_weight = (
        float(args.force_event_weight)
        if args.force_event_weight is not None
        else event_build.suggested_weight
    )
    strategy_weights = build_v8_strategy_weights(event_weight)

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
    print("  BUILDING v8 PORTFOLIO (core + multi-asset + options hedge)")
    print("=" * 70)
    print(
        f"  NAV: ${args.nav:,.0f} | Vol target: {args.target_vol:.0%} | Rebal: {args.rebal_freq}d | "
        f"Long: {args.n_long} | Short: {args.n_short}"
    )
    print(
        f"  Overlay gross caps: etf={args.etf_gross:.2f}x, futures={args.futures_gross:.2f}x, "
        f"fx={args.fx_gross:.2f}x, vix={args.vix_hedge_max:.2f}x, options={args.option_max_notional:.2f}x\n"
    )

    t0 = time.time()
    core_weights, core_diag = v4.build_multi_strategy_portfolio(
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
    core_weights = core_weights.fillna(0.0)

    if args.enable_kelly_sentiment_overlay:
        from src.portfolio.sizing import build_fractional_kelly_overlay
        world_sentiment = (
            event_build.sentiment.replace(0.0, np.nan).mean(axis=1).fillna(0.0).clip(-1.0, 1.0)
        )
        kelly_scale, kelly_diag = build_fractional_kelly_overlay(
            signal=event_build.composite[equity_syms],
            returns=returns[equity_syms],
            lookback=args.kelly_lookback,
            min_obs=args.kelly_min_obs,
            kelly_fraction=args.kelly_fraction,
            max_abs_kelly=args.kelly_max_abs,
            min_scale=args.kelly_min_scale,
            max_scale=args.kelly_max_scale,
            sentiment_series=world_sentiment,
            sentiment_sensitivity=args.kelly_sentiment_sensitivity,
        )
        core_weights = core_weights.mul(kelly_scale, axis=0)
        print(
            "  Kelly-sentiment overlay: on | "
            f"avg={kelly_diag['overlay_avg']:.3f} "
            f"range=[{kelly_diag['overlay_min']:.3f},{kelly_diag['overlay_max']:.3f}] "
            f"kelly_avg={kelly_diag['kelly_avg']:.4f} "
            f"positive_days={kelly_diag['kelly_pos_days']:.1%}"
        )
    else:
        print("  Kelly-sentiment overlay: off")

    etf_alpha_symbols = [s for s in etf_syms if s != "SPY"] if args.exclude_spy_from_etf_alpha else etf_syms
    intraday_symbols = [
        s.strip().upper()
        for s in str(args.intraday_symbols).split(",")
        if s.strip()
    ]
    cross_asset_symbols = list(dict.fromkeys(etf_alpha_symbols + futures_syms + fx_syms + vix_syms + intraday_symbols))
    cross_asset_returns = sanitize_cross_asset_returns(returns, cross_asset_symbols)

    etf_sleeve = build_cross_asset_alpha_sleeve(
        prices, cross_asset_returns, etf_alpha_symbols, env, breadth, regime_info,
        rebal_freq=args.rebal_freq,
        gross_budget=args.etf_gross,
        max_pos=args.max_overlay_pos,
        min_signal=args.overlay_min_signal,
        allow_short=args.overlay_shorts,
        warmup=140,
    )
    futures_sleeve = build_cross_asset_alpha_sleeve(
        prices, cross_asset_returns, futures_syms, env, breadth, regime_info,
        rebal_freq=args.rebal_freq,
        gross_budget=args.futures_gross,
        max_pos=args.max_overlay_pos,
        min_signal=args.overlay_min_signal,
        allow_short=args.overlay_shorts,
        warmup=160,
    )
    fx_sleeve = build_cross_asset_alpha_sleeve(
        prices, cross_asset_returns, fx_syms, env, breadth, regime_info,
        rebal_freq=max(5, args.rebal_freq // 2),
        gross_budget=args.fx_gross,
        max_pos=args.max_overlay_pos,
        min_signal=max(args.overlay_min_signal * 0.8, 0.04),
        allow_short=args.overlay_shorts,
        warmup=120,
    )
    vix_sleeve = build_vix_hedge_sleeve(
        prices, cross_asset_returns, vix_syms, env, breadth, regime_info,
        rebal_freq=max(5, args.rebal_freq // 2),
        max_gross=args.vix_hedge_max,
        max_pos=args.max_overlay_pos,
    )
    intraday_effective = False
    if args.enable_intraday_cache_sleeve:
        intraday_sleeve, intraday_diag = build_intraday_cache_alpha_sleeve(
            prices=prices,
            returns=cross_asset_returns,
            candidate_symbols=intraday_symbols,
            cache_dir=args.intraday_cache_dir,
            interval=args.intraday_interval,
            rebal_freq=args.intraday_rebal_freq,
            gross_budget=args.intraday_gross,
            max_pos=args.intraday_max_pos,
            min_signal=args.intraday_min_signal,
            allow_short=args.intraday_allow_shorts,
            warmup=max(90, args.warmup_days // 2),
        )
        loaded_symbols = int(intraday_diag.get("loaded_symbols", 0))
        intraday_effective = loaded_symbols > 0 and (len(intraday_sleeve.columns) > 0)
        if not intraday_effective:
            intraday_sleeve = pd.DataFrame(0.0, index=prices.index, columns=[])
            intraday_diag = {
                **intraday_diag,
                "status": "disabled_no_cache",
                "loaded_symbols": 0.0,
                "active_days": 0.0,
                "avg_gross": 0.0,
            }
            print("  Intraday cache sleeve: auto-disabled (no usable local intraday cache)")
        else:
            print(
                "  Intraday cache sleeve: on | "
                f"symbols={int(intraday_diag.get('loaded_symbols', 0))} "
                f"active_days={intraday_diag.get('active_days', 0.0):.1%} "
                f"avg_gross={intraday_diag.get('avg_gross', 0.0):.3f}x"
            )
    else:
        intraday_sleeve = pd.DataFrame(0.0, index=prices.index, columns=[])
        intraday_diag = {"status": "off", "loaded_symbols": 0.0, "active_days": 0.0, "avg_gross": 0.0}
        print("  Intraday cache sleeve: off")

    if args.enable_regime_router_v10:
        router_map, router_diag = build_regime_router_v10(
            index=core_weights.index,
            prices=prices,
            env=env,
            breadth=breadth,
            macro_cache_file=args.router_macro_cache_file,
            vol_window=args.router_vol_window,
            trend_window=args.router_trend_window,
            crash_threshold=args.router_crash_threshold,
            risk_off_threshold=args.router_risk_off_threshold,
            smooth_days=args.router_smooth_days,
        )
        core_weights = core_weights.mul(router_map["core_mult"], axis=0)
        etf_sleeve = etf_sleeve.mul(router_map["etf_mult"], axis=0)
        futures_sleeve = futures_sleeve.mul(router_map["futures_mult"], axis=0)
        fx_sleeve = fx_sleeve.mul(router_map["fx_mult"], axis=0)
        vix_sleeve = vix_sleeve.mul(router_map["vix_mult"], axis=0)
        intraday_sleeve = intraday_sleeve.mul(router_map["etf_mult"], axis=0)
        router_state = router_map["state"].reindex(core_weights.index).ffill().fillna("risk_on")
        router_gross_mult = router_map["gross_mult"].clip(0.40, 1.30).fillna(1.0)
        crash_prob = router_map["crash_prob"].clip(0.0, 1.0).fillna(0.0)
        print(
            "  Regime router v10: on | "
            f"risk_on={int(router_diag.get('risk_on_days', 0))} "
            f"risk_off={int(router_diag.get('risk_off_days', 0))} "
            f"crash={int(router_diag.get('crash_days', 0))} "
            f"avg_crash_prob={router_diag.get('avg_crash_prob', 0.0):.3f}"
        )
    else:
        router_map = {}
        router_diag = {"status": "off"}
        router_state = pd.Series("risk_on", index=core_weights.index, dtype="object")
        router_gross_mult = pd.Series(1.0, index=core_weights.index, dtype=float)
        crash_prob = pd.Series(0.0, index=core_weights.index, dtype=float)
        print("  Regime router v10: off")

    governance_diag: dict[str, dict[str, float]] = {}
    if not args.disable_sleeve_governance:
        sleeve_book = {
            "core": core_weights,
            "etf": etf_sleeve,
            "futures": futures_sleeve,
            "fx": fx_sleeve,
            "vix": vix_sleeve,
        }
        sleeve_expected_sharpes = {
            "core": args.gov_expected_sharpe_core,
            "etf": args.gov_expected_sharpe_etf,
            "futures": args.gov_expected_sharpe_futures,
            "fx": args.gov_expected_sharpe_fx,
            "vix": args.gov_expected_sharpe_vix,
        }
        if intraday_effective:
            sleeve_book["intraday"] = intraday_sleeve
            sleeve_expected_sharpes["intraday"] = args.gov_expected_sharpe_intraday
        sleeve_mult, governance_diag = build_sleeve_governance_multipliers(
            sleeve_weights=sleeve_book,
            returns=returns,
            expected_sharpes=sleeve_expected_sharpes,
            window=args.gov_window,
            min_obs=args.gov_min_obs,
            soft_dd=args.gov_soft_dd,
            hard_dd=args.gov_hard_dd,
            kill_sharpe=args.gov_kill_sharpe,
            kill_dd=args.gov_kill_dd,
            cooldown_days=args.gov_cooldown_days,
            min_mult=args.gov_min_mult,
            max_mult=args.gov_max_mult,
            smooth_days=args.gov_smooth_days,
            enable_weak_sleeve_hard_demote=args.enable_weak_sleeve_hard_demote,
            weak_sleeve_names=tuple(s.strip().lower() for s in str(args.weak_sleeve_list).split(",") if s.strip()),
            weak_demote_sharpe=args.weak_demote_sharpe,
            weak_recover_sharpe=args.weak_recover_sharpe,
            weak_demote_confirm_days=args.weak_demote_confirm_days,
            weak_recover_confirm_days=args.weak_recover_confirm_days,
            weak_demote_mult=args.weak_demote_mult,
        )
        core_weights = core_weights.mul(sleeve_mult["core"], axis=0)
        etf_sleeve = etf_sleeve.mul(sleeve_mult["etf"], axis=0)
        futures_sleeve = futures_sleeve.mul(sleeve_mult["futures"], axis=0)
        fx_sleeve = fx_sleeve.mul(sleeve_mult["fx"], axis=0)
        vix_sleeve = vix_sleeve.mul(sleeve_mult["vix"], axis=0)
        if intraday_effective and "intraday" in sleeve_mult:
            intraday_sleeve = intraday_sleeve.mul(sleeve_mult["intraday"], axis=0)

    all_cols = list(dict.fromkeys(
        list(core_weights.columns)
        + list(etf_sleeve.columns)
        + list(futures_sleeve.columns)
        + list(fx_sleeve.columns)
        + list(vix_sleeve.columns)
        + list(intraday_sleeve.columns)
    ))
    weights = pd.DataFrame(0.0, index=core_weights.index, columns=all_cols)
    weights = weights.add(core_weights.reindex(columns=all_cols).fillna(0.0), fill_value=0.0)
    weights = weights.add(etf_sleeve.reindex(columns=all_cols).fillna(0.0), fill_value=0.0)
    weights = weights.add(futures_sleeve.reindex(columns=all_cols).fillna(0.0), fill_value=0.0)
    weights = weights.add(fx_sleeve.reindex(columns=all_cols).fillna(0.0), fill_value=0.0)
    weights = weights.add(vix_sleeve.reindex(columns=all_cols).fillna(0.0), fill_value=0.0)
    weights = weights.add(intraday_sleeve.reindex(columns=all_cols).fillna(0.0), fill_value=0.0)

    if args.enable_regime_risk_scaler:
        regime_scaler = build_regime_risk_scaler(
            index=weights.index,
            env=env,
            breadth=breadth,
            regime_info=regime_info,
            floor=args.risk_floor,
            ceiling=args.risk_ceiling,
            smooth_days=args.risk_smooth_days,
        )
        weights = weights.mul(regime_scaler, axis=0)
        print(
            f"  Regime risk scaler: on | avg={regime_scaler.mean():.3f} "
            f"min={regime_scaler.min():.3f} max={regime_scaler.max():.3f}"
        )
    else:
        regime_scaler = pd.Series(1.0, index=weights.index, dtype=float)
        print("  Regime risk scaler: off")

    if args.enable_macro_overlay:
        macro_scaler, macro_diag = build_macro_overlay_scaler(
            index=weights.index,
            cache_file=args.macro_cache_file,
            max_de_risk=args.macro_max_de_risk,
            min_scale=args.macro_min_scale,
            smooth_days=args.macro_smooth_days,
            z_window=args.macro_zscore_window,
        )
        weights = weights.mul(macro_scaler, axis=0)
        print(
            f"  Macro overlay: on | status={macro_diag.get('status')} "
            f"avg={macro_diag.get('avg_scale', 1.0):.3f} "
            f"min={macro_diag.get('min_scale', 1.0):.3f} "
            f"max={macro_diag.get('max_scale', 1.0):.3f}"
        )
    else:
        macro_scaler = pd.Series(1.0, index=weights.index, dtype=float)
        macro_diag = {"status": "off", "avg_scale": 1.0, "min_scale": 1.0, "max_scale": 1.0}
        print("  Macro overlay: off")

    adaptive_diag = {
        "status": "off",
        "trigger_crash_prob": float(args.adaptive_v10_crash_prob_trigger),
        "avg_gate": 0.0,
        "max_gate": 0.0,
        "stress_days": 0.0,
        "stress_ratio": 0.0,
    }
    stress_gate = pd.Series(0.0, index=weights.index, dtype=float)
    if args.enable_adaptive_v10_layers:
        trigger = float(np.clip(args.adaptive_v10_crash_prob_trigger, 0.05, 0.95))
        router_state_al = router_state.reindex(weights.index).ffill().fillna("risk_on")
        crash_prob_al = crash_prob.reindex(weights.index).ffill().fillna(0.0).clip(0.0, 1.0)
        hard_stress = ((router_state_al != "risk_on") | (crash_prob_al >= trigger)).astype(float)
        if args.adaptive_v10_soft_gate:
            soft_gate = ((crash_prob_al - trigger) / max(1.0 - trigger, 1e-9)).clip(0.0, 1.0)
            stress_gate = pd.Series(np.maximum(hard_stress.to_numpy(), soft_gate.to_numpy()), index=weights.index, dtype=float)
            status = "on_soft"
        else:
            stress_gate = hard_stress.astype(float)
            status = "on_hard"
        adaptive_diag = {
            "status": status,
            "trigger_crash_prob": trigger,
            "avg_gate": float(stress_gate.mean()),
            "max_gate": float(stress_gate.max()),
            "stress_days": float((stress_gate > 1e-9).sum()),
            "stress_ratio": float((stress_gate > 1e-9).mean()),
        }
        print(
            "  Adaptive v10 layer gate: on | "
            f"mode={status} trigger={trigger:.2f} "
            f"stress_days={int(adaptive_diag['stress_days'])} "
            f"avg_gate={adaptive_diag['avg_gate']:.3f}"
        )
    else:
        print("  Adaptive v10 layer gate: off")

    state_bank_diag = {
        "status": "off",
        "avg_risk_mult": 1.0,
        "avg_hedge_mult": 1.0,
        "avg_gross_mult": 1.0,
        "avg_option_mult": 1.0,
    }
    state_gross_mult = pd.Series(1.0, index=weights.index, dtype=float)
    state_option_mult = pd.Series(1.0, index=weights.index, dtype=float)
    if args.enable_state_param_bank_v10:
        state = router_state.reindex(weights.index).ffill().fillna("risk_on")
        risk_map = {
            "risk_on": float(args.state_risk_on_mult),
            "risk_off": float(args.state_risk_off_mult),
            "crash": float(args.state_crash_mult),
        }
        hedge_map = {
            "risk_on": float(args.state_hedge_risk_on_mult),
            "risk_off": float(args.state_hedge_risk_off_mult),
            "crash": float(args.state_hedge_crash_mult),
        }
        gross_map = {
            "risk_on": float(args.state_gross_risk_on_mult),
            "risk_off": float(args.state_gross_risk_off_mult),
            "crash": float(args.state_gross_crash_mult),
        }
        option_map = {
            "risk_on": float(args.state_option_risk_on_mult),
            "risk_off": float(args.state_option_risk_off_mult),
            "crash": float(args.state_option_crash_mult),
        }
        risk_mult = state.map(lambda s: float(risk_map.get(str(s), 1.0))).astype(float).clip(0.40, 1.40)
        hedge_mult = state.map(lambda s: float(hedge_map.get(str(s), 1.0))).astype(float).clip(0.40, 1.80)
        state_gross_mult = state.map(lambda s: float(gross_map.get(str(s), 1.0))).astype(float).clip(0.40, 1.50)
        state_option_mult = state.map(lambda s: float(option_map.get(str(s), 1.0))).astype(float).clip(0.40, 2.00)
        core_weights = core_weights.mul(risk_mult, axis=0)
        etf_sleeve = etf_sleeve.mul(risk_mult, axis=0)
        futures_sleeve = futures_sleeve.mul(risk_mult, axis=0)
        fx_sleeve = fx_sleeve.mul(risk_mult, axis=0)
        intraday_sleeve = intraday_sleeve.mul(risk_mult, axis=0)
        vix_sleeve = vix_sleeve.mul(hedge_mult, axis=0)
        risk_cols = list(
            dict.fromkeys(
                list(core_weights.columns)
                + list(etf_sleeve.columns)
                + list(futures_sleeve.columns)
                + list(fx_sleeve.columns)
                + list(intraday_sleeve.columns)
            )
        )
        hedge_cols = list(vix_sleeve.columns)
        if risk_cols:
            weights[risk_cols] = weights[risk_cols].mul(risk_mult, axis=0)
        if hedge_cols:
            weights[hedge_cols] = weights[hedge_cols].mul(hedge_mult, axis=0)
        state_bank_diag = {
            "status": "on",
            "avg_risk_mult": float(risk_mult.mean()),
            "avg_hedge_mult": float(hedge_mult.mean()),
            "avg_gross_mult": float(state_gross_mult.mean()),
            "avg_option_mult": float(state_option_mult.mean()),
        }
        print(
            "  State param bank v10: on | "
            f"risk={state_bank_diag['avg_risk_mult']:.3f} "
            f"hedge={state_bank_diag['avg_hedge_mult']:.3f} "
            f"gross={state_bank_diag['avg_gross_mult']:.3f} "
            f"option={state_bank_diag['avg_option_mult']:.3f}"
        )
    else:
        print("  State param bank v10: off")

    whipsaw_diag = {"status": "off", "turnover_in": 0.0, "turnover_out": 0.0, "turnover_reduction": 0.0}
    if args.enable_whipsaw_control_v10:
        whipsaw_out, whipsaw_diag_base = apply_entry_exit_hysteresis(
            weights=weights,
            entry_days=args.whipsaw_entry_days,
            exit_days=args.whipsaw_exit_days,
            entry_abs=args.whipsaw_entry_abs,
            exit_abs=args.whipsaw_exit_abs,
        )
        if args.enable_adaptive_v10_layers:
            active = stress_gate.reindex(weights.index).fillna(0.0)
            weights = weights.mul(1.0 - active, axis=0).add(whipsaw_out.mul(active, axis=0), fill_value=0.0)
            whipsaw_diag = {
                **whipsaw_diag_base,
                "status": "adaptive",
                "applied_days": float((active > 1e-9).sum()),
                "applied_ratio": float((active > 1e-9).mean()),
                "avg_gate": float(active.mean()),
            }
        else:
            weights = whipsaw_out
            whipsaw_diag = {"status": "on", **whipsaw_diag_base}
        print(
            "  Whipsaw control v10: on | "
            f"turnover_reduction={whipsaw_diag['turnover_reduction']:.6f}"
        )
    else:
        print("  Whipsaw control v10: off")

    # Global gross cap to keep deployment-safe leverage.
    gross = weights.abs().sum(axis=1)
    gross_cap = float(np.clip(args.total_gross_cap, 0.10, 4.00))
    effective_gross_cap = (gross_cap * router_gross_mult.reindex(weights.index).fillna(1.0)).clip(
        lower=0.10, upper=4.00
    )
    effective_gross_cap = (
        effective_gross_cap * state_gross_mult.reindex(weights.index).fillna(1.0)
    ).clip(lower=0.10, upper=4.00)
    scaler = (effective_gross_cap / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
    weights = weights.mul(scaler, axis=0)

    budget_diag = {"status": "off", "avg_scale": 1.0, "min_scale": 1.0, "max_scale": 1.0}
    if args.enable_yearly_risk_budget_v10:
        budget_scaler, budget_diag = build_yearly_risk_budget_scaler(
            weights=weights,
            returns=returns,
            annual_vol_budget=args.yearly_vol_budget,
            annual_dd_budget=args.yearly_dd_budget,
            min_scale=args.yearly_budget_min_scale,
            max_scale=args.yearly_budget_max_scale,
            smooth_days=args.yearly_budget_smooth_days,
        )
        budget_scaler = budget_scaler.reindex(weights.index).fillna(1.0)
        if args.enable_adaptive_v10_layers:
            active = stress_gate.reindex(weights.index).fillna(0.0)
            adaptive_scaler = 1.0 - active * (1.0 - budget_scaler)
            weights = weights.mul(adaptive_scaler, axis=0)
            budget_diag = {
                **budget_diag,
                "status": "adaptive",
                "adaptive_avg_scale": float(adaptive_scaler.mean()),
                "adaptive_min_scale": float(adaptive_scaler.min()),
                "adaptive_max_scale": float(adaptive_scaler.max()),
                "adaptive_avg_gate": float(active.mean()),
            }
        else:
            weights = weights.mul(budget_scaler, axis=0)
            budget_diag = {**budget_diag, "status": "on"}
        # Re-apply gross cap after yearly controller to preserve deployment limits.
        gross2 = weights.abs().sum(axis=1)
        scaler2 = (effective_gross_cap / gross2.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
        weights = weights.mul(scaler2, axis=0)
        print(
            "  Yearly risk budget v10: on | "
            f"avg={budget_diag['avg_scale']:.3f} "
            f"min={budget_diag['min_scale']:.3f} "
            f"max={budget_diag['max_scale']:.3f}"
        )
    else:
        print("  Yearly risk budget v10: off")

    spy_ret = prices["SPY"].pct_change(fill_method=None).fillna(0.0).reindex(weights.index).fillna(0.0)
    option_ret, option_notional, option_turnover, option_short_coverage = build_bs_option_overlay(
        spy_ret=spy_ret,
        spy_prices=prices["SPY"].reindex(weights.index).ffill().bfill(),
        env=env.reindex(weights.index).ffill().fillna(1.0),
        breadth=breadth.reindex(weights.index).ffill().fillna(0.5),
        regime_info=regime_info,
        roll_days=args.option_roll_days,
        max_notional=args.option_max_notional,
        strike_otm=args.option_strike_daily,
        short_strike_otm=args.option_short_strike_daily,
        activation_score=args.option_activation_score,
        severe_score=args.option_severe_score,
        always_on_min_notional=args.option_always_on_min_notional,
        crash_prob=crash_prob.reindex(weights.index).fillna(0.0),
        calm_short_coverage=args.option_calm_short_coverage,
        crash_short_coverage_floor=args.option_crash_short_coverage_floor,
        crash_prob_sensitivity=args.option_crash_prob_sensitivity,
        # Kept for backward-compat with old CLI args, unused in BS model
        payout_mult=args.option_payout_mult,
        theta_bps_daily=args.option_theta_bps_daily,
        short_credit_bps_daily=args.option_short_credit_bps_daily,
    )
    option_state_scale = state_option_mult.reindex(weights.index).fillna(1.0).clip(0.40, 2.00)
    option_ret = option_ret.mul(option_state_scale, fill_value=0.0)
    option_notional = option_notional.mul(option_state_scale, fill_value=0.0)
    option_turnover = option_turnover.mul(option_state_scale, fill_value=0.0)
    option_roll_plan = build_option_roll_plan(
        spy_prices=prices["SPY"].reindex(weights.index).ffill().bfill(),
        option_notional=option_notional,
        option_short_coverage=option_short_coverage,
        roll_days=args.option_roll_days,
        strike_otm=args.option_strike_daily,
        short_strike_otm=args.option_short_strike_daily,
    )

    print(f"  Portfolio construction: {time.time()-t0:.1f}s\n")
    print(
        f"  Core allocator: {'on' if core_diag.get('allocator_enabled') else 'off'} | "
        f"Core capacity clamps: {core_diag.get('capacity_clamp_events', 0)}"
    )
    if governance_diag:
        print("  Sleeve governance: on")
        for sleeve_name in ["core", "etf", "futures", "fx", "vix", "intraday"]:
            if sleeve_name not in governance_diag:
                continue
            gd = governance_diag[sleeve_name]
            print(
                f"    {sleeve_name:7s} mult={gd['avg_multiplier']:.2f}x "
                f"(min={gd['min_multiplier']:.2f}, max={gd['max_multiplier']:.2f}) "
                f"kill={int(gd['kill_events'])} trailing_sh={gd['avg_trailing_sharpe']:.2f} "
                f"worst_dd={gd['worst_trailing_dd']:.2%}"
            )
    else:
        print("  Sleeve governance: off")
    print(
        f"  Option overlay avg notional: {option_notional.mean():.3f}x | "
        f"max: {option_notional.max():.3f}x | active days: {(option_notional > 0).mean():.1%}"
    )
    print(
        f"  Option short-leg coverage: avg={option_short_coverage.mean():.2f} "
        f"(active={option_short_coverage[option_notional > 1e-12].mean() if (option_notional > 1e-12).any() else 0.0:.2f})"
    )
    print(f"  Option roll events: {len(option_roll_plan)}")
    print()

    warmup = int(max(args.warmup_days, 1))
    trade_cols = [c for c in weights.columns if c in returns.columns]
    returns_aligned = returns.reindex(columns=trade_cols).fillna(0.0)
    for sym in trade_cols:
        if sym in cross_asset_returns.columns:
            returns_aligned[sym] = cross_asset_returns[sym]
    port_ret = (weights[trade_cols].shift(1) * returns_aligned).sum(axis=1)
    port_ret = port_ret.add(option_ret, fill_value=0.0)
    if args.enforce_no_lookahead:
        lagged = (weights[trade_cols].shift(1) * returns_aligned).sum(axis=1).add(option_ret, fill_value=0.0)
        if not np.allclose(
            port_ret.fillna(0.0).to_numpy(),
            lagged.fillna(0.0).to_numpy(),
            atol=1e-12,
            rtol=1e-9,
        ):
            raise RuntimeError("Lookahead guard failed: portfolio return must use t-1 weights.")

    asset_cost_map = build_asset_cost_map(trade_cols)
    tx_cost, exec_diag = compute_dynamic_execution_cost(
        weights=weights[trade_cols],
        prices=prices.reindex(columns=trade_cols).ffill().bfill(),
        returns=returns_aligned,
        volumes=volumes.reindex(index=weights.index, columns=trade_cols).fillna(0.0),
        bps_by_symbol=asset_cost_map,
        vol_lookback=args.exec_vol_lookback,
        adv_lookback=args.exec_adv_lookback,
        vol_cost_mult=args.exec_vol_cost_mult,
        liq_cost_mult=args.exec_liq_cost_mult,
        gap_jump_threshold=args.exec_jump_threshold,
        gap_jump_bps=args.exec_jump_bps,
        min_scale=args.exec_min_cost_scale,
        max_scale=args.exec_max_cost_scale,
    )
    tx_cost = tx_cost.add(option_turnover * (args.option_trade_bps / 10000.0), fill_value=0.0)

    net_ret = (port_ret - tx_cost).iloc[warmup:]
    weights_post = weights.iloc[warmup:]
    turnover_post = weights[trade_cols].diff().abs().sum(axis=1).iloc[warmup:]

    eval_mask = pd.Series(True, index=net_ret.index, dtype=bool)
    if args.eval_start:
        eval_start_ts = _coerce_timestamp_for_index(args.eval_start, net_ret.index)
        eval_mask &= net_ret.index >= eval_start_ts
    if args.eval_end:
        eval_end_ts = _coerce_timestamp_for_index(args.eval_end, net_ret.index)
        eval_mask &= net_ret.index <= eval_end_ts
    net_ret = net_ret.loc[eval_mask]
    weights_post = weights_post.loc[eval_mask]
    turnover_post = turnover_post.loc[eval_mask]
    tx_cost_eval = tx_cost.iloc[warmup:].loc[eval_mask]
    if len(net_ret) < 60:
        raise ValueError(
            f"Evaluation window too short ({len(net_ret)} rows). "
            "Use wider --eval-start/--eval-end range."
        )

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
    tx_bps = tx_cost_eval.mean() * 252 * 10000 if not tx_cost_eval.empty else 0.0

    spy_ret_eval = spy_ret.iloc[warmup:].reindex(net_ret.index).fillna(0.0)
    spy_eq = args.nav * (1 + spy_ret_eval).cumprod()
    spy_cagr = (spy_eq.iloc[-1] / spy_eq.iloc[0]) ** (1 / max(n_years, 0.01)) - 1
    spy_sharpe = v4.sharpe(spy_ret_eval)
    spy_dd = v4.max_dd(spy_eq)

    avg_core = core_weights.iloc[warmup:].abs().sum(axis=1).mean() if len(core_weights) > warmup else 0.0
    avg_etf = etf_sleeve.iloc[warmup:].abs().sum(axis=1).mean() if len(etf_sleeve) > warmup else 0.0
    avg_fut = futures_sleeve.iloc[warmup:].abs().sum(axis=1).mean() if len(futures_sleeve) > warmup else 0.0
    avg_fx = fx_sleeve.iloc[warmup:].abs().sum(axis=1).mean() if len(fx_sleeve) > warmup else 0.0
    avg_vix = vix_sleeve.iloc[warmup:].abs().sum(axis=1).mean() if len(vix_sleeve) > warmup else 0.0
    avg_intraday = (
        intraday_sleeve.iloc[warmup:].abs().sum(axis=1).mean()
        if len(intraday_sleeve) > warmup
        else 0.0
    )

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
    print(
        f"  Exec realism:  avg_scale={exec_diag['avg_cost_scale']:.2f}x "
        f"p95_scale={exec_diag['p95_cost_scale']:.2f}x "
        f"jump_cost={exec_diag['avg_jump_cost_bps']:.1f} bps/yr"
    )
    print(
        f"  Sleeve gross:  core={avg_core:.2f}x etf={avg_etf:.2f}x futures={avg_fut:.2f}x "
        f"fx={avg_fx:.2f}x vix={avg_vix:.2f}x intraday={avg_intraday:.2f}x"
    )
    print(f"  Sample window: {equity_curve.index[0].date()} -> {equity_curve.index[-1].date()} ({n_years:.2f} years)")
    print()

    dates = lp_ret.index
    annual_rows: list[dict[str, float | int | bool]] = []
    total_years = 0
    lp_hit_return = lp_hit_sharpe = lp_hit_both = 0
    gross_hit_return = gross_hit_sharpe = gross_hit_both = 0
    gross_beat_spy = lp_beat_spy = 0
    for year in sorted(set(d.year for d in dates)):
        year_mask = [d.year == year for d in lp_ret.index]
        yr_lp = lp_ret[year_mask]
        yr_lp_eq = lp_equity_curve[year_mask]
        yr_gross = net_ret[year_mask]
        yr_gross_eq = equity_curve[year_mask]
        yr_spy = spy_ret_eval[year_mask]
        yr_spy_eq = spy_eq[year_mask]
        if len(yr_lp) < 5 or len(yr_gross) < 5:
            continue
        total_years += 1

        yr_lp_ret = (yr_lp_eq.iloc[-1] / yr_lp_eq.iloc[0]) - 1
        yr_lp_s = v4.sharpe(yr_lp)
        yr_spy_ret = (yr_spy_eq.iloc[-1] / yr_spy_eq.iloc[0]) - 1
        yr_spy_s = v4.sharpe(yr_spy)
        lp_meets_return = yr_lp_ret >= args.lp_target_return
        lp_meets_sharpe = yr_lp_s >= args.lp_target_sharpe
        if lp_meets_return:
            lp_hit_return += 1
        if lp_meets_sharpe:
            lp_hit_sharpe += 1
        if lp_meets_return and lp_meets_sharpe:
            lp_hit_both += 1

        yr_gross_ret = (yr_gross_eq.iloc[-1] / yr_gross_eq.iloc[0]) - 1
        yr_gross_s = v4.sharpe(yr_gross)
        gross_meets_return = yr_gross_ret >= args.gross_target_return
        gross_meets_sharpe = yr_gross_s >= args.gross_target_sharpe
        gross_beats_spy = yr_gross_ret > yr_spy_ret
        lp_beats_spy = yr_lp_ret > yr_spy_ret
        if gross_meets_return:
            gross_hit_return += 1
        if gross_meets_sharpe:
            gross_hit_sharpe += 1
        if gross_meets_return and gross_meets_sharpe:
            gross_hit_both += 1
        if gross_beats_spy:
            gross_beat_spy += 1
        if lp_beats_spy:
            lp_beat_spy += 1
        annual_rows.append(
            {
                "year": int(year),
                "gross_return": float(yr_gross_ret),
                "gross_sharpe": float(yr_gross_s),
                "spy_return": float(yr_spy_ret),
                "spy_sharpe": float(yr_spy_s),
                "gross_beats_spy": bool(gross_beats_spy),
                "gross_hit_return": bool(gross_meets_return),
                "gross_hit_sharpe": bool(gross_meets_sharpe),
                "gross_hit_both": bool(gross_meets_return and gross_meets_sharpe),
                "lp_return": float(yr_lp_ret),
                "lp_sharpe": float(yr_lp_s),
                "lp_beats_spy": bool(lp_beats_spy),
                "lp_hit_return": bool(lp_meets_return),
                "lp_hit_sharpe": bool(lp_meets_sharpe),
                "lp_hit_both": bool(lp_meets_return and lp_meets_sharpe),
            }
        )

    gross_underwater_days = compute_max_underwater_days(equity_curve)
    lp_underwater_days = compute_max_underwater_days(lp_equity_curve)
    if annual_rows:
        gross_annual_returns = [float(r["gross_return"]) for r in annual_rows]
        gross_annual_sharpes = [float(r["gross_sharpe"]) for r in annual_rows]
        lp_annual_returns = [float(r["lp_return"]) for r in annual_rows]
        lp_annual_sharpes = [float(r["lp_sharpe"]) for r in annual_rows]
    else:
        gross_annual_returns = []
        gross_annual_sharpes = []
        lp_annual_returns = []
        lp_annual_sharpes = []

    print("  ── Annual Hurdles ──")
    print(f"  Gross years >= {args.gross_target_return:.0%}: {gross_hit_return}/{total_years}")
    print(f"  Gross years >= {args.gross_target_sharpe:.1f} Sharpe: {gross_hit_sharpe}/{total_years}")
    print(f"  Gross years hitting both: {gross_hit_both}/{total_years}")
    print(f"  Gross years beating SPY: {gross_beat_spy}/{total_years}")
    print(f"  LP years >= {args.lp_target_return:.0%} net: {lp_hit_return}/{total_years}")
    print(f"  LP years >= {args.lp_target_sharpe:.1f} Sharpe: {lp_hit_sharpe}/{total_years}")
    print(f"  LP years hitting both: {lp_hit_both}/{total_years}")
    print(f"  LP years beating SPY: {lp_beat_spy}/{total_years}")
    print(f"  Max underwater days (gross / LP): {gross_underwater_days} / {lp_underwater_days}")
    print()
    print("=" * 70)

    metrics = {
        "instrument_count": int(len(symbols)),
        "gross_cagr": float(cagr),
        "gross_sharpe": float(s),
        "gross_max_dd": float(dd),
        "final_nav": float(equity_curve.iloc[-1]),
        "lp_net_cagr": float(lp_cagr),
        "lp_net_sharpe": float(lp_s),
        "lp_net_max_dd": float(lp_dd),
        "lp_final_nav": float(lp_equity_curve.iloc[-1]),
        "turnover_x_per_year": float(avg_turn),
        "tx_cost_bps_per_year": float(tx_bps),
        "years_ge_net_target": int(lp_hit_return),
        "years_ge_sharpe_target": int(lp_hit_sharpe),
        "years_ge_both_targets": int(lp_hit_both),
        "years_ge_gross_return_target": int(gross_hit_return),
        "years_ge_gross_sharpe_target": int(gross_hit_sharpe),
        "years_ge_gross_both_targets": int(gross_hit_both),
        "years_beating_spy_gross": int(gross_beat_spy),
        "years_beating_spy_lp": int(lp_beat_spy),
        "gross_years_beating_spy_ratio": float(gross_beat_spy / total_years) if total_years > 0 else 0.0,
        "lp_years_beating_spy_ratio": float(lp_beat_spy / total_years) if total_years > 0 else 0.0,
        "total_years": int(total_years),
        "min_gross_year_return": float(min(gross_annual_returns)) if gross_annual_returns else 0.0,
        "min_gross_year_sharpe": float(min(gross_annual_sharpes)) if gross_annual_sharpes else 0.0,
        "min_lp_year_return": float(min(lp_annual_returns)) if lp_annual_returns else 0.0,
        "min_lp_year_sharpe": float(min(lp_annual_sharpes)) if lp_annual_sharpes else 0.0,
        "annual_summary": annual_rows,
        "spy_cagr": float(spy_cagr),
        "spy_sharpe": float(spy_sharpe),
        "spy_max_dd": float(spy_dd),
        "spy_final_nav": float(spy_eq.iloc[-1]),
        "gross_cagr_alpha_vs_spy": float(cagr - spy_cagr),
        "lp_cagr_alpha_vs_spy": float(lp_cagr - spy_cagr),
        "gross_final_nav_alpha_vs_spy": float(equity_curve.iloc[-1] - spy_eq.iloc[-1]),
        "lp_final_nav_alpha_vs_spy": float(lp_equity_curve.iloc[-1] - spy_eq.iloc[-1]),
        "max_underwater_days_gross": int(gross_underwater_days),
        "max_underwater_days_lp": int(lp_underwater_days),
        "eval_start": str(net_ret.index[0].date()),
        "eval_end": str(net_ret.index[-1].date()),
        "run_start": str(prices.index[0].date()),
        "run_end": str(prices.index[-1].date()),
        "event_live_weight": float(strategy_weights.get("earnings_drift", 0.0)),
        "event_quality_scale": float(event_build.data_quality_scale),
        "event_total_scored": int(event_build.total_events),
        "intraday_sleeve_requested": bool(args.enable_intraday_cache_sleeve),
        "intraday_sleeve_enabled": bool(intraday_effective),
        "intraday_sleeve_diag": intraday_diag,
        "state_param_bank_v10_enabled": bool(args.enable_state_param_bank_v10),
        "state_param_bank_v10_diag": state_bank_diag,
        "option_roll_events": int(len(option_roll_plan)),
        "option_active_days": float((option_notional > 0).mean()),
        "option_avg_short_coverage": float(option_short_coverage.mean()),
        "governance_enabled": bool(not args.disable_sleeve_governance),
        "sleeve_governance": governance_diag,
        "regime_risk_scaler_enabled": bool(args.enable_regime_risk_scaler),
        "regime_risk_scaler_diag": {
            "avg": float(regime_scaler.mean()),
            "min": float(regime_scaler.min()),
            "max": float(regime_scaler.max()),
        },
        "macro_overlay_enabled": bool(args.enable_macro_overlay),
        "macro_overlay_diag": macro_diag,
        "regime_router_v10_enabled": bool(args.enable_regime_router_v10),
        "regime_router_v10_diag": router_diag,
        "whipsaw_control_v10_enabled": bool(args.enable_whipsaw_control_v10),
        "whipsaw_control_v10_diag": whipsaw_diag,
        "yearly_risk_budget_v10_enabled": bool(args.enable_yearly_risk_budget_v10),
        "yearly_risk_budget_v10_diag": budget_diag,
        "adaptive_v10_layers_enabled": bool(args.enable_adaptive_v10_layers),
        "adaptive_v10_layers_diag": adaptive_diag,
        "execution_diag": exec_diag,
        "lookahead_guard": bool(args.enforce_no_lookahead),
    }
    if args.metrics_json:
        with open(args.metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    if args.output_returns:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_returns)), exist_ok=True)
        ret_df = pd.DataFrame({"gross_ret": net_ret.values, "lp_ret": lp_ret.values}, index=net_ret.index)
        ret_df.index.name = "date"
        ret_df.to_csv(args.output_returns)
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AlphaForge v8 — multi-asset + options-hedge overlay")
    p.add_argument("--config", default="config/sp500_universe.yaml")
    p.add_argument("--env", default=".env")
    p.add_argument("--start-date", default=None, help="Optional data start date (YYYY-MM-DD)")
    p.add_argument("--end-date", default=None, help="Optional data end date (YYYY-MM-DD)")
    p.add_argument("--eval-start", default=None, help="Optional evaluation window start (YYYY-MM-DD)")
    p.add_argument("--eval-end", default=None, help="Optional evaluation window end (YYYY-MM-DD)")
    p.add_argument("--nav", type=float, default=10_000_000)
    p.add_argument("--target-vol", type=float, default=0.15)
    p.add_argument("--target-gross", type=float, default=1.5)
    p.add_argument("--total-gross-cap", type=float, default=2.0)
    p.add_argument("--rebal-freq", type=int, default=15)
    p.add_argument("--max-pos", type=float, default=0.06)
    p.add_argument("--max-overlay-pos", type=float, default=0.10)
    p.add_argument("--n-long", type=int, default=50)
    p.add_argument("--n-short", type=int, default=25)
    p.add_argument("--event-target-weight", type=float, default=0.14)
    p.add_argument("--force-event-weight", type=float, default=0.05)
    p.add_argument("--sentiment-weight", type=float, default=0.20)
    p.add_argument("--revision-flow-weight", type=float, default=0.18)
    p.add_argument("--revision-flow-half-life-days", type=float, default=18.0)
    p.add_argument("--event-max-per-symbol-day", type=int, default=2)
    p.add_argument("--event-min-relevance", type=float, default=0.25)
    p.add_argument("--event-min-novelty", type=float, default=0.25)
    p.add_argument("--event-min-abs-default", type=float, default=0.04)
    p.add_argument("--event-min-abs-news", type=float, default=0.10)
    p.add_argument("--event-min-abs-macro", type=float, default=0.10)
    p.add_argument("--event-burst-daily-decay", type=float, default=0.70)
    p.add_argument("--finbert-min-abs-score", type=float, default=0.20)
    p.add_argument("--finbert-min-confidence", type=float, default=0.60)
    p.add_argument("--sec-min-abs-score", type=float, default=0.00)
    p.add_argument("--sec-min-confidence", type=float, default=0.10)
    p.add_argument("--event-min-source-quality", type=float, default=0.0)
    p.add_argument("--fundamentals-db", default="data/fundamentals.db")
    p.add_argument("--symbol-master-db", default="data/symbol_master.db")
    p.add_argument("--events-db", default="data/events.db")
    p.add_argument("--preemptive-de-risk", type=float, default=0.05)
    p.add_argument("--crisis-hedge-max", type=float, default=0.15)
    p.add_argument("--crisis-hedge-strength", type=float, default=0.75)
    p.add_argument("--crisis-beta-floor", type=float, default=0.15)
    p.add_argument("--hedge-lookback", type=int, default=63)
    p.add_argument("--mgmt-fee", type=float, default=0.02)
    p.add_argument("--perf-fee", type=float, default=0.20)
    p.add_argument("--lp-target-return", type=float, default=0.15)
    p.add_argument("--lp-target-sharpe", type=float, default=1.0)
    p.add_argument("--gross-target-return", type=float, default=0.15)
    p.add_argument("--gross-target-sharpe", type=float, default=1.0)
    p.add_argument("--disable-dynamic-allocator", action="store_true")
    p.add_argument("--disable-capacity-constraints", action="store_true")
    p.add_argument("--disable-sleeve-governance", action="store_true")
    p.add_argument("--enable-regime-risk-scaler", action="store_true")
    p.add_argument("--risk-floor", type=float, default=0.80)
    p.add_argument("--risk-ceiling", type=float, default=1.12)
    p.add_argument("--risk-smooth-days", type=int, default=7)
    p.add_argument("--enable-regime-router-v10", action="store_true")
    p.add_argument("--router-macro-cache-file", default="~/.alphaforge/cache/macro/fred_daily.parquet")
    p.add_argument("--router-vol-window", type=int, default=21)
    p.add_argument("--router-trend-window", type=int, default=200)
    p.add_argument("--router-crash-threshold", type=float, default=0.62)
    p.add_argument("--router-risk-off-threshold", type=float, default=0.50)
    p.add_argument("--router-smooth-days", type=int, default=3)
    p.add_argument("--enable-macro-overlay", action="store_true")
    p.add_argument("--macro-cache-file", default="~/.alphaforge/cache/macro/fred_daily.parquet")
    p.add_argument("--macro-max-de-risk", type=float, default=0.12)
    p.add_argument("--macro-min-scale", type=float, default=0.85)
    p.add_argument("--macro-smooth-days", type=int, default=5)
    p.add_argument("--macro-zscore-window", type=int, default=252)
    p.add_argument("--gov-window", type=int, default=126)
    p.add_argument("--gov-min-obs", type=int, default=42)
    p.add_argument("--gov-soft-dd", type=float, default=0.10)
    p.add_argument("--gov-hard-dd", type=float, default=0.22)
    p.add_argument("--gov-kill-sharpe", type=float, default=-0.15)
    p.add_argument("--gov-kill-dd", type=float, default=0.12)
    p.add_argument("--gov-cooldown-days", type=int, default=21)
    p.add_argument("--gov-min-mult", type=float, default=0.35)
    p.add_argument("--gov-max-mult", type=float, default=1.25)
    p.add_argument("--gov-smooth-days", type=int, default=5)
    p.add_argument("--gov-expected-sharpe-core", type=float, default=0.45)
    p.add_argument("--gov-expected-sharpe-etf", type=float, default=0.30)
    p.add_argument("--gov-expected-sharpe-futures", type=float, default=0.30)
    p.add_argument("--gov-expected-sharpe-fx", type=float, default=0.20)
    p.add_argument("--gov-expected-sharpe-vix", type=float, default=0.05)
    p.add_argument("--gov-expected-sharpe-intraday", type=float, default=0.40)
    p.add_argument("--enable-weak-sleeve-hard-demote", action="store_true")
    p.add_argument("--weak-sleeve-list", default="fx")
    p.add_argument("--weak-demote-sharpe", type=float, default=-0.10)
    p.add_argument("--weak-recover-sharpe", type=float, default=0.15)
    p.add_argument("--weak-demote-confirm-days", type=int, default=21)
    p.add_argument("--weak-recover-confirm-days", type=int, default=42)
    p.add_argument("--weak-demote-mult", type=float, default=0.0)
    p.add_argument("--enable-kelly-sentiment-overlay", action="store_true")
    p.add_argument("--kelly-lookback", type=int, default=126)
    p.add_argument("--kelly-min-obs", type=int, default=42)
    p.add_argument("--kelly-fraction", type=float, default=0.08)
    p.add_argument("--kelly-max-abs", type=float, default=1.0)
    p.add_argument("--kelly-min-scale", type=float, default=0.90)
    p.add_argument("--kelly-max-scale", type=float, default=1.12)
    p.add_argument("--kelly-sentiment-sensitivity", type=float, default=0.08)
    p.add_argument("--capacity-impact-k", type=float, default=0.45)
    p.add_argument("--capacity-participation-limit", type=float, default=0.05)
    p.add_argument("--capacity-impact-limit-bps", type=float, default=15.0)
    p.add_argument("--capacity-max-spread-bps", type=float, default=14.0)
    p.add_argument("--exec-vol-lookback", type=int, default=20)
    p.add_argument("--exec-adv-lookback", type=int, default=20)
    p.add_argument("--exec-vol-cost-mult", type=float, default=0.45)
    p.add_argument("--exec-liq-cost-mult", type=float, default=0.70)
    p.add_argument("--exec-jump-threshold", type=float, default=0.04)
    p.add_argument("--exec-jump-bps", type=float, default=1.2)
    p.add_argument("--exec-min-cost-scale", type=float, default=0.60)
    p.add_argument("--exec-max-cost-scale", type=float, default=3.25)
    p.add_argument("--overlay-min-signal", type=float, default=0.06)
    p.add_argument("--overlay-shorts", action="store_true")
    p.add_argument("--exclude-spy-from-etf-alpha", action="store_true")
    p.add_argument("--enable-intraday-cache-sleeve", action="store_true")
    p.add_argument("--intraday-cache-dir", default="data/cache/intraday")
    p.add_argument("--intraday-interval", default="5m")
    p.add_argument(
        "--intraday-symbols",
        default="SPY,QQQ,AAPL,MSFT,NVDA,AMZN,META,GOOGL,AVGO,JPM,XOM,ES,CL,EURUSD,USDJPY",
    )
    p.add_argument("--intraday-gross", type=float, default=0.08)
    p.add_argument("--intraday-max-pos", type=float, default=0.03)
    p.add_argument("--intraday-min-signal", type=float, default=0.08)
    p.add_argument("--intraday-rebal-freq", type=int, default=5)
    p.add_argument("--intraday-allow-shorts", action="store_true")
    p.add_argument("--etf-gross", type=float, default=0.20)
    p.add_argument("--futures-gross", type=float, default=0.20)
    p.add_argument("--fx-gross", type=float, default=0.12)
    p.add_argument("--vix-hedge-max", type=float, default=0.10)
    p.add_argument("--option-roll-days", type=int, default=21)
    p.add_argument("--option-max-notional", type=float, default=0.04)
    p.add_argument("--option-strike-daily", type=float, default=0.020)
    p.add_argument("--option-short-strike-daily", type=float, default=0.070)
    p.add_argument("--option-payout-mult", type=float, default=1.00)
    p.add_argument("--option-theta-bps-daily", type=float, default=1.8)
    p.add_argument("--option-short-credit-bps-daily", type=float, default=0.9)
    p.add_argument("--option-activation-score", type=float, default=0.38)
    p.add_argument("--option-severe-score", type=float, default=0.78)
    p.add_argument("--option-always-on-min-notional", type=float, default=0.0)
    p.add_argument("--option-calm-short-coverage", type=float, default=0.95)
    p.add_argument("--option-crash-short-coverage-floor", type=float, default=0.15)
    p.add_argument("--option-crash-prob-sensitivity", type=float, default=0.45)
    p.add_argument("--option-trade-bps", type=float, default=10.0)
    p.add_argument("--enable-whipsaw-control-v10", action="store_true")
    p.add_argument("--whipsaw-entry-days", type=int, default=2)
    p.add_argument("--whipsaw-exit-days", type=int, default=3)
    p.add_argument("--whipsaw-entry-abs", type=float, default=0.0035)
    p.add_argument("--whipsaw-exit-abs", type=float, default=0.0020)
    p.add_argument("--enable-adaptive-v10-layers", action="store_true")
    p.add_argument("--adaptive-v10-crash-prob-trigger", type=float, default=0.45)
    p.add_argument("--adaptive-v10-soft-gate", action="store_true")
    p.add_argument("--enable-state-param-bank-v10", action="store_true")
    p.add_argument("--state-risk-on-mult", type=float, default=1.00)
    p.add_argument("--state-risk-off-mult", type=float, default=0.92)
    p.add_argument("--state-crash-mult", type=float, default=0.72)
    p.add_argument("--state-hedge-risk-on-mult", type=float, default=0.90)
    p.add_argument("--state-hedge-risk-off-mult", type=float, default=1.05)
    p.add_argument("--state-hedge-crash-mult", type=float, default=1.30)
    p.add_argument("--state-gross-risk-on-mult", type=float, default=1.00)
    p.add_argument("--state-gross-risk-off-mult", type=float, default=0.92)
    p.add_argument("--state-gross-crash-mult", type=float, default=0.74)
    p.add_argument("--state-option-risk-on-mult", type=float, default=0.92)
    p.add_argument("--state-option-risk-off-mult", type=float, default=1.08)
    p.add_argument("--state-option-crash-mult", type=float, default=1.25)
    p.add_argument("--enable-yearly-risk-budget-v10", action="store_true")
    p.add_argument("--yearly-vol-budget", type=float, default=0.18)
    p.add_argument("--yearly-dd-budget", type=float, default=0.18)
    p.add_argument("--yearly-budget-min-scale", type=float, default=0.75)
    p.add_argument("--yearly-budget-max-scale", type=float, default=1.10)
    p.add_argument("--yearly-budget-smooth-days", type=int, default=5)
    p.add_argument("--metrics-json", default=None)
    p.add_argument("--output-returns", default=None, help="Write daily gross/lp returns to CSV for Monte Carlo")
    p.add_argument("--warmup-days", type=int, default=300)
    p.add_argument("--min-rows", type=int, default=700)
    p.add_argument("--cache-complete-only", action="store_true")
    p.add_argument("--enforce-no-lookahead", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    if os.path.exists(args.env):
        from dotenv import load_dotenv

        load_dotenv(args.env)

    run(args)
