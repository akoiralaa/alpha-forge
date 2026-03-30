#!/usr/bin/env python3

import argparse
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

try:
    from ib_async import IB, Stock, Forex, Future
except ImportError:
    from ib_insync import IB, Stock, Forex, Future

from src.paper.engine import PaperConfig, PaperTick, PaperTradingEngine
from src.portfolio.sizing import compute_position_size


ASSET_TYPES = {}

# ── signal weights per asset class ────────────────────────────
# (trend, momentum, mean_reversion, breakout)
ASSET_CLASS_WEIGHTS = {
    "ETF":        (0.30, 0.35, 0.20, 0.15),
    "FUTURE":     (0.35, 0.30, 0.15, 0.20),
    "COMMODITY":  (0.40, 0.30, 0.10, 0.20),
    "BOND":       (0.35, 0.35, 0.20, 0.10),
    "FX":         (0.30, 0.25, 0.30, 0.15),
    "VOLATILITY": (0.25, 0.30, 0.25, 0.20),
}


def load_universe(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    instruments = cfg.get("instruments", {})
    universe = []

    for sym in instruments.get("sector_etfs", []):
        universe.append((sym, "ETF"))
        ASSET_TYPES[sym] = "ETF"
    for sym in instruments.get("equity_index_futures", []):
        universe.append((sym, "FUTURE"))
        ASSET_TYPES[sym] = "FUTURE"
    for sym in instruments.get("commodity_futures", []):
        universe.append((sym, "COMMODITY"))
        ASSET_TYPES[sym] = "COMMODITY"
    for sym in instruments.get("fixed_income_futures", []):
        universe.append((sym, "BOND"))
        ASSET_TYPES[sym] = "BOND"
    for sym in instruments.get("fx_pairs", []):
        universe.append((sym, "FX"))
        ASSET_TYPES[sym] = "FX"
    for sym in instruments.get("vix_futures", []):
        universe.append((sym, "VOLATILITY"))
        ASSET_TYPES[sym] = "VOLATILITY"

    return universe


EXCHANGE_MAP = {
    "CL": "NYMEX", "NG": "NYMEX", "HG": "COMEX",
    "GC": "COMEX", "SI": "COMEX",
    "ZC": "CBOT", "ZW": "CBOT", "ZS": "CBOT",
    "ZN": "CBOT", "ZB": "CBOT", "ZF": "CBOT", "ZT": "CBOT", "GE": "CME",
    "ES": "CME", "NQ": "CME", "RTY": "CME", "YM": "CBOT",
    "VX": "CFE",
}


def make_contract(symbol, asset_type):
    if asset_type in ("ETF", "EQUITY"):
        return Stock(symbol, "SMART", "USD")
    elif asset_type == "FX":
        return Forex(symbol[:3] + symbol[3:])
    elif asset_type in ("FUTURE", "COMMODITY", "BOND", "VOLATILITY"):
        exchange = EXCHANGE_MAP.get(symbol, "CME")
        return Future(symbol, exchange=exchange, includeExpired=True)
    return Stock(symbol, "SMART", "USD")


def qualify_future(ib, symbol, asset_type):
    exchange = EXCHANGE_MAP.get(symbol, "CME")
    try:
        from ib_async import ContFuture
    except ImportError:
        from ib_insync import ContFuture

    contract = ContFuture(symbol, exchange=exchange)
    try:
        qualified = ib.qualifyContracts(contract)
        if qualified:
            return qualified[0]
    except Exception:
        pass

    contract = Future(symbol, exchange=exchange)
    try:
        qualified = ib.qualifyContracts(contract)
        if qualified:
            return qualified[0]
    except Exception:
        pass

    return None


def fetch_max_daily(ib, symbol, asset_type):
    if asset_type in ("FUTURE", "COMMODITY", "BOND", "VOLATILITY"):
        contract = qualify_future(ib, symbol, asset_type)
        if contract is None:
            return pd.DataFrame()
    else:
        contract = make_contract(symbol, asset_type)
        try:
            qualified = ib.qualifyContracts(contract)
            if not qualified:
                return pd.DataFrame()
            contract = qualified[0]
        except Exception as e:
            print(f"    skip {symbol}: {e}")
            return pd.DataFrame()

    what = "MIDPOINT" if asset_type == "FX" else "TRADES"

    durations = ["20 Y", "10 Y", "5 Y", "2 Y", "1 Y"]
    if asset_type not in ("ETF", "EQUITY"):
        durations = ["10 Y", "5 Y", "2 Y", "1 Y"]

    bars = None
    for dur in durations:
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=dur,
                barSizeSetting="1 day",
                whatToShow=what,
                useRTH=True,
                formatDate=2,
            )
            time.sleep(0.5)
            if bars:
                break
        except Exception:
            time.sleep(0.5)
            continue

    if not bars:
        return pd.DataFrame()

    records = []
    for bar in bars:
        dt = bar.date if isinstance(bar.date, datetime) else datetime.fromisoformat(str(bar.date))
        if hasattr(dt, 'tzinfo') and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        records.append({
            "date": dt,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": int(bar.volume) if bar.volume > 0 else 100,
        })

    return pd.DataFrame(records)


# ── Feature engineering ───────────────────────────────────────

def compute_features(df):
    """Build multi-factor feature matrix from daily OHLCV bars."""
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    lo = df["low"].values.astype(float)
    v = df["volume"].values.astype(float)
    n = len(c)

    feat = pd.DataFrame(index=df.index)

    # returns at multiple horizons
    for w in [1, 5, 10, 21, 63, 126, 252]:
        r = np.full(n, np.nan)
        r[w:] = c[w:] / c[:-w] - 1
        feat[f"ret_{w}d"] = r

    # simple moving averages
    for w in [10, 20, 50, 100, 200]:
        sma = np.full(n, np.nan)
        for i in range(w - 1, n):
            sma[i] = np.mean(c[i - w + 1:i + 1])
        feat[f"sma_{w}"] = sma
        feat[f"price_vs_sma_{w}"] = c / sma - 1

    # exponential moving averages
    for w in [12, 26, 50]:
        ema = np.full(n, np.nan)
        alpha = 2.0 / (w + 1)
        ema[w - 1] = np.mean(c[:w])
        for i in range(w, n):
            ema[i] = alpha * c[i] + (1 - alpha) * ema[i - 1]
        feat[f"ema_{w}"] = ema

    # MACD (12/26/9)
    ema12 = feat["ema_12"].values
    ema26 = feat["ema_26"].values
    macd_line = ema12 - ema26
    macd_signal = np.full(n, np.nan)
    alpha9 = 2.0 / 10
    start = np.where(~np.isnan(macd_line))[0]
    if len(start) > 9:
        s = start[0]
        macd_signal[s + 8] = np.mean(macd_line[s:s + 9])
        for i in range(s + 9, n):
            macd_signal[i] = alpha9 * macd_line[i] + (1 - alpha9) * macd_signal[i - 1]
    feat["macd"] = macd_line
    feat["macd_signal"] = macd_signal
    feat["macd_hist"] = macd_line - macd_signal

    # realized volatility
    log_ret = np.full(n, np.nan)
    log_ret[1:] = np.log(c[1:] / c[:-1])
    for w in [10, 20, 60]:
        rv = np.full(n, np.nan)
        for i in range(w, n):
            rv[i] = np.std(log_ret[i - w + 1:i + 1]) * np.sqrt(252)
        feat[f"rvol_{w}d"] = rv

    # vol ratio (short vs long)
    feat["vol_ratio"] = feat["rvol_20d"] / feat["rvol_60d"]

    # z-scores of returns
    for w in [20, 50, 100]:
        z = np.full(n, np.nan)
        for i in range(w, n):
            window = log_ret[i - w + 1:i + 1]
            window = window[~np.isnan(window)]
            if len(window) > 5 and np.std(window) > 1e-10:
                z[i] = (log_ret[i] - np.mean(window)) / np.std(window)
        feat[f"zscore_{w}"] = z

    # RSI (14-period)
    rsi = np.full(n, np.nan)
    delta = np.diff(c, prepend=c[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    if n > 14:
        avg_gain[14] = np.mean(gain[1:15])
        avg_loss[14] = np.mean(loss[1:15])
        for i in range(15, n):
            avg_gain[i] = (avg_gain[i - 1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i - 1] * 13 + loss[i]) / 14
        for i in range(14, n):
            if avg_loss[i] > 0:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - 100 / (1 + rs)
            else:
                rsi[i] = 100
    feat["rsi_14"] = rsi

    # Bollinger band position: (price - lower) / (upper - lower)
    bb_w = 20
    bb_mid = feat["sma_20"].values
    bb_std = np.full(n, np.nan)
    for i in range(bb_w - 1, n):
        bb_std[i] = np.std(c[i - bb_w + 1:i + 1])
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    feat["bb_position"] = np.where(bb_width > 1e-10, (c - bb_lower) / bb_width, 0.5)

    # channel breakout: price vs N-day high/low
    for w in [20, 50]:
        hi = np.full(n, np.nan)
        lo_arr = np.full(n, np.nan)
        for i in range(w - 1, n):
            hi[i] = np.max(h[i - w + 1:i + 1])
            lo_arr[i] = np.min(lo[i - w + 1:i + 1])
        rng = hi - lo_arr
        feat[f"channel_{w}"] = np.where(rng > 1e-10, (c - lo_arr) / rng, 0.5)

    # volume moving average ratio
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(v[i - 19:i + 1])
    feat["vol_surge"] = np.where(vol_ma > 0, v / vol_ma, 1.0)

    # average true range (normalized)
    tr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1]))
    atr14 = np.full(n, np.nan)
    if n > 14:
        atr14[14] = np.mean(tr[1:15])
        for i in range(15, n):
            atr14[i] = (atr14[i - 1] * 13 + tr[i]) / 14
    feat["atr_pct"] = atr14 / c  # ATR as % of price

    return feat


# ── Signal computation ────────────────────────────────────────

def signal_trend(feat, i):
    """Trend-following: SMA crossovers + price vs long-term MA."""
    sma50 = feat["sma_50"].iloc[i]
    sma200 = feat["sma_200"].iloc[i]
    price_vs_50 = feat["price_vs_sma_50"].iloc[i]
    price_vs_200 = feat["price_vs_sma_200"].iloc[i]

    if any(np.isnan(x) for x in [sma50, sma200, price_vs_50, price_vs_200]):
        return 0.0

    # golden/death cross
    cross_signal = 0.0
    if sma50 > sma200:
        cross_signal = 0.4
    elif sma50 < sma200:
        cross_signal = -0.4

    # price distance from 200 SMA
    trend_strength = np.clip(price_vs_200 * 3.0, -0.6, 0.6)

    return np.clip(cross_signal + trend_strength, -1.0, 1.0)


def signal_momentum(feat, i):
    """Time-series momentum: 12-1 month with vol scaling (Moskowitz et al.)."""
    ret_252 = feat["ret_252d"].iloc[i] if not np.isnan(feat["ret_252d"].iloc[i]) else None
    ret_21 = feat["ret_21d"].iloc[i] if not np.isnan(feat["ret_21d"].iloc[i]) else None
    ret_126 = feat["ret_126d"].iloc[i] if not np.isnan(feat["ret_126d"].iloc[i]) else None
    rvol = feat["rvol_60d"].iloc[i] if not np.isnan(feat["rvol_60d"].iloc[i]) else None

    if ret_252 is None or ret_21 is None or rvol is None or rvol < 0.01:
        return 0.0

    # 12-1 month momentum: total 12m return minus last month (skip recent reversal)
    mom_12_1 = ret_252 - ret_21

    # 6-1 month momentum (shorter horizon, catches faster trends)
    mom_6_1 = (ret_126 - ret_21) if ret_126 is not None else 0.0

    # vol-adjusted: divide by realized vol to normalize across assets
    adj_12_1 = mom_12_1 / rvol
    adj_6_1 = mom_6_1 / rvol

    # blend: 60% 12-1m, 40% 6-1m
    raw = 0.6 * adj_12_1 + 0.4 * adj_6_1

    # smooth to [-1, 1] using tanh
    return float(np.tanh(raw * 0.5))


def signal_mean_reversion(feat, i):
    """Mean-reversion with trend filter: only trade reversals in range-bound markets."""
    z20 = feat["zscore_20"].iloc[i]
    rsi = feat["rsi_14"].iloc[i]
    bb = feat["bb_position"].iloc[i]
    price_vs_200 = feat["price_vs_sma_200"].iloc[i]

    if any(np.isnan(x) for x in [z20, rsi, bb, price_vs_200]):
        return 0.0

    # trend filter: don't mean-revert if price is far from 200 SMA (strong trend)
    trend_strength = abs(price_vs_200)
    if trend_strength > 0.15:
        return 0.0  # market is trending, stay out of mean-reversion

    # z-score reversal
    z_signal = np.clip(-z20 * 0.2, -0.5, 0.5)

    # RSI reversal: oversold = buy, overbought = sell
    rsi_signal = 0.0
    if rsi < 30:
        rsi_signal = (30 - rsi) / 30  # stronger as more oversold
    elif rsi > 70:
        rsi_signal = -(rsi - 70) / 30

    # Bollinger band reversal
    bb_signal = 0.0
    if bb < 0.1:
        bb_signal = 0.3
    elif bb > 0.9:
        bb_signal = -0.3

    # combine
    raw = 0.4 * z_signal + 0.35 * rsi_signal + 0.25 * bb_signal
    return float(np.clip(raw, -1.0, 1.0))


def signal_breakout(feat, i):
    """Channel breakout: Donchian-style with volume confirmation."""
    ch20 = feat["channel_20"].iloc[i]
    ch50 = feat["channel_50"].iloc[i]
    vol_surge = feat["vol_surge"].iloc[i]
    macd_hist = feat["macd_hist"].iloc[i]

    if any(np.isnan(x) for x in [ch20, ch50, vol_surge, macd_hist]):
        return 0.0

    # channel position: 0=at low, 1=at high
    # breakout above 0.9 or below 0.1
    breakout_signal = 0.0
    if ch20 > 0.9 and ch50 > 0.7:
        breakout_signal = 0.5
    elif ch20 < 0.1 and ch50 < 0.3:
        breakout_signal = -0.5

    # volume confirmation: breakout on high volume is more reliable
    vol_mult = min(vol_surge / 1.5, 2.0) if vol_surge > 1.0 else 0.5

    # MACD confirmation
    macd_confirm = 1.0 if np.sign(macd_hist) == np.sign(breakout_signal) else 0.5

    return float(np.clip(breakout_signal * vol_mult * macd_confirm, -1.0, 1.0))


def compute_composite_signal(feat, i, asset_type):
    """Combine all four sub-signals with asset-class-specific weights."""
    w_trend, w_mom, w_mr, w_bo = ASSET_CLASS_WEIGHTS.get(asset_type, (0.30, 0.30, 0.20, 0.20))

    s_trend = signal_trend(feat, i)
    s_mom = signal_momentum(feat, i)
    s_mr = signal_mean_reversion(feat, i)
    s_bo = signal_breakout(feat, i)

    # regime adaptation: when volatility is high, boost momentum and reduce mean-reversion
    vol_ratio = feat["vol_ratio"].iloc[i]
    if not np.isnan(vol_ratio):
        if vol_ratio > 1.3:
            # high vol regime: more trend, less mean-reversion
            w_trend *= 1.3
            w_mom *= 1.2
            w_mr *= 0.4
            w_bo *= 1.1
        elif vol_ratio < 0.7:
            # low vol regime: more mean-reversion, less trend
            w_trend *= 0.7
            w_mom *= 0.8
            w_mr *= 1.5
            w_bo *= 0.8

    # renormalize weights
    total_w = w_trend + w_mom + w_mr + w_bo
    if total_w > 0:
        w_trend /= total_w
        w_mom /= total_w
        w_mr /= total_w
        w_bo /= total_w

    composite = w_trend * s_trend + w_mom * s_mom + w_mr * s_mr + w_bo * s_bo

    # conviction scaling: stronger when multiple signals agree
    signals = [s_trend, s_mom, s_mr, s_bo]
    agreeing = sum(1 for s in signals if np.sign(s) == np.sign(composite) and abs(s) > 0.05)
    if agreeing >= 3:
        composite *= 1.3  # boost when consensus
    elif agreeing <= 1:
        composite *= 0.5  # dampen when conflicting

    return float(np.clip(composite, -1.0, 1.0))


def compute_environment_score(feat, i):
    """
    Predict market environment quality: benign vs volatile.
    Returns 0.0 (dangerous, stay out) to 1.0 (benign, full size).
    This is the "second derivative" approach — trade the environment, not direction.
    """
    rvol_20 = feat["rvol_20d"].iloc[i] if i < len(feat) else np.nan
    rvol_60 = feat["rvol_60d"].iloc[i] if i < len(feat) else np.nan
    vol_ratio = feat["vol_ratio"].iloc[i] if i < len(feat) else np.nan
    atr_pct = feat["atr_pct"].iloc[i] if i < len(feat) else np.nan
    rsi = feat["rsi_14"].iloc[i] if i < len(feat) else np.nan

    if any(np.isnan(x) for x in [rvol_20, rvol_60, vol_ratio]):
        return 0.3  # uncertain = cautious

    score = 1.0

    # vol regime: high absolute vol = dangerous
    # typical equity vol: 15-20%. above 30% = crisis territory
    if rvol_20 > 0.40:
        score *= 0.15  # crisis, almost fully out
    elif rvol_20 > 0.30:
        score *= 0.30  # high vol
    elif rvol_20 > 0.25:
        score *= 0.50  # elevated
    elif rvol_20 > 0.20:
        score *= 0.75  # above average
    # below 20% = normal/benign, keep score at 1.0

    # vol expansion: rising vol = incoming trouble
    # vol_ratio > 1 means short-term vol > long-term vol (expanding)
    if not np.isnan(vol_ratio):
        if vol_ratio > 1.5:
            score *= 0.3   # vol exploding
        elif vol_ratio > 1.2:
            score *= 0.5   # vol expanding
        elif vol_ratio < 0.7:
            score *= 1.2   # vol compressing = benign
        elif vol_ratio < 0.8:
            score *= 1.1

    # vol of vol: look at recent ATR volatility
    # high ATR% means unstable markets
    if not np.isnan(atr_pct):
        if atr_pct > 0.03:
            score *= 0.4
        elif atr_pct > 0.02:
            score *= 0.6

    # extreme RSI = unstable environment
    if not np.isnan(rsi):
        if rsi > 80 or rsi < 20:
            score *= 0.5  # extreme readings = choppy ahead

    return float(np.clip(score, 0.05, 1.5))


def compute_vol_target_scalar(rvol_20d, target_vol=0.10):
    """Scale position to target annualized vol."""
    if np.isnan(rvol_20d) or rvol_20d < 0.01:
        return 0.5
    return min(target_vol / rvol_20d, 2.0)


def drawdown_scale(current_dd_pct):
    """
    Smooth de-risking based on current drawdown.
    0% DD = 100% size, 10% DD = 60%, 20% DD = 20%, 30%+ DD = 5%
    """
    if current_dd_pct <= 0.0:
        return 1.0
    if current_dd_pct < 0.05:
        return 1.0 - current_dd_pct * 2  # 100% -> 90%
    if current_dd_pct < 0.10:
        return 0.9 - (current_dd_pct - 0.05) * 6  # 90% -> 60%
    if current_dd_pct < 0.20:
        return 0.6 - (current_dd_pct - 0.10) * 4  # 60% -> 20%
    if current_dd_pct < 0.30:
        return 0.2 - (current_dd_pct - 0.20) * 1.5  # 20% -> 5%
    return 0.05  # minimal exposure in deep drawdown


# ── Backtest engine ───────────────────────────────────────────

def run_backtest_on_bars(all_bars, nav, signal_threshold, slippage_bps, kelly_fraction,
                         target_vol=0.10):
    """
    Multi-factor backtest with vol targeting, regime adaptation, and proper risk management.
    """
    config = PaperConfig(
        initial_nav=nav,
        signal_threshold=signal_threshold,
        slippage_bps=slippage_bps,
        drawdown_auto_kill_pct=0.25,  # high kill only for catastrophic
        kelly_fraction=kelly_fraction,
        risk_budget_per_position=0.003,
        max_position_pct_nav=0.04,
    )
    engine = PaperTradingEngine(config)

    # relax risk thresholds for multi-factor portfolio backtest
    # defaults (10%/20% DD, 1.5% intraday) are too tight for a 36-instrument book
    engine.risk_check.drawdown_level1_pct = 0.20  # allow new entries up to 20% DD
    engine.risk_check.drawdown_level2_pct = 0.35  # hard stop at 35% DD
    engine.risk_check.intraday_stop_loss_pct = 0.04  # 4% daily stop (not 1.5%)
    engine.risk_check.max_gross_leverage = 2.0  # cap leverage at 2x
    engine.risk_check.max_net_exposure = 1.5  # allow some directional bias

    # pre-compute features for every symbol
    print("  Computing features...", end=" ", flush=True)
    sym_features = {}
    for sym, sid, df in all_bars:
        if len(df) < 252:
            continue  # need at least 1 year of history
        feat = compute_features(df)
        sym_features[sid] = {
            "feat": feat,
            "df": df,
            "sym": sym,
            "asset_type": ASSET_TYPES.get(sym, "ETF"),
        }
    print(f"done ({len(sym_features)} symbols)")

    # pre-compute all signals per (sid, date_index)
    print("  Computing signals...", end=" ", flush=True)
    signal_cache = {}  # (sid, date_idx) -> (signal, vol_scalar, price, volume)
    for sid, info in sym_features.items():
        feat = info["feat"]
        df = info["df"]
        asset_type = info["asset_type"]
        for i in range(252, len(df)):  # start after 1 year warmup
            sig = compute_composite_signal(feat, i, asset_type)
            rvol = feat["rvol_20d"].iloc[i]
            vol_scalar = compute_vol_target_scalar(rvol, target_vol)
            env_score = compute_environment_score(feat, i)
            # environment score gates position size: benign = full, dangerous = tiny
            adjusted = sig * vol_scalar * env_score
            signal_cache[(sid, i)] = (
                adjusted,
                vol_scalar,
                df["close"].iloc[i],
                int(df["volume"].iloc[i]),
            )
    print(f"done ({len(signal_cache):,} signal points)")

    # build sorted timeline of (date, sid, date_idx)
    timeline = []
    for sid, info in sym_features.items():
        df = info["df"]
        for i in range(252, len(df)):
            dt = df["date"].iloc[i]
            if hasattr(dt, 'timestamp'):
                ts = int(dt.timestamp() * 1e9)
            else:
                ts = 0
            timeline.append((ts, sid, i))
    timeline.sort()

    if not timeline:
        return None, None, None

    # custom signal function that reads from cache
    current_signal = [0.0]  # mutable closure

    def cached_signal_fn(symbol_id, price, eng):
        return current_signal[0]

    engine.set_signal_function(cached_signal_fn)

    # run through engine
    daily_navs = []
    daily_dates = []
    current_day = None

    for ts, sid, date_idx in timeline:
        dt = datetime.fromtimestamp(ts / 1e9, tz=timezone.utc)
        day = dt.date()

        if current_day is None:
            current_day = day
        if day != current_day:
            daily_navs.append(engine.stats.final_nav)
            daily_dates.append(current_day)
            current_day = day
            if engine.kill_switch.level.value > 0:
                engine.kill_switch.reset()
                engine.portfolio.peak_nav = engine.portfolio.nav

        cached = signal_cache.get((sid, date_idx))
        if cached is None:
            continue

        adj_signal, vol_scalar, price, volume = cached
        current_signal[0] = adj_signal

        tick = PaperTick(
            symbol_id=sid,
            price=price,
            volume=volume,
            timestamp_ns=ts,
        )
        engine.on_tick(tick)

    # last day
    daily_navs.append(engine.stats.final_nav)
    daily_dates.append(current_day)

    return np.array(daily_navs), daily_dates, engine


# ── Statistics ────────────────────────────────────────────────

def compute_sharpe(returns, ann=252):
    if len(returns) < 2 or np.std(returns) < 1e-12:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(ann))


def compute_sortino(returns, ann=252):
    neg = returns[returns < 0]
    if len(neg) < 2:
        return 0.0
    downside = np.std(neg)
    if downside < 1e-12:
        return 0.0
    return float(np.mean(returns) / downside * np.sqrt(ann))


def compute_max_dd(equity):
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1)
    return float(np.min(dd))


def monte_carlo(daily_returns, n_sims=10000, n_days=252):
    rng = np.random.default_rng(42)
    results = np.zeros((n_sims, n_days))
    for i in range(n_sims):
        sampled = rng.choice(daily_returns, size=n_days, replace=True)
        results[i] = np.cumprod(1 + sampled)

    terminal = results[:, -1]
    max_dds = np.zeros(n_sims)
    for i in range(n_sims):
        peak = np.maximum.accumulate(results[i])
        dd = (results[i] - peak) / np.where(peak > 0, peak, 1)
        max_dds[i] = np.min(dd)

    return terminal, max_dds


# ── Main ──────────────────────────────────────────────────────

def run(args):
    universe = load_universe(args.config)
    print(f"Universe: {len(universe)} instruments across {len(set(t for _, t in universe))} asset classes\n")

    print(f"Connecting to IB Gateway at {args.host}:{args.port}...")
    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id)
    ib.reqMarketDataType(4)
    print("Connected\n")

    print("=" * 65)
    print("  FETCHING HISTORICAL DATA")
    print("=" * 65)

    all_bars = []
    symbol_info = {}
    sid = 0

    for sym, asset_type in universe:
        sid += 1
        print(f"  [{sid:2d}/{len(universe)}] {sym:8s} ({asset_type:10s}) ", end="", flush=True)
        df = fetch_max_daily(ib, sym, asset_type)
        if df.empty:
            print("-- no data")
            continue
        date_start = df["date"].iloc[0].strftime("%Y-%m-%d") if hasattr(df["date"].iloc[0], "strftime") else "?"
        date_end = df["date"].iloc[-1].strftime("%Y-%m-%d") if hasattr(df["date"].iloc[-1], "strftime") else "?"
        print(f"-- {len(df):,} days ({date_start} to {date_end})")
        all_bars.append((sym, sid, df))
        symbol_info[sym] = {"sid": sid, "type": asset_type, "bars": len(df)}

    ib.disconnect()

    if not all_bars:
        print("\nNo data fetched. Exiting.")
        sys.exit(1)

    total_bars = sum(len(df) for _, _, df in all_bars)
    print(f"\nTotal: {len(all_bars)} instruments, {total_bars:,} daily bars\n")

    all_dates = set()
    for sym, sid, df in all_bars:
        for d in df["date"]:
            all_dates.add(d.date() if hasattr(d, 'date') else d)
    min_date = min(all_dates)
    max_date = max(all_dates)

    # run backtest
    print("=" * 65)
    print("  RUNNING MULTI-FACTOR BACKTEST")
    print("=" * 65)
    print(f"  Strategy: Trend + Momentum + MeanRev + Breakout (regime-adaptive)")
    print(f"  Vol target: {args.target_vol:.0%} annualized")
    print(f"  Kelly: {args.kelly} | Threshold: {args.signal_threshold} | Slippage: {args.slippage} bps")
    print()

    bt_start = time.time()
    equity, dates, engine = run_backtest_on_bars(
        all_bars, args.nav, args.signal_threshold, args.slippage, args.kelly,
        target_vol=args.target_vol,
    )
    bt_elapsed = time.time() - bt_start

    if equity is None or len(equity) < 2:
        print("Backtest produced no results.")
        sys.exit(1)

    daily_rets = np.diff(equity) / equity[:-1]
    daily_rets = daily_rets[np.isfinite(daily_rets)]
    total_return = (equity[-1] - equity[0]) / equity[0]
    n_years = len(equity) / 252

    print(f"  Period:          {dates[0]} to {dates[-1]} ({len(equity)} days, {n_years:.1f} years)")
    print(f"  Backtest time:   {bt_elapsed:.1f}s")
    print(f"  Ticks processed: {engine.stats.ticks_processed:,}")
    print()

    sharpe = compute_sharpe(daily_rets)
    sortino = compute_sortino(daily_rets)
    max_dd = compute_max_dd(equity)
    cagr = (equity[-1] / equity[0]) ** (1 / max(n_years, 0.01)) - 1
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-12 else 0
    s = engine.stats

    print("  -- Aggregate Performance --")
    print(f"  CAGR:            {cagr:+.2%}")
    print(f"  Total return:    {total_return:+.2%}")
    print(f"  Sharpe:          {sharpe:.2f}")
    print(f"  Sortino:         {sortino:.2f}")
    print(f"  Calmar:          {calmar:.2f}")
    print(f"  Max drawdown:    {max_dd:.2%}")
    print(f"  Final NAV:       ${equity[-1]:,.0f}")
    print(f"  Peak NAV:        ${np.max(equity):,.0f}")
    print(f"  Total PnL:       ${equity[-1] - equity[0]:+,.0f}")
    print(f"  Orders:          {s.orders_submitted} sent, {s.orders_filled} filled, {s.orders_rejected} rejected")
    fill_rate = s.orders_filled / max(s.orders_submitted, 1) * 100
    print(f"  Fill rate:       {fill_rate:.1f}%")
    print()

    # year by year
    print("  -- Year-by-Year Performance --")
    print(f"  {'Year':<6} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8}")
    print("  " + "-" * 42)

    year_data = defaultdict(list)
    for i, d in enumerate(dates):
        year_data[d.year].append(i)

    for year in sorted(year_data.keys()):
        indices = year_data[year]
        if len(indices) < 2:
            continue
        start_idx = indices[0]
        end_idx = indices[-1]
        yr_equity = equity[start_idx:end_idx + 1]
        yr_ret = (yr_equity[-1] - yr_equity[0]) / yr_equity[0] if yr_equity[0] > 0 else 0
        yr_daily = np.diff(yr_equity) / yr_equity[:-1]
        yr_daily = yr_daily[np.isfinite(yr_daily)]
        yr_sharpe = compute_sharpe(yr_daily)
        yr_dd = compute_max_dd(yr_equity)
        yr_trades = int(s.orders_filled * len(indices) / len(dates)) if len(dates) > 0 else 0
        print(f"  {year:<6} {yr_ret:>+7.2%} {yr_sharpe:>8.2f} {yr_dd:>+7.2%} {yr_trades:>8}")

    print()

    # asset class breakdown
    print("  -- Performance by Asset Class --")
    print(f"  {'Class':<14} {'Instruments':>12} {'Avg Bars':>10}")
    print("  " + "-" * 38)

    class_groups = defaultdict(list)
    for sym, sid, df in all_bars:
        class_groups[ASSET_TYPES.get(sym, "OTHER")].append((sym, len(df)))

    for cls in sorted(class_groups.keys()):
        items = class_groups[cls]
        avg_bars = np.mean([b for _, b in items])
        print(f"  {cls:<14} {len(items):>12} {avg_bars:>10,.0f}")

    print()

    # monte carlo
    if len(daily_rets) > 20:
        print("  -- Monte Carlo Simulation (10,000 paths, 1yr forward) --")
        terminal, max_dds = monte_carlo(daily_rets, n_sims=10000, n_days=252)

        print(f"  Median return:     {np.median(terminal) - 1:+.2%}")
        print(f"  Mean return:       {np.mean(terminal) - 1:+.2%}")
        print(f"  5th pctl:          {np.percentile(terminal, 5) - 1:+.2%}")
        print(f"  25th pctl:         {np.percentile(terminal, 25) - 1:+.2%}")
        print(f"  75th pctl:         {np.percentile(terminal, 75) - 1:+.2%}")
        print(f"  95th pctl:         {np.percentile(terminal, 95) - 1:+.2%}")
        print(f"  Prob of loss:      {np.mean(terminal < 1.0):.1%}")
        print(f"  Prob of >10% loss: {np.mean(terminal < 0.90):.1%}")
        print(f"  Prob of >20% gain: {np.mean(terminal > 1.20):.1%}")
        print()
        print(f"  Median max DD:     {np.median(max_dds):.2%}")
        print(f"  95th pctl DD:      {np.percentile(max_dds, 5):.2%}")
        print(f"  Worst case DD:     {np.min(max_dds):.2%}")
    else:
        print("  Not enough data for Monte Carlo simulation.")

    print()
    print("=" * 65)
    print(f"  Strategy:        Multi-factor (trend/mom/meanrev/breakout)")
    print(f"  Vol target:      {args.target_vol:.0%}")
    print(f"  Kelly fraction:  {args.kelly}")
    print(f"  Signal threshold:{args.signal_threshold}")
    print(f"  Slippage:        {args.slippage} bps")
    print(f"  Starting NAV:    ${args.nav:,.0f}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-factor backtest with vol targeting")
    parser.add_argument("--config", default="config/data_layer.yaml", help="instrument config")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=2)
    parser.add_argument("--nav", type=float, default=1_000_000, help="starting capital")
    parser.add_argument("--signal-threshold", type=float, default=0.15, help="min signal to trade")
    parser.add_argument("--slippage", type=float, default=1.0, help="slippage bps")
    parser.add_argument("--kelly", type=float, default=0.25, help="kelly fraction")
    parser.add_argument("--target-vol", type=float, default=0.10, help="target annualized vol")
    args = parser.parse_args()
    run(args)
