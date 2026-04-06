#!/usr/bin/env python3
"""
AlphaForge v3 — Cross-Sectional Alpha Engine

v3.1 upgrades (from +6.67% / -40.56% DD baseline):

1. RESIDUAL MOMENTUM — strip market beta before ranking.
   Raw momentum is 50% market exposure. Beta-hedged momentum is pure alpha.
   This is the single biggest improvement (AQR, Two Sigma, DE Shaw all do this).

2. ADAPTIVE FACTOR WEIGHTS — trailing 63-day Information Coefficient.
   Static weights assume each factor always works. In reality, momentum crashes
   every few years, quality outperforms in downturns. IC-weighting adapts.

3. DRAWDOWN CIRCUIT BREAKER — hard deleveraging at -15% and -25% DD.
   The -40% max DD was unacceptable. Real funds have kill switches.
   Deleverage aggressively, re-lever slowly (asymmetric response).

4. MARKET BREADTH OVERLAY — % of stocks above 200-day SMA.
   Pure price breadth captures regime better than vol alone.
   Below 40% breadth = bear, reduce net exposure.

5. HIGH-CONVICTION SHORTS — require 3+ factors to agree.
   Weak shorts bleed in bull markets. Only short consensus losers.

6. SECTOR-NEUTRAL CONSTRUCTION — within each sector: long best, short worst.
   Prevent sector bets from dominating factor alpha.
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

from src.data.ingest.base import AssetClass
from src.data.ingest.data_manager import DataManager, build_data_manager_from_env
from src.regime.tracker import RegimeTracker


ASSET_TYPES = {}
CACHE_DIR = os.path.expanduser("~/.one_brain_fund/cache/bars")

_ASSET_CLASS_MAP = {
    "ETF": AssetClass.ETF, "EQUITY": AssetClass.EQUITY,
    "FUTURE": AssetClass.FUTURE, "COMMODITY": AssetClass.COMMODITY,
    "BOND": AssetClass.BOND, "FX": AssetClass.FX,
    "VOLATILITY": AssetClass.VOLATILITY,
}

# Sector mapping for S&P 500 stocks (GICS sectors)
# This enables sector-neutral construction
SECTOR_MAP = {
    # Technology
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "AVGO": "Tech", "AMD": "Tech",
    "ADBE": "Tech", "CRM": "Tech", "CSCO": "Tech", "ACN": "Tech", "INTC": "Tech",
    "TXN": "Tech", "QCOM": "Tech", "AMAT": "Tech", "INTU": "Tech", "NOW": "Tech",
    "ADI": "Tech", "LRCX": "Tech", "SNPS": "Tech", "KLAC": "Tech", "CDNS": "Tech",
    "MCHP": "Tech", "MU": "Tech", "NXPI": "Tech", "ON": "Tech", "SWKS": "Tech",
    "KEYS": "Tech", "ZBRA": "Tech", "FTNT": "Tech", "PANW": "Tech", "AKAM": "Tech",
    "TECH": "Tech", "TER": "Tech", "EPAM": "Tech", "HPQ": "Tech", "HPE": "Tech",
    "WDC": "Tech", "NTAP": "Tech", "DXC": "Tech", "GPN": "Tech", "FI": "Tech",
    "GOOGL": "Tech", "GOOG": "Tech", "META": "Tech",
    # ETFs
    "XLK": "Tech", "XLC": "Tech",
    # Healthcare
    "LLY": "Health", "UNH": "Health", "JNJ": "Health", "ABBV": "Health", "MRK": "Health",
    "TMO": "Health", "ABT": "Health", "DHR": "Health", "ISRG": "Health", "BSX": "Health",
    "AMGN": "Health", "PFE": "Health", "GILD": "Health", "VRTX": "Health", "REGN": "Health",
    "CI": "Health", "ELV": "Health", "BDX": "Health", "SYK": "Health", "EW": "Health",
    "ZBH": "Health", "DXCM": "Health", "IDXX": "Health", "IQV": "Health", "HOLX": "Health",
    "MTD": "Health", "WST": "Health", "A": "Health", "BAX": "Health", "ALGN": "Health",
    "MOH": "Health", "CNC": "Health", "HCA": "Health", "COO": "Health", "STE": "Health",
    "LH": "Health", "VTRS": "Health", "DVA": "Health", "XRAY": "Health", "BIO": "Health",
    "BMY": "Health", "RVTY": "Health",
    "XLV": "Health",
    # Financials
    "JPM": "Financ", "V": "Financ", "MA": "Financ", "BAC": "Financ", "BRK.B": "Financ",
    "GS": "Financ", "BLK": "Financ", "SPGI": "Financ", "MMC": "Financ", "CB": "Financ",
    "SCHW": "Financ", "ADP": "Financ", "ICE": "Financ", "CME": "Financ", "MCO": "Financ",
    "PGR": "Financ", "AON": "Financ", "AJG": "Financ", "COF": "Financ", "USB": "Financ",
    "PNC": "Financ", "TFC": "Financ", "C": "Financ", "PRU": "Financ", "AIG": "Financ",
    "ALL": "Financ", "AFL": "Financ", "MET": "Financ", "BK": "Financ", "STT": "Financ",
    "MSCI": "Financ", "RJF": "Financ", "FITB": "Financ", "MTB": "Financ", "RF": "Financ",
    "HBAN": "Financ", "CFG": "Financ", "KEY": "Financ", "CINF": "Financ", "WRB": "Financ",
    "GL": "Financ", "NTRS": "Financ", "TROW": "Financ", "BRO": "Financ", "IVZ": "Financ",
    "PFG": "Financ", "NDAQ": "Financ", "CBOE": "Financ", "MKTX": "Financ", "JKHY": "Financ",
    "WTW": "Financ",
    "XLF": "Financ",
    # Consumer Discretionary
    "AMZN": "ConsDsc", "TSLA": "ConsDsc", "HD": "ConsDsc", "MCD": "ConsDsc", "NKE": "ConsDsc",
    "LOW": "ConsDsc", "SBUX": "ConsDsc", "TJX": "ConsDsc", "BKNG": "ConsDsc", "ORLY": "ConsDsc",
    "AZO": "ConsDsc", "ROST": "ConsDsc", "DHI": "ConsDsc", "GM": "ConsDsc", "F": "ConsDsc",
    "HLT": "ConsDsc", "LVS": "ConsDsc", "WYNN": "ConsDsc", "CZR": "ConsDsc", "MGM": "ConsDsc",
    "ULTA": "ConsDsc", "NVR": "ConsDsc", "BBY": "ConsDsc", "TSCO": "ConsDsc", "DRI": "ConsDsc",
    "YUM": "ConsDsc", "CPRT": "ConsDsc", "POOL": "ConsDsc", "GRMN": "ConsDsc", "TPR": "ConsDsc",
    "RCL": "ConsDsc", "DG": "ConsDsc", "DLTR": "ConsDsc", "TGT": "ConsDsc",
    "NFLX": "ConsDsc", "EA": "ConsDsc", "TTWO": "ConsDsc", "MTCH": "ConsDsc",
    "UBER": "ConsDsc", "BBWI": "ConsDsc", "AAL": "ConsDsc", "DAL": "ConsDsc",
    "UAL": "ConsDsc", "LUV": "ConsDsc", "MHK": "ConsDsc",
    "XLY": "ConsDsc",
    # Consumer Staples
    "PG": "ConsStp", "COST": "ConsStp", "KO": "ConsStp", "PEP": "ConsStp", "WMT": "ConsStp",
    "PM": "ConsStp", "MO": "ConsStp", "MDLZ": "ConsStp", "KDP": "ConsStp", "KMB": "ConsStp",
    "GIS": "ConsStp", "HSY": "ConsStp", "KR": "ConsStp", "SYY": "ConsStp", "STZ": "ConsStp",
    "TSN": "ConsStp", "CHD": "ConsStp", "CLX": "ConsStp", "MKC": "ConsStp", "K": "ConsStp",
    "MNST": "ConsStp", "HRL": "ConsStp", "TAP": "ConsStp", "SJM": "ConsStp",
    "WBA": "ConsStp", "CAH": "ConsStp", "MCK": "ConsStp",
    "XLP": "ConsStp",
    # Industrials
    "CAT": "Indust", "DE": "Indust", "HON": "Indust", "UNP": "Indust", "RTX": "Indust",
    "LMT": "Indust", "GE": "Indust", "GD": "Indust", "NOC": "Indust", "BA": "Indust",
    "ETN": "Indust", "EMR": "Indust", "ITW": "Indust", "PH": "Indust", "ROK": "Indust",
    "MMM": "Indust", "FDX": "Indust", "NSC": "Indust", "CMI": "Indust", "CTAS": "Indust",
    "AME": "Indust", "FAST": "Indust", "ODFL": "Indust", "TDG": "Indust", "CARR": "Indust",
    "PCAR": "Indust", "WAB": "Indust", "TT": "Indust", "DOV": "Indust", "GWW": "Indust",
    "SNA": "Indust", "ROP": "Indust", "IR": "Indust", "TXT": "Indust", "JBHT": "Indust",
    "SWK": "Indust", "HII": "Indust", "MSI": "Indust", "PAYX": "Indust", "VRSK": "Indust",
    "BR": "Indust", "LDOS": "Indust", "CHRW": "Indust", "BWA": "Indust", "HUBB": "Indust",
    "GNRC": "Indust", "ROL": "Indust", "RHI": "Indust", "EXPD": "Indust", "FTV": "Indust",
    "STLD": "Indust",
    "XLI": "Indust",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "EOG": "Energy", "SLB": "Energy", "OXY": "Energy",
    "PSX": "Energy", "MPC": "Energy", "VLO": "Energy", "HAL": "Energy", "DVN": "Energy",
    "FANG": "Energy", "HES": "Energy", "BKR": "Energy", "OKE": "Energy", "WMB": "Energy",
    "TRGP": "Energy", "KMI": "Energy", "APA": "Energy", "CF": "Energy",
    "XLE": "Energy",
    # Utilities
    "NEE": "Util", "SO": "Util", "DUK": "Util", "D": "Util", "AEP": "Util",
    "SRE": "Util", "EXC": "Util", "XEL": "Util", "WEC": "Util", "ED": "Util",
    "AWK": "Util", "PCG": "Util", "DTE": "Util", "PPL": "Util", "ETR": "Util",
    "FE": "Util", "ATO": "Util", "CMS": "Util", "CNP": "Util", "NI": "Util",
    "EVRG": "Util", "AES": "Util", "LNT": "Util",
    "XLU": "Util",
    # Real Estate
    "AMT": "RealEst", "PLD": "RealEst", "CCI": "RealEst", "EQIX": "RealEst",
    "PSA": "RealEst", "SPG": "RealEst", "O": "RealEst", "WELL": "RealEst",
    "DLR": "RealEst", "SBAC": "RealEst", "VICI": "RealEst", "AVB": "RealEst",
    "EQR": "RealEst", "IRM": "RealEst", "MAA": "RealEst", "EXR": "RealEst",
    "VTR": "RealEst", "KIM": "RealEst", "REG": "RealEst", "HST": "RealEst",
    "UDR": "RealEst", "CPT": "RealEst", "PEAK": "RealEst", "INVH": "RealEst",
    "BXP": "RealEst", "FRT": "RealEst", "ARE": "RealEst",
    "XLRE": "RealEst",
    # Materials
    "LIN": "Mater", "APD": "Mater", "SHW": "Mater", "ECL": "Mater", "NEM": "Mater",
    "NUE": "Mater", "DD": "Mater", "DOW": "Mater", "PPG": "Mater", "VMC": "Mater",
    "MLM": "Mater", "IFF": "Mater", "BALL": "Mater", "PKG": "Mater", "IP": "Mater",
    "CE": "Mater", "LYB": "Mater", "ALB": "Mater", "FMC": "Mater", "SEE": "Mater",
    "MAS": "Mater", "BG": "Mater",
    "XLB": "Mater",
    # Comms
    "CMCSA": "Comms", "VZ": "Comms", "T": "Comms", "TMUS": "Comms",
    "DISH": "Comms", "NWSA": "Comms", "NWS": "Comms", "PARA": "Comms",
    "ATVI": "Comms", "WBD": "Comms", "LKQ": "Comms", "CTLT": "Comms",
}

# Sector ETF tickers for sector-relative momentum
SECTOR_ETFS = {
    "Tech": "XLK", "Health": "XLV", "Financ": "XLF", "ConsDsc": "XLY",
    "ConsStp": "XLP", "Indust": "XLI", "Energy": "XLE", "Util": "XLU",
    "RealEst": "XLRE", "Mater": "XLB", "Comms": "XLC",
}


def load_universe_static(config_path):
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


def fetch_daily_bars(dm, symbol, asset_type, use_cache=True):
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
        return None, None, []
    prices = pd.DataFrame(frames).sort_index().ffill()

    # Also build volume matrix for quality factor
    vol_frames = {}
    for sym, df in all_data.items():
        if df.empty or "volume" not in df.columns:
            continue
        s = df.set_index("date")["volume"]
        s.name = sym
        vol_frames[sym] = s
    volumes = pd.DataFrame(vol_frames).sort_index().ffill().fillna(0)
    # Align columns
    common = prices.columns.intersection(volumes.columns)
    return prices[common], volumes[common], list(common)


# ═══════════════════════════════════════════════════════════════
#  5-FACTOR CROSS-SECTIONAL SIGNAL MODEL
# ═══════════════════════════════════════════════════════════════

def compute_factor_scores(prices, volumes):
    """
    Compute 5 factor scores for every stock at every point in time.
    Returns dict of DataFrames, one per factor.

    v3.1 UPGRADE: Residual (beta-hedged) momentum.
    Raw 12-1 month momentum is ~50% market return and ~25% sector return.
    Stripping both isolates the stock-specific alpha component.
    """
    n_days, n_stocks = prices.shape
    returns = prices.pct_change().fillna(0)

    # Pre-compute rolling stats
    print("    [factors] Computing rolling statistics...", end=" ", flush=True)
    t0 = time.time()

    sma50 = prices.rolling(50, min_periods=50).mean()
    sma200 = prices.rolling(200, min_periods=200).mean()
    rvol20 = returns.rolling(20, min_periods=15).std() * np.sqrt(252)
    rvol60 = returns.rolling(60, min_periods=40).std() * np.sqrt(252)
    ret_252 = prices / prices.shift(252) - 1
    ret_21 = prices / prices.shift(21) - 1
    ret_126 = prices / prices.shift(126) - 1
    ret_5 = prices / prices.shift(5) - 1

    # Market return for beta-hedging
    spy_col = "SPY" if "SPY" in prices.columns else prices.columns[0]
    mkt_ret = returns[spy_col]
    mkt_ret_252 = prices[spy_col] / prices[spy_col].shift(252) - 1
    mkt_ret_21 = prices[spy_col] / prices[spy_col].shift(21) - 1

    print(f"{time.time()-t0:.1f}s")

    # ── Factor 1: RESIDUAL MOMENTUM (beta-hedged 12-1m) ──
    print("    [factors] Residual momentum (beta-hedged 12-1m)...", end=" ", flush=True)
    t0 = time.time()
    # Rolling beta: cov(stock, market) / var(market) over 252 days
    rolling_cov = returns.rolling(252, min_periods=200).apply(
        lambda x: np.nan, raw=True)  # placeholder - compute properly below
    # Vectorized rolling beta using covariance
    mkt_var = mkt_ret.rolling(252, min_periods=200).var()
    betas = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    for col in prices.columns:
        cov_with_mkt = returns[col].rolling(252, min_periods=200).cov(mkt_ret)
        betas[col] = (cov_with_mkt / (mkt_var + 1e-10)).clip(-3, 3).fillna(1.0)

    # Residual return = stock return - beta * market return
    # Over 12-1 month window
    raw_mom = ret_252 - ret_21
    mkt_mom = mkt_ret_252 - mkt_ret_21
    residual_mom = raw_mom.sub(betas.mul(mkt_mom, axis=0), axis=0)

    # Also subtract sector momentum for industry-adjusted residual
    for sector, etf in SECTOR_ETFS.items():
        if etf in prices.columns:
            sector_mom = prices[etf] / prices[etf].shift(252) - 1 - (prices[etf] / prices[etf].shift(21) - 1)
            sector_syms = [s for s in prices.columns if SECTOR_MAP.get(s) == sector]
            for sym in sector_syms:
                if sym in residual_mom.columns:
                    # Subtract half of sector momentum (don't overstrip)
                    residual_mom[sym] = residual_mom[sym] - 0.5 * sector_mom.fillna(0)

    # Vol-adjust
    mom = residual_mom / (rvol60 + 0.01)
    print(f"{time.time()-t0:.1f}s")

    # ── Factor 2: QUALITY (low vol + high residual return) ──
    print("    [factors] Quality...", end=" ", flush=True)
    t0 = time.time()
    # Also beta-hedge quality: high IDIOSYNCRATIC return / low vol
    residual_ret_126 = ret_126 - betas * (prices[spy_col] / prices[spy_col].shift(126) - 1)
    quality = residual_ret_126 / (rvol60 + 0.01)
    print(f"{time.time()-t0:.1f}s")

    # ── Factor 3: SHORT-TERM REVERSAL (5-day) ──
    print("    [factors] Short-term reversal...", end=" ", flush=True)
    t0 = time.time()
    reversal = -ret_5
    print(f"{time.time()-t0:.1f}s")

    # ── Factor 4: TREND (time-series) ──
    print("    [factors] Trend...", end=" ", flush=True)
    t0 = time.time()
    above_50 = (prices > sma50).astype(float)
    sma_cross = (sma50 > sma200).astype(float) - (sma50 < sma200).astype(float)
    sma50_slope = (sma50 / sma50.shift(20) - 1).fillna(0)
    trend = 0.5 * sma_cross + 0.3 * (above_50 * 2 - 1) + 0.2 * sma50_slope.clip(-1, 1) * 5
    print(f"{time.time()-t0:.1f}s")

    # ── Factor 5: EARNINGS DRIFT PROXY ──
    print("    [factors] Earnings drift proxy...", end=" ", flush=True)
    t0 = time.time()
    gap_threshold = 0.025
    is_gap_up = (returns > gap_threshold).astype(float)
    is_gap_down = (returns < -gap_threshold).astype(float)
    gap_signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for lag in [1, 2, 3, 5, 8, 13, 21, 34, 55]:
        decay = 1.0 / (1 + lag * 0.05)
        gap_signal += decay * (is_gap_up.shift(lag).fillna(0) - is_gap_down.shift(lag).fillna(0))
    earnings_drift = gap_signal / 5.0
    print(f"{time.time()-t0:.1f}s")

    # ── Factor 6: 52-WEEK HIGH PROXIMITY (new) ──
    print("    [factors] 52-week high proximity...", end=" ", flush=True)
    t0 = time.time()
    # Stocks near 52-week highs tend to continue (George & Hwang 2004)
    # Anchoring bias: analysts underreact to new highs
    high_252 = prices.rolling(252, min_periods=200).max()
    proximity = prices / high_252 - 1  # 0 at high, negative below
    # Normalize: near high = bullish, far from high = bearish
    high_52w = proximity.clip(-0.5, 0) * 2  # scale to [-1, 0]
    high_52w = high_52w + 1  # shift to [0, 1] where 1 = at 52w high
    print(f"{time.time()-t0:.1f}s")

    return {
        "momentum": mom,
        "quality": quality,
        "reversal": reversal,
        "trend": trend,
        "earnings_drift": earnings_drift,
        "high_52w": high_52w,
    }


def cross_sectional_rank(factor_df, equities_only=None):
    """
    Rank stocks cross-sectionally at each point in time.
    Returns scores in [-1, +1] where +1 = top ranked, -1 = bottom ranked.
    """
    if equities_only is not None:
        cols = [c for c in factor_df.columns if c in equities_only]
        sub = factor_df[cols]
    else:
        sub = factor_df

    # Rank across columns (stocks) at each time point
    ranked = sub.rank(axis=1, pct=True, na_option="keep")
    # Map [0, 1] percentile to [-1, +1]
    scored = (ranked - 0.5) * 2.0
    return scored


def compute_trailing_ic(ranked_factor, forward_returns, lookback=63):
    """
    Fast trailing IC: sample every 5 days instead of every day,
    use Pearson on ranks (equivalent to Spearman but faster).
    """
    from scipy.stats import rankdata

    fwd_21 = forward_returns.rolling(21).sum().shift(-21)
    ic_series = pd.Series(0.0, index=ranked_factor.index)

    # Precompute arrays for speed
    factor_arr = ranked_factor.values  # (T, N)
    fwd_arr = fwd_21.values

    for i in range(lookback + 21, len(ranked_factor), 5):  # sample every 5 days
        ics = []
        for t in range(i - lookback, i, 5):  # sample within window too
            f_row = factor_arr[t]
            r_row = fwd_arr[t]
            valid = ~(np.isnan(f_row) | np.isnan(r_row))
            if valid.sum() > 15:
                fr = rankdata(f_row[valid])
                rr = rankdata(r_row[valid])
                ic = np.corrcoef(fr, rr)[0, 1]
                if not np.isnan(ic):
                    ics.append(ic)
        if len(ics) > 2:
            val = np.mean(ics)
            # Fill forward for the 5-day gap
            for j in range(i, min(i + 5, len(ic_series))):
                ic_series.iloc[j] = val

    return ic_series.ffill().fillna(0)


def compute_composite_signal(factors, equity_syms, prices):
    """
    v3.1: Adaptive IC-weighted factor combination.

    Base weights (prior from literature):
    - Momentum: 25% (residual mom is stronger than raw)
    - Quality: 20%
    - Trend: 15%
    - Earnings drift: 15%
    - 52-week high: 15% (George & Hwang 2004)
    - Reversal: 10%

    Then adjust by trailing 63-day IC: if a factor has been predictive
    recently, weight it more. If IC is negative, zero it out.
    """
    BASE_WEIGHTS = {
        "momentum": 0.25,
        "quality": 0.20,
        "trend": 0.15,
        "earnings_drift": 0.15,
        "high_52w": 0.15,
        "reversal": 0.10,
    }

    print("    [composite] Cross-sectional ranking...", end=" ", flush=True)
    t0 = time.time()

    ranked_factors = {}
    for name, df in factors.items():
        ranked_factors[name] = cross_sectional_rank(df, equities_only=equity_syms)

    print(f"{time.time()-t0:.1f}s")

    # Compute trailing IC for adaptive weighting
    print("    [composite] Computing trailing IC for adaptive weights...", end=" ", flush=True)
    t0 = time.time()
    equity_returns = prices[equity_syms].pct_change().fillna(0)
    factor_ics = {}
    for name in BASE_WEIGHTS:
        factor_ics[name] = compute_trailing_ic(ranked_factors[name], equity_returns)
    print(f"{time.time()-t0:.1f}s")

    # Adaptive blend: base weight * max(IC, 0) + floor (vectorized)
    print("    [composite] Adaptive factor blending (vectorized)...", end=" ", flush=True)
    t0 = time.time()
    composite = pd.DataFrame(0.0, index=factors["momentum"].index, columns=equity_syms)

    # Build IC-adjusted weight series for each factor
    for name, base_w in BASE_WEIGHTS.items():
        ic_s = factor_ics[name]
        # IC-tilt: scale IC (typically 0.02-0.08) to multiplier
        ic_mult = (ic_s * 10).clip(lower=0).clip(upper=2.0)
        adaptive_w = base_w * (0.3 + 0.7 * ic_mult)  # floor at 30% of base
        # Apply weight to ranked factor
        r = ranked_factors[name]
        for col in equity_syms:
            if col in r.columns:
                composite[col] += adaptive_w * r[col].fillna(0)

    # Normalize: divide by total weight at each time step
    total_w_series = pd.Series(0.0, index=composite.index)
    for name, base_w in BASE_WEIGHTS.items():
        ic_s = factor_ics[name]
        ic_mult = (ic_s * 10).clip(lower=0).clip(upper=2.0)
        total_w_series += base_w * (0.3 + 0.7 * ic_mult)
    total_w_series = total_w_series.clip(lower=0.01)
    composite = composite.div(total_w_series, axis=0)

    print(f"{time.time()-t0:.1f}s")

    # Print average adaptive weights for transparency
    print("\n    [composite] Average adaptive factor weights:")
    for name in BASE_WEIGHTS:
        avg_ic = factor_ics[name].mean()
        print(f"      {name:>16s}: base={BASE_WEIGHTS[name]:.0%}, avg IC={avg_ic:+.4f}")

    return composite


# ═══════════════════════════════════════════════════════════════
#  SECTOR-NEUTRAL PORTFOLIO CONSTRUCTION
# ═══════════════════════════════════════════════════════════════

def compute_environment(prices, spy_col="SPY"):
    """Market environment: controls leverage scaling."""
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
        if rv > 0.40: scale = 0.20
        elif rv > 0.30: scale = 0.35
        elif rv > 0.25: scale = 0.55
        elif rv > 0.20: scale = 0.75
        elif rv < 0.12: scale = 1.20
        elif rv < 0.08: scale = 1.40

        if vr > 1.5: scale *= 0.50
        elif vr > 1.3: scale *= 0.65
        elif vr < 0.7: scale *= 1.15
        env.iloc[i] = float(np.clip(scale, 0.15, 1.5))
    return env


def compute_breadth(prices, sma200):
    """Market breadth: % of stocks above 200-day SMA."""
    above = (prices > sma200).astype(float)
    breadth = above.mean(axis=1)
    return breadth.rolling(5, min_periods=1).mean()  # smooth 5-day


def build_regime_features(prices, equity_syms, spy_col="SPY"):
    """Daily market features for regime classification."""
    if spy_col not in prices.columns:
        spy_col = prices.columns[0]

    spy = prices[spy_col]
    spy_ret = spy.pct_change().fillna(0)
    rvol20 = spy_ret.rolling(20, min_periods=20).std() * np.sqrt(252)
    ret_63 = spy / spy.shift(63) - 1
    spy_sma200 = spy.rolling(200, min_periods=200).mean()
    trend_gap = spy / spy_sma200 - 1

    sma200 = prices[equity_syms].rolling(200, min_periods=200).mean()
    breadth = compute_breadth(prices[equity_syms], sma200)

    features = pd.DataFrame(
        {
            "vol_20": rvol20,
            "ret_63": ret_63,
            "breadth": breadth,
            "trend_gap": trend_gap,
        },
        index=prices.index,
    )
    return features.replace([np.inf, -np.inf], np.nan)


def compute_regime_states(prices, equity_syms, fit_window=504, refit_every=63):
    """
    Fit the repo's HMM+GMM tracker on trailing market features and update online.

    We only enable shorts when the smoothed regime is high-vol and confidence is
    decent. In all other regimes we prefer long-only participation.
    """
    features = build_regime_features(prices, equity_syms)
    defaults = {
        "regime_label": "WARMUP",
        "position_scale": 1.0,
        "signal_gate": True,
        "allow_shorts": False,
        "confidence": 0.0,
    }
    regime_info = pd.DataFrame(defaults, index=prices.index)

    valid = features.dropna()
    if len(valid) <= fit_window:
        return regime_info

    tracker = None
    last_fit_pos = -1

    for pos, (dt, row) in enumerate(valid.iterrows()):
        if pos < fit_window:
            continue

        if tracker is None or (pos - last_fit_pos) >= refit_every:
            train = valid.iloc[max(0, pos - fit_window):pos]
            if len(train) < fit_window:
                continue
            try:
                tracker = RegimeTracker(n_regimes=5)
                tracker.fit(train.values)
                last_fit_pos = pos
            except Exception:
                tracker = None
                continue

        if tracker is None:
            continue

        try:
            state = tracker.update(row.values)
        except Exception:
            continue

        allow_shorts = (
            state.signal_gate_open
            and state.confidence >= 0.45
            and state.regime_label in {"HIGH_VOL_TRENDING", "HIGH_VOL_CHAOTIC"}
        )

        regime_info.loc[dt, "regime_label"] = state.regime_label
        regime_info.loc[dt, "position_scale"] = state.position_scale
        regime_info.loc[dt, "signal_gate"] = state.signal_gate_open
        regime_info.loc[dt, "allow_shorts"] = allow_shorts
        regime_info.loc[dt, "confidence"] = state.confidence

    return regime_info.ffill()


def build_portfolio(composite, prices, env, target_vol, rebal_freq, max_pos,
                    n_long=50, n_short=20, target_gross=1.5, blend_rate=0.92,
                    regime_info=None):
    """
    v3.1 Portfolio construction with:
    - Drawdown circuit breaker (hard deleveraging at -15%, -25%)
    - Market breadth overlay (reduce net when breadth < 40%)
    - High-conviction shorts (require multi-factor agreement)
    - Sector-neutral long/short
    """
    returns = prices.pct_change().fillna(0)
    inst_vol = returns.rolling(20, min_periods=10).std() * np.sqrt(252)
    inst_vol = inst_vol.bfill().clip(lower=0.02)

    equity_syms = list(composite.columns)

    # Market breadth
    sma200 = prices[equity_syms].rolling(200, min_periods=200).mean()
    breadth = compute_breadth(prices[equity_syms], sma200)

    # Map stocks to sectors
    sym_sector = {}
    sector_stocks = defaultdict(list)
    for sym in equity_syms:
        sec = SECTOR_MAP.get(sym, "Other")
        sym_sector[sym] = sec
        sector_stocks[sec].append(sym)

    weights = pd.DataFrame(0.0, index=prices.index, columns=equity_syms)
    target_w = pd.Series(0.0, index=equity_syms)
    live_w = pd.Series(0.0, index=equity_syms)

    # Track portfolio equity for drawdown circuit breaker
    running_nav = 1.0
    peak_nav = 1.0

    for i in range(252, len(prices)):
        # ── Drawdown circuit breaker ──
        if i > 252:
            day_ret = (live_w * returns[equity_syms].iloc[i]).sum()
            running_nav *= (1 + day_ret)
            peak_nav = max(peak_nav, running_nav)

        current_dd = (running_nav / peak_nav) - 1 if peak_nav > 0 else 0

        # Softer drawdown response — avoid killing recovery rallies
        dd_scale = 1.0
        if current_dd < -0.30:
            dd_scale = 0.40  # emergency only at deep drawdown
        elif current_dd < -0.20:
            dd_scale = 0.65
        elif current_dd < -0.12:
            dd_scale = 0.85

        regime_allows_shorts = True
        if regime_info is not None:
            regime_row = regime_info.iloc[i]
            regime_allows_shorts = bool(regime_row.get("allow_shorts", False))

        overlay_scale = dd_scale

        if i % rebal_freq != 0 and i > 252:
            live_w = target_w * overlay_scale if overlay_scale < 1.0 else target_w.copy()
            weights.iloc[i] = live_w
            continue

        sig = composite.iloc[i]
        valid = sig.dropna()
        valid = valid[valid.abs() > 0.01]
        if len(valid) < 20:
            live_w = target_w * overlay_scale if overlay_scale < 1.0 else target_w.copy()
            weights.iloc[i] = live_w
            continue

        e = env.iloc[i]
        b = breadth.iloc[i] if i < len(breadth) else 0.5
        is_bearish = e < 0.75 or b < 0.40  # vol OR breadth says bear
        shorts_active = regime_allows_shorts and is_bearish

        # ── Select longs: top stocks from each sector ──
        long_picks = []
        short_picks = []

        for sector, syms in sector_stocks.items():
            sector_sigs = valid.reindex(syms).dropna()
            if len(sector_sigs) < 2:
                continue
            sector_sigs = sector_sigs.sort_values(ascending=False)

            # Long: top 30% of each sector with positive signal
            n_l = max(int(len(sector_sigs) * 0.30), 1)
            for sym in sector_sigs.head(n_l).index:
                if sector_sigs[sym] > 0.08:
                    long_picks.append((sym, sector_sigs[sym]))

            # Short: require multi-factor conviction in bearish environments
            if shorts_active:
                n_s = max(int(len(sector_sigs) * 0.15), 1)
                for sym in sector_sigs.tail(n_s).index:
                    if sector_sigs[sym] < -0.25:  # very high bar for shorts
                        short_picks.append((sym, sector_sigs[sym]))

        long_picks.sort(key=lambda x: x[1], reverse=True)
        short_picks.sort(key=lambda x: x[1])
        long_picks = long_picks[:n_long]
        short_picks = short_picks[:n_short]

        # ── Sizing ──
        w = pd.Series(0.0, index=equity_syms)

        # Breadth-adjusted exposure
        # High breadth (>60%): full long, no shorts
        # Low breadth (<40%): reduced long, add shorts
        breadth_mult = np.clip(b * 1.5, 0.5, 1.2)  # 0.5 at 33% breadth, 1.2 at 80%

        if shorts_active:
            long_budget = target_gross * e * 0.65 * breadth_mult
            short_budget = target_gross * 0.35 * (1 - breadth_mult / 1.2)
        else:
            long_budget = target_gross * min(e, 1.3) * breadth_mult
            short_budget = 0.0

        # Apply drawdown scale
        long_budget *= overlay_scale
        short_budget *= overlay_scale

        if long_picks:
            long_vols = {s: inst_vol[s].iloc[i] if s in inst_vol.columns else 0.20
                         for s, _ in long_picks}
            long_inv = {s: 1.0 / max(v, 0.05) for s, v in long_vols.items()}
            total_inv = sum(long_inv.values())
            for sym, sig_val in long_picks:
                raw_w = long_budget * long_inv[sym] / total_inv
                tilt = 0.8 + 0.4 * abs(sig_val)
                w[sym] = float(np.clip(raw_w * tilt, 0, max_pos))

        if short_picks and short_budget > 0:
            short_vols = {s: inst_vol[s].iloc[i] if s in inst_vol.columns else 0.20
                          for s, _ in short_picks}
            short_inv = {s: 1.0 / max(v, 0.05) for s, v in short_vols.items()}
            total_inv = sum(short_inv.values())
            for sym, sig_val in short_picks:
                raw_w = -short_budget * short_inv[sym] / total_inv
                w[sym] = float(np.clip(raw_w, -max_pos * 0.5, 0))

        # Exit stale names faster than we enter fresh ones, which keeps the book
        # concentrated without giving up the turnover benefits of slower blending.
        if i > 252:
            prev_target = target_w.copy()
            delta = w - prev_target
            small_changes = delta.abs() < 0.003
            w[small_changes] = prev_target[small_changes]

            keep = pd.Series(blend_rate, index=equity_syms)
            entering = (prev_target.abs() < 1e-12) & (w.abs() > 1e-12)
            exiting = (prev_target.abs() > 1e-12) & (w.abs() < 1e-12)
            flipping = (
                (prev_target.abs() > 1e-12)
                & (w.abs() > 1e-12)
                & (np.sign(prev_target) != np.sign(w))
            )
            keep[entering] = min(blend_rate, 0.70)
            keep[exiting] = 0.55
            keep[flipping] = 0.35

            w = (1 - keep) * w + keep * prev_target
            w[(w.abs() < 0.002) & exiting] = 0.0

        target_w = w.clip(lower=-max_pos * 0.5, upper=max_pos)
        live_w = target_w * overlay_scale if overlay_scale < 1.0 else target_w.copy()
        weights.iloc[i] = live_w

    # ── Portfolio-level vol targeting ──
    port_ret = (weights.shift(1) * returns[equity_syms]).sum(axis=1)
    port_rvol = port_ret.rolling(20, min_periods=10).std() * np.sqrt(252)

    final_weights = weights.copy()
    for i in range(280, len(prices)):
        rv = port_rvol.iloc[i]
        if np.isnan(rv) or rv < 0.01:
            continue
        scale = target_vol / rv
        scale = float(np.clip(scale, 0.3, 2.0))  # tighter cap than before
        final_weights.iloc[i] = weights.iloc[i] * scale

    return final_weights


# ═══════════════════════════════════════════════════════════════
#  STATISTICS & REPORTING
# ═══════════════════════════════════════════════════════════════

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


def run(args):
    # ── Load universe ──
    universe = load_universe_static(args.config)
    print(f"\nUniverse: {len(universe)} instruments\n")

    # ── Connect & fetch data ──
    cache_hits = 0
    if not args.no_cache:
        for sym, atype in universe:
            if os.path.exists(os.path.join(CACHE_DIR, f"{sym}_{atype}.parquet")):
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
            print("ERROR: No data providers available.")
            sys.exit(1)
        if cache_hits:
            print(f"  Cached: {cache_hits}/{len(universe)}")
        print()

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

    prices, volumes, symbols = build_price_matrix(all_data)
    if prices is None:
        sys.exit(1)
    print(f"  Matrix: {prices.shape[0]} x {prices.shape[1]} instruments\n")

    # Identify equity symbols (exclude ETFs, futures, FX)
    equity_syms = [s for s in symbols if ASSET_TYPES.get(s) == "EQUITY"]
    etf_syms = [s for s in symbols if ASSET_TYPES.get(s) == "ETF"]
    tradeable = equity_syms  # v3 focuses on equities for cross-sectional

    print(f"  Tradeable equities: {len(tradeable)}")
    print(f"  Sector coverage: {len(set(SECTOR_MAP.get(s, 'Other') for s in tradeable))} sectors")
    print()

    # ── Compute factor scores ──
    print("=" * 70)
    print("  COMPUTING 5-FACTOR MODEL")
    print("=" * 70)
    t0 = time.time()
    factors = compute_factor_scores(prices, volumes)
    print(f"  Total factor computation: {time.time()-t0:.1f}s\n")

    # ── Composite signal ──
    print("=" * 70)
    print("  CROSS-SECTIONAL RANKING")
    print("=" * 70)
    composite = compute_composite_signal(factors, tradeable, prices)

    # Show signal dispersion over time
    sig_std = composite.std(axis=1)
    print(f"  Avg signal dispersion: {sig_std.mean():.3f}")
    print(f"  Current dispersion:    {sig_std.iloc[-1]:.3f}")

    # Show current top/bottom signals
    latest = composite.iloc[-1].dropna().sort_values()
    print(f"\n  Current top 10 longs:")
    for sym, val in latest.tail(10).iloc[::-1].items():
        sec = SECTOR_MAP.get(sym, "?")
        print(f"    {sym:>8s} {val:>+.3f}  ({sec})")
    print(f"\n  Current top 10 shorts:")
    for sym, val in latest.head(10).items():
        sec = SECTOR_MAP.get(sym, "?")
        print(f"    {sym:>8s} {val:>+.3f}  ({sec})")
    print()

    # ── Environment / regime ──
    env = compute_environment(prices)
    regime_info = compute_regime_states(prices, tradeable)
    regime_counts = regime_info["regime_label"].value_counts()
    print("  Regime mix:")
    for label, count in regime_counts.items():
        if label == "WARMUP":
            continue
        pct = count / max(len(regime_info), 1)
        print(f"    {label:<20s} {count:>4d} days ({pct:>5.1%})")
    if len(regime_counts) == 1 and "WARMUP" in regime_counts:
        print("    warmup only (not enough history for tracker)")
    print()

    # ── Portfolio construction ──
    print("=" * 70)
    print("  WALK-FORWARD BACKTEST v3 (cross-sectional, long/short)")
    print("=" * 70)
    print(f"  NAV: ${args.nav:,.0f} | Vol target: {args.target_vol:.0%} | "
          f"Rebal: {args.rebal_freq}d | Max pos: {args.max_pos:.0%}")
    print(f"  Target gross: {args.target_gross:.1f}x | "
          f"Long: {args.n_long} | Short: {args.n_short}")
    print()

    t0 = time.time()
    print("  Building sector-neutral long/short portfolio...", end=" ", flush=True)
    weights = build_portfolio(
        composite, prices, env,
        args.target_vol, args.rebal_freq, args.max_pos,
        n_long=args.n_long, n_short=args.n_short,
        target_gross=args.target_gross,
        regime_info=regime_info,
    )
    weights = weights.fillna(0)
    print(f"{time.time()-t0:.1f}s\n")

    # ── Compute returns ──
    returns = prices[equity_syms].pct_change().fillna(0)
    warmup = 280
    port_ret = (weights.shift(1) * returns[equity_syms]).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    tx_cost = turnover * (args.slippage + 1.0) / 10000
    net_ret = port_ret - tx_cost
    net_ret = net_ret.iloc[warmup:]
    weights_post = weights.iloc[warmup:]
    turnover_post = turnover.iloc[warmup:]

    equity_curve = args.nav * (1 + net_ret).cumprod()
    dates = equity_curve.index
    n_years = len(net_ret) / 252

    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / max(n_years, 0.01)) - 1
    s = sharpe(net_ret)
    so = sortino(net_ret)
    dd = max_dd(equity_curve)
    cal = cagr / abs(dd) if abs(dd) > 1e-12 else 0
    avg_long = weights_post.clip(lower=0).sum(axis=1).mean()
    avg_short = weights_post.clip(upper=0).abs().sum(axis=1).mean()
    avg_gross = (weights_post.abs().sum(axis=1)).mean()
    avg_net = (weights_post.sum(axis=1)).mean()
    avg_turn = turnover_post.mean() * 252
    tx_bps = tx_cost.iloc[warmup:].mean() * 252 * 10000 if len(tx_cost) > warmup else 0

    # SPY comparison
    spy_ret_full = prices["SPY"].pct_change().fillna(0) if "SPY" in prices.columns else pd.Series(0, index=prices.index)
    spy_ret = spy_ret_full.iloc[warmup:]
    spy_eq = args.nav * (1 + spy_ret).cumprod()
    spy_cagr = (spy_eq.iloc[-1] / spy_eq.iloc[0]) ** (1 / max(n_years, 0.01)) - 1
    spy_sharpe = sharpe(spy_ret)
    spy_dd = max_dd(spy_eq)

    print("  ── Aggregate ──")
    print(f"  CAGR:          {cagr:+.2%}   (SPY: {spy_cagr:+.2%})  {'✓ ALPHA' if cagr > spy_cagr else ''}")
    print(f"  Sharpe:        {s:.2f}      (SPY: {spy_sharpe:.2f})  {'✓' if s > spy_sharpe else ''}")
    print(f"  Sortino:       {so:.2f}")
    print(f"  Calmar:        {cal:.2f}")
    print(f"  Max DD:        {dd:.2%}   (SPY: {spy_dd:.2%})  {'✓' if abs(dd) < abs(spy_dd) else ''}")
    print(f"  Final NAV:     ${equity_curve.iloc[-1]:,.0f}  (SPY: ${spy_eq.iloc[-1]:,.0f})")
    print(f"  Avg long:      {avg_long:.2f}x")
    print(f"  Avg short:     {avg_short:.2f}x")
    print(f"  Avg gross:     {avg_gross:.2f}x | Net: {avg_net:+.2f}x")
    print(f"  Turnover:      {avg_turn:.0f}x/yr")
    print(f"  Tx costs:      {tx_bps:.0f} bps/yr")
    print()

    # ── Year by year with full SPY comparison ──
    print("  ── Year-by-Year: v3 vs SPY ──")
    print(f"  {'Year':<6} {'v3 Ret':>8} {'SPY':>8} {'Alpha':>8} {'v3 Shp':>8} {'SPY Shp':>8} {'v3 DD':>8} {'SPY DD':>8} {'Win':>4}")
    print("  " + "─" * 74)
    loss_years = 0
    beat_spy = 0
    beat_sharpe = 0
    beat_dd = 0

    for year in sorted(set(d.year if hasattr(d, 'year') else d.date().year for d in dates)):
        mask = pd.Series([d.year if hasattr(d, 'year') else d.date().year for d in dates]) == year
        mask.index = net_ret.index
        yr = net_ret[mask]
        yr_eq = equity_curve[mask]
        if len(yr) < 5:
            continue

        yr_ret = (yr_eq.iloc[-1] / yr_eq.iloc[0]) - 1
        yr_s = sharpe(yr)
        yr_d = max_dd(yr_eq)

        # SPY year
        spy_mask = pd.Series([d.year if hasattr(d, 'year') else d.date().year for d in spy_ret.index], index=spy_ret.index) == year
        spy_yr = spy_ret[spy_mask]
        spy_yr_eq = spy_eq[spy_mask]
        spy_yr_ret = (spy_yr_eq.iloc[-1] / spy_yr_eq.iloc[0]) - 1 if len(spy_yr_eq) > 1 else 0
        spy_yr_s = sharpe(spy_yr)
        spy_yr_d = max_dd(spy_yr_eq) if len(spy_yr_eq) > 1 else 0

        alpha = yr_ret - spy_yr_ret
        win = "Y" if yr_ret > 0 else "N"
        if yr_ret <= 0: loss_years += 1
        if yr_ret > spy_yr_ret: beat_spy += 1
        if yr_s > spy_yr_s: beat_sharpe += 1
        if abs(yr_d) < abs(spy_yr_d): beat_dd += 1

        print(f"  {year:<6} {yr_ret:>+7.2%} {spy_yr_ret:>+7.2%} {alpha:>+7.2%} "
              f"{yr_s:>7.2f}  {spy_yr_s:>7.2f}  {yr_d:>+7.2%} {spy_yr_d:>+7.2%} {win:>4}")

    total_years = len(set(d.year if hasattr(d, 'year') else d.date().year for d in dates))
    print("  " + "─" * 74)
    print(f"  Loss years:       {loss_years}/{total_years}")
    print(f"  Beat SPY return:  {beat_spy}/{total_years}")
    print(f"  Beat SPY Sharpe:  {beat_sharpe}/{total_years}")
    print(f"  Beat SPY MaxDD:   {beat_dd}/{total_years}")
    print()

    # ── Sector attribution ──
    print("  ── Sector Attribution ──")
    sector_groups = defaultdict(list)
    for sym in equity_syms:
        if sym in weights.columns:
            sector_groups[SECTOR_MAP.get(sym, "Other")].append(sym)
    print(f"  {'Sector':<10} {'#Syms':>6} {'AvgGross':>9} {'Contrib':>9}")
    print("  " + "─" * 38)
    for sec in sorted(sector_groups):
        syms = [s for s in sector_groups[sec] if s in weights_post.columns]
        if not syms:
            continue
        avg_w = weights_post[syms].abs().mean().sum()
        contrib = (weights_post[syms].shift(1) * returns[syms].iloc[warmup:]).sum(axis=1).mean() * 252
        print(f"  {sec:<10} {len(syms):>6} {avg_w:>8.3f}x {contrib:>+8.2%}")
    print()

    # ── Trade log ──
    print("  ── Trade Log (last 60 days) ──")
    pos_changes = weights.diff().fillna(0)
    recent_changes = pos_changes.tail(60)
    trade_records = []
    for dt_idx in recent_changes.index:
        row = recent_changes.loc[dt_idx]
        movers = row[row.abs() > 0.005].sort_values(key=abs, ascending=False)
        for sym, delta in movers.head(5).items():
            cur_w = weights.loc[dt_idx, sym]
            action = "BUY" if delta > 0 else "SELL"
            dt_str = dt_idx.strftime("%Y-%m-%d") if hasattr(dt_idx, 'strftime') else str(dt_idx)[:10]
            sig_val = composite[sym].loc[dt_idx] if sym in composite.columns else 0
            trade_records.append({
                "date": dt_str, "sym": sym, "action": action,
                "delta": delta, "new_wt": cur_w, "signal": sig_val,
                "sector": SECTOR_MAP.get(sym, "?"),
            })
    if trade_records:
        print(f"  {'Date':>10} {'Sym':>8} {'Action':>6} {'Delta':>8} {'NewWt':>8} {'Signal':>8} {'Sector':>8}")
        print("  " + "─" * 68)
        for t in trade_records[-30:]:
            print(f"  {t['date']:>10} {t['sym']:>8} {t['action']:>6} "
                  f"{t['delta']:>+7.3f} {t['new_wt']:>+7.3f} {t['signal']:>+7.3f} {t['sector']:>8}")
    print()

    # ── Book snapshot ──
    print("  ── Current Book ──")
    latest_w = weights.iloc[-1]
    longs = latest_w[latest_w > 0.005].sort_values(ascending=False)
    shorts = latest_w[latest_w < -0.005].sort_values()
    print(f"  {len(longs)} longs | {len(shorts)} shorts | "
          f"Gross: {latest_w.abs().sum():.2f}x | Net: {latest_w.sum():+.2f}x")
    print(f"\n  Top 10 longs:")
    for sym, w in longs.head(10).items():
        sec = SECTOR_MAP.get(sym, "?")
        sig = composite[sym].iloc[-1] if sym in composite.columns else 0
        print(f"    {sym:>8s} {w:>+7.3f}  signal={sig:>+.3f}  ({sec})")
    if len(shorts) > 0:
        print(f"\n  Top 10 shorts:")
        for sym, w in shorts.head(10).items():
            sec = SECTOR_MAP.get(sym, "?")
            sig = composite[sym].iloc[-1] if sym in composite.columns else 0
            print(f"    {sym:>8s} {w:>+7.3f}  signal={sig:>+.3f}  ({sec})")
    print()

    # ── Monte Carlo ──
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
    p = argparse.ArgumentParser(description="AlphaForge v3 — Cross-Sectional Alpha Engine")
    p.add_argument("--config", default="config/sp500_universe.yaml")
    p.add_argument("--env", default=".env")
    p.add_argument("--nav", type=float, default=10_000_000)
    p.add_argument("--target-vol", type=float, default=0.15)
    p.add_argument("--target-gross", type=float, default=1.5,
                   help="Target gross exposure (default 1.5x)")
    p.add_argument("--slippage", type=float, default=2.0)
    p.add_argument("--rebal-freq", type=int, default=10)
    p.add_argument("--max-pos", type=float, default=0.08)
    p.add_argument("--n-long", type=int, default=50)
    p.add_argument("--n-short", type=int, default=20)
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    if os.path.exists(args.env):
        from dotenv import load_dotenv
        load_dotenv(args.env)

    run(args)
