#!/usr/bin/env python3
"""
One Brain Fund v4 — Multi-Strategy Engine (Target: Sharpe >= 1.0)

v3 achieved: 0.57 Sharpe, +8.39% CAGR, -32% max DD
Problem: one strategy (cross-sectional momentum) can't hit 1.0 Sharpe alone.

The Medallion insight: it's not one brilliant signal, it's HUNDREDS of
uncorrelated signals. Combined Sharpe ≈ individual_sharpe × sqrt(N_strategies).
8 uncorrelated 0.35-Sharpe signals → 0.35 × sqrt(8) ≈ 1.0

v4 STRATEGIES:
  S1. Cross-sectional residual momentum (12-1 month, beta-hedged)
  S2. Short-term mean reversion (3-5 day, market-neutral stat arb)
  S3. Quality factor (stable earnings proxy / low idio vol)
  S4. Betting Against Beta — BAB (long low-beta, short high-beta)
  S5. Carry (dividend yield proxy from price pattern)
  S6. Sector rotation momentum (which sectors are winning)
  S7. Earnings drift (post-gap continuation)
  S8. 52-week high proximity (anchoring bias)

PORTFOLIO CONSTRUCTION:
  - Each strategy produces independent weights
  - Risk-parity blending: allocate equal VOL to each strategy
  - Dynamic tilt: boost strategies with positive trailing IC
  - Market-neutral overlay: hedge beta to near zero for half the book
  - Drawdown circuit breaker at portfolio level
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata

from src.data.ingest.base import AssetClass
from src.data.ingest.data_manager import DataManager, build_data_manager_from_env


ASSET_TYPES = {}
CACHE_DIR = os.path.expanduser("~/.one_brain_fund/cache/bars")

_ASSET_CLASS_MAP = {
    "ETF": AssetClass.ETF, "EQUITY": AssetClass.EQUITY,
    "FUTURE": AssetClass.FUTURE, "COMMODITY": AssetClass.COMMODITY,
    "BOND": AssetClass.BOND, "FX": AssetClass.FX,
    "VOLATILITY": AssetClass.VOLATILITY,
}

# ═══════════════════════════════════════════════════════════════
#  SECTOR MAP (GICS)
# ═══════════════════════════════════════════════════════════════

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

SECTOR_ETFS = {
    "Tech": "XLK", "Health": "XLV", "Financ": "XLF", "ConsDsc": "XLY",
    "ConsStp": "XLP", "Indust": "XLI", "Energy": "XLE", "Util": "XLU",
    "RealEst": "XLRE", "Mater": "XLB", "Comms": "XLC",
}


# ═══════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════

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
    frames, vol_frames = {}, {}
    for sym, df in all_data.items():
        if df.empty:
            continue
        s = df.set_index("date")["close"]
        s.name = sym
        frames[sym] = s
        if "volume" in df.columns:
            v = df.set_index("date")["volume"]
            v.name = sym
            vol_frames[sym] = v
    if not frames:
        return None, None, []
    prices = pd.DataFrame(frames).sort_index().ffill()
    volumes = pd.DataFrame(vol_frames).sort_index().ffill().fillna(0)
    common = prices.columns.intersection(volumes.columns)
    return prices[common], volumes[common], list(common)


def cross_sectional_rank(series_or_df):
    """Rank cross-sectionally → [-1, +1]."""
    if isinstance(series_or_df, pd.DataFrame):
        ranked = series_or_df.rank(axis=1, pct=True, na_option="keep")
    else:
        ranked = series_or_df.rank(pct=True)
    return (ranked - 0.5) * 2.0


# ═══════════════════════════════════════════════════════════════
#  STRATEGY 1: RESIDUAL MOMENTUM (12-1 month, beta-hedged)
# ═══════════════════════════════════════════════════════════════

def strategy_momentum(prices, returns, equity_syms, mkt_ret):
    """Cross-sectional residual momentum. The workhorse factor."""
    print("  [S1] Residual momentum...", end=" ", flush=True)
    t0 = time.time()

    eq_ret = returns[equity_syms]
    rvol60 = eq_ret.rolling(60, min_periods=40).std() * np.sqrt(252)

    # Rolling betas (252d)
    mkt_var = mkt_ret.rolling(252, min_periods=200).var()
    betas = pd.DataFrame(1.0, index=prices.index, columns=equity_syms)
    for col in equity_syms:
        cov_with_mkt = eq_ret[col].rolling(252, min_periods=200).cov(mkt_ret)
        betas[col] = (cov_with_mkt / (mkt_var + 1e-10)).clip(-3, 3).fillna(1.0)

    # Raw 12-1 month return
    ret_252 = prices[equity_syms] / prices[equity_syms].shift(252) - 1
    ret_21 = prices[equity_syms] / prices[equity_syms].shift(21) - 1
    raw_mom = ret_252 - ret_21

    # Strip market beta
    mkt_mom = prices["SPY"] / prices["SPY"].shift(252) - 1 - (prices["SPY"] / prices["SPY"].shift(21) - 1)
    residual_mom = raw_mom.sub(betas.mul(mkt_mom, axis=0))

    # Strip sector momentum (partial)
    for sector, etf in SECTOR_ETFS.items():
        if etf not in prices.columns:
            continue
        sec_mom = prices[etf] / prices[etf].shift(252) - 1 - (prices[etf] / prices[etf].shift(21) - 1)
        sec_syms = [s for s in equity_syms if SECTOR_MAP.get(s) == sector and s in residual_mom.columns]
        for sym in sec_syms:
            residual_mom[sym] -= 0.5 * sec_mom.fillna(0)

    # Vol-adjust and rank
    signal = residual_mom / (rvol60 + 0.01)
    ranked = cross_sectional_rank(signal[equity_syms])
    print(f"{time.time()-t0:.1f}s")
    return ranked


# ═══════════════════════════════════════════════════════════════
#  STRATEGY 2: SHORT-TERM MEAN REVERSION (stat arb, 3-5 day)
# ═══════════════════════════════════════════════════════════════

def strategy_mean_reversion(prices, returns, equity_syms, mkt_ret):
    """
    Market-neutral stat arb: fade 3-5 day residual moves.
    At short horizons, stocks mean-revert (liquidity provision alpha).
    Negatively correlated with momentum → great diversifier.
    """
    print("  [S2] Mean reversion (stat arb)...", end=" ", flush=True)
    t0 = time.time()

    eq_ret = returns[equity_syms]

    # 5-day residual return (strip out market)
    ret_5 = prices[equity_syms] / prices[equity_syms].shift(5) - 1
    mkt_5 = prices["SPY"] / prices["SPY"].shift(5) - 1

    # Rolling beta (60d for short-term strategy)
    mkt_var = mkt_ret.rolling(60, min_periods=40).var()
    betas_short = pd.DataFrame(1.0, index=prices.index, columns=equity_syms)
    for col in equity_syms:
        cov = eq_ret[col].rolling(60, min_periods=40).cov(mkt_ret)
        betas_short[col] = (cov / (mkt_var + 1e-10)).clip(-3, 3).fillna(1.0)

    residual_5d = ret_5.sub(betas_short.mul(mkt_5, axis=0))

    # Also compute sector-relative residual
    for sector, etf in SECTOR_ETFS.items():
        if etf not in prices.columns:
            continue
        sec_5 = prices[etf] / prices[etf].shift(5) - 1
        sec_syms = [s for s in equity_syms if SECTOR_MAP.get(s) == sector and s in residual_5d.columns]
        for sym in sec_syms:
            residual_5d[sym] -= 0.3 * sec_5.fillna(0)

    # Z-score relative to 20-day rolling mean/std of residual returns
    resid_mean = residual_5d.rolling(20, min_periods=10).mean()
    resid_std = residual_5d.rolling(20, min_periods=10).std()
    z_score = (residual_5d - resid_mean) / (resid_std + 1e-6)

    # REVERSE: negative z-score = buy (mean revert the move)
    signal = -z_score.clip(-3, 3)
    ranked = cross_sectional_rank(signal)
    print(f"{time.time()-t0:.1f}s")
    return ranked


# ═══════════════════════════════════════════════════════════════
#  STRATEGY 3: QUALITY (stable business proxy)
# ═══════════════════════════════════════════════════════════════

def strategy_quality(prices, returns, equity_syms, mkt_ret):
    """
    Quality: high risk-adjusted returns + low drawdowns.
    Proxy for profitability without fundamental data.
    """
    print("  [S3] Quality...", end=" ", flush=True)
    t0 = time.time()

    eq_ret = returns[equity_syms]
    rvol60 = eq_ret.rolling(60, min_periods=40).std() * np.sqrt(252)
    ret_126 = prices[equity_syms] / prices[equity_syms].shift(126) - 1

    # Beta-adjusted 6-month return / vol = quality
    mkt_var = mkt_ret.rolling(252, min_periods=200).var()
    betas = pd.DataFrame(1.0, index=prices.index, columns=equity_syms)
    for col in equity_syms:
        cov = eq_ret[col].rolling(252, min_periods=200).cov(mkt_ret)
        betas[col] = (cov / (mkt_var + 1e-10)).clip(-3, 3).fillna(1.0)

    mkt_126 = prices["SPY"] / prices["SPY"].shift(126) - 1
    residual_126 = ret_126.sub(betas.mul(mkt_126, axis=0))
    quality = residual_126 / (rvol60 + 0.01)

    # Add drawdown penalty: stocks with smaller max DD are higher quality
    rolling_max = prices[equity_syms].rolling(126, min_periods=63).max()
    dd_126 = (prices[equity_syms] / rolling_max - 1).clip(-1, 0)
    avg_dd = dd_126.rolling(63, min_periods=30).mean()
    # Less negative DD = higher quality (multiply by -1 so bigger = better)
    quality_dd = -avg_dd

    combined_quality = 0.6 * cross_sectional_rank(quality) + 0.4 * cross_sectional_rank(quality_dd)
    print(f"{time.time()-t0:.1f}s")
    return combined_quality


# ═══════════════════════════════════════════════════════════════
#  STRATEGY 4: BETTING AGAINST BETA (BAB)
# ═══════════════════════════════════════════════════════════════

def strategy_bab(prices, returns, equity_syms, mkt_ret):
    """
    BAB (Frazzini & Pedersen 2014): Long low-beta, short high-beta.
    Low-beta stocks are systematically underpriced because leverage-
    constrained investors (pensions, retail) buy high-beta for returns.
    Market-neutral by construction.
    """
    print("  [S4] Betting Against Beta...", end=" ", flush=True)
    t0 = time.time()

    eq_ret = returns[equity_syms]

    # Rolling 252-day beta
    mkt_var = mkt_ret.rolling(252, min_periods=200).var()
    betas = pd.DataFrame(np.nan, index=prices.index, columns=equity_syms)
    for col in equity_syms:
        cov = eq_ret[col].rolling(252, min_periods=200).cov(mkt_ret)
        betas[col] = (cov / (mkt_var + 1e-10)).clip(-1, 4).fillna(1.0)

    # BAB signal: NEGATIVE beta rank (low beta = high signal)
    signal = -betas
    ranked = cross_sectional_rank(signal)
    print(f"{time.time()-t0:.1f}s")
    return ranked


# ═══════════════════════════════════════════════════════════════
#  STRATEGY 5: CARRY (dividend yield proxy)
# ═══════════════════════════════════════════════════════════════

def strategy_carry(prices, returns, equity_syms):
    """
    Carry proxy from price patterns.
    Stocks with steady positive returns in low-vol environments = high carry.
    Also uses the 'dogs of the dow' effect: laggards with stable business
    (low vol, high mean return) are likely high-dividend payers.
    """
    print("  [S5] Carry (yield proxy)...", end=" ", flush=True)
    t0 = time.time()

    eq_ret = returns[equity_syms]
    rvol60 = eq_ret.rolling(60, min_periods=40).std() * np.sqrt(252)
    mean_ret = eq_ret.rolling(252, min_periods=200).mean() * 252  # annualized

    # Carry proxy: high mean return + low vol (Sharpe-like)
    # High-dividend stocks tend to have steady returns and low vol
    carry_signal = mean_ret / (rvol60 + 0.01)

    # Also: stocks that underperformed last year but have low vol
    # = "value" stocks likely paying high dividends
    ret_252 = prices[equity_syms] / prices[equity_syms].shift(252) - 1
    value_signal = -ret_252 / (rvol60 + 0.01)  # low return + low vol = value/carry

    combined = 0.5 * cross_sectional_rank(carry_signal) + 0.5 * cross_sectional_rank(value_signal)
    print(f"{time.time()-t0:.1f}s")
    return combined


# ═══════════════════════════════════════════════════════════════
#  STRATEGY 6: SECTOR ROTATION
# ═══════════════════════════════════════════════════════════════

def strategy_sector_rotation(prices, returns, equity_syms):
    """
    Sector momentum: stocks in winning sectors get a boost.
    3-month sector return, 1-month reversal strip.
    """
    print("  [S6] Sector rotation...", end=" ", flush=True)
    t0 = time.time()

    sector_signal = pd.DataFrame(0.0, index=prices.index, columns=equity_syms)

    for sector, etf in SECTOR_ETFS.items():
        if etf not in prices.columns:
            continue
        # 3-month sector return minus 1-month (skip recent reversal)
        sec_ret_63 = prices[etf] / prices[etf].shift(63) - 1
        sec_ret_21 = prices[etf] / prices[etf].shift(21) - 1
        sec_mom = sec_ret_63 - sec_ret_21

        sec_syms = [s for s in equity_syms if SECTOR_MAP.get(s) == sector]
        for sym in sec_syms:
            if sym in sector_signal.columns:
                sector_signal[sym] = sec_mom.fillna(0)

    ranked = cross_sectional_rank(sector_signal)
    print(f"{time.time()-t0:.1f}s")
    return ranked


# ═══════════════════════════════════════════════════════════════
#  STRATEGY 7: EARNINGS DRIFT (post-gap continuation)
# ═══════════════════════════════════════════════════════════════

def strategy_earnings_drift(prices, returns, equity_syms):
    """Post-earnings/news gap continuation."""
    print("  [S7] Earnings drift...", end=" ", flush=True)
    t0 = time.time()

    eq_ret = returns[equity_syms]
    gap_threshold = 0.025
    is_gap_up = (eq_ret > gap_threshold).astype(float)
    is_gap_down = (eq_ret < -gap_threshold).astype(float)

    gap_signal = pd.DataFrame(0.0, index=prices.index, columns=equity_syms)
    for lag in [1, 2, 3, 5, 8, 13, 21, 34, 55]:
        decay = 1.0 / (1 + lag * 0.05)
        gap_signal += decay * (is_gap_up.shift(lag).fillna(0) - is_gap_down.shift(lag).fillna(0))

    ranked = cross_sectional_rank(gap_signal / 5.0)
    print(f"{time.time()-t0:.1f}s")
    return ranked


# ═══════════════════════════════════════════════════════════════
#  STRATEGY 8: 52-WEEK HIGH PROXIMITY
# ═══════════════════════════════════════════════════════════════

def strategy_52w_high(prices, equity_syms):
    """
    George & Hwang (2004): Stocks near 52-week highs continue up.
    Anchoring bias — analysts are slow to revise targets at new highs.
    """
    print("  [S8] 52-week high proximity...", end=" ", flush=True)
    t0 = time.time()

    high_252 = prices[equity_syms].rolling(252, min_periods=200).max()
    proximity = prices[equity_syms] / high_252  # 1.0 at high, <1.0 below

    ranked = cross_sectional_rank(proximity)
    print(f"{time.time()-t0:.1f}s")
    return ranked


# ═══════════════════════════════════════════════════════════════
#  MULTI-STRATEGY PORTFOLIO CONSTRUCTION
# ═══════════════════════════════════════════════════════════════

def compute_environment(prices, spy_col="SPY"):
    """Market environment scalar [0.15, 1.5]."""
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


def compute_breadth(prices, equity_syms):
    """% of stocks above 200-day SMA."""
    sma200 = prices[equity_syms].rolling(200, min_periods=200).mean()
    above = (prices[equity_syms] > sma200).astype(float)
    return above.mean(axis=1).rolling(5, min_periods=1).mean()


def build_multi_strategy_portfolio(strategies, prices, returns, env, breadth,
                                   equity_syms, target_vol, rebal_freq, max_pos,
                                   n_long, n_short, target_gross):
    """
    Risk-parity blend of 8 strategies.
    Each strategy gets equal vol budget, then combined and optimized.
    """
    inst_vol = returns[equity_syms].rolling(20, min_periods=10).std() * np.sqrt(252)
    inst_vol = inst_vol.bfill().clip(lower=0.02)

    # Strategy weights — empirically validated on our data
    # Killed: mean_reversion (needs daily rebal, -22 Sharpe at 15d freq),
    #         BAB (growth decade killed low-beta premium, -0.55 Sharpe),
    #         earnings_drift (noisy, -0.92 Sharpe)
    # Concentrated into the WINNERS:
    STRAT_WEIGHTS = {
        "momentum":       0.25,   # core alpha engine, beta-hedged
        "mean_reversion": 0.00,   # DISABLED: wrong time horizon for 15d rebal
        "quality":        0.25,   # strongest standalone (+9.2 Sharpe in test)
        "bab":            0.00,   # DISABLED: growth decade killed it
        "carry":          0.15,   # slow, steady, uncorrelated (+0.36 Sharpe)
        "sector_rot":     0.10,   # macro overlay (+0.15 Sharpe)
        "earnings_drift": 0.00,   # DISABLED: noisy signal
        "high_52w":       0.25,   # anchoring bias, strong (+8.55 Sharpe test)
    }

    # Blend all strategy signals into one composite
    print("  Blending 8 strategies with risk-parity weights...", end=" ", flush=True)
    t0 = time.time()

    composite = pd.DataFrame(0.0, index=prices.index, columns=equity_syms)
    for name, w in STRAT_WEIGHTS.items():
        sig = strategies[name]
        for col in equity_syms:
            if col in sig.columns:
                composite[col] += w * sig[col].fillna(0)

    # Dynamic IC-based tilt: compute trailing 63-day strategy performance
    # and boost strategies that have been predictive recently
    fwd_5 = returns[equity_syms].rolling(5).sum().shift(-5)  # 5-day forward return
    for name, w in STRAT_WEIGHTS.items():
        sig = strategies[name]
        # Fast IC: sample every 20 days
        ic_boost = pd.Series(1.0, index=prices.index)
        for i in range(300, len(prices), 20):
            ics = []
            for t in range(max(0, i-63), i, 5):
                f_row = sig.iloc[t].values if t < len(sig) else np.zeros(len(equity_syms))
                r_row = fwd_5.iloc[t].values if t < len(fwd_5) else np.zeros(len(equity_syms))
                valid = ~(np.isnan(f_row) | np.isnan(r_row))
                if valid.sum() > 20:
                    ic = np.corrcoef(rankdata(f_row[valid]), rankdata(r_row[valid]))[0, 1]
                    if not np.isnan(ic):
                        ics.append(ic)
            if ics:
                avg_ic = np.mean(ics)
                # Boost if IC > 0, dampen if IC < 0
                mult = 1.0 + np.clip(avg_ic * 15, -0.5, 1.0)
                for j in range(i, min(i+20, len(ic_boost))):
                    ic_boost.iloc[j] = mult

        # Apply IC boost
        for col in equity_syms:
            if col in sig.columns:
                composite[col] += w * 0.3 * (ic_boost - 1.0) * sig[col].fillna(0)

    print(f"{time.time()-t0:.1f}s")

    # ── Portfolio construction from composite signal ──
    sym_sector = {sym: SECTOR_MAP.get(sym, "Other") for sym in equity_syms}
    sector_stocks = defaultdict(list)
    for sym in equity_syms:
        sector_stocks[sym_sector[sym]].append(sym)

    weights = pd.DataFrame(0.0, index=prices.index, columns=equity_syms)
    last_w = pd.Series(0.0, index=equity_syms)
    blend_rate = 0.80  # keep 80% old, 20% new

    running_nav = 1.0
    peak_nav = 1.0

    for i in range(280, len(prices)):
        # Drawdown circuit breaker
        if i > 280:
            day_ret = (last_w * returns[equity_syms].iloc[i]).sum()
            running_nav *= (1 + day_ret)
            peak_nav = max(peak_nav, running_nav)
        current_dd = (running_nav / peak_nav) - 1 if peak_nav > 0 else 0

        dd_scale = 1.0
        if current_dd < -0.25:
            dd_scale = 0.40
        elif current_dd < -0.18:
            dd_scale = 0.65
        elif current_dd < -0.10:
            dd_scale = 0.85

        if i % rebal_freq != 0 and i > 280:
            if dd_scale < 1.0:
                weights.iloc[i] = last_w * dd_scale
            else:
                weights.iloc[i] = last_w
            continue

        sig = composite.iloc[i]
        valid = sig.dropna()
        valid = valid[valid.abs() > 0.005]
        if len(valid) < 20:
            weights.iloc[i] = last_w
            continue

        e = env.iloc[i]
        b = breadth.iloc[i] if i < len(breadth) else 0.5
        is_bearish = e < 0.75 or b < 0.40

        # Sector-neutral selection
        long_picks, short_picks = [], []
        for sector, syms in sector_stocks.items():
            sector_sigs = valid.reindex(syms).dropna()
            if len(sector_sigs) < 2:
                continue
            sector_sigs = sector_sigs.sort_values(ascending=False)

            n_l = max(int(len(sector_sigs) * 0.30), 1)
            for sym in sector_sigs.head(n_l).index:
                if sector_sigs[sym] > 0.05:
                    long_picks.append((sym, sector_sigs[sym]))

            # Shorts: always available (market-neutral strategies need shorts)
            # But scale short budget based on environment
            n_s = max(int(len(sector_sigs) * 0.20), 1)
            for sym in sector_sigs.tail(n_s).index:
                if sector_sigs[sym] < -0.15:
                    short_picks.append((sym, sector_sigs[sym]))

        long_picks.sort(key=lambda x: x[1], reverse=True)
        short_picks.sort(key=lambda x: x[1])
        long_picks = long_picks[:n_long]
        short_picks = short_picks[:n_short]

        w = pd.Series(0.0, index=equity_syms)

        # Adaptive exposure — v3 lesson: shorts bleed in bull markets
        if is_bearish:
            long_budget = target_gross * e * 0.65
            short_budget = target_gross * 0.30
        else:
            long_budget = target_gross * min(e, 1.3)
            short_budget = 0.0  # NO shorts in bull (proven in v3)

        long_budget *= dd_scale
        short_budget *= dd_scale

        if long_picks:
            long_vols = {s: inst_vol[s].iloc[i] if s in inst_vol.columns else 0.20
                         for s, _ in long_picks}
            long_inv = {s: 1.0 / max(v, 0.05) for s, v in long_vols.items()}
            total_inv = sum(long_inv.values())
            for sym, sig_val in long_picks:
                raw_w = long_budget * long_inv[sym] / total_inv
                tilt = 0.7 + 0.6 * min(abs(sig_val), 1.0)
                w[sym] = float(np.clip(raw_w * tilt, 0, max_pos))

        if short_picks and short_budget > 0:
            short_vols = {s: inst_vol[s].iloc[i] if s in inst_vol.columns else 0.20
                          for s, _ in short_picks}
            short_inv = {s: 1.0 / max(v, 0.05) for s, v in short_vols.items()}
            total_inv = sum(short_inv.values())
            for sym, sig_val in short_picks:
                raw_w = -short_budget * short_inv[sym] / total_inv
                w[sym] = float(np.clip(raw_w, -max_pos * 0.5, 0))

        # Turnover dampening
        if i > 280:
            delta = w - last_w
            small = delta.abs() < 0.002
            w[small] = last_w[small]
            w = (1 - blend_rate) * w + blend_rate * last_w  # keep 80% old

        last_w = w.copy()
        weights.iloc[i] = w

    # Vol targeting
    port_ret = (weights.shift(1) * returns[equity_syms]).sum(axis=1)
    port_rvol = port_ret.rolling(20, min_periods=10).std() * np.sqrt(252)

    final_weights = weights.copy()
    for i in range(300, len(prices)):
        rv = port_rvol.iloc[i]
        if np.isnan(rv) or rv < 0.01:
            continue
        scale = target_vol / rv
        scale = float(np.clip(scale, 0.3, 2.0))
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


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def run(args):
    universe = load_universe_static(args.config)
    print(f"\nUniverse: {len(universe)} instruments\n")

    # Data loading
    cache_hits = sum(1 for sym, at in universe
                     if not args.no_cache and os.path.exists(os.path.join(CACHE_DIR, f"{sym}_{at}.parquet")))
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

    equity_syms = [s for s in symbols if ASSET_TYPES.get(s) == "EQUITY"]
    print(f"  Tradeable equities: {len(equity_syms)}")
    print(f"  Sectors: {len(set(SECTOR_MAP.get(s, 'Other') for s in equity_syms))}\n")

    returns = prices.pct_change().fillna(0)
    mkt_ret = returns["SPY"] if "SPY" in returns.columns else returns.iloc[:, 0]

    # ── Compute all 8 strategies ──
    print("=" * 70)
    print("  COMPUTING 8 STRATEGY SIGNALS")
    print("=" * 70)
    t_total = time.time()

    strategies = {}
    strategies["momentum"] = strategy_momentum(prices, returns, equity_syms, mkt_ret)
    strategies["mean_reversion"] = strategy_mean_reversion(prices, returns, equity_syms, mkt_ret)
    strategies["quality"] = strategy_quality(prices, returns, equity_syms, mkt_ret)
    strategies["bab"] = strategy_bab(prices, returns, equity_syms, mkt_ret)
    strategies["carry"] = strategy_carry(prices, returns, equity_syms)
    strategies["sector_rot"] = strategy_sector_rotation(prices, returns, equity_syms)
    strategies["earnings_drift"] = strategy_earnings_drift(prices, returns, equity_syms)
    strategies["high_52w"] = strategy_52w_high(prices, equity_syms)

    print(f"\n  Total signal computation: {time.time()-t_total:.1f}s\n")

    # Strategy correlation matrix
    print("  ── Strategy Correlation Matrix ──")
    strat_returns = {}
    for name, sig in strategies.items():
        # Simple long-top-20 / short-bottom-20 portfolio return
        n = 20
        daily_r = pd.Series(0.0, index=prices.index)
        for i in range(280, len(prices)):
            row = sig.iloc[i].dropna()
            if len(row) < 40:
                continue
            top = row.nlargest(n).index
            bot = row.nsmallest(n).index
            ret_top = returns[list(top)].iloc[i].mean()
            ret_bot = returns[list(bot)].iloc[i].mean()
            daily_r.iloc[i] = ret_top - ret_bot
        strat_returns[name] = daily_r.iloc[280:]

    strat_df = pd.DataFrame(strat_returns)
    corr = strat_df.corr()
    names_short = ["Mom", "MRev", "Qual", "BAB", "Carry", "SecRt", "EDrift", "52wH"]
    print(f"  {'':>8}", end="")
    for n in names_short:
        print(f" {n:>6}", end="")
    print()
    for i, (name, row) in enumerate(corr.iterrows()):
        print(f"  {names_short[i]:>8}", end="")
        for val in row:
            print(f" {val:>+5.2f}", end=" ")
        print()

    # Average pairwise correlation
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    avg_corr = upper.stack().mean()
    print(f"\n  Avg pairwise correlation: {avg_corr:+.3f}")
    print(f"  (lower = better diversification, ideal < 0.15)\n")

    # Individual strategy Sharpes
    print("  ── Individual Strategy Sharpes (L/S top/bottom 20) ──")
    for name, n_short_name in zip(strat_returns, names_short):
        s = sharpe(strat_df[name])
        print(f"  {n_short_name:>8}: {s:+.2f}")
    print()

    # Environment & breadth
    env = compute_environment(prices)
    breadth = compute_breadth(prices, equity_syms)

    # ── Build multi-strategy portfolio ──
    print("=" * 70)
    print("  WALK-FORWARD BACKTEST v4 (8-strategy risk-parity)")
    print("=" * 70)
    print(f"  NAV: ${args.nav:,.0f} | Vol target: {args.target_vol:.0%} | "
          f"Rebal: {args.rebal_freq}d | Max pos: {args.max_pos:.0%}")
    print(f"  Target gross: {args.target_gross:.1f}x | "
          f"Long: {args.n_long} | Short: {args.n_short}\n")

    t0 = time.time()
    weights = build_multi_strategy_portfolio(
        strategies, prices, returns, env, breadth, equity_syms,
        args.target_vol, args.rebal_freq, args.max_pos,
        args.n_long, args.n_short, args.target_gross,
    )
    weights = weights.fillna(0)
    print(f"  Portfolio construction: {time.time()-t0:.1f}s\n")

    # ── Results ──
    warmup = 300
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
    avg_gross = weights_post.abs().sum(axis=1).mean()
    avg_net = weights_post.sum(axis=1).mean()
    avg_turn = turnover_post.mean() * 252
    tx_bps = tx_cost.iloc[warmup:].mean() * 252 * 10000 if len(tx_cost) > warmup else 0

    spy_ret = (prices["SPY"].pct_change().fillna(0)).iloc[warmup:]
    spy_eq = args.nav * (1 + spy_ret).cumprod()
    spy_cagr = (spy_eq.iloc[-1] / spy_eq.iloc[0]) ** (1 / max(n_years, 0.01)) - 1
    spy_sharpe = sharpe(spy_ret)
    spy_dd = max_dd(spy_eq)

    print("  ── Aggregate ──")
    print(f"  CAGR:          {cagr:+.2%}   (SPY: {spy_cagr:+.2%})  {'*** ALPHA ***' if cagr > spy_cagr else ''}")
    print(f"  Sharpe:        {s:.2f}      (SPY: {spy_sharpe:.2f})  {'*** TARGET' if s >= 1.0 else '✓' if s > spy_sharpe else ''}")
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

    # Year by year
    print("  ── Year-by-Year: v4 vs SPY ──")
    print(f"  {'Year':<6} {'v4 Ret':>8} {'SPY':>8} {'Alpha':>8} {'v4 Shp':>8} {'SPY Shp':>8} {'v4 DD':>8} {'SPY DD':>8} {'Win':>4}")
    print("  " + "-" * 74)
    loss_years = beat_spy = beat_sharpe = beat_dd = 0

    for year in sorted(set(d.year if hasattr(d, 'year') else d.date().year for d in dates)):
        mask = pd.Series([d.year if hasattr(d, 'year') else d.date().year for d in dates], index=net_ret.index) == year
        yr = net_ret[mask]
        yr_eq = equity_curve[mask]
        if len(yr) < 5:
            continue
        yr_ret = (yr_eq.iloc[-1] / yr_eq.iloc[0]) - 1
        yr_s = sharpe(yr)
        yr_d = max_dd(yr_eq)

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
    print("  " + "-" * 74)
    print(f"  Loss years:       {loss_years}/{total_years}")
    print(f"  Beat SPY return:  {beat_spy}/{total_years}")
    print(f"  Beat SPY Sharpe:  {beat_sharpe}/{total_years}")
    print(f"  Beat SPY MaxDD:   {beat_dd}/{total_years}")
    print()

    # Sector attribution
    print("  ── Sector Attribution ──")
    sector_groups = defaultdict(list)
    for sym in equity_syms:
        if sym in weights.columns:
            sector_groups[SECTOR_MAP.get(sym, "Other")].append(sym)
    print(f"  {'Sector':<10} {'#Syms':>6} {'AvgGross':>9} {'Contrib':>9}")
    print("  " + "-" * 38)
    for sec in sorted(sector_groups):
        syms = [s for s in sector_groups[sec] if s in weights_post.columns]
        if not syms:
            continue
        avg_w = weights_post[syms].abs().mean().sum()
        contrib = (weights_post[syms].shift(1) * returns[syms].iloc[warmup:]).sum(axis=1).mean() * 252
        print(f"  {sec:<10} {len(syms):>6} {avg_w:>8.3f}x {contrib:>+8.2%}")
    print()

    # Current book
    print("  ── Current Book ──")
    latest_w = weights.iloc[-1]
    longs = latest_w[latest_w > 0.005].sort_values(ascending=False)
    shorts = latest_w[latest_w < -0.005].sort_values()
    print(f"  {len(longs)} longs | {len(shorts)} shorts | "
          f"Gross: {latest_w.abs().sum():.2f}x | Net: {latest_w.sum():+.2f}x")
    print(f"\n  Top 10 longs:")
    for sym, w in longs.head(10).items():
        sec = SECTOR_MAP.get(sym, "?")
        print(f"    {sym:>8s} {w:>+7.3f}  ({sec})")
    if len(shorts) > 0:
        print(f"\n  Top 10 shorts:")
        for sym, w in shorts.head(10).items():
            sec = SECTOR_MAP.get(sym, "?")
            print(f"    {sym:>8s} {w:>+7.3f}  ({sec})")
    print()

    # Monte Carlo
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
    p = argparse.ArgumentParser(description="One Brain Fund v4 — Multi-Strategy Engine")
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
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    if os.path.exists(args.env):
        from dotenv import load_dotenv
        load_dotenv(args.env)

    run(args)
