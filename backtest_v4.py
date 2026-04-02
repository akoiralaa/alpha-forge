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
from src.portfolio.allocator import CentralRiskAllocator, StrategyExpectation
from src.portfolio.capacity import LiquidityCapacityModel, LiquiditySnapshot
from src.regime.tracker import RegimeTracker


ASSET_TYPES = {}
CACHE_DIR = os.path.expanduser("~/.one_brain_fund/cache/bars")

_ASSET_CLASS_MAP = {
    "ETF": AssetClass.ETF, "EQUITY": AssetClass.EQUITY,
    "FUTURE": AssetClass.FUTURE, "COMMODITY": AssetClass.COMMODITY,
    "BOND": AssetClass.BOND, "FX": AssetClass.FX,
    "VOLATILITY": AssetClass.VOLATILITY,
}

MIN_BARS_BY_TYPE = {
    "ETF": 756,
    "EQUITY": 756,
    "FUTURE": 252,
    "COMMODITY": 252,
    "BOND": 252,
    "FX": 252,
    "VOLATILITY": 252,
}

MAX_STALE_DAYS_BY_TYPE = {
    "ETF": 45,
    "EQUITY": 45,
    "FUTURE": 10,
    "COMMODITY": 10,
    "BOND": 10,
    "FX": 10,
    "VOLATILITY": 10,
}

CORE_STRATEGY_BASE_WEIGHTS = {
    "momentum": 0.32,
    "quality": 0.36,
    "carry": 0.06,
    "sector_rot": 0.14,
    "high_52w": 0.12,
}

FAST_MREV_REBAL_FREQ = 5
FAST_MREV_TARGET_GROSS = 0.08
FAST_MREV_MAX_POS = 0.010

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


def default_core_strategy_weights():
    return dict(CORE_STRATEGY_BASE_WEIGHTS)


def filter_histories_for_backtest(
    all_data,
    universe,
    min_bars_by_type=None,
    max_stale_days_by_type=None,
):
    min_bars_by_type = min_bars_by_type or MIN_BARS_BY_TYPE
    max_stale_days_by_type = max_stale_days_by_type or MAX_STALE_DAYS_BY_TYPE

    non_empty = {
        sym: df
        for sym, df in all_data.items()
        if df is not None and not df.empty and "date" in df.columns
    }
    if not non_empty:
        return {}, [], None

    latest_date = max(pd.to_datetime(df["date"], utc=True).max() for df in non_empty.values())
    filtered = {}
    dropped = []

    for sym, atype in universe:
        df = non_empty.get(sym)
        if df is None:
            continue

        n_bars = len(df)
        end_dt = pd.to_datetime(df["date"], utc=True).max()
        stale_days = int((latest_date - end_dt).days)
        min_bars = int(min_bars_by_type.get(atype, 252))
        max_stale = int(max_stale_days_by_type.get(atype, 45))

        if n_bars < min_bars:
            dropped.append((sym, atype, "short_history", n_bars, str(end_dt.date())))
            continue
        if stale_days > max_stale:
            dropped.append((sym, atype, "stale_history", n_bars, str(end_dt.date())))
            continue

        filtered[sym] = df

    return filtered, dropped, latest_date


def compute_strategy_test_returns(strategies, returns, equity_syms, warmup=280, top_n=20, bottom_n=20):
    strat_returns = {}
    for name, sig in strategies.items():
        aligned = sig.reindex(columns=equity_syms).fillna(0.0)
        daily_r = pd.Series(0.0, index=returns.index, dtype=float)
        for i in range(warmup, len(returns) - 1):
            row = aligned.iloc[i].dropna()
            if len(row) < (top_n + bottom_n):
                continue
            top = row.nlargest(top_n).index
            bot = row.nsmallest(bottom_n).index
            ret_top = returns[top].iloc[i + 1].mean()
            ret_bot = returns[bot].iloc[i + 1].mean()
            daily_r.iloc[i + 1] = ret_top - ret_bot
        strat_returns[name] = daily_r.iloc[warmup:]
    return pd.DataFrame(strat_returns)


def build_strategy_evidence_multipliers(strategy_returns, target_index):
    if strategy_returns.empty:
        return pd.DataFrame(1.0, index=target_index, columns=[])

    multipliers = pd.DataFrame(1.0, index=target_index, columns=strategy_returns.columns)
    for name in strategy_returns.columns:
        r = strategy_returns[name].reindex(target_index).fillna(0.0)
        trailing_mean = r.rolling(126, min_periods=42).mean()
        trailing_vol = r.rolling(126, min_periods=42).std()
        trailing_sharpe = (trailing_mean / (trailing_vol + 1e-12)) * np.sqrt(252)
        hit_rate = (r > 0).astype(float).rolling(63, min_periods=30).mean()

        mult = 0.80 + 0.20 * trailing_sharpe.clip(-1.5, 2.5) + 0.70 * (hit_rate - 0.50)
        mult = mult.clip(0.05, 1.50)

        weak = (trailing_sharpe < -0.10) & (hit_rate < 0.48)
        mult[weak] = mult[weak] * 0.35
        multipliers[name] = mult.shift(1).fillna(1.0)

    return multipliers.fillna(1.0)


def build_high_52w_kill_switch(target_index, regime_info, strategy_returns):
    """
    Disable or damp the 52w-high sleeve in regimes where it historically adds
    churn without enough edge.
    """
    gate = pd.Series(1.0, index=target_index, dtype=float)

    if regime_info is not None and not regime_info.empty:
        labels = regime_info["regime_label"].reindex(target_index).fillna("WARMUP")
        confidence = regime_info["confidence"].reindex(target_index).fillna(0.0).clip(0.0, 1.0)
        signal_gate = regime_info["signal_gate"].reindex(target_index).fillna(True).astype(bool)

        hard_off = (~signal_gate) | labels.isin({"HIGH_VOL_CHAOTIC", "LIQUIDITY_CRISIS"})
        gate.loc[hard_off] = 0.0

        medium_off = (labels == "HIGH_VOL_TRENDING") & (confidence >= 0.55)
        gate.loc[medium_off] = np.minimum(gate.loc[medium_off], 0.35)

        low_conf = confidence < 0.35
        gate.loc[low_conf] = np.minimum(gate.loc[low_conf], 0.55)

    if strategy_returns is not None and "high_52w" in strategy_returns.columns:
        r = strategy_returns["high_52w"].reindex(target_index).fillna(0.0)
        trailing_sharpe = (
            r.rolling(126, min_periods=42).mean()
            / (r.rolling(126, min_periods=42).std() + 1e-12)
        ) * np.sqrt(252)
        weak = (trailing_sharpe < -0.15) & (trailing_sharpe >= -0.35)
        very_weak = trailing_sharpe < -0.35
        gate.loc[weak] = np.minimum(gate.loc[weak], 0.25)
        gate.loc[very_weak] = 0.0

    return gate.clip(0.0, 1.0).ffill().fillna(1.0)


def build_strategy_signal_weights_for_capacity(signal_row, gross_weight, top_k=80):
    signal_row = signal_row.replace([np.inf, -np.inf], np.nan).dropna()
    if signal_row.empty or abs(gross_weight) < 1e-12:
        return {}
    signal_row = signal_row[signal_row.abs() > 0.02]
    if signal_row.empty:
        return {}
    signal_row = signal_row.reindex(signal_row.abs().sort_values(ascending=False).index).iloc[:top_k]
    denom = signal_row.abs().sum()
    if denom < 1e-12:
        return {}
    return {sym: float(gross_weight * val / denom) for sym, val in signal_row.items()}


def build_liquidity_snapshot(
    symbol_id,
    symbol,
    i,
    prices,
    adv_usd_frame,
    daily_vol_frame,
    participation_limit=0.05,
    impact_limit_bps=15.0,
    max_spread_bps=14.0,
):
    price = float(prices[symbol].iloc[i]) if symbol in prices.columns else 0.0
    if not np.isfinite(price) or price <= 0:
        price = 1.0

    adv_usd = 5_000_000.0
    if adv_usd_frame is not None and symbol in adv_usd_frame.columns:
        adv_val = float(adv_usd_frame[symbol].iloc[i])
        if np.isfinite(adv_val) and adv_val > 0:
            adv_usd = adv_val

    realized_vol = 0.02
    if daily_vol_frame is not None and symbol in daily_vol_frame.columns:
        vol_val = float(daily_vol_frame[symbol].iloc[i])
        if np.isfinite(vol_val) and vol_val > 0:
            realized_vol = vol_val

    spread_bps = float(np.clip(1.8 + 180.0 * realized_vol, 1.2, 40.0))
    return LiquiditySnapshot(
        symbol_id=int(symbol_id),
        price=price,
        adv_usd=adv_usd,
        spread_bps=spread_bps,
        realized_vol_daily=max(realized_vol, 1e-4),
        participation_limit=float(participation_limit),
        impact_limit_bps=float(impact_limit_bps),
        max_spread_bps=float(max_spread_bps),
    )


def build_dynamic_allocator_scales(
    active_strategies,
    base_weights,
    strategy_frames,
    strategy_returns,
    prices,
    returns,
    equity_syms,
    target_gross,
    rebal_freq,
    nav_usd,
    regime_info=None,
    volumes=None,
    capacity_impact_k=0.45,
    capacity_participation_limit=0.05,
    capacity_impact_limit_bps=15.0,
    capacity_max_spread_bps=14.0,
):
    scales = pd.DataFrame(1.0, index=prices.index, columns=active_strategies, dtype=float)
    diagnostics = {
        "allocator_enabled": False,
        "allocator_capacity_updates": 0,
        "allocator_weight_entropy": np.nan,
    }
    if len(active_strategies) <= 1:
        return scales, diagnostics

    expectations = []
    for name in active_strategies:
        r = strategy_returns[name].dropna() if name in strategy_returns.columns else pd.Series(dtype=float)
        ann_ret = float(r.mean() * 252) if len(r) > 20 else 0.08
        ann_vol = float(r.std() * np.sqrt(252)) if len(r) > 20 else 0.18
        expected_return = float(np.clip(max(ann_ret, 0.04), 0.04, 0.30))
        expected_vol = float(np.clip(ann_vol, 0.08, 0.55))
        max_weight = 0.55 if name != "high_52w" else 0.30
        expectations.append(
            StrategyExpectation(
                name=name,
                expected_return_annual=expected_return,
                expected_vol_annual=expected_vol,
                base_weight=float(max(base_weights.get(name, 0.0), 1e-6)),
                min_weight=0.01,
                max_weight=max_weight,
                drawdown_limit=0.12,
                hard_stop_drawdown=0.25,
            )
        )

    allocator = CentralRiskAllocator(
        expectations=expectations,
        ewma_decay=0.98,
        min_observations=126,
        exploration_floor=0.60,
        capacity_soft_limit=0.75,
        score_temperature=0.55,
    )
    capacity_model = LiquidityCapacityModel(impact_k=capacity_impact_k, min_adv_usd=1_000_000.0)
    diagnostics["allocator_enabled"] = True

    base_total = sum(max(base_weights.get(name, 0.0), 0.0) for name in active_strategies)
    if base_total <= 0:
        base_norm = {name: 1.0 / len(active_strategies) for name in active_strategies}
    else:
        base_norm = {name: max(base_weights.get(name, 0.0), 0.0) / base_total for name in active_strategies}

    standalone = strategy_returns.reindex(prices.index).fillna(0.0)
    symbol_to_id = {sym: idx + 1 for idx, sym in enumerate(equity_syms)}
    daily_vol = returns[equity_syms].rolling(20, min_periods=10).std().fillna(0.02)
    adv_usd = None
    if volumes is not None and not volumes.empty:
        adv_usd = (volumes[equity_syms].rolling(20, min_periods=5).mean() * prices[equity_syms]).fillna(0.0)

    for i in range(280, len(prices)):
        dt = prices.index[i]
        if i > 280:
            prev_dt = prices.index[i - 1]
            for name in active_strategies:
                allocator.observe(name, float(standalone.at[prev_dt, name]))

        if i % rebal_freq == 0:
            alloc_w = allocator.target_weights()
            for name in active_strategies:
                strat_gross = float(target_gross * alloc_w.get(name, 0.0))
                if strat_gross <= 1e-8:
                    continue
                sig_row = strategy_frames[name].iloc[i]
                symbol_weights = build_strategy_signal_weights_for_capacity(sig_row, strat_gross, top_k=70)
                if not symbol_weights:
                    continue
                snapshots = {}
                id_weights = {}
                for sym, weight in symbol_weights.items():
                    sid = symbol_to_id.get(sym)
                    if sid is None:
                        continue
                    snapshots[sid] = build_liquidity_snapshot(
                        sid,
                        sym,
                        i,
                        prices,
                        adv_usd,
                        daily_vol,
                        participation_limit=capacity_participation_limit,
                        impact_limit_bps=capacity_impact_limit_bps,
                        max_spread_bps=capacity_max_spread_bps,
                    )
                    id_weights[sid] = weight
                if not id_weights:
                    continue
                estimate = capacity_model.estimate_strategy_capacity(
                    strategy_name=name,
                    symbol_weights=id_weights,
                    snapshots=snapshots,
                    nav_usd=float(max(nav_usd, 1_000_000.0)),
                    turnover=max(1.0 / max(rebal_freq, 1), 0.05),
                )
                allocator.observe_capacity(
                    name,
                    utilization=estimate.utilization,
                    capacity_nav_limit=estimate.nav_capacity_usd,
                    impact_bps=estimate.weighted_impact_bps,
                )
                diagnostics["allocator_capacity_updates"] += 1

        alloc_w = allocator.target_weights()
        for name in active_strategies:
            base_w = max(base_norm.get(name, 1.0 / len(active_strategies)), 1e-6)
            scales.loc[dt, name] = float(np.clip(alloc_w.get(name, base_w) / base_w, 0.70, 1.30))

    final_weights = allocator.target_weights()
    entropy = -sum(w * np.log(max(w, 1e-12)) for w in final_weights.values())
    diagnostics["allocator_weight_entropy"] = float(entropy)
    return scales.ffill().fillna(1.0), diagnostics


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
    breadth = compute_breadth(prices, equity_syms)

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

    We use regimes for two things only:
    1. tilting multi-strategy weights toward the signals that fit the tape
    2. allowing shorts only when the market is hostile enough to justify them
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
            and state.regime_label in {"HIGH_VOL_TRENDING", "HIGH_VOL_CHAOTIC", "LIQUIDITY_CRISIS"}
        )

        regime_info.loc[dt, "regime_label"] = state.regime_label
        regime_info.loc[dt, "position_scale"] = state.position_scale
        regime_info.loc[dt, "signal_gate"] = state.signal_gate_open
        regime_info.loc[dt, "allow_shorts"] = allow_shorts
        regime_info.loc[dt, "confidence"] = state.confidence

    return regime_info.ffill()


def apply_hedge_fund_fees(returns, management_fee=0.02, incentive_fee=0.20):
    """
    Approximate standard 2-and-20 LP economics.

    Management fee accrues daily. Incentive fee is crystallized on the last
    trading day of each calendar year on gains above the running high-water mark.
    """
    if len(returns) == 0:
        return returns.copy()

    lp_ret = returns.astype(float).copy()
    if management_fee:
        lp_ret -= management_fee / 252.0

    nav = 1.0
    high_water = 1.0
    year_index = pd.Series(lp_ret.index.year, index=lp_ret.index)

    for year in sorted(year_index.unique()):
        yr_idx = lp_ret.index[year_index == year]
        if len(yr_idx) == 0:
            continue

        pre_fee_eq = nav * (1 + lp_ret.loc[yr_idx]).cumprod()
        pre_fee_end = float(pre_fee_eq.iloc[-1])

        if incentive_fee > 0 and pre_fee_end > high_water:
            incentive_charge = incentive_fee * (pre_fee_end - high_water)
            last_nav_before_fee = float(pre_fee_eq.iloc[-2]) if len(pre_fee_eq) > 1 else nav
            if last_nav_before_fee > 0:
                lp_ret.loc[yr_idx[-1]] -= incentive_charge / last_nav_before_fee
            pre_fee_end -= incentive_charge

        nav = pre_fee_end
        high_water = max(high_water, nav)

    return lp_ret


def compute_crisis_overlay(env_value, breadth_value, regime_row, current_dd, preemptive_de_risk):
    """
    Convert market stress into a mild pre-emptive de-risking signal.

    Unlike the hard drawdown circuit breaker, this aims to trim risk before the
    portfolio is already deep in a hole. It should be gentle enough to preserve
    most upside in normal tapes.
    """
    crisis_score = 0.0

    if regime_row is not None:
        label = regime_row.get("regime_label", "WARMUP")
        confidence = float(np.clip(regime_row.get("confidence", 0.0), 0.0, 1.0))
        position_scale = float(np.clip(regime_row.get("position_scale", 1.0), 0.30, 1.20))

        if label == "LIQUIDITY_CRISIS":
            crisis_score += 0.55 + 0.25 * confidence
        elif label == "HIGH_VOL_CHAOTIC":
            crisis_score += 0.35 + 0.25 * confidence
        elif label == "HIGH_VOL_TRENDING":
            crisis_score += 0.18 + 0.20 * confidence

        crisis_score += np.clip((0.95 - position_scale) / 0.65, 0.0, 0.25)

    crisis_score += np.clip((0.45 - breadth_value) / 0.20, 0.0, 0.30)
    crisis_score += np.clip((0.72 - env_value) / 0.35, 0.0, 0.25)

    if current_dd < -0.06:
        crisis_score += np.clip((abs(current_dd) - 0.06) / 0.16, 0.0, 0.20)

    crisis_score = float(np.clip(crisis_score, 0.0, 1.0))
    proactive_scale = float(np.clip(1.0 - preemptive_de_risk * crisis_score, 0.82, 1.0))
    return crisis_score, proactive_scale


def build_fast_mean_reversion_overlay(
    mean_reversion_signal,
    prices,
    returns,
    equity_syms,
    regime_info=None,
    target_gross=FAST_MREV_TARGET_GROSS,
    rebal_freq=FAST_MREV_REBAL_FREQ,
    max_pos=FAST_MREV_MAX_POS,
):
    overlay = pd.DataFrame(0.0, index=prices.index, columns=equity_syms)
    if mean_reversion_signal is None or mean_reversion_signal.empty:
        return overlay

    inst_vol = returns[equity_syms].rolling(20, min_periods=10).std() * np.sqrt(252)
    inst_vol = inst_vol.bfill().clip(lower=0.05)
    live_row = pd.Series(0.0, index=equity_syms)

    for i in range(60, len(prices)):
        if regime_info is not None:
            regime_row = regime_info.iloc[i]
            label = regime_row.get("regime_label", "WARMUP")
            confidence = float(np.clip(regime_row.get("confidence", 0.0), 0.0, 1.0))
            signal_gate = bool(regime_row.get("signal_gate", True))
        else:
            label = "MEAN_REVERTING_RANGE"
            confidence = 0.5
            signal_gate = True

        overlay_active = (
            signal_gate
            and confidence >= 0.35
            and label in {"MEAN_REVERTING_RANGE", "HIGH_VOL_CHAOTIC"}
        )

        if (i % rebal_freq != 0 and i > 60) or not overlay_active:
            if not overlay_active:
                live_row = pd.Series(0.0, index=equity_syms)
            overlay.iloc[i] = live_row
            continue

        sig = mean_reversion_signal.iloc[i].dropna()
        sig = sig[sig.abs() > 0.20]
        if len(sig) < 16:
            live_row = pd.Series(0.0, index=equity_syms)
            overlay.iloc[i] = live_row
            continue

        long_picks = list(sig.nlargest(8).items())
        short_picks = list(sig.nsmallest(8).items())
        budget = target_gross * (0.75 + 0.25 * confidence)
        long_budget = budget * 0.50
        short_budget = budget * 0.50

        w = pd.Series(0.0, index=equity_syms)
        long_inv = {
            sym: 1.0 / max(float(inst_vol[sym].iloc[i]), 0.05)
            for sym, _ in long_picks
        }
        short_inv = {
            sym: 1.0 / max(float(inst_vol[sym].iloc[i]), 0.05)
            for sym, _ in short_picks
        }
        long_total = sum(long_inv.values()) or 1.0
        short_total = sum(short_inv.values()) or 1.0

        for sym, sig_val in long_picks:
            tilt = 0.75 + 0.50 * min(abs(float(sig_val)), 1.0)
            w[sym] = float(np.clip(long_budget * long_inv[sym] / long_total * tilt, 0.0, max_pos))
        for sym, sig_val in short_picks:
            tilt = 0.75 + 0.50 * min(abs(float(sig_val)), 1.0)
            w[sym] = float(np.clip(-short_budget * short_inv[sym] / short_total * tilt, -max_pos, 0.0))

        live_row = w
        overlay.iloc[i] = live_row

    return overlay


def build_multi_strategy_portfolio(strategies, prices, returns, env, breadth,
                                   equity_syms, target_vol, rebal_freq, max_pos,
                                   n_long, n_short, target_gross, regime_info=None,
                                   hedge_symbol="SPY", crisis_hedge_max=0.35,
                                   crisis_hedge_strength=0.75, crisis_beta_floor=0.15,
                                   preemptive_de_risk=0.18, hedge_lookback=63,
                                   strategy_weights=None,
                                   volumes=None, nav_usd=10_000_000,
                                   use_dynamic_allocator=True,
                                   use_capacity_constraints=True,
                                   capacity_impact_k=0.45,
                                   capacity_participation_limit=0.05,
                                   capacity_impact_limit_bps=15.0,
                                   capacity_max_spread_bps=14.0):
    """
    Risk-parity blend of 8 strategies.
    Each strategy gets equal vol budget, then combined and optimized.
    """
    hedge_available = hedge_symbol in prices.columns and hedge_symbol in returns.columns
    trade_cols = list(equity_syms)
    if hedge_available and hedge_symbol not in trade_cols:
        trade_cols.append(hedge_symbol)

    inst_vol = returns[equity_syms].rolling(20, min_periods=10).std() * np.sqrt(252)
    inst_vol = inst_vol.bfill().clip(lower=0.02)
    mkt_ret = returns[hedge_symbol] if hedge_available else returns.iloc[:, 0]
    mkt_var = mkt_ret.rolling(hedge_lookback, min_periods=max(20, hedge_lookback // 2)).var()
    rolling_beta = pd.DataFrame(1.0, index=prices.index, columns=equity_syms)
    for col in equity_syms:
        cov = returns[col].rolling(hedge_lookback, min_periods=max(20, hedge_lookback // 2)).cov(mkt_ret)
        rolling_beta[col] = (cov / (mkt_var + 1e-10)).clip(-1.5, 3.0).fillna(1.0)

    # Core sleeves stay medium-horizon; fast mean reversion is handled separately
    # so it does not pollute the slower 15-day book.
    default_strategy_weights = {
        "momentum":       CORE_STRATEGY_BASE_WEIGHTS["momentum"],
        "mean_reversion": 0.00,
        "quality":        CORE_STRATEGY_BASE_WEIGHTS["quality"],
        "bab":            0.00,
        "carry":          CORE_STRATEGY_BASE_WEIGHTS["carry"],
        "sector_rot":     CORE_STRATEGY_BASE_WEIGHTS["sector_rot"],
        "earnings_drift": 0.00,
        "high_52w":       CORE_STRATEGY_BASE_WEIGHTS["high_52w"],
    }
    STRAT_WEIGHTS = dict(default_strategy_weights)
    if strategy_weights is not None:
        for name, value in strategy_weights.items():
            if name in STRAT_WEIGHTS:
                STRAT_WEIGHTS[name] = float(value)

    REGIME_TILTS = {
        "LOW_VOL_TRENDING": {
            "momentum": 1.15,
            "quality": 0.95,
            "carry": 0.90,
            "sector_rot": 1.00,
            "high_52w": 1.20,
        },
        "MEAN_REVERTING_RANGE": {
            "momentum": 0.90,
            "quality": 1.05,
            "carry": 1.00,
            "sector_rot": 1.15,
            "high_52w": 0.85,
        },
        "HIGH_VOL_TRENDING": {
            "momentum": 1.00,
            "quality": 1.10,
            "carry": 1.05,
            "sector_rot": 1.05,
            "high_52w": 0.90,
        },
        "HIGH_VOL_CHAOTIC": {
            "momentum": 0.80,
            "quality": 1.20,
            "carry": 1.10,
            "sector_rot": 0.95,
            "high_52w": 0.75,
        },
        "LIQUIDITY_CRISIS": {
            "momentum": 0.75,
            "quality": 1.25,
            "carry": 1.10,
            "sector_rot": 1.05,
            "high_52w": 0.65,
        },
    }

    # Blend all strategy signals into one composite
    active_strategies = [
        name for name, base in STRAT_WEIGHTS.items()
        if abs(base) > 1e-12 and name in strategies
    ]
    if not active_strategies:
        active_strategies = list(strategies.keys())

    print(f"  Blending {len(active_strategies)} active strategies with risk-parity weights...", end=" ", flush=True)
    t0 = time.time()

    composite = pd.DataFrame(0.0, index=prices.index, columns=equity_syms)
    strategy_frames = {
        name: strategies[name].reindex(columns=equity_syms).fillna(0.0)
        for name in active_strategies
    }

    strategy_scale = pd.DataFrame(1.0, index=prices.index, columns=active_strategies)
    if regime_info is not None:
        regime_labels = regime_info["regime_label"].reindex(prices.index).fillna("WARMUP")
        confidences = regime_info["confidence"].reindex(prices.index).fillna(0.0).clip(lower=0.0, upper=1.0)
        for label, tilt_map in REGIME_TILTS.items():
            mask = regime_labels == label
            if not mask.any():
                continue
            intensity = 0.50 + 0.50 * confidences.loc[mask]
            for strat_name, multiplier in tilt_map.items():
                if strat_name in strategy_scale.columns:
                    strategy_scale.loc[mask, strat_name] = 1.0 + (multiplier - 1.0) * intensity

    standalone_returns = compute_strategy_test_returns(
        {name: strategy_frames[name] for name in active_strategies},
        returns,
        equity_syms,
    )
    evidence_scale = build_strategy_evidence_multipliers(standalone_returns, prices.index)

    allocator_scale = pd.DataFrame(1.0, index=prices.index, columns=active_strategies, dtype=float)
    allocator_diag = {
        "allocator_enabled": False,
        "allocator_capacity_updates": 0,
        "allocator_weight_entropy": np.nan,
    }
    if use_dynamic_allocator:
        allocator_scale, allocator_diag = build_dynamic_allocator_scales(
            active_strategies=active_strategies,
            base_weights=STRAT_WEIGHTS,
            strategy_frames=strategy_frames,
            strategy_returns=standalone_returns,
            prices=prices,
            returns=returns,
            equity_syms=equity_syms,
            target_gross=target_gross,
            rebal_freq=rebal_freq,
            nav_usd=nav_usd,
            regime_info=regime_info,
            volumes=volumes,
            capacity_impact_k=capacity_impact_k,
            capacity_participation_limit=capacity_participation_limit,
            capacity_impact_limit_bps=capacity_impact_limit_bps,
            capacity_max_spread_bps=capacity_max_spread_bps,
        )

    effective_weights = pd.DataFrame(
        {
            name: STRAT_WEIGHTS[name]
            * strategy_scale[name]
            * evidence_scale.get(name, 1.0)
            * allocator_scale.get(name, 1.0)
            for name in active_strategies
        },
        index=prices.index,
    )

    high_52w_active_pct = 1.0
    if "high_52w" in effective_weights.columns:
        high_52w_gate = build_high_52w_kill_switch(prices.index, regime_info, standalone_returns)
        effective_weights["high_52w"] = effective_weights["high_52w"] * high_52w_gate
        high_52w_active_pct = float((high_52w_gate > 0.01).mean())

    weight_norm = effective_weights.sum(axis=1).replace(0, 1.0)

    for name in active_strategies:
        sig = strategy_frames[name]
        w = effective_weights[name].div(weight_norm)
        composite += sig.mul(w, axis=0)

    # Dynamic IC-based tilt: compute trailing 63-day strategy performance
    # and boost strategies that have been predictive recently
    fwd_5 = returns[equity_syms].rolling(5).sum().shift(-5)  # 5-day forward return
    for name in active_strategies:
        sig = strategy_frames[name]
        ic_boost = pd.Series(1.0, index=prices.index, dtype=float)
        for i in range(300, len(prices), 20):
            ics = []
            for t in range(max(0, i - 63), i, 5):
                f_row = sig.iloc[t].values if t < len(sig) else np.zeros(len(equity_syms))
                r_row = fwd_5.iloc[t].values if t < len(fwd_5) else np.zeros(len(equity_syms))
                valid = ~(np.isnan(f_row) | np.isnan(r_row))
                if valid.sum() > 20:
                    f_valid = f_row[valid]
                    r_valid = r_row[valid]
                    if np.nanstd(f_valid) < 1e-12 or np.nanstd(r_valid) < 1e-12:
                        continue
                    ic = np.corrcoef(rankdata(f_valid), rankdata(r_valid))[0, 1]
                    if not np.isnan(ic):
                        ics.append(ic)
            if ics:
                avg_ic = np.mean(ics)
                mult = 1.0 + np.clip(avg_ic * 15, -0.5, 1.0)
                for j in range(i, min(i + 20, len(ic_boost))):
                    ic_boost.iloc[j] = mult

        w = effective_weights[name].div(weight_norm)
        composite += sig.mul(w * 0.3 * (ic_boost - 1.0), axis=0)

    print(f"{time.time()-t0:.1f}s")

    # ── Portfolio construction from composite signal ──
    sym_sector = {sym: SECTOR_MAP.get(sym, "Other") for sym in equity_syms}
    sector_stocks = defaultdict(list)
    for sym in equity_syms:
        sector_stocks[sym_sector[sym]].append(sym)

    quality_signal = strategies.get(
        "quality",
        pd.DataFrame(0.0, index=prices.index, columns=equity_syms),
    ).reindex(columns=equity_syms).fillna(0.0)
    sma200_eq = prices[equity_syms].rolling(200, min_periods=120).mean()
    trend_gap_rank = cross_sectional_rank(-(prices[equity_syms] / sma200_eq - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0))
    beta_rank = cross_sectional_rank(rolling_beta.fillna(1.0))
    short_signal = (
        0.55 * (-composite)
        + 0.20 * (-quality_signal)
        + 0.15 * trend_gap_rank
        + 0.10 * beta_rank
    ).clip(-1.5, 1.5)

    weights = pd.DataFrame(0.0, index=prices.index, columns=trade_cols)
    target_w = pd.Series(0.0, index=equity_syms)
    target_hedge = 0.0
    live_row = pd.Series(0.0, index=trade_cols)
    blend_rate = 0.90

    running_nav = 1.0
    peak_nav = 1.0
    capacity_clamp_events = 0
    max_capacity_util = 0.0

    capacity_model = None
    adv_usd = None
    daily_vol_for_capacity = None
    symbol_to_id = {sym: idx + 1 for idx, sym in enumerate(equity_syms)}
    if use_capacity_constraints:
        capacity_model = LiquidityCapacityModel(
            impact_k=capacity_impact_k,
            min_adv_usd=1_000_000.0,
        )
        if volumes is not None and not volumes.empty:
            adv_usd = (volumes[equity_syms].rolling(20, min_periods=5).mean() * prices[equity_syms]).fillna(0.0)
        daily_vol_for_capacity = returns[equity_syms].rolling(20, min_periods=10).std().fillna(0.02)

    for i in range(280, len(prices)):
        # Drawdown circuit breaker
        if i > 280:
            day_ret = float((live_row * returns[trade_cols].iloc[i]).sum())
            running_nav *= (1 + day_ret)
            peak_nav = max(peak_nav, running_nav)
        current_dd = (running_nav / peak_nav) - 1 if peak_nav > 0 else 0

        dd_scale = 1.0
        if current_dd < -0.30:
            dd_scale = 0.45
        elif current_dd < -0.22:
            dd_scale = 0.70
        elif current_dd < -0.12:
            dd_scale = 0.85

        regime_row = regime_info.iloc[i] if regime_info is not None else None
        regime_allows_shorts = bool(regime_row.get("allow_shorts", False)) if regime_row is not None else False
        e = float(env.iloc[i]) if i < len(env) else 1.0
        b = float(breadth.iloc[i]) if i < len(breadth) else 0.5
        crisis_score, proactive_scale = compute_crisis_overlay(
            e, b, regime_row, current_dd, preemptive_de_risk
        )
        overlay_scale = float(np.clip(dd_scale * proactive_scale, 0.55, 1.0))
        prev_target = target_w.copy()

        if i % rebal_freq != 0 and i > 280:
            live_row = pd.Series(0.0, index=trade_cols)
            live_row.loc[equity_syms] = target_w * overlay_scale if overlay_scale < 1.0 else target_w.copy()
            if hedge_available:
                live_row.loc[hedge_symbol] = target_hedge
            weights.iloc[i] = live_row
            continue

        sig = composite.iloc[i]
        valid = sig.dropna()
        valid = valid[valid.abs() > 0.005]
        if len(valid) < 20:
            live_row = pd.Series(0.0, index=trade_cols)
            live_row.loc[equity_syms] = target_w * overlay_scale if overlay_scale < 1.0 else target_w.copy()
            if hedge_available:
                live_row.loc[hedge_symbol] = target_hedge
            weights.iloc[i] = live_row
            continue

        is_bearish = e < 0.85 or b < 0.50
        shorts_active = regime_allows_shorts and (is_bearish or crisis_score > 0.35)

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

            if shorts_active:
                sector_short = short_signal.iloc[i].reindex(syms).dropna().sort_values(ascending=False)
                n_s = max(int(len(sector_short) * 0.15), 1)
                for sym in sector_short.head(n_s).index:
                    if (
                        sector_short[sym] > 0.18
                        and quality_signal.iloc[i].get(sym, 0.0) < 0.05
                        and prices[sym].iloc[i] < sma200_eq[sym].iloc[i]
                    ):
                        short_picks.append((sym, sector_short[sym]))

        long_picks.sort(key=lambda x: x[1], reverse=True)
        short_picks.sort(key=lambda x: x[1])
        long_picks = long_picks[:n_long]
        short_picks = short_picks[:n_short]

        w = pd.Series(0.0, index=equity_syms)

        breadth_mult = np.clip(b * 1.5, 0.55, 1.20)
        if shorts_active:
            long_budget = target_gross * min(e, 1.05) * 0.72 * breadth_mult
            short_budget = target_gross * (0.10 + 0.18 * crisis_score) * max(0.35, 1 - breadth_mult / 1.10)
        else:
            long_budget = target_gross * min(e, 1.35) * breadth_mult
            short_budget = 0.0

        long_budget *= overlay_scale
        short_budget *= overlay_scale

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
                w[sym] = float(np.clip(raw_w, -max_pos * 0.45, 0))

        # Exit stale names faster than we enter fresh ones.
        if i > 280:
            delta = w - prev_target
            small = delta.abs() < 0.003
            w[small] = prev_target[small]

            keep = pd.Series(blend_rate, index=equity_syms)
            entering = (prev_target.abs() < 1e-12) & (w.abs() > 1e-12)
            exiting = (prev_target.abs() > 1e-12) & (w.abs() < 1e-12)
            flipping = (
                (prev_target.abs() > 1e-12)
                & (w.abs() > 1e-12)
                & (np.sign(prev_target) != np.sign(w))
            )
            keep[entering] = min(blend_rate, 0.72)
            keep[exiting] = 0.50
            keep[flipping] = 0.30

            w = (1 - keep) * w + keep * prev_target
            w[(w.abs() < 0.002) & exiting] = 0.0

        if capacity_model is not None:
            nav_now_usd = float(max(nav_usd * running_nav, nav_usd * 0.30))
            delta_w = w - prev_target
            for sym, delta in delta_w.items():
                if abs(delta) < 1e-4:
                    continue
                sid = symbol_to_id.get(sym)
                if sid is None:
                    continue
                snapshot = build_liquidity_snapshot(
                    symbol_id=sid,
                    symbol=sym,
                    i=i,
                    prices=prices,
                    adv_usd_frame=adv_usd,
                    daily_vol_frame=daily_vol_for_capacity,
                    participation_limit=capacity_participation_limit,
                    impact_limit_bps=capacity_impact_limit_bps,
                    max_spread_bps=capacity_max_spread_bps,
                )
                order_notional = abs(float(delta)) * nav_now_usd
                estimate = capacity_model.estimate_order(snapshot, order_notional)
                max_capacity_util = max(max_capacity_util, float(estimate.utilization))
                if estimate.utilization <= 1.0:
                    continue
                allowed_delta = float(estimate.max_order_notional_usd / max(nav_now_usd, 1e-12))
                adjusted_delta = np.sign(delta) * max(allowed_delta, 0.0)
                w[sym] = prev_target[sym] + adjusted_delta
                capacity_clamp_events += 1

        target_w = w.clip(lower=-max_pos * 0.45, upper=max_pos)
        target_hedge = 0.0
        if hedge_available and crisis_hedge_max > 0 and crisis_score > 0.10:
            beta_row = rolling_beta.iloc[i].reindex(equity_syms).fillna(1.0)
            portfolio_beta = float((target_w * beta_row).sum())
            excess_beta = max(portfolio_beta - crisis_beta_floor, 0.0)
            if excess_beta > 0:
                hedge_mult = min(1.0, 0.35 + crisis_score * crisis_hedge_strength)
                target_hedge = -float(np.clip(excess_beta * hedge_mult, 0.0, crisis_hedge_max))

        live_row = pd.Series(0.0, index=trade_cols)
        live_row.loc[equity_syms] = target_w * overlay_scale if overlay_scale < 1.0 else target_w.copy()
        if hedge_available:
            live_row.loc[hedge_symbol] = target_hedge
        weights.iloc[i] = live_row

    fast_mean_reversion = build_fast_mean_reversion_overlay(
        strategies.get("mean_reversion"),
        prices,
        returns,
        equity_syms,
        regime_info=regime_info,
    )
    weights.loc[:, equity_syms] = weights[equity_syms].add(fast_mean_reversion, fill_value=0.0)

    # Vol targeting
    port_ret = (weights.shift(1) * returns[trade_cols]).sum(axis=1)
    port_rvol = port_ret.rolling(20, min_periods=10).std() * np.sqrt(252)

    final_weights = weights.copy()
    for i in range(300, len(prices)):
        rv = port_rvol.iloc[i]
        if np.isnan(rv) or rv < 0.01:
            continue
        scale = target_vol / rv
        scale = float(np.clip(scale, 0.3, 2.0))
        final_weights.iloc[i] = weights.iloc[i] * scale

    diagnostics = {
        "allocator_enabled": bool(allocator_diag.get("allocator_enabled", False)),
        "allocator_capacity_updates": int(allocator_diag.get("allocator_capacity_updates", 0)),
        "allocator_weight_entropy": float(allocator_diag.get("allocator_weight_entropy", np.nan)),
        "high_52w_active_pct": float(high_52w_active_pct),
        "capacity_constraints_enabled": bool(capacity_model is not None),
        "capacity_clamp_events": int(capacity_clamp_events),
        "max_capacity_utilization_seen": float(max_capacity_util),
    }

    return final_weights, diagnostics


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


def elapsed_years_from_index(index):
    if index is None or len(index) < 2:
        return max((len(index) if index is not None else 0) / 252.0, 1 / 252.0)
    start = pd.Timestamp(index[0])
    end = pd.Timestamp(index[-1])
    days = max((end - start).days, 1)
    return max(days / 365.25, 1 / 252.0)


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

    all_data, dropped_histories, latest_date = filter_histories_for_backtest(all_data, universe)
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
    strat_df = compute_strategy_test_returns(strategies, returns, equity_syms)
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
    for name, n_short_name in zip(strat_df.columns, names_short):
        s = sharpe(strat_df[name])
        print(f"  {n_short_name:>8}: {s:+.2f}")
    print()

    # Environment & breadth
    env = compute_environment(prices)
    breadth = compute_breadth(prices, equity_syms)
    regime_info = compute_regime_states(prices, equity_syms)
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

    # ── Build multi-strategy portfolio ──
    print("=" * 70)
    print("  WALK-FORWARD BACKTEST v4 (8-strategy risk-parity)")
    print("=" * 70)
    print(f"  NAV: ${args.nav:,.0f} | Vol target: {args.target_vol:.0%} | "
          f"Rebal: {args.rebal_freq}d | Max pos: {args.max_pos:.0%}")
    print(f"  Target gross: {args.target_gross:.1f}x | "
          f"Long: {args.n_long} | Short: {args.n_short}\n")
    print(f"  Crisis hedge max: {args.crisis_hedge_max:.0%} | "
          f"Pre-emptive de-risk: {args.preemptive_de_risk:.0%}\n")

    t0 = time.time()
    weights, portfolio_diag = build_multi_strategy_portfolio(
        strategies, prices, returns, env, breadth, equity_syms,
        args.target_vol, args.rebal_freq, args.max_pos,
        args.n_long, args.n_short, args.target_gross,
        regime_info=regime_info,
        hedge_symbol="SPY",
        crisis_hedge_max=args.crisis_hedge_max,
        crisis_hedge_strength=args.crisis_hedge_strength,
        crisis_beta_floor=args.crisis_beta_floor,
        preemptive_de_risk=args.preemptive_de_risk,
        hedge_lookback=args.hedge_lookback,
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

    # ── Results ──
    warmup = 300
    trade_cols = list(weights.columns)
    port_ret = (weights.shift(1) * returns[trade_cols]).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    tx_cost = turnover * (args.slippage + 1.0) / 10000
    net_ret = port_ret - tx_cost
    net_ret = net_ret.iloc[warmup:]
    weights_post = weights.iloc[warmup:]
    turnover_post = turnover.iloc[warmup:]

    equity_curve = args.nav * (1 + net_ret).cumprod()
    lp_ret = apply_hedge_fund_fees(net_ret, args.mgmt_fee, args.perf_fee)
    lp_equity_curve = args.nav * (1 + lp_ret).cumprod()
    dates = equity_curve.index
    n_years = elapsed_years_from_index(dates)

    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / max(n_years, 0.01)) - 1
    s = sharpe(net_ret)
    so = sortino(net_ret)
    dd = max_dd(equity_curve)
    cal = cagr / abs(dd) if abs(dd) > 1e-12 else 0
    lp_cagr = (lp_equity_curve.iloc[-1] / lp_equity_curve.iloc[0]) ** (1 / max(n_years, 0.01)) - 1
    lp_s = sharpe(lp_ret)
    lp_so = sortino(lp_ret)
    lp_dd = max_dd(lp_equity_curve)
    avg_long = weights_post.clip(lower=0).sum(axis=1).mean()
    avg_short = weights_post.clip(upper=0).abs().sum(axis=1).mean()
    avg_gross = weights_post.abs().sum(axis=1).mean()
    avg_net = weights_post.sum(axis=1).mean()
    avg_turn = turnover_post.mean() * 252
    tx_bps = tx_cost.iloc[warmup:].mean() * 252 * 10000 if len(tx_cost) > warmup else 0

    spy_ret = (prices["SPY"].pct_change().fillna(0)).iloc[warmup:]
    spy_ret = spy_ret.reindex(net_ret.index).fillna(0.0)
    spy_eq = args.nav * (1 + spy_ret).cumprod()
    spy_cagr = (spy_eq.iloc[-1] / spy_eq.iloc[0]) ** (1 / max(n_years, 0.01)) - 1
    spy_sharpe = sharpe(spy_ret)
    spy_dd = max_dd(spy_eq)

    print("  ── Aggregate ──")
    print(f"  Gross CAGR:    {cagr:+.2%}   (SPY: {spy_cagr:+.2%})  {'*** ALPHA ***' if cagr > spy_cagr else ''}")
    print(f"  Gross Sharpe:  {s:.2f}      (SPY: {spy_sharpe:.2f})  {'*** TARGET' if s >= args.lp_target_sharpe else '✓' if s > spy_sharpe else ''}")
    print(f"  Gross Sortino: {so:.2f}")
    print(f"  Calmar:        {cal:.2f}")
    print(f"  LP Net CAGR:   {lp_cagr:+.2%}   (after {args.mgmt_fee:.0%}/{args.perf_fee:.0%})")
    print(f"  LP Net Sharpe: {lp_s:.2f}")
    print(f"  LP Net Sortino:{lp_so:.2f}")
    print(f"  LP Net Max DD: {lp_dd:.2%}")
    print(f"  Gross Max DD:  {dd:.2%}   (SPY: {spy_dd:.2%})  {'✓' if abs(dd) < abs(spy_dd) else ''}")
    print(f"  Final NAV:     ${equity_curve.iloc[-1]:,.0f}  (SPY: ${spy_eq.iloc[-1]:,.0f})")
    print(f"  LP NAV:        ${lp_equity_curve.iloc[-1]:,.0f}")
    print(f"  Avg long:      {avg_long:.2f}x")
    print(f"  Avg short:     {avg_short:.2f}x")
    print(f"  Avg gross:     {avg_gross:.2f}x | Net: {avg_net:+.2f}x")
    print(f"  Turnover:      {avg_turn:.0f}x/yr")
    print(f"  Tx costs:      {tx_bps:.0f} bps/yr")
    print(f"  Sample window: {dates[0].date()} -> {dates[-1].date()} ({n_years:.2f} years)")
    print()

    # Year by year
    print("  ── Year-by-Year: LP Net vs SPY ──")
    print(f"  {'Year':<6} {'LP Ret':>8} {'SPY':>8} {'Alpha':>8} {'LP Shp':>8} {'SPY Shp':>8} {'LP DD':>8} {'SPY DD':>8} {'Hit':>4}")
    print("  " + "-" * 74)
    loss_years = beat_spy = beat_sharpe = beat_dd = 0
    hit_return = hit_sharpe = hit_both = 0

    for year in sorted(set(d.year if hasattr(d, 'year') else d.date().year for d in dates)):
        mask = pd.Series([d.year if hasattr(d, 'year') else d.date().year for d in dates], index=lp_ret.index) == year
        yr = lp_ret[mask]
        yr_eq = lp_equity_curve[mask]
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
        meets_return = yr_ret >= args.lp_target_return
        meets_sharpe = yr_s >= args.lp_target_sharpe
        win = "Y" if meets_return and meets_sharpe else "N"
        if yr_ret <= 0: loss_years += 1
        if yr_ret > spy_yr_ret: beat_spy += 1
        if yr_s > spy_yr_s: beat_sharpe += 1
        if abs(yr_d) < abs(spy_yr_d): beat_dd += 1
        if meets_return: hit_return += 1
        if meets_sharpe: hit_sharpe += 1
        if meets_return and meets_sharpe: hit_both += 1

        print(f"  {year:<6} {yr_ret:>+7.2%} {spy_yr_ret:>+7.2%} {alpha:>+7.2%} "
              f"{yr_s:>7.2f}  {spy_yr_s:>7.2f}  {yr_d:>+7.2%} {spy_yr_d:>+7.2%} {win:>4}")

    total_years = len(set(d.year if hasattr(d, 'year') else d.date().year for d in dates))
    print("  " + "-" * 74)
    print(f"  Loss years:       {loss_years}/{total_years}")
    print(f"  Beat SPY return:  {beat_spy}/{total_years}")
    print(f"  Beat SPY Sharpe:  {beat_sharpe}/{total_years}")
    print(f"  Beat SPY MaxDD:   {beat_dd}/{total_years}")
    print(f"  Years >= {args.lp_target_return:.0%} net: {hit_return}/{total_years}")
    print(f"  Years >= {args.lp_target_sharpe:.1f} Sharpe: {hit_sharpe}/{total_years}")
    print(f"  Years hitting both hurdles: {hit_both}/{total_years}")
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
    if len(lp_ret) > 50:
        print("  ── Monte Carlo (10K paths, 1yr) ──")
        terminal, dds_mc = monte_carlo(lp_ret)
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
    p.add_argument("--preemptive-de-risk", type=float, default=0.0,
                   help="Maximum proactive gross trim from crisis signals before drawdown breaker")
    p.add_argument("--crisis-hedge-max", type=float, default=0.0,
                   help="Maximum SPY hedge weight when market stress is elevated")
    p.add_argument("--crisis-hedge-strength", type=float, default=0.75,
                   help="How aggressively excess beta is hedged during crisis regimes")
    p.add_argument("--crisis-beta-floor", type=float, default=0.15,
                   help="Leave this much net beta unhedged before crisis hedge kicks in")
    p.add_argument("--hedge-lookback", type=int, default=63,
                   help="Lookback window for portfolio beta estimation used by crisis hedge")
    p.add_argument("--mgmt-fee", type=float, default=0.02,
                   help="Annual management fee for LP net reporting")
    p.add_argument("--perf-fee", type=float, default=0.20,
                   help="Incentive fee on gains above high-water mark")
    p.add_argument("--lp-target-return", type=float, default=0.15,
                   help="Annual LP net return hurdle")
    p.add_argument("--lp-target-sharpe", type=float, default=1.0,
                   help="Annual LP net Sharpe hurdle")
    p.add_argument("--disable-dynamic-allocator", action="store_true",
                   help="Disable dynamic allocator-driven strategy reweighting")
    p.add_argument("--disable-capacity-constraints", action="store_true",
                   help="Disable capacity-aware trade clamping during rebalances")
    p.add_argument("--capacity-impact-k", type=float, default=0.45,
                   help="Square-root impact coefficient used by liquidity capacity model")
    p.add_argument("--capacity-participation-limit", type=float, default=0.05,
                   help="Max participation of ADV per rebalance trade")
    p.add_argument("--capacity-impact-limit-bps", type=float, default=15.0,
                   help="Max allowed impact per rebalance trade in bps before clamping")
    p.add_argument("--capacity-max-spread-bps", type=float, default=14.0,
                   help="Spread threshold for capacity penalty")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    if os.path.exists(args.env):
        from dotenv import load_dotenv
        load_dotenv(args.env)

    run(args)
