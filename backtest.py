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


# asset type lookup
ASSET_TYPES = {}

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
    # commodity futures
    "CL": "NYMEX", "NG": "NYMEX", "HG": "COMEX",
    "GC": "COMEX", "SI": "COMEX",
    "ZC": "CBOT", "ZW": "CBOT", "ZS": "CBOT",
    # fixed income
    "ZN": "CBOT", "ZB": "CBOT", "ZF": "CBOT", "ZT": "CBOT", "GE": "CME",
    # equity index
    "ES": "CME", "NQ": "CME", "RTY": "CME", "YM": "CBOT",
    # volatility
    "VX": "CFE",
}


def make_contract(symbol, asset_type):
    if asset_type in ("ETF", "EQUITY"):
        return Stock(symbol, "SMART", "USD")
    elif asset_type == "FX":
        return Forex(symbol[:3] + symbol[3:])
    elif asset_type in ("FUTURE", "COMMODITY", "BOND", "VOLATILITY"):
        exchange = EXCHANGE_MAP.get(symbol, "CME")
        # use continuous front month
        return Future(symbol, exchange=exchange, includeExpired=True)
    return Stock(symbol, "SMART", "USD")


def qualify_future(ib, symbol, asset_type):
    """Get the front-month future contract."""
    exchange = EXCHANGE_MAP.get(symbol, "CME")
    # try continuous first
    from datetime import date
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

    # fallback: generic future, pick the nearest expiry
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

    # try longest duration first, fall back to shorter
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


def run_backtest_on_bars(all_bars, nav, signal_threshold, slippage_bps, kelly_fraction):
    """
    Run the mean-reversion strategy on daily bars across all symbols.
    Uses Kelly criterion for position sizing.
    Returns daily NAV series and stats.
    """
    config = PaperConfig(
        initial_nav=nav,
        signal_threshold=signal_threshold,
        slippage_bps=slippage_bps,
        drawdown_auto_kill_pct=0.25,
        kelly_fraction=kelly_fraction,
        risk_budget_per_position=0.005,
    )
    engine = PaperTradingEngine(config)

    # merge all bars into one timeline sorted by date
    all_ticks = []
    for sym, sid, df in all_bars:
        for _, row in df.iterrows():
            dt = row["date"]
            if hasattr(dt, 'timestamp'):
                ts = int(dt.timestamp() * 1e9)
            else:
                ts = 0
            all_ticks.append(PaperTick(
                symbol_id=sid,
                price=row["close"],
                volume=int(row["volume"]),
                timestamp_ns=ts,
            ))

    all_ticks.sort(key=lambda t: t.timestamp_ns)

    if not all_ticks:
        return None, None, None

    # run through engine, capture daily NAV
    # reset kill switch each day so drawdown doesn't permanently stop trading
    daily_navs = []
    daily_dates = []
    current_day = None

    for tick in all_ticks:
        ts = datetime.fromtimestamp(tick.timestamp_ns / 1e9, tz=timezone.utc)
        day = ts.date()
        if current_day is None:
            current_day = day
        if day != current_day:
            daily_navs.append(engine.stats.final_nav)
            daily_dates.append(current_day)
            current_day = day
            # in backtest mode, reset kill switch and peak NAV at start of each day
            # so a single bad period doesn't permanently stop the strategy
            if engine.kill_switch.level.value > 0:
                engine.kill_switch.reset()
                engine.portfolio.peak_nav = engine.portfolio.nav
        engine.on_tick(tick)
    # last day
    daily_navs.append(engine.stats.final_nav)
    daily_dates.append(current_day)

    return np.array(daily_navs), daily_dates, engine


def run(args):
    universe = load_universe(args.config)
    print(f"Universe: {len(universe)} instruments across {len(set(t for _, t in universe))} asset classes\n")

    # connect to IB
    print(f"Connecting to IB Gateway at {args.host}:{args.port}...")
    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id)
    ib.reqMarketDataType(4)  # delayed if no live sub
    print("Connected\n")

    # fetch all historical data
    print("=" * 65)
    print("  FETCHING HISTORICAL DATA")
    print("=" * 65)

    all_bars = []  # (symbol, symbol_id, dataframe)
    symbol_info = {}
    sid = 0

    for sym, asset_type in universe:
        sid += 1
        print(f"  [{sid:2d}/{len(universe)}] {sym:8s} ({asset_type:10s}) ", end="", flush=True)
        df = fetch_max_daily(ib, sym, asset_type)
        if df.empty:
            print("-- no data")
            continue
        print(f"-- {len(df):,} days ({df['date'].iloc[0].strftime('%Y-%m-%d') if hasattr(df['date'].iloc[0], 'strftime') else '?'} to {df['date'].iloc[-1].strftime('%Y-%m-%d') if hasattr(df['date'].iloc[-1], 'strftime') else '?'})")
        all_bars.append((sym, sid, df))
        symbol_info[sym] = {"sid": sid, "type": asset_type, "bars": len(df)}

    ib.disconnect()

    if not all_bars:
        print("\nNo data fetched. Exiting.")
        sys.exit(1)

    total_bars = sum(len(df) for _, _, df in all_bars)
    print(f"\nTotal: {len(all_bars)} instruments, {total_bars:,} daily bars\n")

    # figure out date range
    all_dates = set()
    for sym, sid, df in all_bars:
        for d in df["date"]:
            if hasattr(d, 'date'):
                all_dates.add(d.date() if hasattr(d, 'date') else d)
            else:
                all_dates.add(d)
    min_date = min(all_dates)
    max_date = max(all_dates)
    total_years = (max_date - min_date).days / 365.25

    # ── FULL PERIOD BACKTEST ──────────────────────────────────
    print("=" * 65)
    print("  RUNNING FULL PERIOD BACKTEST")
    print("=" * 65)

    bt_start = time.time()
    equity, dates, engine = run_backtest_on_bars(
        all_bars, args.nav, args.signal_threshold, args.slippage, args.kelly,
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

    # ── FULL PERIOD METRICS ───────────────────────────────────
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
    print()

    # ── YEAR-BY-YEAR BREAKDOWN ────────────────────────────────
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
        # rough trade count (proportional)
        yr_trades = int(s.orders_filled * len(indices) / len(dates)) if len(dates) > 0 else 0
        print(f"  {year:<6} {yr_ret:>+7.2%} {yr_sharpe:>8.2f} {yr_dd:>+7.2%} {yr_trades:>8}")

    print()

    # ── ASSET CLASS BREAKDOWN ─────────────────────────────────
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

    # ── MONTE CARLO ───────────────────────────────────────────
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
    print(f"  Kelly fraction:    {args.kelly}")
    print(f"  Signal threshold:  {args.signal_threshold}")
    print(f"  Slippage:          {args.slippage} bps")
    print(f"  Starting NAV:      ${args.nav:,.0f}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full-universe backtest with Kelly sizing and Monte Carlo")
    parser.add_argument("--config", default="config/data_layer.yaml", help="instrument config")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=2)
    parser.add_argument("--nav", type=float, default=1_000_000, help="starting capital")
    parser.add_argument("--signal-threshold", type=float, default=0.08, help="min signal to trade")
    parser.add_argument("--slippage", type=float, default=1.0, help="slippage bps")
    parser.add_argument("--kelly", type=float, default=0.25, help="kelly fraction (0.25 = quarter kelly)")
    args = parser.parse_args()
    run(args)
