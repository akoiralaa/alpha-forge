#!/usr/bin/env python3

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

try:
    from ib_async import IB, Stock, Forex, Future
except ImportError:
    from ib_insync import IB, Stock, Forex, Future

from src.paper.engine import PaperConfig, PaperTick, PaperTradingEngine


def make_contract(symbol, asset_type):
    if asset_type in ("ETF", "EQUITY"):
        return Stock(symbol, "SMART", "USD")
    elif asset_type == "FX":
        return Forex(symbol[:3] + symbol[3:])
    elif asset_type in ("FUTURE", "COMMODITY", "BOND", "VOLATILITY"):
        return Future(symbol, exchange="CME")
    return Stock(symbol, "SMART", "USD")


def fetch_bars(ib, symbol, asset_type, duration_str, bar_size="1 min"):
    contract = make_contract(symbol, asset_type)
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        print(f"  Could not qualify {symbol}")
        return pd.DataFrame()
    contract = qualified[0]

    what = "MIDPOINT" if asset_type == "FX" else "TRADES"

    print(f"  Fetching {symbol} ({duration_str}, {bar_size} bars)...")
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=duration_str,
        barSizeSetting=bar_size,
        whatToShow=what,
        useRTH=True,
        formatDate=2,
    )
    time.sleep(0.5)

    if not bars:
        print(f"  No data returned for {symbol}")
        return pd.DataFrame()

    records = []
    for bar in bars:
        dt = bar.date if isinstance(bar.date, datetime) else datetime.fromisoformat(str(bar.date))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        records.append({
            "timestamp": dt,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": int(bar.volume),
        })

    df = pd.DataFrame(records)
    print(f"  Got {len(df)} bars for {symbol}")
    return df


def bars_to_ticks(df, symbol_id):
    ticks = []
    for _, row in df.iterrows():
        ts = int(row["timestamp"].timestamp() * 1e9) if hasattr(row["timestamp"], "timestamp") else 0
        ticks.append(PaperTick(
            symbol_id=symbol_id,
            price=row["close"],
            volume=int(row["volume"]) if row["volume"] > 0 else 100,
            timestamp_ns=ts,
        ))
    return ticks


def compute_sharpe(returns, ann=252):
    if len(returns) < 2 or np.std(returns) < 1e-12:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(ann))


def compute_max_drawdown(equity_curve):
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    return float(np.min(dd))


def compute_calmar(total_return, max_dd, years):
    if abs(max_dd) < 1e-12 or years < 0.01:
        return 0.0
    annual_return = total_return / years
    return float(annual_return / abs(max_dd))


def monte_carlo(daily_returns, n_sims=10000, n_days=252):
    rng = np.random.default_rng(42)
    results = np.zeros((n_sims, n_days))

    for i in range(n_sims):
        # resample daily returns with replacement
        sampled = rng.choice(daily_returns, size=n_days, replace=True)
        results[i] = np.cumprod(1 + sampled)

    terminal = results[:, -1]
    paths_equity = results  # each row is a simulated equity curve (starting from 1.0)

    # compute drawdowns for each path
    max_dds = np.zeros(n_sims)
    for i in range(n_sims):
        peak = np.maximum.accumulate(results[i])
        dd = (results[i] - peak) / peak
        max_dds[i] = np.min(dd)

    return {
        "terminal_wealth": terminal,
        "max_drawdowns": max_dds,
        "paths": results,
    }


def print_report(engine, daily_returns, equity_curve, mc_results, elapsed_s, n_days_actual):
    s = engine.stats
    total_return = (s.final_nav - engine.config.initial_nav) / engine.config.initial_nav
    years = n_days_actual / 252
    sharpe = compute_sharpe(daily_returns)
    max_dd = compute_max_drawdown(equity_curve)
    calmar = compute_calmar(total_return, max_dd, years)
    win_rate = s.orders_filled / max(s.orders_submitted, 1)
    sortino_denom = np.std(daily_returns[daily_returns < 0]) if np.any(daily_returns < 0) else 1e-12
    sortino = float(np.mean(daily_returns) / sortino_denom * np.sqrt(252))

    print()
    print("=" * 65)
    print("  BACKTEST RESULTS")
    print("=" * 65)
    print(f"  Period:            {n_days_actual} trading days ({years:.1f} years)")
    print(f"  Backtest time:     {elapsed_s:.1f}s")
    print()
    print("  -- Performance --")
    print(f"  Total return:      {total_return:+.2%}")
    print(f"  Annual return:     {total_return/years:+.2%}" if years > 0.01 else "")
    print(f"  Sharpe ratio:      {sharpe:.2f}")
    print(f"  Sortino ratio:     {sortino:.2f}")
    print(f"  Calmar ratio:      {calmar:.2f}")
    print(f"  Max drawdown:      {max_dd:.2%}")
    print()
    print("  -- Trading --")
    print(f"  Orders submitted:  {s.orders_submitted}")
    print(f"  Orders filled:     {s.orders_filled}")
    print(f"  Orders rejected:   {s.orders_rejected} ({s.risk_blocks} risk)")
    print(f"  Win rate:          {win_rate:.1%}")
    print()
    print("  -- Portfolio --")
    print(f"  Starting NAV:      ${engine.config.initial_nav:,.0f}")
    print(f"  Final NAV:         ${s.final_nav:,.0f}")
    print(f"  Peak NAV:          ${s.peak_nav:,.0f}")
    print(f"  Total PnL:         ${s.total_pnl:+,.0f}")
    print()

    # Monte Carlo
    terminal = mc_results["terminal_wealth"]
    max_dds = mc_results["max_drawdowns"]

    print("  -- Monte Carlo (10,000 simulations, 1yr forward) --")
    print(f"  Median return:     {np.median(terminal) - 1:+.2%}")
    print(f"  5th pctl return:   {np.percentile(terminal, 5) - 1:+.2%}")
    print(f"  95th pctl return:  {np.percentile(terminal, 95) - 1:+.2%}")
    print(f"  Prob of loss:      {np.mean(terminal < 1.0):.1%}")
    print(f"  Prob of >10% loss: {np.mean(terminal < 0.90):.1%}")
    print(f"  Median max DD:     {np.median(max_dds):.2%}")
    print(f"  95th pctl max DD:  {np.percentile(max_dds, 5):.2%}")  # 5th pctl of DD = worst
    print(f"  Worst case DD:     {np.min(max_dds):.2%}")
    print("=" * 65)


def run(args):
    print(f"Connecting to IB Gateway at {args.host}:{args.port}...")
    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id)
    print("Connected\n")

    # figure out duration
    duration_map = {
        "1m": ("1 M", "1 min"),
        "3m": ("3 M", "1 min"),
        "6m": ("6 M", "5 mins"),
        "1y": ("1 Y", "5 mins"),
        "2y": ("2 Y", "15 mins"),
        "5y": ("5 Y", "1 day"),
    }

    if args.period not in duration_map:
        print(f"Invalid period: {args.period}. Choose from: {list(duration_map.keys())}")
        sys.exit(1)

    duration_str, bar_size = duration_map[args.period]

    # fetch data for each symbol
    all_ticks = []
    symbols = [s.upper() for s in args.symbols]

    for i, sym in enumerate(symbols):
        # guess asset type
        asset_type = "ETF"
        if sym in ("ES", "NQ", "RTY", "YM"):
            asset_type = "FUTURE"
        elif sym in ("CL", "NG", "GC", "SI", "HG", "ZC", "ZW", "ZS"):
            asset_type = "COMMODITY"
        elif sym in ("ZN", "ZB", "ZF", "ZT", "GE"):
            asset_type = "BOND"
        elif len(sym) == 6 and sym[:3].isalpha() and sym[3:].isalpha():
            asset_type = "FX"
        elif sym == "VX":
            asset_type = "VOLATILITY"

        df = fetch_bars(ib, sym, asset_type, duration_str, bar_size)
        if df.empty:
            continue

        sid = i + 1
        ticks = bars_to_ticks(df, sid)
        all_ticks.extend(ticks)

    ib.disconnect()
    print(f"\nTotal ticks: {len(all_ticks):,}")

    if not all_ticks:
        print("No data fetched. Check market data subscriptions.")
        sys.exit(1)

    # sort by timestamp
    all_ticks.sort(key=lambda t: t.timestamp_ns)

    # run backtest
    print("\nRunning backtest...")
    config = PaperConfig(
        initial_nav=args.nav,
        signal_threshold=args.signal_threshold,
        slippage_bps=args.slippage,
        drawdown_auto_kill_pct=0.20,
    )
    engine = PaperTradingEngine(config)

    start = time.time()
    stats = engine.run_session(all_ticks)
    elapsed = time.time() - start

    # build daily equity curve from tick-level NAV
    # group ticks by day and take end-of-day NAV
    nav_series = []
    day_nav = {}

    # replay to get per-tick NAV (use the engine's final state)
    # simpler: reconstruct from ticks timestamps
    engine2 = PaperTradingEngine(config)
    current_day = None
    for tick in all_ticks:
        engine2.on_tick(tick)
        ts = datetime.fromtimestamp(tick.timestamp_ns / 1e9, tz=timezone.utc)
        day = ts.date()
        if current_day is None:
            current_day = day
        if day != current_day:
            nav_series.append(engine2.stats.final_nav)
            current_day = day
    nav_series.append(engine2.stats.final_nav)  # last day

    equity_curve = np.array(nav_series)
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]

    n_days = len(nav_series)

    # monte carlo
    if len(daily_returns) > 10:
        print("Running Monte Carlo (10,000 simulations)...")
        mc = monte_carlo(daily_returns, n_sims=10000, n_days=252)
    else:
        print("Not enough data for Monte Carlo")
        mc = {
            "terminal_wealth": np.array([1.0]),
            "max_drawdowns": np.array([0.0]),
            "paths": np.array([[1.0]]),
        }

    # use engine2 since it has the replay
    engine2.stats.final_nav = nav_series[-1]
    engine2.stats.peak_nav = max(nav_series)
    engine2.stats.total_pnl = nav_series[-1] - config.initial_nav
    print_report(engine2, daily_returns, equity_curve, mc, elapsed, n_days)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest with historical IB data + Monte Carlo")
    parser.add_argument("symbols", nargs="+", help="symbols to backtest (e.g. SPY QQQ)")
    parser.add_argument("--period", default="1y", help="1m, 3m, 6m, 1y, 2y, 5y")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=2)
    parser.add_argument("--nav", type=float, default=1_000_000)
    parser.add_argument("--signal-threshold", type=float, default=0.1)
    parser.add_argument("--slippage", type=float, default=1.0)
    args = parser.parse_args()
    run(args)
