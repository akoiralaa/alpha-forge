#!/usr/bin/env python3
"""
AlphaForge — Alpaca Paper Trading Runner
=========================================
Drop-in replacement for run.py that uses Alpaca's free websocket feed
instead of IB Gateway.  Feeds PaperTick events into the existing
PaperTradingEngine — all fills, risk checks, kill switch, and WAL work
exactly as in run.py.

Alpaca free tier streams US equities only (no futures/FX), so this runner
trades sector ETFs from config/data_layer.yaml.

Usage
─────
    python run_alpaca.py
    python run_alpaca.py --symbols SPY QQQ XLK --nav 500000
    python run_alpaca.py --symbols SPY QQQ --signal-threshold 0.05

Keys are read from .env  (ALPACA_API_KEY, ALPACA_API_SECRET).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.paper.engine import PaperConfig, PaperTick, PaperTradingEngine

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("alpaca_runner")

# ── symbol helpers ────────────────────────────────────────────────────────────

def load_equity_symbols(config_path: str) -> list[str]:
    """Return the sector ETF list from the instrument config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("instruments", {}).get("sector_etfs", [])


def build_symbol_map(symbols: list[str]) -> dict[str, int]:
    """symbol → symbol_id (stable integer used by PaperTradingEngine)."""
    return {sym: idx for idx, sym in enumerate(sorted(symbols))}


# ── terminal stats ────────────────────────────────────────────────────────────

def print_stats(engine: PaperTradingEngine, elapsed_s: float) -> None:
    s = engine.stats
    nav = s.final_nav
    pnl = s.total_pnl
    dd = s.max_drawdown_pct
    pos = len(engine.broker.get_positions())

    green = "\033[32m"
    red   = "\033[31m"
    reset = "\033[0m"
    color = green if pnl >= 0 else red

    sys.stdout.write(
        f"\r  ticks: {s.ticks_processed:,}  |  "
        f"orders: {s.orders_submitted} sent / {s.orders_filled} filled  |  "
        f"NAV: ${nav:,.0f}  |  "
        f"PnL: {color}${pnl:+,.0f}{reset}  |  "
        f"DD: {dd:.2%}  |  "
        f"positions: {pos}  |  "
        f"uptime: {elapsed_s:.0f}s    "
    )
    sys.stdout.flush()


# ── main runner ───────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    api_key    = os.environ.get("ALPACA_API_KEY", "")
    api_secret = os.environ.get("ALPACA_API_SECRET", "")

    if not api_key or not api_secret:
        log.error("ALPACA_API_KEY / ALPACA_API_SECRET not set in .env")
        sys.exit(1)

    # Symbol list — equity-only (Alpaca free)
    all_symbols = load_equity_symbols(args.config)
    if args.symbols:
        requested = {s.upper() for s in args.symbols}
        symbols = [s for s in all_symbols if s in requested]
        unknown = requested - set(symbols)
        if unknown:
            log.warning("Unknown symbols (not in sector_etfs): %s", unknown)
    else:
        symbols = all_symbols

    if not symbols:
        log.error("No symbols to trade.")
        sys.exit(1)

    sym_to_id = build_symbol_map(symbols)
    log.info("Trading %d symbols: %s", len(symbols), " ".join(symbols))

    # Engine
    config = PaperConfig(
        initial_nav=args.nav,
        signal_threshold=args.signal_threshold,
        slippage_bps=args.slippage,
        drawdown_auto_kill_pct=args.max_drawdown,
    )
    engine = PaperTradingEngine(config)
    log.info("PaperTradingEngine initialised (NAV=$%,.0f)", args.nav)

    # Shared state
    running  = True
    start_ts = time.time()
    last_stats = 0.0

    def _shutdown(sig, frame):  # noqa: ARG001
        nonlocal running
        log.info("\nShutdown signal received — finishing session...")
        running = False

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Alpaca websocket ──────────────────────────────────────────────────────
    from alpaca.data.live import StockDataStream

    stream = StockDataStream(api_key, api_secret)

    async def on_bar(bar) -> None:
        """Called by Alpaca for each 1-minute bar."""
        nonlocal last_stats
        sym = bar.symbol
        if sym not in sym_to_id:
            return

        sid   = sym_to_id[sym]
        price = float(bar.close)
        vol   = int(bar.volume) if bar.volume else 100

        # Alpaca bars don't carry bid/ask — estimate from VWAP spread heuristic
        bid = float(bar.vwap) * 0.9999 if bar.vwap else price * 0.9999
        ask = float(bar.vwap) * 1.0001 if bar.vwap else price * 1.0001

        tick = PaperTick(
            symbol_id=sid,
            price=price,
            volume=vol,
            timestamp_ns=time.time_ns(),
            bid=bid,
            ask=ask,
        )
        engine.on_tick(tick)

        now = time.time()
        if now - last_stats >= 1.0:
            print_stats(engine, now - start_ts)
            last_stats = now

        if not running:
            await stream.stop_ws()

    async def on_trade(trade) -> None:
        """Called by Alpaca for each trade print — higher frequency than bars."""
        nonlocal last_stats
        sym = trade.symbol
        if sym not in sym_to_id:
            return

        sid   = sym_to_id[sym]
        price = float(trade.price)
        vol   = int(trade.size) if trade.size else 1

        tick = PaperTick(
            symbol_id=sid,
            price=price,
            volume=vol,
            timestamp_ns=time.time_ns(),
        )
        engine.on_tick(tick)

        now = time.time()
        if now - last_stats >= 1.0:
            print_stats(engine, now - start_ts)
            last_stats = now

        if not running:
            await stream.stop_ws()

    if args.use_trades:
        stream.subscribe_trades(on_trade, *symbols)
        log.info("Subscribed to trade feed (tick-by-tick)")
    else:
        stream.subscribe_bars(on_bar, *symbols)
        log.info("Subscribed to 1-minute bar feed")

    log.info("Connecting to Alpaca websocket...")
    try:
        stream.run()
    except KeyboardInterrupt:
        pass

    # ── session report ────────────────────────────────────────────────────────
    elapsed = time.time() - start_ts
    s = engine.stats
    print()
    print()
    print("=" * 62)
    print("  SESSION REPORT")
    print("=" * 62)
    print(f"  Duration:        {elapsed:.0f}s")
    print(f"  Symbols:         {' '.join(symbols)}")
    print(f"  Ticks processed: {s.ticks_processed:,}")
    print(f"  Orders sent:     {s.orders_submitted}")
    print(f"  Orders filled:   {s.orders_filled}")
    print(f"  Orders blocked:  {s.orders_rejected}  ({s.risk_blocks} by risk)")
    print(f"  Kill activations:{s.kill_switch_activations}")
    print(f"  Final NAV:       ${s.final_nav:,.2f}")
    print(f"  Total PnL:       ${s.total_pnl:+,.2f}")
    print(f"  Peak NAV:        ${s.peak_nav:,.2f}")
    print(f"  Max drawdown:    {s.max_drawdown_pct:.2%}")
    print(f"  Alerts fired:    {s.alerts_fired}")
    print(f"  Recon runs:      {s.reconciliation_runs}")
    print(f"  Recon breaks:    {s.reconciliation_breaks}")
    print("=" * 62)

    # Save prometheus snapshot
    snap_dir = Path("data/sessions")
    snap_dir.mkdir(parents=True, exist_ok=True)
    snap_path = snap_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prom"
    with open(snap_path, "wb") as f:
        f.write(engine.metrics.snapshot())
    log.info("Metrics snapshot → %s", snap_path)

    # Save journal entry
    _write_journal(engine, symbols, elapsed)


def _write_journal(
    engine: PaperTradingEngine,
    symbols: list[str],
    elapsed_s: float,
) -> None:
    """Write a daily recap JSON to journals/YYYY-MM-DD.json."""
    import json
    from datetime import date

    s = engine.stats
    nav_start = engine.config.initial_nav
    nav_end   = s.final_nav
    gross_pnl = s.total_pnl
    net_pnl   = gross_pnl  # no financing cost model yet
    gross_pct = gross_pnl / nav_start if nav_start else 0.0

    positions = engine.broker.get_positions()
    pos_values = {
        sid: qty * engine.prices.get(sid, 0)
        for sid, qty in positions.items()
        if abs(qty) > 1e-9
    }
    gross_exp  = sum(abs(v) for v in pos_values.values())
    net_exp    = sum(pos_values.values())
    hit_ratio  = (
        s.orders_filled / s.orders_submitted
        if s.orders_submitted else 0.0
    )

    entry = {
        "_schema": "AlphaForge daily trading journal v1.0",
        "_description": "Automated daily recap — generated by run_alpaca.py",
        "date": date.today().isoformat(),
        "session": {
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(net_pnl, 2),
            "gross_pnl_pct": round(gross_pct, 6),
            "nav_start": round(nav_start, 2),
            "nav_end": round(nav_end, 2),
        },
        "execution": {
            "trades_executed": s.orders_filled,
            "hit_ratio": round(hit_ratio, 4),
            "profit_factor": 0.0,         # requires trade-level P&L log
            "avg_fill_vs_open": 0.0,
            "implementation_shortfall_bps": 0.0,
            "estimated_spread_cost_bps": 0.0,
            "passive_fills_pct": 0.0,
            "adverse_selection_events": 0,
        },
        "risk": {
            "portfolio_drawdown_pct": round(s.max_drawdown_pct, 6),
            "daily_var_95_pct": 0.0,
            "gross_exposure": round(gross_exp, 2),
            "net_exposure": round(net_exp, 2),
            "beta_to_spy": 0.0,
            "active_regime": "UNKNOWN",
            "regime_confidence": 0.0,
            "circuit_breaker_status": (
                "HALTED" if engine.kill_switch.level > 0 else "CLEAR"
            ),
        },
        "compliance": {
            "kill_switch_status": (
                "HALTED" if engine.kill_switch.level > 0 else "CLEAR"
            ),
            "bad_print_rejections": 0,
            "order_size_rejections": 0,
            "wash_trade_rejections": 0,
            "event_blackout_active": False,
            "active_blackout_events": [],
        },
        "sleeves": {
            "equity_core_pnl_pct": round(gross_pct, 6),
            "etf_sleeve_pnl_pct": round(gross_pct, 6),
            "futures_sleeve_pnl_pct": 0.0,
            "fx_sleeve_pnl_pct": 0.0,
            "options_overlay_pnl_pct": 0.0,
            "event_alpha_pnl_pct": 0.0,
        },
        "factor_attribution": {
            "beta_pnl_pct": 0.0,
            "sector_pnl_pct": 0.0,
            "momentum_factor_pnl_pct": 0.0,
            "quality_factor_pnl_pct": 0.0,
            "pure_alpha_pct": 0.0,
        },
        "top_contributors": [],
        "top_detractors": [],
        "notes": f"Alpaca paper session | {len(symbols)} symbols | {elapsed_s:.0f}s uptime",
    }

    journal_dir = Path("journals")
    journal_dir.mkdir(exist_ok=True)
    out_path = journal_dir / f"{date.today().isoformat()}.json"
    with open(out_path, "w") as f:
        json.dump(entry, f, indent=2)
    log.info("Journal entry → %s", out_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AlphaForge paper trading via Alpaca")
    p.add_argument("--config",           default="config/data_layer.yaml")
    p.add_argument("--symbols",          nargs="*",
                   help="Trade only these symbols (default: all sector ETFs)")
    p.add_argument("--nav",              type=float, default=1_000_000,
                   help="Starting NAV in USD (default: 1,000,000)")
    p.add_argument("--signal-threshold", type=float, default=0.1,
                   help="Min signal magnitude to submit an order (default: 0.1)")
    p.add_argument("--slippage",         type=float, default=1.0,
                   help="Slippage assumption in bps (default: 1.0)")
    p.add_argument("--max-drawdown",     type=float, default=0.15,
                   help="Auto-kill drawdown threshold (default: 0.15 = 15%%)")
    p.add_argument("--use-trades",       action="store_true",
                   help="Subscribe to trade prints instead of 1-min bars "
                        "(higher frequency, uses more of the free rate limit)")
    args = p.parse_args()
    run(args)
