#!/usr/bin/env python3

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timezone

import yaml
try:
    from ib_async import IB, Stock, Forex, Future, Contract
except ImportError:
    from ib_insync import IB, Stock, Forex, Future, Contract

from src.paper.engine import PaperConfig, PaperTick, PaperTradingEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("runner")

# maps symbol -> (contract, symbol_id)
SYMBOL_MAP: dict[str, tuple[Contract, int]] = {}

def load_symbols(config_path: str) -> list[tuple[str, str]]:
    # returns list of (symbol, asset_type)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    instruments = cfg.get("instruments", {})
    symbols = []
    for sym in instruments.get("sector_etfs", []):
        symbols.append((sym, "ETF"))
    for sym in instruments.get("equity_index_futures", []):
        symbols.append((sym, "FUTURE"))
    for sym in instruments.get("commodity_futures", []):
        symbols.append((sym, "COMMODITY"))
    for sym in instruments.get("fixed_income_futures", []):
        symbols.append((sym, "BOND"))
    for sym in instruments.get("fx_pairs", []):
        symbols.append((sym, "FX"))
    for sym in instruments.get("vix_futures", []):
        symbols.append((sym, "VOLATILITY"))
    return symbols


def make_contract(symbol: str, asset_type: str) -> Contract:
    if asset_type in ("ETF", "EQUITY"):
        return Stock(symbol, "SMART", "USD")
    elif asset_type == "FX":
        return Forex(symbol[:3] + symbol[3:])
    elif asset_type in ("FUTURE", "COMMODITY", "BOND", "VOLATILITY"):
        return Future(symbol, exchange="CME")
    return Stock(symbol, "SMART", "USD")


def print_stats(engine: PaperTradingEngine, elapsed_s: float):
    s = engine.stats
    nav = s.final_nav
    pnl = s.total_pnl
    dd = s.max_drawdown_pct
    pos = len(engine.broker.get_positions())

    pnl_color = "\033[32m" if pnl >= 0 else "\033[31m"
    reset = "\033[0m"

    sys.stdout.write(
        f"\r  ticks: {s.ticks_processed:,}  |  "
        f"orders: {s.orders_submitted} sent / {s.orders_filled} filled  |  "
        f"NAV: ${nav:,.0f}  |  "
        f"PnL: {pnl_color}${pnl:+,.0f}{reset}  |  "
        f"DD: {dd:.2%}  |  "
        f"positions: {pos}  |  "
        f"uptime: {elapsed_s:.0f}s    "
    )
    sys.stdout.flush()


def run(args):
    # load IB connection settings
    ib_host = args.host
    ib_port = args.port
    client_id = args.client_id

    # load symbols
    symbols = load_symbols(args.config)
    if args.symbols:
        # filter to just the ones the user asked for
        requested = set(s.upper() for s in args.symbols)
        symbols = [(s, t) for s, t in symbols if s in requested]

    if not symbols:
        log.error("No symbols to trade. Check --config or --symbols")
        sys.exit(1)

    log.info("Starting paper trading session")
    log.info("Connecting to IB Gateway at %s:%d (client_id=%d)", ib_host, ib_port, client_id)

    ib = IB()
    ib.connect(ib_host, ib_port, clientId=client_id)
    log.info("Connected to IB Gateway")

    # set up engine
    config = PaperConfig(
        initial_nav=args.nav,
        signal_threshold=args.signal_threshold,
        slippage_bps=args.slippage,
        drawdown_auto_kill_pct=args.max_drawdown,
    )
    engine = PaperTradingEngine(config)

    # qualify and subscribe to contracts
    qualified = []
    for i, (sym, asset_type) in enumerate(symbols):
        contract = make_contract(sym, asset_type)
        try:
            result = ib.qualifyContracts(contract)
            if result:
                contract = result[0]
                sid = i + 1  # symbol_id starts at 1
                SYMBOL_MAP[sym] = (contract, sid)
                engine.broker.set_price(sid, 0)  # will be set on first tick
                qualified.append(sym)
                log.info("  [%d] %s (%s) qualified", sid, sym, asset_type)
            else:
                log.warning("  Could not qualify %s — skipping", sym)
        except Exception as e:
            log.warning("  Failed to qualify %s: %s — skipping", sym, e)

    if not qualified:
        log.error("No contracts qualified. Check IB Gateway and market data subscriptions.")
        ib.disconnect()
        sys.exit(1)

    log.info("Qualified %d/%d symbols", len(qualified), len(symbols))

    # request delayed data if live isn't available
    ib.reqMarketDataType(4)  # 4 = delayed frozen, 3 = delayed, 1 = live
    log.info("Requesting delayed market data (upgrade IB subscription for live)")

    # subscribe to market data
    contract_to_sid = {}
    for sym, (contract, sid) in SYMBOL_MAP.items():
        ib.reqMktData(contract, "", False, False)
        contract_to_sid[contract.conId] = (sym, sid)

    log.info("Subscribed to market data. Waiting for ticks...")
    log.info("Press Ctrl+C to stop\n")

    # graceful shutdown
    running = True
    def on_signal(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    start_time = time.time()
    last_stats_print = 0

    while running and ib.isConnected():
        ib.sleep(0.05)  # IB event loop — processes callbacks

        for ticker in ib.tickers():
            if ticker.contract is None or ticker.contract.conId not in contract_to_sid:
                continue

            # skip if no valid price yet
            last = ticker.last
            if last != last or last <= 0:  # NaN check
                bid = ticker.bid if ticker.bid == ticker.bid else 0
                ask = ticker.ask if ticker.ask == ticker.ask else 0
                if bid > 0 and ask > 0:
                    last = (bid + ask) / 2
                else:
                    continue

            sym, sid = contract_to_sid[ticker.contract.conId]
            vol = int(ticker.volume) if ticker.volume == ticker.volume and ticker.volume > 0 else 100

            tick = PaperTick(
                symbol_id=sid,
                price=last,
                volume=vol,
                timestamp_ns=time.time_ns(),
                bid=ticker.bid if ticker.bid == ticker.bid else 0,
                ask=ticker.ask if ticker.ask == ticker.ask else 0,
            )
            engine.on_tick(tick)

        # print stats every second
        now = time.time()
        if now - last_stats_print >= 1.0:
            print_stats(engine, now - start_time)
            last_stats_print = now

    # shutdown
    print()
    log.info("Shutting down...")

    # cancel market data
    for sym, (contract, sid) in SYMBOL_MAP.items():
        try:
            ib.cancelMktData(contract)
        except Exception:
            pass

    ib.disconnect()

    # final report
    s = engine.stats
    print()
    print("=" * 60)
    print("  SESSION REPORT")
    print("=" * 60)
    print(f"  Duration:       {time.time() - start_time:.0f}s")
    print(f"  Ticks:          {s.ticks_processed:,}")
    print(f"  Orders sent:    {s.orders_submitted}")
    print(f"  Orders filled:  {s.orders_filled}")
    print(f"  Orders blocked: {s.orders_rejected} ({s.risk_blocks} risk)")
    print(f"  Final NAV:      ${s.final_nav:,.2f}")
    print(f"  Total PnL:      ${s.total_pnl:+,.2f}")
    print(f"  Peak NAV:       ${s.peak_nav:,.2f}")
    print(f"  Max drawdown:   {s.max_drawdown_pct:.2%}")
    print(f"  Alerts fired:   {s.alerts_fired}")
    print(f"  Recon runs:     {s.reconciliation_runs}")
    print(f"  Recon breaks:   {s.reconciliation_breaks}")
    print("=" * 60)

    # dump prometheus snapshot
    snap_path = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prom"
    with open(snap_path, "wb") as f:
        f.write(engine.metrics.snapshot())
    log.info("Metrics snapshot saved to %s", snap_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paper trading session")
    parser.add_argument("--host", default="127.0.0.1", help="IB Gateway host")
    parser.add_argument("--port", type=int, default=7497, help="IB Gateway port (7497=paper, 7496=live)")
    parser.add_argument("--client-id", type=int, default=1, help="IB client ID")
    parser.add_argument("--config", default="config/data_layer.yaml", help="instrument config")
    parser.add_argument("--symbols", nargs="*", help="trade only these symbols (e.g. SPY QQQ)")
    parser.add_argument("--nav", type=float, default=1_000_000, help="starting NAV")
    parser.add_argument("--signal-threshold", type=float, default=0.1, help="min signal to trade")
    parser.add_argument("--slippage", type=float, default=1.0, help="slippage in bps")
    parser.add_argument("--max-drawdown", type=float, default=0.15, help="auto-kill drawdown pct")
    args = parser.parse_args()
    run(args)
