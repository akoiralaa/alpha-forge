#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ingest.daily_bar_cache import load_universe_static
from src.data.ingest.polygon_event_backfill import (
    build_active_equity_tickers,
    ensure_equity_universe_in_symbol_master,
)
from src.data.ingest.yahoo_event_backfill import YahooEventBackfiller


DEFAULT_INTRADAY_CACHE_DIR = Path("data/cache/intraday")


def parse_args():
    p = argparse.ArgumentParser(description="Backfill Yahoo point-in-time events/revisions and optional intraday bars")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--config", default="config/sp500_universe.yaml")
    p.add_argument("--symbol-master-db", default="data/symbol_master.db")
    p.add_argument("--fundamentals-db", default="data/fundamentals.db")
    p.add_argument("--events-db", default="data/events.db")
    p.add_argument("--cache-dir", default="data/cache/pit")
    p.add_argument("--symbols", default="", help="Comma-separated subset")
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument(
        "--quote-modules",
        default="earningsHistory,earningsTrend,upgradeDowngradeHistory,recommendationTrend,calendarEvents",
    )
    p.add_argument("--chunk-size", type=int, default=40)
    p.add_argument("--sleep-seconds", type=float, default=0.25)
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--cache-only", action="store_true")
    p.add_argument("--with-intraday", action="store_true")
    p.add_argument("--intraday-cache-dir", default=str(DEFAULT_INTRADAY_CACHE_DIR))
    p.add_argument("--intraday-interval", default="1m")
    p.add_argument("--intraday-range", default="60d")
    return p.parse_args()


def resolve_tickers(args) -> list[str]:
    tickers: list[str] = []
    if args.symbols.strip():
        tickers = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        tickers = build_active_equity_tickers(args.symbol_master_db)
        if not tickers:
            universe = load_universe_static(args.config)
            tickers = sorted({sym for sym, atype in universe if atype == "EQUITY"})
    if args.max_symbols > 0:
        tickers = tickers[: args.max_symbols]
    return tickers


def main():
    args = parse_args()
    load_dotenv(args.env_file)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    tickers = resolve_tickers(args)
    if not tickers:
        raise RuntimeError("No tickers resolved for Yahoo PIT backfill")

    inserted = ensure_equity_universe_in_symbol_master(args.symbol_master_db, tickers)
    if inserted:
        logging.info("Inserted %d missing symbols into symbol master", inserted)
    logging.info("Yahoo PIT scope: %d tickers", len(tickers))

    modules = [m.strip() for m in args.quote_modules.split(",") if m.strip()]
    backfiller = YahooEventBackfiller(
        symbol_master_path=args.symbol_master_db,
        fundamentals_path=args.fundamentals_db,
        events_path=args.events_db,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
        cache_only=args.cache_only,
        sleep_seconds=args.sleep_seconds,
    )
    try:
        backfiller.backfill_quote_summary(
            tickers,
            modules=modules,
            chunk_size=max(args.chunk_size, 1),
        )
        if args.with_intraday:
            intraday_stats = backfiller.backfill_intraday_bars(
                tickers,
                cache_dir=args.intraday_cache_dir,
                interval=args.intraday_interval,
                range_name=args.intraday_range,
            )
            logging.info("Intraday cache stats: %s", intraday_stats)
        print(json.dumps(backfiller.summary(), indent=2, sort_keys=True))
    finally:
        backfiller.close()


if __name__ == "__main__":
    main()
