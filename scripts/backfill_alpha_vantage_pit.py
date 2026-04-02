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

from src.data.ingest.alpha_vantage_event_backfill import AlphaVantageEventBackfiller
from src.data.ingest.daily_bar_cache import load_universe_static
from src.data.ingest.polygon_event_backfill import (
    build_active_equity_tickers,
    ensure_equity_universe_in_symbol_master,
)


def parse_args():
    p = argparse.ArgumentParser(description="Backfill Alpha Vantage PIT events and revisions")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--api-key", default="")
    p.add_argument("--config", default="config/sp500_universe.yaml")
    p.add_argument("--symbol-master-db", default="data/symbol_master.db")
    p.add_argument("--fundamentals-db", default="data/fundamentals.db")
    p.add_argument("--events-db", default="data/events.db")
    p.add_argument("--cache-dir", default="data/cache/pit")
    p.add_argument("--symbols", default="", help="Comma-separated subset")
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--requests-per-minute", type=int, default=5)
    p.add_argument("--chunk-size", type=int, default=25)
    p.add_argument("--no-news", action="store_true")
    p.add_argument("--no-calendar", action="store_true")
    p.add_argument("--allow-demo", action="store_true")
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--cache-only", action="store_true")
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

    api_key = args.api_key.strip() or os.getenv("ALPHAVANTAGE_API_KEY", "").strip() or "demo"
    if api_key == "demo" and not args.allow_demo:
        raise RuntimeError(
            "ALPHAVANTAGE_API_KEY is not set. Add it to .env or pass --api-key. "
            "Use --allow-demo only for quick smoke tests."
        )
    tickers = resolve_tickers(args)
    if not tickers:
        raise RuntimeError("No tickers resolved for Alpha Vantage backfill")

    inserted = ensure_equity_universe_in_symbol_master(args.symbol_master_db, tickers)
    if inserted:
        logging.info("Inserted %d missing symbols into symbol master", inserted)
    logging.info("Alpha Vantage scope: %d tickers | api_key=%s", len(tickers), "demo" if api_key == "demo" else "configured")

    backfiller = AlphaVantageEventBackfiller(
        api_key=api_key,
        symbol_master_path=args.symbol_master_db,
        fundamentals_path=args.fundamentals_db,
        events_path=args.events_db,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
        cache_only=args.cache_only,
        requests_per_minute=max(args.requests_per_minute, 1),
    )
    try:
        backfiller.backfill(
            tickers,
            include_news=not args.no_news,
            include_calendar=not args.no_calendar,
            chunk_size=max(args.chunk_size, 1),
        )
        print(json.dumps(backfiller.summary(), indent=2, sort_keys=True))
    finally:
        backfiller.close()


if __name__ == "__main__":
    main()
