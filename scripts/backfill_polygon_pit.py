#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ingest.polygon_event_backfill import (
    PolygonEventBackfiller,
    build_active_equity_tickers,
    ensure_equity_universe_in_symbol_master,
)


def parse_args():
    p = argparse.ArgumentParser(description="Backfill point-in-time Polygon/Benzinga event data")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--symbol-master-db", default="data/symbol_master.db")
    p.add_argument("--fundamentals-db", default="data/fundamentals.db")
    p.add_argument("--events-db", default="data/events.db")
    p.add_argument("--cache-dir", default="data/cache/pit")
    p.add_argument("--config", default="config/sp500_universe.yaml")
    p.add_argument("--start-date", default="2016-01-01")
    p.add_argument("--end-date", default="")
    p.add_argument("--tickers", default="", help="Comma-separated ticker subset")
    p.add_argument("--max-tickers", type=int, default=0)
    p.add_argument("--chunk-size", type=int, default=25)
    p.add_argument("--prefer-hf", action="store_true")
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--cache-only", action="store_true")
    p.add_argument("--skip-earnings", action="store_true")
    p.add_argument("--skip-guidance", action="store_true")
    p.add_argument("--skip-analyst", action="store_true")
    p.add_argument("--skip-news", action="store_true")
    p.add_argument("--skip-financials", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    load_dotenv(args.env_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    api_key = os.getenv("POLYGON_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY is missing")

    if args.tickers.strip():
        tickers = [tok.strip().upper() for tok in args.tickers.split(",") if tok.strip()]
    else:
        tickers = []
        config_path = Path(args.config)
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
            tickers = list((raw.get("instruments") or {}).get("equities") or [])
        if not tickers:
            tickers = build_active_equity_tickers(args.symbol_master_db)
    if args.max_tickers > 0:
        tickers = tickers[:args.max_tickers]
    if not tickers:
        raise RuntimeError("No tickers resolved for backfill")

    inserted = ensure_equity_universe_in_symbol_master(args.symbol_master_db, tickers)
    if inserted:
        logging.info("Inserted %d missing equity tickers into symbol master", inserted)

    end_date = args.end_date
    if not end_date:
        from datetime import datetime, timezone
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    logging.info("Backfilling Polygon PIT data for %d tickers (%s to %s)", len(tickers), args.start_date, end_date)
    backfiller = PolygonEventBackfiller(
        api_key=api_key,
        symbol_master_path=args.symbol_master_db,
        fundamentals_path=args.fundamentals_db,
        events_path=args.events_db,
        rate_limit_per_minute=int(os.getenv("POLYGON_RATE_LIMIT_PER_MIN", "5")),
        prefer_hf=args.prefer_hf,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
        cache_only=args.cache_only,
    )
    try:
        if not args.skip_earnings:
            backfiller.backfill_earnings(tickers, args.start_date, end_date, chunk_size=args.chunk_size)
        if not args.skip_guidance:
            backfiller.backfill_guidance(tickers, args.start_date, end_date, chunk_size=args.chunk_size)
        if not args.skip_analyst:
            backfiller.backfill_analyst_insights(tickers, args.start_date, end_date, chunk_size=args.chunk_size)
        if not args.skip_news:
            backfiller.backfill_news(tickers, args.start_date, end_date, chunk_size=args.chunk_size)
        if not args.skip_financials:
            backfiller.backfill_income_statements(tickers, args.start_date, end_date, chunk_size=args.chunk_size)
        print(backfiller.summary())
    finally:
        backfiller.close()


if __name__ == "__main__":
    main()
