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

from src.data.ingest.polygon_event_backfill import build_active_equity_tickers
from src.data.ingest.polygon_event_backfill import ensure_equity_universe_in_symbol_master
from src.data.ingest.sec_companyfacts_backfill import SecCompanyFactsBackfiller


def parse_args():
    p = argparse.ArgumentParser(description="Backfill SEC company facts into PIT stores")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--symbol-master-db", default="data/symbol_master.db")
    p.add_argument("--fundamentals-db", default="data/fundamentals.db")
    p.add_argument("--events-db", default="data/events.db")
    p.add_argument("--cache-dir", default="data/cache/pit")
    p.add_argument("--config", default="config/sp500_universe.yaml")
    p.add_argument("--tickers", default="", help="Comma-separated ticker subset")
    p.add_argument("--max-tickers", type=int, default=0)
    p.add_argument("--request-sleep", type=float, default=0.2)
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--cache-only", action="store_true")
    p.add_argument("--prefer-hf", action="store_true")
    p.add_argument("--model-name", default="ProsusAI/finbert")
    return p.parse_args()


def main():
    args = parse_args()
    load_dotenv(args.env_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

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
        raise RuntimeError("No tickers resolved for SEC backfill")

    inserted = ensure_equity_universe_in_symbol_master(args.symbol_master_db, tickers)
    if inserted:
        logging.info("Inserted %d missing equity tickers into symbol master", inserted)

    user_agent = os.getenv(
        "SEC_USER_AGENT",
        "alphaforge/0.1 (research)",
    )
    logging.info("Backfilling SEC company facts for %d tickers", len(tickers))
    backfiller = SecCompanyFactsBackfiller(
        user_agent=user_agent,
        symbol_master_path=args.symbol_master_db,
        fundamentals_path=args.fundamentals_db,
        events_path=args.events_db,
        request_sleep=args.request_sleep,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
        cache_only=args.cache_only,
        prefer_hf=args.prefer_hf,
        hf_model_name=args.model_name,
    )
    try:
        backfiller.backfill(tickers)
        print(backfiller.summary())
    finally:
        backfiller.close()


if __name__ == "__main__":
    main()
