#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ingest.finbert_event_backfill import FinBertEventBackfiller


def parse_args():
    p = argparse.ArgumentParser(description="Backfill FinBERT sentiment events into local PIT store")
    p.add_argument("--events-db", default="data/events.db")
    p.add_argument("--cache-dir", default="data/cache/pit")
    p.add_argument("--model-name", default="ProsusAI/finbert")
    p.add_argument("--prefer-hf", action="store_true", default=True)
    p.add_argument("--allow-lexicon-fallback", action="store_true")
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--cache-only", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--since", default="", help="ISO datetime or ns timestamp")
    p.add_argument(
        "--event-types",
        default="news,guidance,earnings,estimate_revision,m&a,product",
        help="Comma-separated event types to enrich",
    )
    p.add_argument(
        "--sources",
        default="",
        help="Optional comma-separated source allowlist (default: all non-FINBERT sources)",
    )
    p.add_argument("--min-confidence", type=float, default=0.05)
    return p.parse_args()


def _csv_arg(value: str) -> list[str]:
    text = (value or "").strip()
    if not text:
        return []
    return [chunk.strip() for chunk in text.split(",") if chunk.strip()]


def main():
    args = parse_args()
    backfiller = FinBertEventBackfiller(
        events_path=args.events_db,
        cache_dir=args.cache_dir,
        model_name=args.model_name,
        prefer_hf=bool(args.prefer_hf),
        allow_lexicon_fallback=bool(args.allow_lexicon_fallback),
        refresh_cache=bool(args.refresh_cache),
        cache_only=bool(args.cache_only),
    )
    try:
        backfiller.backfill(
            source_filter=_csv_arg(args.sources),
            event_types=_csv_arg(args.event_types),
            since=args.since.strip() or None,
            limit=max(int(args.limit), 0),
            min_confidence=float(args.min_confidence),
        )
        print(json.dumps(backfiller.summary(), indent=2, sort_keys=True))
    finally:
        backfiller.close()


if __name__ == "__main__":
    main()

