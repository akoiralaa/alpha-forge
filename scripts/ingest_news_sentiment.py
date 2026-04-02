#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd

from src.data.events import EventRecord, EventStore
from src.data.symbol_master import SymbolMaster
from src.signals.sentiment import build_sentiment_model


def parse_args():
    p = argparse.ArgumentParser(description="Score event/news CSV into EventStore")
    p.add_argument("--input", required=True, help="CSV with ticker,published_at,headline[,body,event_type,source]")
    p.add_argument("--events-db", default="data/events.db")
    p.add_argument("--symbol-master-db", default="data/symbol_master.db")
    p.add_argument("--prefer-hf", action="store_true", help="Use Hugging Face sentiment if transformers/model are available")
    p.add_argument("--model-name", default="ProsusAI/finbert")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    required = {"ticker", "published_at", "headline"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    model = build_sentiment_model(prefer_hf=args.prefer_hf, model_name=args.model_name)
    sm = SymbolMaster(args.symbol_master_db)
    store = EventStore(args.events_db)
    records: list[EventRecord] = []

    try:
        for row in df.itertuples(index=False):
            ts = pd.Timestamp(getattr(row, "published_at"))
            published_ns = int(ts.value)
            ticker = str(getattr(row, "ticker"))
            canonical_id = sm.resolve_ticker(ticker, published_ns)
            if canonical_id is None:
                continue

            headline = str(getattr(row, "headline"))
            body = str(getattr(row, "body", "") or "")
            score = model.score_text(f"{headline}. {body}".strip())

            records.append(
                EventRecord(
                    canonical_id=canonical_id,
                    ticker=ticker,
                    published_at_ns=published_ns,
                    event_type=str(getattr(row, "event_type", "news")),
                    source=str(getattr(row, "source", "csv_ingest")),
                    headline=headline,
                    body=body,
                    sentiment_score=score.score,
                    relevance=float(getattr(row, "relevance", 1.0) or 1.0),
                    novelty=float(getattr(row, "novelty", 1.0) or 1.0),
                    confidence=score.confidence,
                    metadata=f"model={score.model_name};label={score.label}",
                )
            )

        if records:
            store.add_records_batch(records)
        print(f"Inserted {len(records)} event records into {args.events_db}")
    finally:
        sm.close()
        store.close()


if __name__ == "__main__":
    main()
