from __future__ import annotations

from src.data.events import EventRecord, EventStore
from src.data.ingest.base import date_to_ns
from src.data.ingest.finbert_event_backfill import FinBertEventBackfiller


def test_finbert_backfill_inserts_enriched_records(tmp_path):
    events_path = tmp_path / "events.db"
    store = EventStore(events_path)
    try:
        store.add_record(
            EventRecord(
                canonical_id=1,
                ticker="AAPL",
                published_at_ns=date_to_ns(2025, 1, 15),
                event_type="news",
                source="ALPHA_VANTAGE_NEWS",
                headline="AAPL beats earnings expectations",
                body="Revenue and margin expanded strongly.",
                sentiment_score=0.1,
                confidence=0.4,
            )
        )
    finally:
        store.close()

    backfiller = FinBertEventBackfiller(
        events_path=str(events_path),
        cache_dir=str(tmp_path / "cache"),
        allow_lexicon_fallback=True,
    )
    try:
        backfiller.backfill(limit=100)
        summary = backfiller.summary()
    finally:
        backfiller.close()

    assert summary["inserted"] == 1
    assert summary["raw_cache_writes"] == 1

    verify = EventStore(events_path)
    try:
        df = verify.to_dataframe()
    finally:
        verify.close()

    finbert_rows = df[df["source"] == "FINBERT_ENRICHED"]
    assert len(finbert_rows) == 1
    assert finbert_rows.iloc[0]["event_type"] == "news"
    assert abs(float(finbert_rows.iloc[0]["sentiment_score"])) <= 1.0


def test_finbert_backfill_cache_only_uses_cached_scores(tmp_path):
    events_path = tmp_path / "events.db"
    store = EventStore(events_path)
    try:
        store.add_record(
            EventRecord(
                canonical_id=2,
                ticker="MSFT",
                published_at_ns=date_to_ns(2025, 2, 20),
                event_type="guidance",
                source="ALPHA_VANTAGE_NEWS",
                headline="MSFT guidance raised with robust cloud demand",
                body="Management raises full-year guide.",
                sentiment_score=0.0,
                confidence=0.3,
            )
        )
    finally:
        store.close()

    warm = FinBertEventBackfiller(
        events_path=str(events_path),
        cache_dir=str(tmp_path / "cache"),
        allow_lexicon_fallback=True,
    )
    try:
        warm.backfill(limit=100)
    finally:
        warm.close()

    offline = FinBertEventBackfiller(
        events_path=str(events_path),
        cache_dir=str(tmp_path / "cache"),
        allow_lexicon_fallback=True,
        cache_only=True,
    )
    try:
        offline.backfill(limit=100)
        summary = offline.summary()
    finally:
        offline.close()

    assert summary["raw_cache_hits"] >= 1
    assert summary["raw_cache_misses"] == 0
    assert summary["inserted"] == 0
