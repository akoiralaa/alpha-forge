from __future__ import annotations

from src.data.events import EventRecord, EventStore
from src.data.ingest.base import date_to_ns


class TestEventStore:
    def test_add_and_recent_lookup(self, tmp_dir):
        store = EventStore(tmp_dir / "events.db")
        try:
            store.add_records_batch(
                [
                    EventRecord(
                        canonical_id=1,
                        ticker="AAPL",
                        published_at_ns=date_to_ns(2024, 1, 5),
                        event_type="earnings",
                        source="test",
                        headline="AAPL beats earnings",
                        sentiment_score=0.8,
                    ),
                    EventRecord(
                        canonical_id=1,
                        ticker="AAPL",
                        published_at_ns=date_to_ns(2024, 2, 5),
                        event_type="guidance",
                        source="test",
                        headline="AAPL raises outlook",
                        sentiment_score=0.7,
                    ),
                ]
            )

            recent = store.get_recent(
                canonical_id=1,
                as_of_ns=date_to_ns(2024, 2, 10),
                lookback_ns=40 * 86_400_000_000_000,
            )
            assert len(recent) == 2
            assert recent[0].headline == "AAPL raises outlook"
            assert store.count_records() == 2
        finally:
            store.close()

    def test_recent_by_ticker_respects_as_of(self, tmp_dir):
        store = EventStore(tmp_dir / "events.db")
        try:
            store.add_record(
                EventRecord(
                    canonical_id=2,
                    ticker="MSFT",
                    published_at_ns=date_to_ns(2024, 3, 1),
                    event_type="news",
                    source="test",
                    headline="MSFT launches new product",
                    sentiment_score=0.4,
                )
            )
            before = store.get_recent_by_ticker(
                "MSFT",
                as_of_ns=date_to_ns(2024, 2, 28),
                lookback_ns=10 * 86_400_000_000_000,
            )
            after = store.get_recent_by_ticker(
                "MSFT",
                as_of_ns=date_to_ns(2024, 3, 5),
                lookback_ns=10 * 86_400_000_000_000,
            )
            assert before == []
            assert len(after) == 1
        finally:
            store.close()
