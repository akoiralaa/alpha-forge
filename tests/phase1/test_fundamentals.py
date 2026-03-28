"""Tests for point-in-time fundamental data store."""

import pytest

from src.data.fundamentals import FundamentalRecord, FundamentalsStore
from src.data.ingest.base import date_to_ns


class TestFundamentalsStore:
    def test_add_and_retrieve(self, fundamentals: FundamentalsStore):
        record = FundamentalRecord(
            canonical_id=1,
            metric_name="EPS",
            value=3.50,
            published_at_ns=date_to_ns(2019, 4, 15),
            period_end_ns=date_to_ns(2019, 3, 31),
            source="SEC_FILING",
        )
        fundamentals.add_record(record)

        result = fundamentals.get_as_of(1, "EPS", date_to_ns(2019, 6, 1))
        assert result is not None
        assert result.value == 3.50
        assert result.published_at_ns == date_to_ns(2019, 4, 15)

    def test_pit_integrity_no_lookahead(self, fundamentals: FundamentalsStore):
        """The critical PIT test: querying before publication returns nothing."""
        record = FundamentalRecord(
            canonical_id=1,
            metric_name="EPS",
            value=3.50,
            published_at_ns=date_to_ns(2019, 4, 15),  # published Apr 15
            period_end_ns=date_to_ns(2019, 3, 31),
            source="SEC_FILING",
        )
        fundamentals.add_record(record)

        # Query as-of April 10 — before publication. Must return None.
        result = fundamentals.get_as_of(1, "EPS", date_to_ns(2019, 4, 10))
        assert result is None

    def test_pit_returns_latest_known(self, fundamentals: FundamentalsStore):
        """Multiple publications: as-of query returns the most recent known value."""
        # Q4 2018 EPS published Jan 2019
        fundamentals.add_record(FundamentalRecord(
            canonical_id=1, metric_name="EPS", value=2.80,
            published_at_ns=date_to_ns(2019, 1, 20),
            period_end_ns=date_to_ns(2018, 12, 31), source="SEC",
        ))
        # Q1 2019 EPS published Apr 2019
        fundamentals.add_record(FundamentalRecord(
            canonical_id=1, metric_name="EPS", value=3.50,
            published_at_ns=date_to_ns(2019, 4, 15),
            period_end_ns=date_to_ns(2019, 3, 31), source="SEC",
        ))
        # Q2 2019 EPS published Jul 2019
        fundamentals.add_record(FundamentalRecord(
            canonical_id=1, metric_name="EPS", value=3.80,
            published_at_ns=date_to_ns(2019, 7, 20),
            period_end_ns=date_to_ns(2019, 6, 30), source="SEC",
        ))

        # As-of March 2019: should get Q4 2018 value (2.80)
        result = fundamentals.get_as_of(1, "EPS", date_to_ns(2019, 3, 1))
        assert result.value == 2.80

        # As-of June 2019: should get Q1 2019 value (3.50), NOT Q2 2019
        result = fundamentals.get_as_of(1, "EPS", date_to_ns(2019, 6, 1))
        assert result.value == 3.50
        assert result.published_at_ns <= date_to_ns(2019, 6, 1)

        # As-of August 2019: should get Q2 2019 value (3.80)
        result = fundamentals.get_as_of(1, "EPS", date_to_ns(2019, 8, 1))
        assert result.value == 3.80

    def test_restated_values_dont_leak(self, fundamentals: FundamentalsStore):
        """A restated value published later should not appear in earlier queries."""
        # Original Q1 EPS published Apr 15
        fundamentals.add_record(FundamentalRecord(
            canonical_id=1, metric_name="EPS", value=3.50,
            published_at_ns=date_to_ns(2019, 4, 15),
            period_end_ns=date_to_ns(2019, 3, 31), source="SEC",
        ))
        # Restated Q1 EPS published Aug 1 (correction)
        fundamentals.add_record(FundamentalRecord(
            canonical_id=1, metric_name="EPS", value=3.42,
            published_at_ns=date_to_ns(2019, 8, 1),
            period_end_ns=date_to_ns(2019, 3, 31), source="SEC_RESTATEMENT",
        ))

        # As-of June: should get original 3.50, NOT restated 3.42
        result = fundamentals.get_as_of(1, "EPS", date_to_ns(2019, 6, 1))
        assert result.value == 3.50

        # As-of September: should get restated 3.42
        result = fundamentals.get_as_of(1, "EPS", date_to_ns(2019, 9, 1))
        assert result.value == 3.42

    def test_get_all_as_of(self, fundamentals: FundamentalsStore):
        fundamentals.add_record(FundamentalRecord(
            canonical_id=1, metric_name="EPS", value=3.50,
            published_at_ns=date_to_ns(2019, 4, 15),
            period_end_ns=date_to_ns(2019, 3, 31), source="SEC",
        ))
        fundamentals.add_record(FundamentalRecord(
            canonical_id=1, metric_name="MARKET_CAP", value=1_000_000_000,
            published_at_ns=date_to_ns(2019, 4, 15),
            period_end_ns=date_to_ns(2019, 3, 31), source="CALC",
        ))

        all_metrics = fundamentals.get_all_as_of(1, date_to_ns(2019, 6, 1))
        assert "EPS" in all_metrics
        assert "MARKET_CAP" in all_metrics
        assert all_metrics["EPS"].value == 3.50

    def test_batch_insert(self, fundamentals: FundamentalsStore):
        records = [
            FundamentalRecord(
                canonical_id=i, metric_name="EPS", value=float(i),
                published_at_ns=date_to_ns(2019, 4, 15),
                period_end_ns=date_to_ns(2019, 3, 31), source="SEC",
            )
            for i in range(1, 101)
        ]
        fundamentals.add_records_batch(records)
        assert fundamentals.count_records() == 100

    def test_history(self, fundamentals: FundamentalsStore):
        for q, month in enumerate([1, 4, 7, 10], 1):
            fundamentals.add_record(FundamentalRecord(
                canonical_id=1, metric_name="EPS", value=float(q),
                published_at_ns=date_to_ns(2019, month, 15),
                period_end_ns=date_to_ns(2019, month, 1), source="SEC",
            ))

        history = fundamentals.get_history(1, "EPS")
        assert len(history) == 4
        assert history[0].value == 1.0
        assert history[3].value == 4.0
