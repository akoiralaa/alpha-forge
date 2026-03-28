"""Tests for the data quality pipeline."""

import pytest

from src.data.ingest.base import Tick, date_to_ns
from src.data.quality_pipeline import DataQualityPipeline, RejectionReason
from tests.phase1.conftest import make_tick, make_tick_series


class TestDataQualityPipeline:
    def _warmup(self, pipeline: DataQualityPipeline, symbol_id: int = 1, n: int = 200):
        """Feed enough good ticks to establish rolling statistics."""
        base = date_to_ns(2024, 1, 2)
        for i in range(n):
            tick = make_tick(
                symbol_id=symbol_id,
                exchange_time_ns=base + i * 1_000_000,
                bid=100.0,
                ask=100.05,
                last_price=100.02,
            )
            pipeline.check_tick(tick)

    def test_accept_valid_tick(self, quality_pipeline: DataQualityPipeline):
        tick = make_tick()
        result = quality_pipeline.check_tick(tick)
        assert result.accepted

    def test_reject_zero_price(self, quality_pipeline: DataQualityPipeline):
        tick = make_tick(last_price=0.0)
        result = quality_pipeline.check_tick(tick)
        assert not result.accepted
        assert RejectionReason.NEGATIVE_PRICE in result.rejection_reasons

    def test_reject_negative_price(self, quality_pipeline: DataQualityPipeline):
        tick = make_tick(last_price=-5.0)
        result = quality_pipeline.check_tick(tick)
        assert not result.accepted
        assert RejectionReason.NEGATIVE_PRICE in result.rejection_reasons

    def test_reject_negative_bid(self, quality_pipeline: DataQualityPipeline):
        tick = make_tick(bid=-1.0)
        result = quality_pipeline.check_tick(tick)
        assert not result.accepted
        assert RejectionReason.NEGATIVE_BID in result.rejection_reasons

    def test_reject_negative_ask(self, quality_pipeline: DataQualityPipeline):
        tick = make_tick(ask=-1.0)
        result = quality_pipeline.check_tick(tick)
        assert not result.accepted
        assert RejectionReason.NEGATIVE_ASK in result.rejection_reasons

    def test_reject_crossed_book(self, quality_pipeline: DataQualityPipeline):
        tick = make_tick(bid=101.0, ask=100.0)
        result = quality_pipeline.check_tick(tick)
        assert not result.accepted
        assert RejectionReason.CROSSED_BOOK in result.rejection_reasons

    def test_reject_equal_bid_ask(self, quality_pipeline: DataQualityPipeline):
        tick = make_tick(bid=100.0, ask=100.0)
        result = quality_pipeline.check_tick(tick)
        assert not result.accepted
        assert RejectionReason.CROSSED_BOOK in result.rejection_reasons

    def test_reject_time_travel(self, quality_pipeline: DataQualityPipeline):
        tick = Tick(
            exchange_time_ns=2_000_000_000,
            capture_time_ns=1_000_000_000,  # capture before exchange
            symbol_id=1, bid=100.0, ask=100.05, bid_size=100, ask_size=100,
            last_price=100.02, last_size=50,
        )
        result = quality_pipeline.check_tick(tick)
        assert not result.accepted
        assert RejectionReason.TIME_TRAVEL in result.rejection_reasons

    def test_reject_stale_data(self, quality_pipeline: DataQualityPipeline):
        tick = Tick(
            exchange_time_ns=1_000_000_000_000,
            capture_time_ns=1_000_000_000_000 + 15_000_000_000,  # 15s lag
            symbol_id=1, bid=100.0, ask=100.05, bid_size=100, ask_size=100,
            last_price=100.02, last_size=50,
        )
        result = quality_pipeline.check_tick(tick)
        assert not result.accepted
        assert RejectionReason.STALE_DATA in result.rejection_reasons

    def test_reject_fat_finger_price(self, quality_pipeline: DataQualityPipeline):
        self._warmup(quality_pipeline)
        # Price 50% above rolling mean
        tick = make_tick(
            exchange_time_ns=date_to_ns(2024, 1, 2) + 500_000_000,
            last_price=150.0,
            bid=149.95,
            ask=150.05,
        )
        result = quality_pipeline.check_tick(tick)
        assert not result.accepted
        assert RejectionReason.FAT_FINGER_PRICE in result.rejection_reasons

    def test_reject_fat_finger_volume(self, quality_pipeline: DataQualityPipeline):
        self._warmup(quality_pipeline)
        # Volume 200x rolling average
        tick = Tick(
            exchange_time_ns=date_to_ns(2024, 1, 2) + 500_000_000,
            capture_time_ns=date_to_ns(2024, 1, 2) + 500_100_000,
            symbol_id=1, bid=100.0, ask=100.05, bid_size=100, ask_size=100,
            last_price=100.02, last_size=10_000,  # 200x normal
        )
        result = quality_pipeline.check_tick(tick)
        assert not result.accepted
        assert RejectionReason.FAT_FINGER_VOLUME in result.rejection_reasons

    def test_reject_non_monotonic_time(self, quality_pipeline: DataQualityPipeline):
        t1 = make_tick(exchange_time_ns=2_000_000_000)
        t2 = make_tick(exchange_time_ns=1_000_000_000)  # goes backward
        quality_pipeline.check_tick(t1)
        result = quality_pipeline.check_tick(t2)
        assert not result.accepted
        assert RejectionReason.NON_MONOTONIC_TIME in result.rejection_reasons

    def test_multiple_rejections(self, quality_pipeline: DataQualityPipeline):
        # Zero price AND crossed book
        tick = make_tick(last_price=0.0, bid=101.0, ask=100.0)
        result = quality_pipeline.check_tick(tick)
        assert not result.accepted
        assert len(result.rejection_reasons) >= 2

    def test_batch_check(self, quality_pipeline: DataQualityPipeline):
        good = [make_tick(exchange_time_ns=1000 + i * 1000) for i in range(5)]
        bad = [make_tick(last_price=0.0, exchange_time_ns=99000 + i * 1000) for i in range(3)]
        accepted, rejected = quality_pipeline.check_batch(good + bad)
        assert len(accepted) == 5
        assert len(rejected) == 3

    def test_stats_tracking(self, quality_pipeline: DataQualityPipeline):
        for i in range(10):
            quality_pipeline.check_tick(make_tick(exchange_time_ns=1000 + i * 1000))
        quality_pipeline.check_tick(make_tick(last_price=0.0, exchange_time_ns=99000))

        stats = quality_pipeline.get_stats(1)
        assert stats.total_received == 11
        assert stats.total_rejected == 1

    def test_reset(self, quality_pipeline: DataQualityPipeline):
        quality_pipeline.check_tick(make_tick())
        quality_pipeline.reset()
        stats = quality_pipeline.get_stats()
        assert len(stats) == 0
