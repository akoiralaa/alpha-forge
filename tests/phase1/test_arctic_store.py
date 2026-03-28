"""Tests for ArcticDB tick store."""

import numpy as np
import pandas as pd
import pytest

from src.data.arctic_store import TickStore
from src.data.ingest.base import Tick
from tests.phase1.conftest import make_tick, make_tick_series


class TestTickStore:
    def test_write_and_read_raw_ticks(self, tick_store: TickStore):
        ticks = make_tick_series(symbol_id=1, n=100)
        df = Tick.ticks_to_dataframe(ticks)
        tick_store.write_ticks_raw(1, df)

        result = tick_store.read_ticks_raw(1)
        assert len(result) == 100
        assert set(result.columns) >= set(Tick.schema_dtypes().keys())

    def test_read_with_time_filter(self, tick_store: TickStore):
        ticks = make_tick_series(symbol_id=1, n=100, start_ns=1_000_000_000_000)
        df = Tick.ticks_to_dataframe(ticks)
        tick_store.write_ticks_raw(1, df)

        mid_time = ticks[50].capture_time_ns
        result = tick_store.read_ticks_raw(1, start_ns=mid_time)
        assert len(result) <= 51  # roughly second half

    def test_append_ticks(self, tick_store: TickStore):
        ticks1 = make_tick_series(symbol_id=1, n=50, start_ns=1_000_000_000_000)
        ticks2 = make_tick_series(symbol_id=1, n=50, start_ns=2_000_000_000_000)
        df1 = Tick.ticks_to_dataframe(ticks1)
        df2 = Tick.ticks_to_dataframe(ticks2)

        tick_store.write_ticks_raw(1, df1)
        tick_store.write_ticks_raw(1, df2)

        result = tick_store.read_ticks_raw(1)
        assert len(result) == 100

    def test_list_symbols(self, tick_store: TickStore):
        for sid in [1, 5, 10]:
            ticks = make_tick_series(symbol_id=sid, n=10)
            df = Tick.ticks_to_dataframe(ticks)
            tick_store.write_ticks_raw(sid, df)

        symbols = tick_store.list_symbols()
        assert symbols == [1, 5, 10]

    def test_has_symbol(self, tick_store: TickStore):
        assert not tick_store.has_symbol(1)
        ticks = make_tick_series(symbol_id=1, n=10)
        tick_store.write_ticks_raw(1, Tick.ticks_to_dataframe(ticks))
        assert tick_store.has_symbol(1)
        assert not tick_store.has_symbol(2)

    def test_tick_count(self, tick_store: TickStore):
        ticks = make_tick_series(symbol_id=1, n=42)
        tick_store.write_ticks_raw(1, Tick.ticks_to_dataframe(ticks))
        assert tick_store.tick_count(1) == 42

    def test_rejections_storage(self, tick_store: TickStore):
        df = pd.DataFrame({
            "exchange_time_ns": [1000],
            "capture_time_ns": [1100],
            "symbol_id": [1],
            "bid": [0.0],
            "ask": [100.0],
            "bid_size": [100],
            "ask_size": [100],
            "last_price": [0.0],
            "last_size": [50],
            "trade_condition": [0],
            "rejection_reason": ["last_price <= 0"],
        })
        tick_store.write_rejections(1, df)
        result = tick_store.read_rejections(1)
        assert len(result) == 1
        assert result["rejection_reason"].iloc[0] == "last_price <= 0"

    def test_validate_unsorted_raises(self, tick_store: TickStore):
        df = pd.DataFrame({
            "exchange_time_ns": np.array([200, 100], dtype=np.int64),
            "capture_time_ns": np.array([300, 200], dtype=np.int64),
            "symbol_id": np.array([1, 1], dtype=np.int32),
            "bid": np.array([100.0, 100.0]),
            "ask": np.array([100.05, 100.05]),
            "bid_size": np.array([100, 100], dtype=np.int64),
            "ask_size": np.array([100, 100], dtype=np.int64),
            "last_price": np.array([100.02, 100.02]),
            "last_size": np.array([50, 50], dtype=np.int64),
            "trade_condition": np.array([0, 0], dtype=np.uint8),
        })
        with pytest.raises(ValueError, match="sorted"):
            tick_store.write_ticks_raw(1, df)

    def test_sample_random_records(self, tick_store: TickStore):
        for sid in [1, 2, 3]:
            ticks = make_tick_series(symbol_id=sid, n=100)
            tick_store.write_ticks_raw(sid, Tick.ticks_to_dataframe(ticks))

        sample = tick_store.sample_random_records(n=50)
        assert len(sample) == 50

    def test_bars_write_read(self, tick_store: TickStore):
        df = pd.DataFrame({
            "timestamp_ns": np.array([1000, 2000, 3000], dtype=np.int64),
            "open": [100.0, 100.1, 100.2],
            "high": [100.5, 100.6, 100.7],
            "low": [99.5, 99.6, 99.7],
            "close": [100.2, 100.3, 100.4],
            "volume": np.array([1000, 1100, 1200], dtype=np.int64),
            "vwap": [100.1, 100.2, 100.3],
        })
        tick_store.write_bars_raw(1, df)
        result = tick_store.read_bars_raw(1)
        assert len(result) == 3
