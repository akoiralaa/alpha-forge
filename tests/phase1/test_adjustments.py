"""Tests for price adjustment engine."""

import numpy as np
import pandas as pd
import pytest

from src.data.adjustments import PriceAdjuster
from src.data.ingest.base import AssetClass, Tick, date_to_ns
from src.data.symbol_master import SymbolMaster


class TestPriceAdjuster:
    def _setup_stock_with_split(self, sm: SymbolMaster) -> int:
        """Create a stock with a 2:1 split and return its canonical_id."""
        cid = sm.add_instrument(
            exchange="NASDAQ", ticker="TEST", valid_from_ns=date_to_ns(2020, 1, 1),
            asset_class=AssetClass.EQUITY, currency="USD",
        )
        sm.record_corporate_action(
            cid, "SPLIT", effective_ns=date_to_ns(2020, 7, 1), split_ratio=2.0
        )
        return cid

    def _make_price_df(self, n_before: int, n_after: int, split_ns: int) -> pd.DataFrame:
        """Create a tick DataFrame with prices before and after a 2:1 split."""
        rows = []
        # Before split: stock trading at ~$200
        for i in range(n_before):
            t = split_ns - (n_before - i) * 86_400_000_000_000  # 1 day intervals
            rows.append({
                "exchange_time_ns": t,
                "capture_time_ns": t + 100_000,
                "symbol_id": 1,
                "bid": 199.95,
                "ask": 200.05,
                "bid_size": 100,
                "ask_size": 100,
                "last_price": 200.0,
                "last_size": 50,
                "trade_condition": 0,
            })
        # After split: stock trading at ~$100 (halved after 2:1)
        for i in range(n_after):
            t = split_ns + i * 86_400_000_000_000
            rows.append({
                "exchange_time_ns": t,
                "capture_time_ns": t + 100_000,
                "symbol_id": 1,
                "bid": 99.95,
                "ask": 100.05,
                "bid_size": 200,
                "ask_size": 200,
                "last_price": 100.0,
                "last_size": 100,
                "trade_condition": 0,
            })
        df = pd.DataFrame(rows)
        for col, dtype in Tick.schema_dtypes().items():
            df[col] = df[col].astype(dtype)
        return df

    def test_no_splits_returns_unchanged(self, symbol_master: SymbolMaster):
        cid = symbol_master.add_instrument(
            exchange="NYSE", ticker="NOSPLIT", valid_from_ns=date_to_ns(2020, 1, 1),
            asset_class=AssetClass.EQUITY, currency="USD",
        )
        adjuster = PriceAdjuster(symbol_master)
        timestamps = np.array([date_to_ns(2024, 1, 1)], dtype=np.int64)
        factors = adjuster.compute_adjustment_factors(cid, timestamps)
        assert np.allclose(factors, 1.0)

    def test_2_to_1_split_adjustment(self, symbol_master: SymbolMaster):
        cid = self._setup_stock_with_split(symbol_master)
        adjuster = PriceAdjuster(symbol_master)

        split_ns = date_to_ns(2020, 7, 1)
        df = self._make_price_df(10, 10, split_ns)
        adjusted = adjuster.adjust_dataframe(cid, df)

        # Before split: $200 raw should become $100 adjusted (factor = 0.5)
        before = adjusted[adjusted["capture_time_ns"] < split_ns]
        assert np.allclose(before["last_price"].values, 100.0, atol=0.1)

        # After split: $100 raw should remain $100 adjusted (factor = 1.0)
        after = adjusted[adjusted["capture_time_ns"] >= split_ns]
        assert np.allclose(after["last_price"].values, 100.0, atol=0.1)

    def test_split_adjustment_continuity(self, symbol_master: SymbolMaster):
        """Adjusted price should be continuous across split boundary."""
        cid = self._setup_stock_with_split(symbol_master)
        adjuster = PriceAdjuster(symbol_master)
        split_ns = date_to_ns(2020, 7, 1)
        df = self._make_price_df(10, 10, split_ns)
        adjusted = adjuster.adjust_dataframe(cid, df)

        passes = adjuster.verify_adjustment(cid, df, adjusted, tolerance_pct=0.01)
        assert passes

    def test_size_adjusted_inversely(self, symbol_master: SymbolMaster):
        """On a 2:1 split, sizes should double for pre-split records."""
        cid = self._setup_stock_with_split(symbol_master)
        adjuster = PriceAdjuster(symbol_master)
        split_ns = date_to_ns(2020, 7, 1)
        df = self._make_price_df(5, 5, split_ns)
        adjusted = adjuster.adjust_dataframe(cid, df)

        # Before split: last_size was 50, should become 100 (doubled)
        before = adjusted[adjusted["capture_time_ns"] < split_ns]
        assert np.all(before["last_size"].values == 100)

    def test_multiple_splits(self, symbol_master: SymbolMaster):
        cid = symbol_master.add_instrument(
            exchange="NASDAQ", ticker="MULTI", valid_from_ns=date_to_ns(2015, 1, 1),
            asset_class=AssetClass.EQUITY, currency="USD",
        )
        # Two splits: 2:1 in 2018, 4:1 in 2022
        symbol_master.record_corporate_action(
            cid, "SPLIT", effective_ns=date_to_ns(2018, 1, 1), split_ratio=2.0
        )
        symbol_master.record_corporate_action(
            cid, "SPLIT", effective_ns=date_to_ns(2022, 1, 1), split_ratio=4.0
        )

        adjuster = PriceAdjuster(symbol_master)
        # Price before both splits should be adjusted by 1/(2*4) = 0.125
        timestamps = np.array([date_to_ns(2016, 1, 1)], dtype=np.int64)
        factors = adjuster.compute_adjustment_factors(cid, timestamps)
        assert np.isclose(factors[0], 0.125)

        # Price between splits: adjusted by 1/4 = 0.25
        timestamps = np.array([date_to_ns(2020, 1, 1)], dtype=np.int64)
        factors = adjuster.compute_adjustment_factors(cid, timestamps)
        assert np.isclose(factors[0], 0.25)

        # Price after both splits: no adjustment
        timestamps = np.array([date_to_ns(2024, 1, 1)], dtype=np.int64)
        factors = adjuster.compute_adjustment_factors(cid, timestamps)
        assert np.isclose(factors[0], 1.0)
