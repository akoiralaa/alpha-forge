"""Tests for the symbol master."""

import pytest

from src.data.ingest.base import AssetClass, date_to_ns
from src.data.symbol_master import SymbolMaster


class TestSymbolMaster:
    def test_add_and_get_instrument(self, symbol_master: SymbolMaster):
        cid = symbol_master.add_instrument(
            exchange="NYSE",
            ticker="AAPL",
            valid_from_ns=date_to_ns(2000, 1, 1),
            asset_class=AssetClass.EQUITY,
            currency="USD",
            sector="TECHNOLOGY",
        )
        inst = symbol_master.get_by_id(cid)
        assert inst is not None
        assert inst.ticker == "AAPL"
        assert inst.exchange == "NYSE"
        assert inst.asset_class == AssetClass.EQUITY

    def test_resolve_ticker(self, symbol_master: SymbolMaster):
        cid = symbol_master.add_instrument(
            exchange="NASDAQ",
            ticker="GOOG",
            valid_from_ns=date_to_ns(2004, 8, 1),
            asset_class=AssetClass.EQUITY,
            currency="USD",
        )
        resolved = symbol_master.resolve_ticker("GOOG", date_to_ns(2024, 1, 1))
        assert resolved == cid

    def test_resolve_ticker_before_ipo_returns_none(self, symbol_master: SymbolMaster):
        symbol_master.add_instrument(
            exchange="NASDAQ",
            ticker="META",
            valid_from_ns=date_to_ns(2012, 5, 18),
            asset_class=AssetClass.EQUITY,
            currency="USD",
        )
        assert symbol_master.resolve_ticker("META", date_to_ns(2010, 1, 1)) is None

    def test_delist(self, symbol_master: SymbolMaster):
        cid = symbol_master.add_instrument(
            exchange="NYSE",
            ticker="LMNA",
            valid_from_ns=date_to_ns(2005, 1, 1),
            asset_class=AssetClass.EQUITY,
            currency="USD",
        )
        symbol_master.record_corporate_action(
            cid, "DELIST", effective_ns=date_to_ns(2018, 6, 1)
        )
        # Should not resolve after delist
        assert symbol_master.resolve_ticker("LMNA", date_to_ns(2020, 1, 1)) is None
        # Should resolve before delist
        assert symbol_master.resolve_ticker("LMNA", date_to_ns(2017, 1, 1)) == cid

    def test_split_recording(self, symbol_master: SymbolMaster):
        cid = symbol_master.add_instrument(
            exchange="NASDAQ",
            ticker="AAPL",
            valid_from_ns=date_to_ns(2000, 1, 1),
            asset_class=AssetClass.EQUITY,
            currency="USD",
        )
        symbol_master.record_corporate_action(
            cid, "SPLIT", effective_ns=date_to_ns(2020, 8, 31), split_ratio=4.0
        )
        splits = symbol_master.get_splits(cid)
        assert len(splits) == 1
        assert splits[0][1] == 4.0

    def test_dividend_recording(self, symbol_master: SymbolMaster):
        cid = symbol_master.add_instrument(
            exchange="NYSE",
            ticker="JNJ",
            valid_from_ns=date_to_ns(2000, 1, 1),
            asset_class=AssetClass.EQUITY,
            currency="USD",
        )
        symbol_master.record_corporate_action(
            cid, "DIVIDEND", effective_ns=date_to_ns(2024, 3, 15), dividend_amount=1.19
        )
        divs = symbol_master.get_dividends(cid)
        assert len(divs) == 1
        assert divs[0][1] == 1.19

    def test_get_active_instruments(self, symbol_master: SymbolMaster):
        # Add 3 instruments, delist 1
        cid1 = symbol_master.add_instrument(
            exchange="NYSE", ticker="A", valid_from_ns=date_to_ns(2010, 1, 1),
            asset_class=AssetClass.EQUITY, currency="USD",
        )
        cid2 = symbol_master.add_instrument(
            exchange="NYSE", ticker="B", valid_from_ns=date_to_ns(2010, 1, 1),
            asset_class=AssetClass.EQUITY, currency="USD",
        )
        cid3 = symbol_master.add_instrument(
            exchange="NYSE", ticker="C", valid_from_ns=date_to_ns(2010, 1, 1),
            asset_class=AssetClass.EQUITY, currency="USD",
        )
        symbol_master.record_corporate_action(
            cid2, "DELIST", effective_ns=date_to_ns(2015, 6, 1)
        )

        # At 2014: all 3 active
        active_2014 = symbol_master.get_active_instruments(date_to_ns(2014, 1, 1))
        active_ids = {i.canonical_id for i in active_2014}
        assert {cid1, cid2, cid3} == active_ids

        # At 2020: only 2 active (B delisted)
        active_2020 = symbol_master.get_active_instruments(date_to_ns(2020, 1, 1))
        active_ids = {i.canonical_id for i in active_2020}
        assert cid2 not in active_ids
        assert {cid1, cid3} == active_ids

    def test_get_delisted_instruments(self, symbol_master: SymbolMaster):
        cid1 = symbol_master.add_instrument(
            exchange="NYSE", ticker="DEAD1", valid_from_ns=date_to_ns(2008, 1, 1),
            asset_class=AssetClass.EQUITY, currency="USD",
        )
        symbol_master.record_corporate_action(
            cid1, "DELIST", effective_ns=date_to_ns(2015, 1, 1)
        )
        cid2 = symbol_master.add_instrument(
            exchange="NYSE", ticker="ALIVE", valid_from_ns=date_to_ns(2008, 1, 1),
            asset_class=AssetClass.EQUITY, currency="USD",
        )

        delisted = symbol_master.get_delisted_instruments(
            existed_at_ns=date_to_ns(2010, 1, 1),
            delisted_before_ns=date_to_ns(2023, 1, 1),
        )
        delisted_ids = {i.canonical_id for i in delisted}
        assert cid1 in delisted_ids
        assert cid2 not in delisted_ids

    def test_rename(self, symbol_master: SymbolMaster):
        cid = symbol_master.add_instrument(
            exchange="NASDAQ", ticker="FB", valid_from_ns=date_to_ns(2012, 5, 18),
            asset_class=AssetClass.EQUITY, currency="USD",
        )
        symbol_master.record_corporate_action(
            cid, "RENAME", effective_ns=date_to_ns(2021, 10, 28), new_ticker="META"
        )

        # Before rename: FB
        assert symbol_master.get_ticker_at(cid, date_to_ns(2020, 1, 1)) == "FB"
        # After rename: META
        assert symbol_master.get_ticker_at(cid, date_to_ns(2022, 1, 1)) == "META"

    def test_multi_asset_class(self, symbol_master: SymbolMaster):
        symbol_master.add_instrument(
            exchange="CME", ticker="ES", valid_from_ns=date_to_ns(2000, 1, 1),
            asset_class=AssetClass.FUTURE, currency="USD",
        )
        symbol_master.add_instrument(
            exchange="CME", ticker="EURUSD", valid_from_ns=date_to_ns(2000, 1, 1),
            asset_class=AssetClass.FX, currency="USD",
        )

        futures = symbol_master.get_active_instruments(
            date_to_ns(2024, 1, 1), asset_class=AssetClass.FUTURE
        )
        assert len(futures) == 1
        assert futures[0].ticker == "ES"

        fx = symbol_master.get_active_instruments(
            date_to_ns(2024, 1, 1), asset_class=AssetClass.FX
        )
        assert len(fx) == 1
        assert fx[0].ticker == "EURUSD"
