from __future__ import annotations

from src.data.ingest.base import AssetClass, date_to_ns
from src.data.ingest.sec_companyfacts_backfill import SecCompanyFactsBackfiller
from src.data.symbol_master import SymbolMaster


class TestSecCompanyFactsBackfill:
    def test_sec_backfiller_reuses_raw_cache(self, tmp_path, monkeypatch):
        sm = SymbolMaster(tmp_path / "symbol_master.db")
        sm.add_instrument("NASDAQ", "AAPL", date_to_ns(2020, 1, 1), AssetClass.EQUITY, "USD")
        sm.close()

        def fake_fetch(self, url: str):
            if "company_tickers.json" in url:
                return {
                    "0": {
                        "ticker": "AAPL",
                        "cik_str": "320193",
                    }
                }
            if "CIK0000320193.json" in url:
                return {
                    "facts": {
                        "us-gaap": {
                            "EarningsPerShareDiluted": {
                                "units": {
                                    "USD/shares": [
                                        {
                                            "form": "10-Q",
                                            "filed": "2024-01-25",
                                            "end": "2023-12-31",
                                            "val": 1.23,
                                        }
                                    ]
                                }
                            }
                        }
                    }
                }
            raise AssertionError(f"unexpected url {url}")

        monkeypatch.setattr(SecCompanyFactsBackfiller, "_fetch_json_from_source", fake_fetch)
        first = SecCompanyFactsBackfiller(
            user_agent="test-agent",
            symbol_master_path=str(tmp_path / "symbol_master.db"),
            fundamentals_path=str(tmp_path / "fundamentals.db"),
            events_path=str(tmp_path / "events.db"),
            cache_dir=str(tmp_path / "cache"),
            request_sleep=0.0,
        )
        try:
            first.backfill(["AAPL"])
            assert first.stats.raw_cache_writes >= 2
            assert first.fund.count_records() >= 1
            assert first.events.count_records() >= 1
        finally:
            first.close()

        def fail_fetch(self, url: str):
            raise AssertionError(f"should not fetch from network: {url}")

        monkeypatch.setattr(SecCompanyFactsBackfiller, "_fetch_json_from_source", fail_fetch)
        second = SecCompanyFactsBackfiller(
            user_agent="test-agent",
            symbol_master_path=str(tmp_path / "symbol_master.db"),
            fundamentals_path=str(tmp_path / "fundamentals.db"),
            events_path=str(tmp_path / "events.db"),
            cache_dir=str(tmp_path / "cache"),
            request_sleep=0.0,
            cache_only=True,
        )
        try:
            second.backfill(["AAPL"])
            assert second.stats.raw_cache_hits >= 2
        finally:
            second.close()

    def test_sec_backfiller_adds_submission_events(self, tmp_path, monkeypatch):
        sm = SymbolMaster(tmp_path / "symbol_master.db")
        sm.add_instrument("NASDAQ", "AAPL", date_to_ns(2020, 1, 1), AssetClass.EQUITY, "USD")
        sm.close()

        def fake_fetch(self, url: str):
            if "company_tickers.json" in url:
                return {"0": {"ticker": "AAPL", "cik_str": "320193"}}
            if "companyfacts" in url:
                return {
                    "facts": {
                        "us-gaap": {
                            "EarningsPerShareDiluted": {
                                "units": {
                                    "USD/shares": [
                                        {"form": "10-Q", "filed": "2024-01-25", "end": "2023-12-31", "val": 1.23}
                                    ]
                                }
                            }
                        }
                    }
                }
            if url.endswith("CIK0000320193.json"):
                return {
                    "filings": {
                        "recent": {
                            "accessionNumber": ["0000320193-24-000010"],
                            "filingDate": ["2024-02-02"],
                            "reportDate": ["2024-02-02"],
                            "form": ["8-K"],
                            "items": ["2.02 8.01"],
                            "primaryDocument": ["earnings-release.htm"],
                            "primaryDocDescription": ["AAPL announces quarterly results and raises outlook"],
                        },
                        "files": [],
                    }
                }
            raise AssertionError(f"unexpected url {url}")

        monkeypatch.setattr(SecCompanyFactsBackfiller, "_fetch_json_from_source", fake_fetch)
        backfiller = SecCompanyFactsBackfiller(
            user_agent="test-agent",
            symbol_master_path=str(tmp_path / "symbol_master.db"),
            fundamentals_path=str(tmp_path / "fundamentals.db"),
            events_path=str(tmp_path / "events.db"),
            cache_dir=str(tmp_path / "cache"),
            request_sleep=0.0,
        )
        try:
            backfiller.backfill(["AAPL"])
            events = backfiller.events.to_dataframe()
            assert "SEC_SUBMISSIONS" in set(events["source"])
            assert "earnings" in set(events["event_type"])
            assert any("raises outlook" in str(h).lower() for h in events["headline"])
        finally:
            backfiller.close()
