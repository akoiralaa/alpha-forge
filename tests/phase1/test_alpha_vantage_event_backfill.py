from __future__ import annotations

from src.data.fundamentals import FundamentalsStore
from src.data.ingest.alpha_vantage_event_backfill import AlphaVantageEventBackfiller
from src.data.ingest.base import AssetClass, date_to_ns
from src.data.symbol_master import SymbolMaster


class TestAlphaVantageEventBackfill:
    def _build_backfiller(self, tmp_path):
        sm = SymbolMaster(tmp_path / "symbol_master.db")
        sm.add_instrument("NYSE", "IBM", date_to_ns(2010, 1, 1), AssetClass.EQUITY, "USD")
        sm.close()
        return AlphaVantageEventBackfiller(
            api_key="demo",
            symbol_master_path=str(tmp_path / "symbol_master.db"),
            fundamentals_path=str(tmp_path / "fundamentals.db"),
            events_path=str(tmp_path / "events.db"),
            cache_dir=str(tmp_path / "cache"),
            cache_only=True,
        )

    def test_parse_earnings_and_estimates_payloads(self, tmp_path):
        backfiller = self._build_backfiller(tmp_path)
        try:
            earnings_payload = {
                "quarterlyEarnings": [
                    {
                        "fiscalDateEnding": "2025-12-31",
                        "reportedDate": "2026-01-28",
                        "reportedEPS": "2.10",
                        "estimatedEPS": "1.95",
                        "surprisePercentage": "7.69",
                    }
                ]
            }
            est_payload = {
                "estimates": [
                    {
                        "date": "2026-12-31",
                        "horizon": "fiscal year",
                        "eps_estimate_average": "13.2",
                        "eps_estimate_average_30_days_ago": "12.8",
                        "eps_estimate_revision_up_trailing_30_days": "3",
                        "eps_estimate_revision_down_trailing_30_days": "1",
                    }
                ]
            }
            fund_a, event_a = backfiller.parse_earnings_payload("IBM", earnings_payload)
            fund_b, event_b = backfiller.parse_estimates_payload(
                "IBM", est_payload, as_of_ns=date_to_ns(2026, 2, 1)
            )
        finally:
            backfiller.close()

        metrics = {r.metric_name for r in (fund_a + fund_b)}
        assert FundamentalsStore.METRIC_EPS in metrics
        assert FundamentalsStore.METRIC_ANALYST_EST_EPS in metrics
        assert FundamentalsStore.METRIC_ANALYST_REVISION in metrics
        event_types = {e.event_type for e in (event_a + event_b)}
        assert "earnings" in event_types
        assert "estimate_revision" in event_types

    def test_parse_news_and_calendar(self, tmp_path):
        backfiller = self._build_backfiller(tmp_path)
        try:
            news_payload = {
                "feed": [
                    {
                        "time_published": "20260401T140000",
                        "title": "IBM wins major contract",
                        "summary": "Positive demand momentum",
                        "overall_sentiment_score": "0.30",
                        "ticker_sentiment": [
                            {
                                "ticker": "IBM",
                                "relevance_score": "0.85",
                                "ticker_sentiment_score": "0.44",
                            }
                        ],
                    }
                ]
            }
            calendar_csv = (
                "symbol,name,reportDate,fiscalDateEnding,estimate,currency,timeOfTheDay\n"
                "IBM,INTERNATIONAL BUSINESS MACHINES CORP,2026-04-22,2026-03-31,1.78,USD,post-market\n"
            )
            news_events = backfiller.parse_news_payload("IBM", news_payload)
            cal_events = backfiller.parse_earnings_calendar_csv("IBM", calendar_csv)
        finally:
            backfiller.close()

        assert len(news_events) == 1
        assert news_events[0].event_type == "news"
        assert len(cal_events) == 1
        assert cal_events[0].event_type == "earnings"

    def test_cache_only_uses_local_payloads_without_network(self, tmp_path):
        backfiller = self._build_backfiller(tmp_path)
        try:
            backfiller.raw_cache.write_json(
                "alpha_vantage/json",
                "EARNINGS:IBM",
                {"quarterlyEarnings": []},
            )
            backfiller.raw_cache.write_json(
                "alpha_vantage/json",
                "EARNINGS_ESTIMATES:IBM",
                {"estimates": []},
            )
            backfiller.raw_cache.write_json(
                "alpha_vantage/news",
                "NEWS_SENTIMENT:IBM:200",
                {"feed": []},
            )
            backfiller.raw_cache.write_json(
                "alpha_vantage/csv",
                "EARNINGS_CALENDAR:IBM:12month",
                "symbol,name,reportDate,fiscalDateEnding,estimate,currency,timeOfTheDay\n",
            )
            backfiller.backfill(["IBM"], include_news=True, include_calendar=True, chunk_size=1)
            summary = backfiller.summary()
        finally:
            backfiller.close()

        assert summary["network_requests"] == 0
        assert summary["raw_cache_misses"] == 0
        assert summary["raw_cache_hits"] == 4
