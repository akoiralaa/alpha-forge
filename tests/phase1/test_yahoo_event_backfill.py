from __future__ import annotations

from src.data.fundamentals import FundamentalsStore
from src.data.ingest.base import AssetClass, date_to_ns
from src.data.ingest.yahoo_event_backfill import YahooEventBackfiller, score_upgrade_downgrade
from src.data.symbol_master import SymbolMaster


class TestYahooEventBackfill:
    def test_score_upgrade_downgrade(self):
        assert score_upgrade_downgrade("upgrade", "hold", "buy") > 0
        assert score_upgrade_downgrade("downgrade", "buy", "sell") < 0

    def test_parse_quote_summary_generates_fundamental_and_event_records(self, tmp_path):
        sm = SymbolMaster(tmp_path / "symbol_master.db")
        sm.add_instrument("NASDAQ", "AAPL", date_to_ns(2015, 1, 1), AssetClass.EQUITY, "USD")
        sm.close()

        backfiller = YahooEventBackfiller(
            symbol_master_path=str(tmp_path / "symbol_master.db"),
            fundamentals_path=str(tmp_path / "fundamentals.db"),
            events_path=str(tmp_path / "events.db"),
            cache_dir=str(tmp_path / "cache"),
            cache_only=True,
        )
        try:
            payload = {
                "quoteSummary": {
                    "result": [
                        {
                            "earningsHistory": {
                                "history": [
                                    {
                                        "quarter": {"raw": 1704067200},
                                        "epsActual": {"raw": 2.10},
                                        "epsEstimate": {"raw": 1.95},
                                        "surprisePercent": {"raw": 7.7},
                                    }
                                ]
                            },
                            "upgradeDowngradeHistory": {
                                "history": [
                                    {
                                        "epochGradeDate": 1706745600,
                                        "firm": "Broker X",
                                        "fromGrade": "Hold",
                                        "toGrade": "Buy",
                                        "action": "upgrade",
                                    }
                                ]
                            },
                            "earningsTrend": {
                                "trend": [
                                    {
                                        "endDate": {"raw": 1711929600},
                                        "epsTrend": {
                                            "current": {"raw": 2.35},
                                            "30daysAgo": {"raw": 2.10},
                                        },
                                    }
                                ]
                            },
                        }
                    ]
                }
            }
            fund_records, event_records = backfiller.parse_quote_summary(
                "AAPL", payload, as_of_ns=date_to_ns(2024, 2, 1)
            )
        finally:
            backfiller.close()

        metric_names = {r.metric_name for r in fund_records}
        assert FundamentalsStore.METRIC_EPS in metric_names
        assert FundamentalsStore.METRIC_ANALYST_EST_EPS in metric_names
        assert FundamentalsStore.METRIC_ANALYST_REVISION in metric_names
        event_types = {e.event_type for e in event_records}
        assert "earnings" in event_types
        assert "estimate_revision" in event_types

    def test_parse_intraday_chart_returns_sorted_rows(self, tmp_path):
        sm = SymbolMaster(tmp_path / "symbol_master.db")
        sm.add_instrument("NASDAQ", "QQQ", date_to_ns(2015, 1, 1), AssetClass.EQUITY, "USD")
        sm.close()

        backfiller = YahooEventBackfiller(
            symbol_master_path=str(tmp_path / "symbol_master.db"),
            fundamentals_path=str(tmp_path / "fundamentals.db"),
            events_path=str(tmp_path / "events.db"),
            cache_dir=str(tmp_path / "cache"),
            cache_only=True,
        )
        try:
            payload = {
                "chart": {
                    "result": [
                        {
                            "timestamp": [1704067260, 1704067200, 1704067260],
                            "indicators": {
                                "quote": [
                                    {
                                        "open": [100.2, 100.0, 100.25],
                                        "high": [100.3, 100.1, 100.35],
                                        "low": [100.1, 99.9, 100.2],
                                        "close": [100.25, 100.05, 100.30],
                                        "volume": [1200, 1000, 1300],
                                    }
                                ]
                            },
                        }
                    ]
                }
            }
            df = backfiller.parse_intraday_chart(payload)
        finally:
            backfiller.close()

        assert len(df) == 2
        assert list(df["timestamp_ns"]) == sorted(df["timestamp_ns"].tolist())

