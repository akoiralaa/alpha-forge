from __future__ import annotations

from types import SimpleNamespace

from src.data.ingest.base import AssetClass, date_to_ns
from src.data.ingest.polygon_event_backfill import (
    PolygonEventBackfiller,
    parse_market_datetime_ns,
    score_analyst_revision,
    score_earnings_surprise,
    score_guidance_change,
)
from src.data.symbol_master import SymbolMaster


class TestPolygonEventBackfill:
    def test_score_earnings_surprise_positive(self):
        score, confidence = score_earnings_surprise(12.0, 6.0)
        assert score > 0
        assert 0.25 <= confidence <= 0.95

    def test_score_guidance_change_uses_prior_guidance(self):
        guidance = SimpleNamespace(
            estimated_eps_guidance=5.2,
            previous_min_eps_guidance=4.6,
            previous_max_eps_guidance=4.8,
            estimated_revenue_guidance=110.0,
            previous_min_revenue_guidance=100.0,
            previous_max_revenue_guidance=102.0,
            positioning="raised",
        )
        score, confidence = score_guidance_change(guidance)
        assert score > 0
        assert confidence > 0.2

    def test_score_analyst_revision_handles_downgrade(self):
        insight = SimpleNamespace(
            rating_action="downgrade",
            rating="underperform",
            insight="firm cuts rating after weak demand",
            price_target=90.0,
        )
        assert score_analyst_revision(insight) < 0

    def test_parse_market_datetime_ns_handles_bmo_and_amc(self):
        bmo = parse_market_datetime_ns("2026-01-30", "bmo")
        amc = parse_market_datetime_ns("2026-01-30", "amc")
        assert bmo is not None
        assert amc is not None
        assert amc > bmo

    def test_polygon_raw_cache_reuses_local_rows(self, tmp_path):
        sm = SymbolMaster(tmp_path / "symbol_master.db")
        sm.add_instrument("NASDAQ", "AAPL", date_to_ns(2020, 1, 1), AssetClass.EQUITY, "USD")
        sm.close()

        backfiller = PolygonEventBackfiller(
            api_key="test",
            symbol_master_path=str(tmp_path / "symbol_master.db"),
            fundamentals_path=str(tmp_path / "fundamentals.db"),
            events_path=str(tmp_path / "events.db"),
            cache_dir=str(tmp_path / "cache"),
        )
        try:
            rows = backfiller._fetch_cached_polygon_rows(
                "polygon/test",
                "chunk-aapl",
                lambda: [SimpleNamespace(ticker="AAPL", actual_eps=1.23)],
            )
            assert rows[0]["ticker"] == "AAPL"
            assert backfiller.stats.raw_cache_writes == 1
        finally:
            backfiller.close()

        cached = PolygonEventBackfiller(
            api_key="test",
            symbol_master_path=str(tmp_path / "symbol_master.db"),
            fundamentals_path=str(tmp_path / "fundamentals.db"),
            events_path=str(tmp_path / "events.db"),
            cache_dir=str(tmp_path / "cache"),
            cache_only=True,
        )
        try:
            rows = cached._fetch_cached_polygon_rows(
                "polygon/test",
                "chunk-aapl",
                lambda: (_ for _ in ()).throw(RuntimeError("should not fetch")),
            )
            assert rows[0]["actual_eps"] == 1.23
            assert cached.stats.raw_cache_hits == 1
        finally:
            cached.close()
