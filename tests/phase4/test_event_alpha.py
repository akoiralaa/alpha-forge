from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.events import EventRecord, EventStore
from src.data.fundamentals import FundamentalRecord, FundamentalsStore
from src.data.ingest.base import AssetClass, date_to_ns
from src.data.symbol_master import SymbolMaster
from src.signals.event_alpha import EventAlphaConfig, EventDrivenAlphaSleeve, build_event_alpha_signal


@pytest.fixture
def symbol_master(tmp_path):
    sm = SymbolMaster(tmp_path / "symbol_master.db")
    yield sm
    sm.close()


@pytest.fixture
def fundamentals(tmp_path):
    store = FundamentalsStore(tmp_path / "fundamentals.db")
    yield store
    store.close()


class TestEventAlpha:
    def test_build_event_alpha_signal_combines_fundamental_and_sentiment(
        self,
        tmp_path,
        symbol_master,
        fundamentals,
    ):
        symbol_master.add_instrument("NASDAQ", "AAPL", date_to_ns(2020, 1, 1), AssetClass.EQUITY, "USD")
        symbol_master.add_instrument("NASDAQ", "MSFT", date_to_ns(2020, 1, 1), AssetClass.EQUITY, "USD")

        fundamentals.add_records_batch(
            [
                FundamentalRecord(
                    canonical_id=1,
                    metric_name="ANALYST_EST_EPS",
                    value=1.00,
                    published_at_ns=date_to_ns(2024, 1, 10),
                    period_end_ns=date_to_ns(2023, 12, 31),
                    source="test",
                ),
                FundamentalRecord(
                    canonical_id=1,
                    metric_name="ANALYST_EST_EPS",
                    value=1.10,
                    published_at_ns=date_to_ns(2024, 1, 20),
                    period_end_ns=date_to_ns(2023, 12, 31),
                    source="test",
                ),
                FundamentalRecord(
                    canonical_id=1,
                    metric_name="EPS",
                    value=1.25,
                    published_at_ns=date_to_ns(2024, 1, 25),
                    period_end_ns=date_to_ns(2023, 12, 31),
                    source="test",
                ),
            ]
        )

        event_store = EventStore(tmp_path / "events.db")
        try:
            event_store.add_record(
                EventRecord(
                    canonical_id=2,
                    ticker="MSFT",
                    published_at_ns=date_to_ns(2024, 1, 18),
                    event_type="news",
                    source="POLYGON_BENZINGA_NEWS",
                    headline="MSFT signs major AI partnership",
                    sentiment_score=0.85,
                    confidence=0.90,
                )
            )

            idx = pd.date_range("2024-01-01", periods=40, freq="D")
            prices = pd.DataFrame(
                {
                    "AAPL": np.linspace(100, 110, len(idx)),
                    "MSFT": np.linspace(200, 212, len(idx)),
                },
                index=idx,
            )

            build = build_event_alpha_signal(
                prices,
                ["AAPL", "MSFT"],
                fundamentals_path=str(tmp_path / "fundamentals.db"),
                symbol_master_path=str(tmp_path / "symbol_master.db"),
                event_store_path=str(tmp_path / "events.db"),
                config=EventAlphaConfig(
                    coverage_floor=0,
                    coverage_full=2,
                    event_target_weight=0.12,
                    sentiment_weight=0.35,
                ),
            )

            assert build.fundamental_coverage == 1
            assert build.estimate_coverage == 1
            assert build.revision_coverage == 1
            assert build.sentiment_coverage == 1
            assert build.total_events == 1
            assert build.sentiment_quality_scale == pytest.approx(1.0)
            assert build.data_quality_scale == pytest.approx(1.0)
            assert build.suggested_weight > 0
            assert build.composite.loc["2024-01-26", "AAPL"] > 0
            assert build.composite.loc["2024-01-19", "MSFT"] > 0
        finally:
            event_store.close()

    def test_actual_only_fundamentals_are_quality_capped(self, tmp_path, symbol_master, fundamentals):
        symbol_master.add_instrument("NASDAQ", "AAPL", date_to_ns(2020, 1, 1), AssetClass.EQUITY, "USD")
        fundamentals.add_record(
            FundamentalRecord(
                canonical_id=1,
                metric_name="EPS",
                value=1.00,
                published_at_ns=date_to_ns(2023, 10, 11),
                period_end_ns=date_to_ns(2023, 9, 30),
                source="sec",
            )
        )
        fundamentals.add_record(
            FundamentalRecord(
                canonical_id=1,
                metric_name="EPS",
                value=1.20,
                published_at_ns=date_to_ns(2024, 1, 11),
                period_end_ns=date_to_ns(2023, 12, 31),
                source="sec",
            )
        )

        idx = pd.date_range("2024-01-01", periods=40, freq="D")
        prices = pd.DataFrame({"AAPL": np.linspace(100, 110, len(idx))}, index=idx)

        build = build_event_alpha_signal(
            prices,
            ["AAPL"],
            fundamentals_path=str(tmp_path / "fundamentals.db"),
            symbol_master_path=str(tmp_path / "symbol_master.db"),
            event_store_path=str(tmp_path / "events.db"),
            config=EventAlphaConfig(
                coverage_floor=0,
                coverage_full=1,
                event_target_weight=0.14,
            ),
        )

        assert build.fundamental_coverage == 1
        assert build.estimate_coverage == 0
        assert build.revision_coverage == 0
        assert build.sentiment_coverage == 0
        assert build.sentiment_quality_scale == pytest.approx(0.0)
        assert build.data_quality_scale == pytest.approx(0.35)
        assert build.suggested_weight == pytest.approx(0.0196)

    def test_event_sleeve_scores_paper_engine_view(self, tmp_path, symbol_master, fundamentals):
        symbol_master.add_instrument("NASDAQ", "AAPL", date_to_ns(2020, 1, 1), AssetClass.EQUITY, "USD")
        fundamentals.add_records_batch(
            [
                FundamentalRecord(
                    canonical_id=1,
                    metric_name="ANALYST_EST_EPS",
                    value=1.00,
                    published_at_ns=date_to_ns(2024, 1, 10),
                    period_end_ns=date_to_ns(2023, 12, 31),
                    source="test",
                ),
                FundamentalRecord(
                    canonical_id=1,
                    metric_name="EPS",
                    value=1.20,
                    published_at_ns=date_to_ns(2024, 1, 11),
                    period_end_ns=date_to_ns(2023, 12, 31),
                    source="test",
                ),
            ]
        )

        event_store = EventStore(tmp_path / "events.db")
        try:
            event_store.add_record(
                EventRecord(
                    canonical_id=1,
                    ticker="AAPL",
                    published_at_ns=date_to_ns(2024, 1, 12),
                    event_type="guidance",
                    source="POLYGON_BENZINGA_GUIDANCE",
                    headline="AAPL raises margin outlook",
                    sentiment_score=0.70,
                    confidence=0.95,
                )
            )

            sleeve = EventDrivenAlphaSleeve(
                symbol_map={101: "AAPL"},
                fundamentals_path=str(tmp_path / "fundamentals.db"),
                symbol_master_path=str(tmp_path / "symbol_master.db"),
                event_store_path=str(tmp_path / "events.db"),
                config=EventAlphaConfig(sentiment_weight=0.35),
            )
            engine = SimpleNamespace(
                latest_ticks={101: SimpleNamespace(timestamp_ns=date_to_ns(2024, 1, 13))},
                returns={101: [0.01, 0.004, 0.006]},
            )
            score = sleeve.score_paper(101, 101.0, engine)
            sleeve.close()
            assert score > 0
        finally:
            event_store.close()
