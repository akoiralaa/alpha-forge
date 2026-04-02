from __future__ import annotations

import pandas as pd

from src.data.ingest.daily_bar_cache import cache_is_stale, merge_bar_frames


class TestDailyBarCache:
    def test_merge_bar_frames_dedupes_by_date_and_prefers_newest(self):
        existing = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
                "open": [10.0, 11.0],
                "high": [10.5, 11.5],
                "low": [9.5, 10.5],
                "close": [10.2, 11.2],
                "volume": [100, 110],
            }
        )
        incoming = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True),
                "open": [12.0, 13.0],
                "high": [12.5, 13.5],
                "low": [11.5, 12.5],
                "close": [12.2, 13.2],
                "volume": [120, 130],
            }
        )
        merged = merge_bar_frames(existing, incoming)
        assert list(merged["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-01", "2024-01-02", "2024-01-03"]
        assert float(merged.loc[merged["date"] == pd.Timestamp("2024-01-02", tz="UTC"), "close"].iloc[0]) == 12.2

    def test_cache_is_stale_uses_end_date(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-03"], utc=True),
                "close": [1.0, 1.1],
            }
        )
        assert cache_is_stale(df, stale_days=1, as_of=pd.Timestamp("2024-01-05", tz="UTC")) is True
        assert cache_is_stale(df, stale_days=2, as_of=pd.Timestamp("2024-01-05", tz="UTC")) is False
