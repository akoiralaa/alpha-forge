from __future__ import annotations

from src.data.ingest.base import DataProvider, Tick

__all__ = [
    "DataProvider",
    "Tick",
    "IBKRProvider",
    "PolygonProvider",
    "PolygonEventBackfiller",
    "build_active_equity_tickers",
    "ensure_equity_universe_in_symbol_master",
    "SecCompanyFactsBackfiller",
    "RawDataCache",
    "YahooEventBackfiller",
    "AlphaVantageEventBackfiller",
    "AlpacaProvider",
    "DataManager",
    "build_data_manager_from_env",
]


def __getattr__(name: str):
    if name == "IBKRProvider":
        from src.data.ingest.ibkr import IBKRProvider
        return IBKRProvider
    if name == "PolygonProvider":
        from src.data.ingest.polygon_provider import PolygonProvider
        return PolygonProvider
    if name == "PolygonEventBackfiller":
        from src.data.ingest.polygon_event_backfill import PolygonEventBackfiller
        return PolygonEventBackfiller
    if name == "build_active_equity_tickers":
        from src.data.ingest.polygon_event_backfill import build_active_equity_tickers
        return build_active_equity_tickers
    if name == "ensure_equity_universe_in_symbol_master":
        from src.data.ingest.polygon_event_backfill import ensure_equity_universe_in_symbol_master
        return ensure_equity_universe_in_symbol_master
    if name == "SecCompanyFactsBackfiller":
        from src.data.ingest.sec_companyfacts_backfill import SecCompanyFactsBackfiller
        return SecCompanyFactsBackfiller
    if name == "RawDataCache":
        from src.data.ingest.raw_cache import RawDataCache
        return RawDataCache
    if name == "YahooEventBackfiller":
        from src.data.ingest.yahoo_event_backfill import YahooEventBackfiller
        return YahooEventBackfiller
    if name == "AlphaVantageEventBackfiller":
        from src.data.ingest.alpha_vantage_event_backfill import AlphaVantageEventBackfiller
        return AlphaVantageEventBackfiller
    if name == "AlpacaProvider":
        from src.data.ingest.alpaca_provider import AlpacaProvider
        return AlpacaProvider
    if name in {"DataManager", "build_data_manager_from_env"}:
        from src.data.ingest.data_manager import DataManager, build_data_manager_from_env
        if name == "DataManager":
            return DataManager
        return build_data_manager_from_env
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
