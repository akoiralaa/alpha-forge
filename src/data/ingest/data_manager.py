
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Iterator, Optional

import pandas as pd

from src.data.ingest.base import AssetClass, DataProvider, ProviderConfig, Tick

logger = logging.getLogger(__name__)

# Provider priority for historical bars by asset class
# IBKR first everywhere — provides 20yr history for equities, proper contract
# handling for futures, best FX data. Polygon/Alpaca as fallback when IBKR
# is not connected or fails.
_HISTORICAL_PRIORITY = {
    AssetClass.EQUITY: ["ibkr", "polygon", "alpaca"],
    AssetClass.ETF: ["ibkr", "polygon", "alpaca"],
    AssetClass.FX: ["ibkr", "polygon"],
    AssetClass.FUTURE: ["ibkr", "polygon"],
    AssetClass.COMMODITY: ["ibkr", "polygon"],
    AssetClass.BOND: ["ibkr", "polygon"],
    AssetClass.VOLATILITY: ["ibkr", "polygon"],
}


class DataManager:
    """
    Orchestrates IBKR + Polygon + Alpaca data providers.

    Routing logic:
    - Historical bars: provider priority based on asset class (see _HISTORICAL_PRIORITY)
    - Corporate actions: always Polygon (only source with proper splits/dividends)
    - Live streaming: always IBKR (only source with real-time feeds)
    - Instrument info: IBKR for futures/FX, Polygon for equities

    Falls back through providers automatically on failure.
    """

    def __init__(
        self,
        ibkr: Optional[DataProvider] = None,
        polygon: Optional[DataProvider] = None,
        alpaca: Optional[DataProvider] = None,
    ):
        self._providers: dict[str, DataProvider] = {}
        if ibkr is not None:
            self._providers["ibkr"] = ibkr
        if polygon is not None:
            self._providers["polygon"] = polygon
        if alpaca is not None:
            self._providers["alpaca"] = alpaca

    @property
    def ibkr(self) -> Optional[DataProvider]:
        return self._providers.get("ibkr")

    @property
    def polygon(self) -> Optional[DataProvider]:
        return self._providers.get("polygon")

    @property
    def alpaca(self) -> Optional[DataProvider]:
        return self._providers.get("alpaca")

    def connect_all(self) -> dict[str, bool]:
        """Connect all configured providers. Returns {name: success}."""
        results = {}
        for name, provider in self._providers.items():
            try:
                provider.connect()
                results[name] = True
                logger.info("Connected: %s", name)
            except Exception as e:
                results[name] = False
                logger.warning("Failed to connect %s: %s", name, e)
        return results

    def disconnect_all(self) -> None:
        for name, provider in self._providers.items():
            try:
                provider.disconnect()
            except Exception as e:
                logger.warning("Error disconnecting %s: %s", name, e)

    def _get_priority(self, asset_class: AssetClass) -> list[str]:
        return _HISTORICAL_PRIORITY.get(asset_class, ["polygon", "alpaca", "ibkr"])

    def get_historical_bars(
        self,
        symbol: str,
        asset_class: AssetClass,
        start_ns: int,
        end_ns: int,
        bar_size: str = "1day",
    ) -> pd.DataFrame:
        """Fetch historical bars with automatic provider fallback."""
        priority = self._get_priority(asset_class)

        for provider_name in priority:
            provider = self._providers.get(provider_name)
            if provider is None or not provider.is_connected:
                continue

            try:
                df = provider.get_historical_bars(symbol, asset_class, start_ns, end_ns, bar_size)
                if df is not None and not df.empty:
                    logger.debug("Got %d bars for %s from %s", len(df), symbol, provider_name)
                    return df
            except Exception as e:
                logger.warning("Failed to get bars for %s from %s: %s", symbol, provider_name, e)

        logger.warning("No provider returned bars for %s", symbol)
        return pd.DataFrame()

    def fetch_daily_bars(
        self,
        symbol: str,
        asset_class: AssetClass,
    ) -> pd.DataFrame:
        """
        Fetch maximum available daily bars for a symbol.
        Returns DataFrame with columns: date, open, high, low, close, volume.
        This is the primary interface for backtests.
        """
        priority = self._get_priority(asset_class)

        for provider_name in priority:
            provider = self._providers.get(provider_name)
            if provider is None or not provider.is_connected:
                continue

            try:
                df = self._fetch_daily_from_provider(provider, provider_name, symbol, asset_class)
                if df is not None and not df.empty:
                    logger.debug(
                        "Got %d daily bars for %s from %s", len(df), symbol, provider_name
                    )
                    return df
            except Exception as e:
                logger.warning(
                    "Failed to get daily bars for %s from %s: %s", symbol, provider_name, e
                )

        logger.warning("No provider returned daily bars for %s", symbol)
        return pd.DataFrame()

    def _fetch_daily_from_provider(
        self,
        provider: DataProvider,
        provider_name: str,
        symbol: str,
        asset_class: AssetClass,
    ) -> pd.DataFrame:
        """Fetch daily bars from a specific provider, normalizing to backtest format."""
        # Use the convenience method if available (Alpaca)
        if provider_name == "alpaca" and hasattr(provider, "get_historical_bars_daily"):
            return provider.get_historical_bars_daily(symbol, asset_class)

        # For Polygon/IBKR: use standard interface, then convert timestamp_ns -> date
        # Request max history: 20 years back
        end_dt = datetime.now(tz=timezone.utc)
        start_dt = datetime(end_dt.year - 20, 1, 1, tzinfo=timezone.utc)
        start_ns = int(start_dt.timestamp() * 1_000_000_000)
        end_ns = int(end_dt.timestamp() * 1_000_000_000)

        df = provider.get_historical_bars(symbol, asset_class, start_ns, end_ns, bar_size="1day")
        if df is None or df.empty:
            return pd.DataFrame()

        # Convert timestamp_ns to datetime for backtest compatibility
        df["date"] = pd.to_datetime(df["timestamp_ns"], unit="ns", utc=True)
        result = df[["date", "open", "high", "low", "close", "volume"]].copy()

        # Ensure volume is never 0 (some providers report 0 for FX)
        result.loc[result["volume"] <= 0, "volume"] = 100

        return result.reset_index(drop=True)

    def get_corporate_actions(
        self,
        symbol: str,
        start_ns: int,
        end_ns: int,
    ) -> list[dict]:
        """Always Polygon — the only source with proper split/dividend data."""
        polygon = self._providers.get("polygon")
        if polygon is not None and polygon.is_connected:
            try:
                return polygon.get_corporate_actions(symbol, start_ns, end_ns)
            except Exception as e:
                logger.warning("Polygon corporate actions failed for %s: %s", symbol, e)
        return []

    def stream_ticks(
        self,
        symbols: list[str],
        asset_class: AssetClass,
    ) -> Iterator[Tick]:
        """Always IBKR — the only source with real-time streaming."""
        ibkr = self._providers.get("ibkr")
        if ibkr is None or not ibkr.is_connected:
            raise RuntimeError("IBKR not connected — required for live streaming")
        return ibkr.stream_ticks(symbols, asset_class)

    def get_instrument_info(
        self,
        symbol: str,
        asset_class: AssetClass,
    ) -> dict:
        """IBKR for futures/FX (contract details), Polygon for equities (sector, market cap)."""
        priority = self._get_priority(asset_class)

        for provider_name in priority:
            provider = self._providers.get(provider_name)
            if provider is None or not provider.is_connected:
                continue

            try:
                info = provider.get_instrument_info(symbol, asset_class)
                if info:
                    return info
            except Exception as e:
                logger.warning(
                    "Failed to get instrument info for %s from %s: %s",
                    symbol, provider_name, e,
                )

        return {}


def build_data_manager_from_env() -> DataManager:
    """
    Factory: build a DataManager from environment variables.

    Env vars:
      IB_HOST, IB_PORT, IB_PAPER, IB_CLIENT_ID
      POLYGON_API_KEY, POLYGON_RATE_LIMIT_PER_MIN
      ALPACA_API_KEY, ALPACA_API_SECRET
    """
    ibkr = None
    polygon = None
    alpaca = None

    # IBKR
    ib_host = os.environ.get("IB_HOST", "")
    ib_port = os.environ.get("IB_PORT", "")
    if ib_host and ib_port:
        from src.data.ingest.ibkr import IBKRProvider

        ibkr = IBKRProvider(ProviderConfig(
            host=ib_host,
            port=int(ib_port),
            paper=os.environ.get("IB_PAPER", "true").lower() == "true",
        ))

    # Polygon
    poly_key = os.environ.get("POLYGON_API_KEY", "")
    if poly_key:
        from src.data.ingest.polygon_provider import PolygonProvider

        polygon = PolygonProvider(ProviderConfig(
            api_key=poly_key,
            rate_limit_per_minute=int(os.environ.get("POLYGON_RATE_LIMIT_PER_MIN", "5")),
        ))

    # Alpaca
    alpaca_key = os.environ.get("ALPACA_API_KEY", "")
    alpaca_secret = os.environ.get("ALPACA_API_SECRET", "")
    if alpaca_key and alpaca_secret:
        from src.data.ingest.alpaca_provider import AlpacaProvider

        alpaca = AlpacaProvider(ProviderConfig(
            api_key=alpaca_key,
            api_secret=alpaca_secret,
            rate_limit_per_minute=int(os.environ.get("ALPACA_RATE_LIMIT_PER_MIN", "200")),
        ))

    return DataManager(ibkr=ibkr, polygon=polygon, alpaca=alpaca)
