
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Iterator

import pandas as pd

from src.data.ingest.base import (
    AssetClass,
    DataProvider,
    ProviderConfig,
    Tick,
    TradeCondition,
    ns_now,
)

logger = logging.getLogger(__name__)


def _ns_to_datetime(ns: int) -> datetime:
    return datetime.fromtimestamp(ns / 1_000_000_000, tz=timezone.utc)


class AlpacaProvider(DataProvider):
    """
    Alpaca Markets data provider — US equities + crypto via SIP consolidated tape.
    Uses alpaca-py SDK. Supplementary source alongside IBKR and Polygon.

    Limitations:
    - US equities and crypto only (no futures, FX, bonds)
    - No live streaming in free tier (use IBKR for live)
    - Corporate actions not available (use Polygon)
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._stock_client = None
        self._rate_limit_sleep = 60.0 / max(config.rate_limit_per_minute, 1)

    def connect(self) -> None:
        try:
            from alpaca.data.historical import StockHistoricalDataClient
        except ImportError:
            raise ImportError("alpaca-py is required: pip install alpaca-py")

        self._stock_client = StockHistoricalDataClient(
            api_key=self.config.api_key,
            secret_key=self.config.api_secret,
        )
        self._connected = True
        logger.info("Connected to Alpaca Markets")

    def disconnect(self) -> None:
        self._stock_client = None
        self._connected = False
        logger.info("Disconnected from Alpaca Markets")

    def _rate_limit(self) -> None:
        time.sleep(self._rate_limit_sleep)

    def _supports_asset_class(self, asset_class: AssetClass) -> bool:
        return asset_class in (AssetClass.EQUITY, AssetClass.ETF)

    def get_historical_bars(
        self,
        symbol: str,
        asset_class: AssetClass,
        start_ns: int,
        end_ns: int,
        bar_size: str = "1day",
    ) -> pd.DataFrame:
        if self._stock_client is None:
            raise RuntimeError("Not connected")

        if not self._supports_asset_class(asset_class):
            logger.debug("Alpaca does not support %s — skipping %s", asset_class.value, symbol)
            return pd.DataFrame()

        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
        except ImportError:
            raise ImportError("alpaca-py is required: pip install alpaca-py")

        timeframe = {
            "1min": TimeFrame.Minute,
            "5min": TimeFrame(5, "Min"),
            "15min": TimeFrame(15, "Min"),
            "1hour": TimeFrame.Hour,
            "1day": TimeFrame.Day,
        }.get(bar_size, TimeFrame.Day)

        start_dt = _ns_to_datetime(start_ns)
        end_dt = _ns_to_datetime(end_ns)

        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start_dt,
                end=end_dt,
            )
            bars = self._stock_client.get_stock_bars(request)
            self._rate_limit()
        except Exception as e:
            logger.error("Alpaca bars fetch failed for %s: %s", symbol, e)
            return pd.DataFrame()

        bar_set = bars[symbol] if symbol in bars else []
        if not bar_set:
            return pd.DataFrame()

        records = []
        for bar in bar_set:
            ts = bar.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            ts_ns = int(ts.timestamp() * 1_000_000_000)
            records.append({
                "timestamp_ns": ts_ns,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": int(bar.volume) if bar.volume else 0,
                "vwap": bar.vwap if bar.vwap else (bar.open + bar.close) / 2,
            })

        df = pd.DataFrame(records)
        df = df[(df["timestamp_ns"] >= start_ns) & (df["timestamp_ns"] <= end_ns)]
        return df.reset_index(drop=True)

    def get_historical_bars_daily(
        self,
        symbol: str,
        asset_class: AssetClass,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Convenience method returning date/OHLCV DataFrame matching backtest format."""
        if self._stock_client is None:
            raise RuntimeError("Not connected")

        if not self._supports_asset_class(asset_class):
            return pd.DataFrame()

        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
        except ImportError:
            raise ImportError("alpaca-py is required: pip install alpaca-py")

        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date,
            )
            bars = self._stock_client.get_stock_bars(request)
            self._rate_limit()
        except Exception as e:
            logger.error("Alpaca daily bars fetch failed for %s: %s", symbol, e)
            return pd.DataFrame()

        bar_set = bars[symbol] if symbol in bars else []
        if not bar_set:
            return pd.DataFrame()

        records = []
        for bar in bar_set:
            ts = bar.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            records.append({
                "date": ts,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": int(bar.volume) if bar.volume else 100,
            })

        return pd.DataFrame(records)

    def get_historical_ticks(
        self,
        symbol: str,
        asset_class: AssetClass,
        start_ns: int,
        end_ns: int,
    ) -> Iterator[Tick]:
        if self._stock_client is None:
            raise RuntimeError("Not connected")

        if not self._supports_asset_class(asset_class):
            return

        try:
            from alpaca.data.requests import StockTradesRequest
        except ImportError:
            raise ImportError("alpaca-py is required: pip install alpaca-py")

        start_dt = _ns_to_datetime(start_ns)
        end_dt = _ns_to_datetime(end_ns)

        try:
            request = StockTradesRequest(
                symbol_or_symbols=symbol,
                start=start_dt,
                end=end_dt,
            )
            trades = self._stock_client.get_stock_trades(request)
            self._rate_limit()
        except Exception as e:
            logger.error("Alpaca ticks fetch failed for %s: %s", symbol, e)
            return

        trade_set = trades[symbol] if symbol in trades else []
        for t in trade_set:
            ts = t.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            exchange_ns = int(ts.timestamp() * 1_000_000_000)

            if exchange_ns > end_ns:
                break

            yield Tick(
                exchange_time_ns=exchange_ns,
                capture_time_ns=exchange_ns,
                symbol_id=0,  # Caller resolves via symbol master
                bid=0.0,      # Trade data doesn't have bid/ask
                ask=0.0,
                bid_size=0,
                ask_size=0,
                last_price=t.price if t.price else 0.0,
                last_size=int(t.size) if t.size else 0,
                trade_condition=TradeCondition.REGULAR,
            )

    def stream_ticks(
        self,
        symbols: list[str],
        asset_class: AssetClass,
    ) -> Iterator[Tick]:
        raise NotImplementedError(
            "Alpaca real-time streaming requires paid subscription. Use IBKRProvider for live data."
        )

    def get_instrument_info(self, symbol: str, asset_class: AssetClass) -> dict:
        if self._stock_client is None:
            raise RuntimeError("Not connected")

        if not self._supports_asset_class(asset_class):
            return {}

        # Alpaca doesn't have a dedicated instrument info endpoint in the data client
        # Use the trading client for asset details if needed
        return {
            "symbol": symbol,
            "source": "alpaca",
            "asset_class": asset_class.value,
        }

    def get_corporate_actions(
        self,
        symbol: str,
        start_ns: int,
        end_ns: int,
    ) -> list[dict]:
        logger.debug("Alpaca does not provide corporate actions API; use PolygonProvider")
        return []
