"""Polygon.io data provider — supplementary historical data for US equities.

Used for:
- Deep tick-level historical backfill (2+ years) where IB's history runs out
- Corporate action history (splits, dividends, mergers)
- Reference data (tickers, exchanges, market status)

Free tier: 5 API calls/minute. Paid tiers for higher throughput.
"""

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


def _date_str(ns: int) -> str:
    """Convert nanosecond timestamp to YYYY-MM-DD string."""
    dt = datetime.fromtimestamp(ns / 1_000_000_000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


class PolygonProvider(DataProvider):
    """Polygon.io REST API data provider.

    Primary use: historical US equity tick data backfill and corporate actions.
    Not used for futures/FX/bonds (IB covers those).
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = None
        self._rate_limit_sleep = 60.0 / max(config.rate_limit_per_minute, 1)

    def connect(self) -> None:
        try:
            from polygon import RESTClient
        except ImportError:
            raise ImportError("polygon-api-client is required: pip install polygon-api-client")

        self._client = RESTClient(api_key=self.config.api_key)
        self._connected = True
        logger.info("Connected to Polygon.io")

    def disconnect(self) -> None:
        self._client = None
        self._connected = False
        logger.info("Disconnected from Polygon.io")

    def _rate_limit(self) -> None:
        time.sleep(self._rate_limit_sleep)

    def get_historical_bars(
        self,
        symbol: str,
        asset_class: AssetClass,
        start_ns: int,
        end_ns: int,
        bar_size: str = "1min",
    ) -> pd.DataFrame:
        """Fetch historical bars from Polygon."""
        if self._client is None:
            raise RuntimeError("Not connected")

        multiplier, timespan = {
            "1min": (1, "minute"),
            "5min": (5, "minute"),
            "15min": (15, "minute"),
            "1hour": (1, "hour"),
            "1day": (1, "day"),
        }.get(bar_size, (1, "minute"))

        start_date = _date_str(start_ns)
        end_date = _date_str(end_ns)

        ticker = self._resolve_polygon_ticker(symbol, asset_class)

        try:
            aggs = list(self._client.list_aggs(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_=start_date,
                to=end_date,
                limit=50000,
            ))
            self._rate_limit()
        except Exception as e:
            logger.error("Polygon bars fetch failed for %s: %s", symbol, e)
            return pd.DataFrame()

        if not aggs:
            return pd.DataFrame()

        records = []
        for a in aggs:
            ts_ns = int(a.timestamp * 1_000_000) if a.timestamp else 0  # ms -> ns
            records.append({
                "timestamp_ns": ts_ns,
                "open": a.open,
                "high": a.high,
                "low": a.low,
                "close": a.close,
                "volume": int(a.volume) if a.volume else 0,
                "vwap": a.vwap if a.vwap else (a.open + a.close) / 2,
            })

        df = pd.DataFrame(records)
        df = df[(df["timestamp_ns"] >= start_ns) & (df["timestamp_ns"] <= end_ns)]
        return df.reset_index(drop=True)

    def get_historical_ticks(
        self,
        symbol: str,
        asset_class: AssetClass,
        start_ns: int,
        end_ns: int,
    ) -> Iterator[Tick]:
        """Fetch historical trades from Polygon. Requires paid tier for full history."""
        if self._client is None:
            raise RuntimeError("Not connected")

        ticker = self._resolve_polygon_ticker(symbol, asset_class)
        date = _date_str(start_ns)

        try:
            trades = list(self._client.list_trades(
                ticker=ticker,
                timestamp_gte=start_ns,
                timestamp_lte=end_ns,
                limit=50000,
                order="asc",
            ))
            self._rate_limit()
        except Exception as e:
            logger.error("Polygon ticks fetch failed for %s: %s", symbol, e)
            return

        for t in trades:
            exchange_ns = t.sip_timestamp if t.sip_timestamp else 0
            participant_ns = t.participant_timestamp if t.participant_timestamp else exchange_ns

            yield Tick(
                exchange_time_ns=exchange_ns,
                capture_time_ns=participant_ns or exchange_ns,
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
        """Polygon WebSocket streaming — requires paid tier."""
        raise NotImplementedError(
            "Polygon real-time streaming requires paid subscription. Use IBKRProvider for live data."
        )

    def get_instrument_info(self, symbol: str, asset_class: AssetClass) -> dict:
        """Get ticker details from Polygon."""
        if self._client is None:
            raise RuntimeError("Not connected")

        try:
            details = self._client.get_ticker_details(symbol)
            self._rate_limit()
        except Exception as e:
            logger.error("Polygon instrument info failed for %s: %s", symbol, e)
            return {}

        return {
            "symbol": symbol,
            "name": details.name if details.name else "",
            "market": details.market if details.market else "",
            "locale": details.locale if details.locale else "",
            "currency": details.currency_name if details.currency_name else "USD",
            "primary_exchange": details.primary_exchange if details.primary_exchange else "",
            "type": details.type if details.type else "",
            "sic_code": details.sic_code if details.sic_code else "",
            "market_cap": details.market_cap if details.market_cap else 0,
            "share_class_shares_outstanding": (
                details.share_class_shares_outstanding
                if details.share_class_shares_outstanding else 0
            ),
        }

    def get_corporate_actions(
        self,
        symbol: str,
        start_ns: int,
        end_ns: int,
    ) -> list[dict]:
        """Fetch splits and dividends from Polygon."""
        if self._client is None:
            raise RuntimeError("Not connected")

        actions = []

        # Splits
        try:
            splits = list(self._client.list_splits(
                ticker=symbol,
                execution_date_gte=_date_str(start_ns),
                execution_date_lte=_date_str(end_ns),
            ))
            self._rate_limit()
            for s in splits:
                exec_date = datetime.strptime(s.execution_date, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
                actions.append({
                    "type": "SPLIT",
                    "effective_ns": int(exec_date.timestamp() * 1_000_000_000),
                    "split_ratio": s.split_to / s.split_from if s.split_from else 1.0,
                    "split_from": s.split_from,
                    "split_to": s.split_to,
                })
        except Exception as e:
            logger.warning("Polygon splits fetch failed for %s: %s", symbol, e)

        # Dividends
        try:
            divs = list(self._client.list_dividends(
                ticker=symbol,
                ex_dividend_date_gte=_date_str(start_ns),
                ex_dividend_date_lte=_date_str(end_ns),
            ))
            self._rate_limit()
            for d in divs:
                ex_date = datetime.strptime(d.ex_dividend_date, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
                actions.append({
                    "type": "DIVIDEND",
                    "effective_ns": int(ex_date.timestamp() * 1_000_000_000),
                    "dividend_amount": d.cash_amount if d.cash_amount else 0.0,
                    "frequency": d.frequency if d.frequency else 0,
                })
        except Exception as e:
            logger.warning("Polygon dividends fetch failed for %s: %s", symbol, e)

        return sorted(actions, key=lambda a: a["effective_ns"])

    @staticmethod
    def _resolve_polygon_ticker(symbol: str, asset_class: AssetClass) -> str:
        """Map our symbol naming to Polygon's ticker format."""
        if asset_class == AssetClass.FX:
            # Polygon uses C:EURUSD format for forex
            return f"C:{symbol}"
        elif asset_class in (AssetClass.FUTURE, AssetClass.COMMODITY, AssetClass.VOLATILITY):
            # Polygon uses different format for futures
            return symbol
        return symbol
