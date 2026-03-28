"""Interactive Brokers data provider — primary data + broker for all asset classes.

Uses ib_insync for TWS API connectivity. Provides:
- Historical bars (daily/minute) for all asset classes
- Historical ticks (when available, ~1 year depth)
- Live tick streaming
- Instrument metadata and corporate actions
- Paper trading account (free, no minimum)

The same IB connection is used for data in Phase 1 and execution in Phase 7+.
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

# Seconds between API calls to respect IB rate limits
IB_RATE_LIMIT_SLEEP = 0.5


def _ns_to_datetime(ns: int) -> datetime:
    return datetime.fromtimestamp(ns / 1_000_000_000, tz=timezone.utc)


def _datetime_to_ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000_000)


class IBKRProvider(DataProvider):
    """Interactive Brokers data provider via TWS API (ib_insync).

    Requires TWS or IB Gateway running locally or on a specified host.
    Paper account: no minimum balance, free delayed data for all asset classes.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._ib = None
        self._contracts_cache: dict[str, object] = {}

    def connect(self) -> None:
        """Connect to TWS/IB Gateway."""
        try:
            from ib_insync import IB
        except ImportError:
            raise ImportError("ib_insync is required: pip install ib_insync")

        self._ib = IB()
        host = self.config.host or "127.0.0.1"
        port = self.config.port or (7497 if self.config.paper else 7496)
        client_id = 1

        self._ib.connect(host, port, clientId=client_id)
        self._connected = True
        logger.info(
            "Connected to IB %s:%d (paper=%s)", host, port, self.config.paper
        )

    def disconnect(self) -> None:
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
        self._connected = False
        logger.info("Disconnected from IB")

    def _make_contract(self, symbol: str, asset_class: AssetClass):
        """Create an IB contract object for the given symbol and asset class."""
        from ib_insync import Contract, Forex, Future, Stock

        cache_key = f"{symbol}:{asset_class.value}"
        if cache_key in self._contracts_cache:
            return self._contracts_cache[cache_key]

        contract = None
        if asset_class in (AssetClass.EQUITY, AssetClass.ETF):
            contract = Stock(symbol, "SMART", "USD")
        elif asset_class == AssetClass.FX:
            # FX pairs like EURUSD -> Forex('EUR', 'USD')
            if len(symbol) == 6:
                contract = Forex(symbol[:3] + symbol[3:])
            else:
                contract = Forex(symbol)
        elif asset_class in (AssetClass.FUTURE, AssetClass.COMMODITY, AssetClass.VOLATILITY):
            # Futures: use front month by default
            contract = Future(symbol, exchange="CME")
        elif asset_class == AssetClass.BOND:
            contract = Future(symbol, exchange="CME")
        else:
            contract = Stock(symbol, "SMART", "USD")

        if contract is not None:
            # Qualify to get full contract details
            try:
                qualified = self._ib.qualifyContracts(contract)
                if qualified:
                    contract = qualified[0]
            except Exception as e:
                logger.warning("Failed to qualify contract %s: %s", symbol, e)

        self._contracts_cache[cache_key] = contract
        return contract

    def get_historical_bars(
        self,
        symbol: str,
        asset_class: AssetClass,
        start_ns: int,
        end_ns: int,
        bar_size: str = "1min",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars from IB.

        IB bar sizes: '1 min', '5 mins', '15 mins', '1 hour', '1 day'
        IB limits: ~1 year of minute data, 10+ years of daily data.
        """
        contract = self._make_contract(symbol, asset_class)
        if contract is None:
            return pd.DataFrame()

        end_dt = _ns_to_datetime(end_ns)
        start_dt = _ns_to_datetime(start_ns)
        duration_days = (end_dt - start_dt).days + 1

        # Map our bar_size to IB format
        ib_bar_size = {
            "1min": "1 min",
            "5min": "5 mins",
            "15min": "15 mins",
            "1hour": "1 hour",
            "1day": "1 day",
        }.get(bar_size, "1 min")

        # IB duration string
        if duration_days <= 1:
            duration_str = "1 D"
        elif duration_days <= 30:
            duration_str = f"{duration_days} D"
        elif duration_days <= 365:
            months = duration_days // 30
            duration_str = f"{months} M"
        else:
            years = duration_days // 365
            duration_str = f"{years} Y"

        what_to_show = "TRADES" if asset_class != AssetClass.FX else "MIDPOINT"

        try:
            bars = self._ib.reqHistoricalData(
                contract,
                endDateTime=end_dt,
                durationStr=duration_str,
                barSizeSetting=ib_bar_size,
                whatToShow=what_to_show,
                useRTH=False,  # Include extended hours
                formatDate=2,  # UTC timestamps
            )
            time.sleep(IB_RATE_LIMIT_SLEEP)
        except Exception as e:
            logger.error("Failed to fetch bars for %s: %s", symbol, e)
            return pd.DataFrame()

        if not bars:
            return pd.DataFrame()

        records = []
        for bar in bars:
            bar_dt = bar.date if isinstance(bar.date, datetime) else datetime.fromisoformat(str(bar.date))
            if bar_dt.tzinfo is None:
                bar_dt = bar_dt.replace(tzinfo=timezone.utc)
            ts_ns = _datetime_to_ns(bar_dt)
            records.append({
                "timestamp_ns": ts_ns,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": int(bar.volume),
                "vwap": bar.average if hasattr(bar, "average") else (bar.open + bar.close) / 2,
            })

        df = pd.DataFrame(records)
        # Filter to requested range
        df = df[(df["timestamp_ns"] >= start_ns) & (df["timestamp_ns"] <= end_ns)]
        return df.reset_index(drop=True)

    def get_historical_ticks(
        self,
        symbol: str,
        asset_class: AssetClass,
        start_ns: int,
        end_ns: int,
    ) -> Iterator[Tick]:
        """Fetch historical ticks from IB. Limited to ~1 year of history."""
        contract = self._make_contract(symbol, asset_class)
        if contract is None:
            return

        start_dt = _ns_to_datetime(start_ns)

        try:
            ticks = self._ib.reqHistoricalTicks(
                contract,
                startDateTime=start_dt,
                endDateTime="",
                numberOfTicks=1000,
                whatToShow="BID_ASK",
                useRth=False,
            )
            time.sleep(IB_RATE_LIMIT_SLEEP)
        except Exception as e:
            logger.error("Failed to fetch ticks for %s: %s", symbol, e)
            return

        capture_time = ns_now()
        for t in ticks:
            tick_dt = t.time if isinstance(t.time, datetime) else datetime.fromisoformat(str(t.time))
            if tick_dt.tzinfo is None:
                tick_dt = tick_dt.replace(tzinfo=timezone.utc)
            exchange_ns = _datetime_to_ns(tick_dt)

            if exchange_ns > end_ns:
                break

            yield Tick(
                exchange_time_ns=exchange_ns,
                capture_time_ns=capture_time,
                symbol_id=0,  # Caller must resolve via symbol master
                bid=t.priceBid if hasattr(t, "priceBid") else 0.0,
                ask=t.priceAsk if hasattr(t, "priceAsk") else 0.0,
                bid_size=int(t.sizeBid) if hasattr(t, "sizeBid") else 0,
                ask_size=int(t.sizeAsk) if hasattr(t, "sizeAsk") else 0,
                last_price=(t.priceBid + t.priceAsk) / 2 if hasattr(t, "priceBid") else 0.0,
                last_size=0,
                trade_condition=TradeCondition.REGULAR,
            )
            capture_time += 1  # Increment to maintain monotonicity

    def stream_ticks(
        self,
        symbols: list[str],
        asset_class: AssetClass,
    ) -> Iterator[Tick]:
        """Subscribe to live tick stream from IB.

        Yields Ticks as they arrive. This is a blocking generator — runs until
        disconnect() is called.
        """
        contracts = [self._make_contract(s, asset_class) for s in symbols]

        for contract in contracts:
            if contract is not None:
                self._ib.reqMktData(contract, "", False, False)

        while self._connected and self._ib.isConnected():
            self._ib.sleep(0.01)  # IB event loop tick
            for ticker in self._ib.tickers():
                if ticker.last != ticker.last:  # NaN check
                    continue

                capture_ns = ns_now()
                exchange_ns = capture_ns  # IB doesn't provide exchange timestamp on live

                yield Tick(
                    exchange_time_ns=exchange_ns,
                    capture_time_ns=capture_ns,
                    symbol_id=0,  # Caller resolves
                    bid=ticker.bid if ticker.bid == ticker.bid else 0.0,
                    ask=ticker.ask if ticker.ask == ticker.ask else 0.0,
                    bid_size=int(ticker.bidSize) if ticker.bidSize == ticker.bidSize else 0,
                    ask_size=int(ticker.askSize) if ticker.askSize == ticker.askSize else 0,
                    last_price=ticker.last if ticker.last == ticker.last else 0.0,
                    last_size=int(ticker.lastSize) if ticker.lastSize == ticker.lastSize else 0,
                    trade_condition=TradeCondition.REGULAR,
                )

    def get_instrument_info(self, symbol: str, asset_class: AssetClass) -> dict:
        """Get instrument metadata from IB."""
        contract = self._make_contract(symbol, asset_class)
        if contract is None:
            return {}

        details = self._ib.reqContractDetails(contract)
        time.sleep(IB_RATE_LIMIT_SLEEP)

        if not details:
            return {}

        d = details[0]
        return {
            "symbol": symbol,
            "exchange": d.contract.exchange,
            "currency": d.contract.currency,
            "sec_type": d.contract.secType,
            "long_name": d.longName,
            "industry": d.industry,
            "category": d.category,
            "subcategory": d.subcategory,
            "min_tick": d.minTick,
            "multiplier": d.contract.multiplier,
            "trading_hours": d.tradingHours,
        }

    def get_corporate_actions(
        self,
        symbol: str,
        start_ns: int,
        end_ns: int,
    ) -> list[dict]:
        """IB doesn't have a direct corporate actions API.
        Use Polygon or manual data for corporate action history.
        Returns empty list — supplemented by PolygonProvider.
        """
        logger.debug("IB does not provide corporate actions API; use PolygonProvider")
        return []
