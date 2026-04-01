
from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd

class AssetClass(enum.Enum):
    EQUITY = "EQUITY"
    FUTURE = "FUTURE"
    FX = "FX"
    BOND = "BOND"
    COMMODITY = "COMMODITY"
    ETF = "ETF"
    VOLATILITY = "VOLATILITY"

class TradeCondition(enum.IntEnum):
    REGULAR = 0
    ODD_LOT = 1
    FORM_T = 2          # extended hours
    AVERAGE_PRICE = 3
    CASH_SALE = 4
    INTERMARKET_SWEEP = 5
    DERIVATIVELY_PRICED = 6
    OPENING = 7
    CLOSING = 8
    CORRECTED = 9
    UNKNOWN = 255

@dataclass(slots=True, frozen=True)
class Tick:
    exchange_time_ns: int          # when exchange matching engine processed the event
    capture_time_ns: int           # when our system received the packet
    symbol_id: int                 # canonical ID from symbol master, never raw ticker
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last_price: float
    last_size: int
    trade_condition: int = TradeCondition.REGULAR

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        mid = self.mid
        if mid <= 0:
            return 0.0
        return (self.ask - self.bid) / mid * 10_000

    def to_dict(self) -> dict:
        return {
            "exchange_time_ns": self.exchange_time_ns,
            "capture_time_ns": self.capture_time_ns,
            "symbol_id": self.symbol_id,
            "bid": self.bid,
            "ask": self.ask,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "last_price": self.last_price,
            "last_size": self.last_size,
            "trade_condition": self.trade_condition,
        }

    @staticmethod
    def schema_dtypes() -> dict[str, np.dtype]:
        return {
            "exchange_time_ns": np.dtype("int64"),
            "capture_time_ns": np.dtype("int64"),
            "symbol_id": np.dtype("int32"),
            "bid": np.dtype("float64"),
            "ask": np.dtype("float64"),
            "bid_size": np.dtype("int64"),
            "ask_size": np.dtype("int64"),
            "last_price": np.dtype("float64"),
            "last_size": np.dtype("int64"),
            "trade_condition": np.dtype("uint8"),
        }

    @staticmethod
    def ticks_to_dataframe(ticks: list[Tick]) -> pd.DataFrame:
        if not ticks:
            return pd.DataFrame(columns=list(Tick.schema_dtypes().keys())).astype(
                Tick.schema_dtypes()
            )
        records = [t.to_dict() for t in ticks]
        df = pd.DataFrame(records)
        for col, dtype in Tick.schema_dtypes().items():
            df[col] = df[col].astype(dtype)
        return df

@dataclass
class ProviderConfig:
    api_key: str = ""
    api_secret: str = ""
    host: str = ""
    port: int = 0
    paper: bool = True
    rate_limit_per_minute: int = 200

class DataProvider(ABC):

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._connected = False

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def get_historical_bars(
        self,
        symbol: str,
        asset_class: AssetClass,
        start_ns: int,
        end_ns: int,
        bar_size: str = "1min",
    ) -> pd.DataFrame: ...

    @abstractmethod
    def get_historical_ticks(
        self,
        symbol: str,
        asset_class: AssetClass,
        start_ns: int,
        end_ns: int,
    ) -> Iterator[Tick]: ...

    @abstractmethod
    def stream_ticks(
        self,
        symbols: list[str],
        asset_class: AssetClass,
    ) -> Iterator[Tick]: ...

    @abstractmethod
    def get_instrument_info(
        self,
        symbol: str,
        asset_class: AssetClass,
    ) -> dict: ...

    @abstractmethod
    def get_corporate_actions(
        self,
        symbol: str,
        start_ns: int,
        end_ns: int,
    ) -> list[dict]: ...

    @property
    def is_connected(self) -> bool:
        return self._connected

def ns_now() -> int:
    import time
    return int(time.time_ns())

def date_to_ns(year: int, month: int, day: int) -> int:
    import datetime
    dt = datetime.datetime(year, month, day, tzinfo=datetime.timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)
