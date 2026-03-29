
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum, IntEnum
from pathlib import Path
from typing import Optional

import numpy as np

class Side(IntEnum):
    BUY = 1
    SELL = -1

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"

class StressScenario(Enum):
    CRISIS_2008 = "CRISIS_2008"
    FLASH_CRASH_2010 = "FLASH_CRASH_2010"
    VIX_EXPLOSION_2018 = "VIX_EXPLOSION_2018"
    COVID_CRASH_2020 = "COVID_CRASH_2020"
    RATE_SHOCK_2022 = "RATE_SHOCK_2022"
    LIQUIDITY_BLACKOUT = "LIQUIDITY_BLACKOUT"
    GAP_DOWN_10PCT = "GAP_DOWN_10PCT"
    CORRELATION_CRISIS = "CORRELATION_CRISIS"
    FEED_OUTAGE_5S = "FEED_OUTAGE_5S"

@dataclass(frozen=True)
class OrderIntent:
    symbol_id: int
    side: Side
    order_type: OrderType
    size: int
    limit_price: Optional[float] = None
    signal_time_ns: int = 0
    signal_strength: float = 0.0

@dataclass(frozen=True)
class BookSnapshot:
    symbol_id: int
    timestamp_ns: int
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last_price: float
    vpin: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        m = self.mid
        if m <= 0:
            return 0.0
        return (self.ask - self.bid) / m * 10000.0

@dataclass(frozen=True)
class SimFill:
    symbol_id: int
    side: Side
    fill_price: float
    fill_size: int
    fill_time_ns: int
    cost_bps: float = 0.0
    impact_bps: float = 0.0
    latency_ns: int = 0

@dataclass
class InstrumentSpec:
    symbol_id: int
    commission_bps: float = 0.5
    taker_fee_bps: float = 0.3
    maker_rebate_bps: float = -0.1
    overnight_rate_annual: float = 0.05
    adv_20d: float = 1_000_000.0
    realized_vol_daily_bps: float = 100.0

@dataclass
class Position:
    symbol_id: int
    quantity: int = 0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    last_price: float = 0.0

    def update_mark(self, price: float):
        self.last_price = price
        if self.quantity != 0:
            self.unrealized_pnl = (price - self.avg_price) * self.quantity

    def apply_fill(self, fill: SimFill):
        qty = fill.fill_size if fill.side == Side.BUY else -fill.fill_size
        if self.quantity == 0:
            self.avg_price = fill.fill_price
            self.quantity = qty
        elif (self.quantity > 0 and qty > 0) or (self.quantity < 0 and qty < 0):
            # Adding to position
            total = self.quantity + qty
            self.avg_price = (self.avg_price * self.quantity + fill.fill_price * qty) / total
            self.quantity = total
        else:
            # Reducing or flipping
            close_qty = min(abs(self.quantity), abs(qty))
            pnl_per_unit = fill.fill_price - self.avg_price
            if self.quantity < 0:
                pnl_per_unit = -pnl_per_unit
            self.realized_pnl += pnl_per_unit * close_qty
            remaining = self.quantity + qty
            if remaining == 0:
                self.avg_price = 0.0
            elif abs(remaining) > abs(self.quantity):
                # Flipped
                self.avg_price = fill.fill_price
            self.quantity = remaining
        self.update_mark(fill.fill_price)

@dataclass
class BacktestResult:
    pnl_series: np.ndarray
    total_pnl: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    n_trades: int = 0
    total_cost_bps: float = 0.0
    avg_latency_ns: float = 0.0
    fills: list = field(default_factory=list)

    def compute_metrics(self):
        if len(self.pnl_series) < 2:
            return
        self.total_pnl = float(self.pnl_series[-1])
        returns = np.diff(self.pnl_series)
        if len(returns) == 0:
            return
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        self.sharpe = float(mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

        # Max drawdown
        cummax = np.maximum.accumulate(self.pnl_series)
        drawdowns = self.pnl_series - cummax
        self.max_drawdown = float(np.min(drawdowns))
        peak = np.max(np.abs(cummax))
        self.max_drawdown_pct = float(abs(self.max_drawdown) / peak) if peak > 0 else 0.0

@dataclass
class WalkForwardResult:
    train_range: tuple[date, date]
    embargo_range: tuple[date, date]
    test_range: tuple[date, date]
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    n_trades: int = 0

@dataclass
class LockBox:
    start_date: date
    end_date: date
    data_hash: str = ""
    access_count: int = 0
    _path: Optional[Path] = field(default=None, repr=False)

    def save(self, path: Path):
        self._path = path
        data = {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "data_hash": self.data_hash,
            "access_count": self.access_count,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> LockBox:
        data = json.loads(path.read_text())
        return cls(
            start_date=date.fromisoformat(data["start_date"]),
            end_date=date.fromisoformat(data["end_date"]),
            data_hash=data["data_hash"],
            access_count=data["access_count"],
            _path=path,
        )

    def access(self) -> None:
        self.access_count += 1
        if self._path:
            self.save(self._path)

    @staticmethod
    def compute_hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()
