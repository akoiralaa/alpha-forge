"""Dynamic universe management — recomputed at market open every trading day.

The tradeable universe is not static. It adapts based on liquidity filters per
asset class. Assets below thresholds are removed; assets above are added.
Universe is stored as date-partitioned snapshots for point-in-time backtesting.

This prevents:
1. Trading illiquid instruments (adverse market impact)
2. Survivorship bias in live trading (universe adapts as markets change)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.data.arctic_store import TickStore
from src.data.fundamentals import FundamentalsStore
from src.data.ingest.base import AssetClass, date_to_ns
from src.data.symbol_master import SymbolMaster

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class UniverseFilter:
    """Liquidity filter thresholds per asset class."""
    # Equities
    equity_adv_min_usd: float = 10_000_000.0     # 20-day ADV >= $10M
    equity_price_min: float = 5.0                  # price >= $5
    equity_market_cap_min: float = 500_000_000.0   # market cap >= $500M

    # Equity futures
    eq_futures_oi_min: int = 10_000                # 20-day avg OI >= 10,000 contracts

    # FX
    fx_daily_notional_min: float = 100_000_000.0   # 20-day avg daily notional >= $100M

    # Commodity futures
    commodity_oi_min: int = 5_000                  # 20-day avg OI >= 5,000 contracts

    # Bond futures: always included if listed on major exchange
    # (no filter — they pass by default)


# Standard instruments per the build protocol
EQUITY_ETFS = [
    "SPY", "QQQ", "XLK", "XLV", "XLE", "XLF", "XLI", "XLB", "XLU", "XLRE", "XLC", "XLP", "XLY",
]
EQUITY_INDEX_FUTURES = ["ES", "NQ", "RTY", "YM"]
COMMODITY_FUTURES = ["CL", "NG", "GC", "SI", "HG", "ZC", "ZW", "ZS"]
FIXED_INCOME_FUTURES = ["ZN", "ZB", "ZF", "ZT", "GE"]
FX_PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]
VIX_FUTURES = ["VX"]


@dataclass
class UniverseSnapshot:
    """A point-in-time snapshot of the tradeable universe."""
    date_ns: int
    symbol_ids: list[int]
    asset_classes: dict[int, str]   # symbol_id -> asset_class
    sectors: dict[int, str | None]  # symbol_id -> sector (equities only)

    @property
    def count(self) -> int:
        return len(self.symbol_ids)

    def contains(self, symbol_id: int) -> bool:
        return symbol_id in self.symbol_ids

    def by_asset_class(self, ac: AssetClass) -> list[int]:
        return [sid for sid in self.symbol_ids if self.asset_classes.get(sid) == ac.value]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "symbol_id": self.symbol_ids,
            "asset_class": [self.asset_classes.get(sid, "") for sid in self.symbol_ids],
            "sector": [self.sectors.get(sid) for sid in self.symbol_ids],
            "date_ns": [self.date_ns] * len(self.symbol_ids),
        })


class UniverseManager:
    """Manages the dynamic tradeable universe with date-partitioned snapshots.

    Recomputes the universe based on liquidity filters applied to each asset class.
    Stores snapshots in ArcticDB for point-in-time backtesting queries.
    """

    SNAPSHOT_KEY_PREFIX = "universe_"

    def __init__(
        self,
        symbol_master: SymbolMaster,
        fundamentals: FundamentalsStore,
        tick_store: TickStore,
        filters: UniverseFilter | None = None,
    ):
        self._sm = symbol_master
        self._fund = fundamentals
        self._ts = tick_store
        self._filters = filters or UniverseFilter()

    def compute_universe(self, as_of_ns: int) -> UniverseSnapshot:
        """Compute the tradeable universe as of a specific timestamp.

        Applies per-asset-class liquidity filters to all active instruments.
        """
        active = self._sm.get_active_instruments(as_of_ns)
        included_ids = []
        asset_classes: dict[int, str] = {}
        sectors: dict[int, str | None] = {}

        for inst in active:
            passes = self._check_instrument(inst, as_of_ns)
            if passes:
                included_ids.append(inst.canonical_id)
                asset_classes[inst.canonical_id] = inst.asset_class.value
                sectors[inst.canonical_id] = inst.sector

        snapshot = UniverseSnapshot(
            date_ns=as_of_ns,
            symbol_ids=sorted(included_ids),
            asset_classes=asset_classes,
            sectors=sectors,
        )
        logger.info(
            "Universe as-of %d: %d instruments", as_of_ns, snapshot.count
        )
        return snapshot

    def _check_instrument(self, inst, as_of_ns: int) -> bool:
        """Check if an instrument passes its asset-class-specific liquidity filter."""
        ac = inst.asset_class

        if ac == AssetClass.EQUITY or ac == AssetClass.ETF:
            return self._check_equity(inst.canonical_id, as_of_ns)
        elif ac == AssetClass.FUTURE:
            # Differentiate equity futures, commodity futures, fixed income futures
            # by sector or ticker convention
            return self._check_future(inst, as_of_ns)
        elif ac == AssetClass.FX:
            return self._check_fx(inst.canonical_id, as_of_ns)
        elif ac == AssetClass.BOND:
            # Bond futures: always included if listed on major exchange
            return True
        elif ac == AssetClass.COMMODITY:
            return self._check_commodity(inst.canonical_id, as_of_ns)
        elif ac == AssetClass.VOLATILITY:
            return True  # VIX futures: included if listed

        return False

    def _check_equity(self, canonical_id: int, as_of_ns: int) -> bool:
        """Equity filter: ADV >= $10M, price >= $5, market cap >= $500M."""
        adv = self._fund.get_as_of(canonical_id, FundamentalsStore.METRIC_ADV_20D, as_of_ns)
        if adv is None or adv.value < self._filters.equity_adv_min_usd:
            return False

        mcap = self._fund.get_as_of(canonical_id, FundamentalsStore.METRIC_MARKET_CAP, as_of_ns)
        if mcap is None or mcap.value < self._filters.equity_market_cap_min:
            return False

        # Price check — get last known price from tick store
        df = self._ts.read_ticks_raw(canonical_id, end_ns=as_of_ns)
        if df.empty:
            return False
        last_price = df["last_price"].iloc[-1]
        if last_price < self._filters.equity_price_min:
            return False

        return True

    def _check_future(self, inst, as_of_ns: int) -> bool:
        """Futures filter by type: equity index futures vs commodity futures."""
        oi = self._fund.get_as_of(
            inst.canonical_id, FundamentalsStore.METRIC_OPEN_INTEREST, as_of_ns
        )
        if oi is None:
            return False

        # Commodity futures have a lower threshold
        if inst.sector and inst.sector.upper() in ("COMMODITY", "ENERGY", "AGRICULTURE", "METALS"):
            return oi.value >= self._filters.commodity_oi_min

        # Equity index futures
        return oi.value >= self._filters.eq_futures_oi_min

    def _check_fx(self, canonical_id: int, as_of_ns: int) -> bool:
        """FX filter: 20-day avg daily notional >= $100M."""
        adv = self._fund.get_as_of(canonical_id, FundamentalsStore.METRIC_ADV_20D, as_of_ns)
        if adv is None:
            return False
        return adv.value >= self._filters.fx_daily_notional_min

    def _check_commodity(self, canonical_id: int, as_of_ns: int) -> bool:
        """Commodity filter: 20-day avg OI >= 5,000 contracts."""
        oi = self._fund.get_as_of(
            canonical_id, FundamentalsStore.METRIC_OPEN_INTEREST, as_of_ns
        )
        if oi is None:
            return False
        return oi.value >= self._filters.commodity_oi_min

    # ── Snapshot storage ─────────────────────────────────────────────────

    def save_snapshot(self, snapshot: UniverseSnapshot) -> None:
        """Save a universe snapshot to the tick store metadata library."""
        key = f"{self.SNAPSHOT_KEY_PREFIX}{snapshot.date_ns}"
        self._ts.write_metadata(key, snapshot.to_dataframe())
        logger.debug("Saved universe snapshot: %s (%d instruments)", key, snapshot.count)

    def load_snapshot(self, date_ns: int) -> UniverseSnapshot | None:
        """Load a universe snapshot for a specific date."""
        key = f"{self.SNAPSHOT_KEY_PREFIX}{date_ns}"
        df = self._ts.read_metadata(key)
        if df.empty:
            return None
        return UniverseSnapshot(
            date_ns=date_ns,
            symbol_ids=df["symbol_id"].tolist(),
            asset_classes=dict(zip(df["symbol_id"], df["asset_class"])),
            sectors=dict(zip(df["symbol_id"], df["sector"])),
        )

    def load_nearest_snapshot(self, date_ns: int) -> UniverseSnapshot | None:
        """Load the most recent universe snapshot on or before date_ns.

        Used by backtester for point-in-time universe queries.
        """
        # List all universe snapshots from metadata
        lib = self._ts._lib("metadata")
        all_keys = lib.list_symbols()
        universe_keys = [k for k in all_keys if k.startswith(self.SNAPSHOT_KEY_PREFIX)]

        if not universe_keys:
            return None

        # Parse timestamps and find the nearest one <= date_ns
        candidates = []
        for k in universe_keys:
            try:
                ts = int(k[len(self.SNAPSHOT_KEY_PREFIX):])
                if ts <= date_ns:
                    candidates.append(ts)
            except ValueError:
                continue

        if not candidates:
            return None

        nearest = max(candidates)
        return self.load_snapshot(nearest)

    def get_universe_diff(
        self, date_ns_old: int, date_ns_new: int
    ) -> tuple[list[int], list[int]]:
        """Compare two universe snapshots. Returns (added, removed) symbol_ids."""
        old = self.load_snapshot(date_ns_old)
        new = self.load_snapshot(date_ns_new)
        if old is None or new is None:
            return [], []
        old_set = set(old.symbol_ids)
        new_set = set(new.symbol_ids)
        return sorted(new_set - old_set), sorted(old_set - new_set)
