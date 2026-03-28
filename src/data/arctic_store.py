"""ArcticDB tick store — columnar time-series storage for all market data.

Stores both raw and adjusted price series. Every record has dual timestamps
(exchange_time_ns and capture_time_ns). Backtesting uses capture_time_ns
exclusively to prevent look-ahead bias.

ArcticDB (Man Group) is purpose-built for financial time-series:
- Columnar storage (fast time-range queries across thousands of symbols)
- Native versioning (point-in-time data retrieval)
- Billions of rows with sub-millisecond queries
"""

from __future__ import annotations

import logging
from pathlib import Path

import arcticdb as adb
import numpy as np
import pandas as pd

from src.data.ingest.base import Tick

logger = logging.getLogger(__name__)

# Library names within the ArcticDB instance
LIB_TICKS_RAW = "ticks_raw"
LIB_TICKS_ADJUSTED = "ticks_adjusted"
LIB_BARS_RAW = "bars_raw"
LIB_BARS_ADJUSTED = "bars_adjusted"
LIB_QUALITY_REJECTIONS = "quality_rejections"
LIB_METADATA = "metadata"


class TickStore:
    """ArcticDB-backed tick and bar storage.

    Data is organized by symbol_id. Each symbol's ticks are stored as a single
    ArcticDB symbol (keyed by canonical_id) with append-only writes.
    Versioning is enabled so we can do point-in-time queries for backtesting.
    """

    def __init__(self, storage_path: str | Path):
        """Initialize ArcticDB connection.

        Args:
            storage_path: Path to the LMDB storage directory.
                         E.g. "~/one_brain_fund/data/arcticdb"
        """
        self._storage_path = Path(storage_path).expanduser().resolve()
        self._storage_path.mkdir(parents=True, exist_ok=True)
        uri = f"lmdb://{self._storage_path}"
        self._ac = adb.Arctic(uri)
        self._ensure_libraries()
        logger.info("TickStore initialized at %s", uri)

    def _ensure_libraries(self) -> None:
        """Create all required libraries if they don't exist."""
        for lib_name in [
            LIB_TICKS_RAW,
            LIB_TICKS_ADJUSTED,
            LIB_BARS_RAW,
            LIB_BARS_ADJUSTED,
            LIB_QUALITY_REJECTIONS,
            LIB_METADATA,
        ]:
            if lib_name not in self._ac.list_libraries():
                self._ac.create_library(lib_name)
                logger.info("Created library: %s", lib_name)

    def _lib(self, name: str) -> adb.library.Library:
        return self._ac[name]

    @staticmethod
    def _symbol_key(symbol_id: int) -> str:
        """Convert canonical symbol_id to ArcticDB key string."""
        return f"sym_{symbol_id:08d}"

    # ── Tick writes ──────────────────────────────────────────────────────

    def write_ticks_raw(self, symbol_id: int, df: pd.DataFrame) -> None:
        """Append raw tick data for a symbol.

        Args:
            symbol_id: Canonical instrument ID from symbol master.
            df: DataFrame matching Tick.schema_dtypes(). Must be sorted by
                capture_time_ns (ascending).
        """
        self._validate_tick_df(df)
        key = self._symbol_key(symbol_id)
        lib = self._lib(LIB_TICKS_RAW)
        if lib.has_symbol(key):
            lib.append(key, df)
        else:
            lib.write(key, df)
        logger.debug("Wrote %d raw ticks for symbol %d", len(df), symbol_id)

    def write_ticks_adjusted(self, symbol_id: int, df: pd.DataFrame) -> None:
        """Write adjusted tick data (post split/dividend adjustment)."""
        key = self._symbol_key(symbol_id)
        lib = self._lib(LIB_TICKS_ADJUSTED)
        # Adjusted series is rewritten fully on adjustment recalculation
        lib.write(key, df)
        logger.debug("Wrote %d adjusted ticks for symbol %d", len(df), symbol_id)

    def write_rejections(self, symbol_id: int, df: pd.DataFrame) -> None:
        """Store rejected records from data quality pipeline.

        The DataFrame has all Tick columns plus 'rejection_reason' (str).
        """
        key = self._symbol_key(symbol_id)
        lib = self._lib(LIB_QUALITY_REJECTIONS)
        if lib.has_symbol(key):
            lib.append(key, df)
        else:
            lib.write(key, df)

    # ── Tick reads ───────────────────────────────────────────────────────

    def read_ticks_raw(
        self,
        symbol_id: int,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> pd.DataFrame:
        """Read raw tick data for a symbol, optionally filtered by time range.

        Time filtering uses capture_time_ns (backtest-safe).
        """
        key = self._symbol_key(symbol_id)
        lib = self._lib(LIB_TICKS_RAW)
        if not lib.has_symbol(key):
            return pd.DataFrame(columns=list(Tick.schema_dtypes().keys()))

        df = lib.read(key).data
        if start_ns is not None:
            df = df[df["capture_time_ns"] >= start_ns]
        if end_ns is not None:
            df = df[df["capture_time_ns"] <= end_ns]
        return df

    def read_ticks_adjusted(
        self,
        symbol_id: int,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> pd.DataFrame:
        """Read adjusted tick data. This is what the feature engine consumes."""
        key = self._symbol_key(symbol_id)
        lib = self._lib(LIB_TICKS_ADJUSTED)
        if not lib.has_symbol(key):
            return pd.DataFrame(columns=list(Tick.schema_dtypes().keys()))

        df = lib.read(key).data
        if start_ns is not None:
            df = df[df["capture_time_ns"] >= start_ns]
        if end_ns is not None:
            df = df[df["capture_time_ns"] <= end_ns]
        return df

    def read_rejections(self, symbol_id: int) -> pd.DataFrame:
        """Read rejected records for inspection."""
        key = self._symbol_key(symbol_id)
        lib = self._lib(LIB_QUALITY_REJECTIONS)
        if not lib.has_symbol(key):
            return pd.DataFrame()
        return lib.read(key).data

    # ── Bar writes/reads ─────────────────────────────────────────────────

    def write_bars_raw(self, symbol_id: int, df: pd.DataFrame) -> None:
        """Write raw OHLCV bar data."""
        key = self._symbol_key(symbol_id)
        lib = self._lib(LIB_BARS_RAW)
        if lib.has_symbol(key):
            lib.append(key, df)
        else:
            lib.write(key, df)

    def write_bars_adjusted(self, symbol_id: int, df: pd.DataFrame) -> None:
        """Write adjusted OHLCV bar data."""
        key = self._symbol_key(symbol_id)
        lib = self._lib(LIB_BARS_ADJUSTED)
        lib.write(key, df)

    def read_bars_raw(
        self,
        symbol_id: int,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> pd.DataFrame:
        key = self._symbol_key(symbol_id)
        lib = self._lib(LIB_BARS_RAW)
        if not lib.has_symbol(key):
            return pd.DataFrame()
        df = lib.read(key).data
        if start_ns is not None:
            df = df[df["timestamp_ns"] >= start_ns]
        if end_ns is not None:
            df = df[df["timestamp_ns"] <= end_ns]
        return df

    def read_bars_adjusted(
        self,
        symbol_id: int,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> pd.DataFrame:
        key = self._symbol_key(symbol_id)
        lib = self._lib(LIB_BARS_ADJUSTED)
        if not lib.has_symbol(key):
            return pd.DataFrame()
        df = lib.read(key).data
        if start_ns is not None:
            df = df[df["timestamp_ns"] >= start_ns]
        if end_ns is not None:
            df = df[df["timestamp_ns"] <= end_ns]
        return df

    # ── Metadata ─────────────────────────────────────────────────────────

    def write_metadata(self, key: str, df: pd.DataFrame) -> None:
        """Store arbitrary metadata (universe snapshots, etc.)."""
        self._lib(LIB_METADATA).write(key, df)

    def read_metadata(self, key: str) -> pd.DataFrame:
        lib = self._lib(LIB_METADATA)
        if not lib.has_symbol(key):
            return pd.DataFrame()
        return lib.read(key).data

    # ── Queries ──────────────────────────────────────────────────────────

    def list_symbols(self, library: str = LIB_TICKS_RAW) -> list[int]:
        """List all symbol_ids that have data in a library."""
        keys = self._lib(library).list_symbols()
        symbol_ids = []
        for k in keys:
            if k.startswith("sym_"):
                try:
                    symbol_ids.append(int(k[4:]))
                except ValueError:
                    pass
        return sorted(symbol_ids)

    def tick_count(self, symbol_id: int, library: str = LIB_TICKS_RAW) -> int:
        """Get the number of ticks stored for a symbol."""
        key = self._symbol_key(symbol_id)
        lib = self._lib(library)
        if not lib.has_symbol(key):
            return 0
        info = lib.read(key)
        return len(info.data)

    def has_symbol(self, symbol_id: int, library: str = LIB_TICKS_RAW) -> bool:
        key = self._symbol_key(symbol_id)
        return self._lib(library).has_symbol(key)

    # ── Sampling ─────────────────────────────────────────────────────────

    def sample_random_records(
        self,
        library: str = LIB_TICKS_RAW,
        n: int = 10_000,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Sample n random records across all symbols. For validation checks."""
        rng = np.random.default_rng(seed)
        all_symbols = self.list_symbols(library)
        if not all_symbols:
            return pd.DataFrame()

        # Proportional sampling across symbols
        frames = []
        per_symbol = max(1, n // len(all_symbols))
        for sid in all_symbols:
            df = self._lib(library).read(self._symbol_key(sid)).data
            if len(df) == 0:
                continue
            sample_n = min(per_symbol, len(df))
            idx = rng.choice(len(df), size=sample_n, replace=False)
            frames.append(df.iloc[idx])

        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        # If we got more than n, subsample
        if len(result) > n:
            idx = rng.choice(len(result), size=n, replace=False)
            result = result.iloc[idx]
        return result.reset_index(drop=True)

    # ── Validation helpers ───────────────────────────────────────────────

    @staticmethod
    def _validate_tick_df(df: pd.DataFrame) -> None:
        """Validate a tick DataFrame has the correct schema."""
        expected = set(Tick.schema_dtypes().keys())
        actual = set(df.columns)
        missing = expected - actual
        if missing:
            raise ValueError(f"Missing columns in tick DataFrame: {missing}")

        # Verify sorted by capture_time_ns
        if len(df) > 1:
            capture_times = df["capture_time_ns"].values
            if not np.all(capture_times[1:] >= capture_times[:-1]):
                raise ValueError(
                    "Tick DataFrame must be sorted by capture_time_ns (ascending)"
                )
