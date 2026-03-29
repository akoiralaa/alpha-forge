
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

    def __init__(self, storage_path: str | Path):
        self._storage_path = Path(storage_path).expanduser().resolve()
        self._storage_path.mkdir(parents=True, exist_ok=True)
        uri = f"lmdb://{self._storage_path}"
        self._ac = adb.Arctic(uri)
        self._ensure_libraries()
        logger.info("TickStore initialized at %s", uri)

    def _ensure_libraries(self) -> None:
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
        return f"sym_{symbol_id:08d}"

    # ── Tick writes ──────────────────────────────────────────────────────

    def write_ticks_raw(self, symbol_id: int, df: pd.DataFrame) -> None:
        self._validate_tick_df(df)
        key = self._symbol_key(symbol_id)
        lib = self._lib(LIB_TICKS_RAW)
        if lib.has_symbol(key):
            lib.append(key, df)
        else:
            lib.write(key, df)
        logger.debug("Wrote %d raw ticks for symbol %d", len(df), symbol_id)

    def write_ticks_adjusted(self, symbol_id: int, df: pd.DataFrame) -> None:
        key = self._symbol_key(symbol_id)
        lib = self._lib(LIB_TICKS_ADJUSTED)
        # Adjusted series is rewritten fully on adjustment recalculation
        lib.write(key, df)
        logger.debug("Wrote %d adjusted ticks for symbol %d", len(df), symbol_id)

    def write_rejections(self, symbol_id: int, df: pd.DataFrame) -> None:
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
        key = self._symbol_key(symbol_id)
        lib = self._lib(LIB_QUALITY_REJECTIONS)
        if not lib.has_symbol(key):
            return pd.DataFrame()
        return lib.read(key).data

    # ── Bar writes/reads ─────────────────────────────────────────────────

    def write_bars_raw(self, symbol_id: int, df: pd.DataFrame) -> None:
        key = self._symbol_key(symbol_id)
        lib = self._lib(LIB_BARS_RAW)
        if lib.has_symbol(key):
            lib.append(key, df)
        else:
            lib.write(key, df)

    def write_bars_adjusted(self, symbol_id: int, df: pd.DataFrame) -> None:
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
        self._lib(LIB_METADATA).write(key, df)

    def read_metadata(self, key: str) -> pd.DataFrame:
        lib = self._lib(LIB_METADATA)
        if not lib.has_symbol(key):
            return pd.DataFrame()
        return lib.read(key).data

    # ── Queries ──────────────────────────────────────────────────────────

    def list_symbols(self, library: str = LIB_TICKS_RAW) -> list[int]:
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
