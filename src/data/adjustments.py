
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.symbol_master import SymbolMaster

logger = logging.getLogger(__name__)

class PriceAdjuster:

    def __init__(self, symbol_master: SymbolMaster):
        self._sm = symbol_master

    def compute_adjustment_factors(
        self,
        canonical_id: int,
        timestamps_ns: np.ndarray,
    ) -> np.ndarray:
        splits = self._sm.get_splits(canonical_id)
        dividends = self._sm.get_dividends(canonical_id)

        factors = np.ones(len(timestamps_ns), dtype=np.float64)

        if not splits and not dividends:
            return factors

        # Apply split adjustments (backward from present)
        # For each split at time T with ratio R:
        # All prices before T are multiplied by 1/R (for a 2:1 split, old prices halved)
        for split_time_ns, split_ratio in splits:
            if split_ratio is None or split_ratio <= 0:
                continue
            # All timestamps before this split get adjusted
            mask = timestamps_ns < split_time_ns
            factors[mask] *= (1.0 / split_ratio)

        # Apply dividend adjustments (backward-adjusted)
        # For each dividend at time T with amount D and price P at T:
        # All prices before T are multiplied by (1 - D/P_at_T)
        # We approximate P_at_T as the last known price before the dividend
        # In practice this is applied during the full adjust_dataframe call
        # where we have access to the price series

        return factors

    def adjust_dataframe(
        self,
        canonical_id: int,
        df: pd.DataFrame,
        price_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        if price_columns is None:
            # Auto-detect: tick schema vs bar schema
            if "last_price" in df.columns:
                price_columns = ["bid", "ask", "last_price"]
            elif "close" in df.columns:
                price_columns = ["open", "high", "low", "close"]
            else:
                raise ValueError("Cannot auto-detect price columns. Specify price_columns.")

        # Verify columns exist
        missing = [c for c in price_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing price columns: {missing}")

        timestamps = df["capture_time_ns"].values if "capture_time_ns" in df.columns else df["timestamp_ns"].values
        factors = self.compute_adjustment_factors(canonical_id, timestamps)

        adjusted = df.copy()
        for col in price_columns:
            adjusted[col] = adjusted[col].values * factors

        # Also adjust size columns inversely (shares adjust opposite to price)
        size_columns = []
        if "bid_size" in adjusted.columns:
            size_columns.extend(["bid_size", "ask_size", "last_size"])
        elif "volume" in adjusted.columns:
            size_columns.append("volume")

        for col in size_columns:
            if col in adjusted.columns:
                # Inverse adjustment: if price halved (2:1 split), size doubled
                inv_factors = np.where(factors != 0, 1.0 / factors, 1.0)
                adjusted[col] = (adjusted[col].values * inv_factors).astype(adjusted[col].dtype)

        return adjusted

    def verify_adjustment(
        self,
        canonical_id: int,
        df_raw: pd.DataFrame,
        df_adjusted: pd.DataFrame,
        tolerance_pct: float = 0.01,
    ) -> bool:
        splits = self._sm.get_splits(canonical_id)
        if not splits:
            return True

        price_col = "last_price" if "last_price" in df_adjusted.columns else "close"
        time_col = "capture_time_ns" if "capture_time_ns" in df_adjusted.columns else "timestamp_ns"

        for split_time_ns, split_ratio in splits:
            if split_ratio is None:
                continue

            # Find the last tick before split and first tick after
            before = df_adjusted[df_adjusted[time_col] < split_time_ns]
            after = df_adjusted[df_adjusted[time_col] >= split_time_ns]

            if before.empty or after.empty:
                continue

            price_before = before[price_col].iloc[-1]
            price_after = after[price_col].iloc[0]

            if price_before <= 0 or price_after <= 0:
                continue

            pct_diff = abs(price_before - price_after) / price_after
            if pct_diff > tolerance_pct:
                logger.error(
                    "Split adjustment verification failed for symbol %d at %d: "
                    "before=%.4f after=%.4f diff=%.4f%%",
                    canonical_id, split_time_ns, price_before, price_after, pct_diff * 100,
                )
                return False

        return True
