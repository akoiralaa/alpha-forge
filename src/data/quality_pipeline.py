"""Data quality pipeline — every data point passes through this gate before storage.

Implements all 8 rejection rules from the build protocol. Rejected records are stored
separately with rejection reason. Nothing is silently dropped.

Rejection rate is monitored per symbol per session. Alert if rate exceeds 0.1%.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.ingest.base import Tick

logger = logging.getLogger(__name__)


class RejectionReason(enum.Enum):
    NEGATIVE_PRICE = "last_price <= 0"
    NEGATIVE_BID = "bid <= 0"
    NEGATIVE_ASK = "ask <= 0"
    CROSSED_BOOK = "bid >= ask"
    TIME_TRAVEL = "capture_time_ns < exchange_time_ns"
    STALE_DATA = "capture_time_ns - exchange_time_ns > 10s"
    FAT_FINGER_PRICE = "price outside 10% of rolling mean"
    FAT_FINGER_VOLUME = "volume > 100x rolling average"
    NON_MONOTONIC_TIME = "capture_time_ns not monotonically increasing"


@dataclass
class QualityStats:
    """Per-symbol quality statistics for a session."""
    symbol_id: int
    total_received: int = 0
    total_rejected: int = 0
    rejections_by_reason: dict[str, int] = field(default_factory=dict)

    @property
    def rejection_rate(self) -> float:
        if self.total_received == 0:
            return 0.0
        return self.total_rejected / self.total_received

    @property
    def exceeds_alert_threshold(self) -> bool:
        return self.rejection_rate > 0.001  # 0.1%


@dataclass
class QualityResult:
    """Result of running a tick through the quality pipeline."""
    tick: Tick
    accepted: bool
    rejection_reasons: list[RejectionReason] = field(default_factory=list)


class DataQualityPipeline:
    """Stateful quality pipeline with rolling statistics per symbol.

    Maintains rolling mean/std for price and rolling average for volume
    per symbol to detect fat-finger trades and volume spikes.
    """

    def __init__(
        self,
        price_deviation_pct: float = 0.10,     # 10% from rolling mean
        volume_spike_factor: float = 100.0,     # 100x rolling average
        max_capture_lag_ns: int = 10_000_000_000,  # 10 seconds
        rolling_price_window: int = 20,
        rolling_volume_window: int = 100,
        alert_rejection_rate: float = 0.001,    # 0.1%
    ):
        self._price_deviation_pct = price_deviation_pct
        self._volume_spike_factor = volume_spike_factor
        self._max_capture_lag_ns = max_capture_lag_ns
        self._rolling_price_window = rolling_price_window
        self._rolling_volume_window = rolling_volume_window
        self._alert_rejection_rate = alert_rejection_rate

        # Per-symbol rolling state
        self._price_buffer: dict[int, list[float]] = {}
        self._volume_buffer: dict[int, list[float]] = {}
        self._last_capture_time: dict[int, int] = {}
        self._stats: dict[int, QualityStats] = {}

    def _get_stats(self, symbol_id: int) -> QualityStats:
        if symbol_id not in self._stats:
            self._stats[symbol_id] = QualityStats(symbol_id=symbol_id)
        return self._stats[symbol_id]

    def _rolling_mean(self, buf: list[float]) -> float:
        if not buf:
            return 0.0
        return sum(buf) / len(buf)

    def check_tick(self, tick: Tick) -> QualityResult:
        """Run all quality checks on a single tick.

        Returns a QualityResult indicating acceptance or rejection with reasons.
        All checks are run even if early ones fail — we want the full rejection profile.
        """
        reasons: list[RejectionReason] = []
        stats = self._get_stats(tick.symbol_id)
        stats.total_received += 1

        # 1. Negative/zero price
        if tick.last_price <= 0:
            reasons.append(RejectionReason.NEGATIVE_PRICE)

        # 2. Negative/zero bid
        if tick.bid <= 0:
            reasons.append(RejectionReason.NEGATIVE_BID)

        # 3. Negative/zero ask
        if tick.ask <= 0:
            reasons.append(RejectionReason.NEGATIVE_ASK)

        # 4. Crossed book (bid >= ask)
        if tick.bid >= tick.ask:
            reasons.append(RejectionReason.CROSSED_BOOK)

        # 5. Time travel (capture before exchange)
        if tick.capture_time_ns < tick.exchange_time_ns:
            reasons.append(RejectionReason.TIME_TRAVEL)

        # 6. Stale data (capture > 10s after exchange)
        if (tick.capture_time_ns - tick.exchange_time_ns) > self._max_capture_lag_ns:
            reasons.append(RejectionReason.STALE_DATA)

        # 7. Fat finger — price outside 10% of rolling mean
        price_buf = self._price_buffer.get(tick.symbol_id, [])
        if len(price_buf) >= self._rolling_price_window:
            rolling_mean = self._rolling_mean(price_buf[-self._rolling_price_window:])
            if rolling_mean > 0:
                upper = rolling_mean * (1.0 + self._price_deviation_pct)
                lower = rolling_mean * (1.0 - self._price_deviation_pct)
                if tick.last_price > upper or tick.last_price < lower:
                    reasons.append(RejectionReason.FAT_FINGER_PRICE)

        # 8. Volume spike > 100x rolling average
        vol_buf = self._volume_buffer.get(tick.symbol_id, [])
        if len(vol_buf) >= self._rolling_volume_window:
            rolling_avg = self._rolling_mean(vol_buf[-self._rolling_volume_window:])
            if rolling_avg > 0 and tick.last_size > rolling_avg * self._volume_spike_factor:
                reasons.append(RejectionReason.FAT_FINGER_VOLUME)

        # 9. Non-monotonic capture timestamp
        last_ct = self._last_capture_time.get(tick.symbol_id)
        if last_ct is not None and tick.capture_time_ns < last_ct:
            reasons.append(RejectionReason.NON_MONOTONIC_TIME)

        accepted = len(reasons) == 0

        # Update rolling state only for accepted ticks
        if accepted:
            if tick.symbol_id not in self._price_buffer:
                self._price_buffer[tick.symbol_id] = []
            self._price_buffer[tick.symbol_id].append(tick.last_price)
            # Keep buffer bounded
            if len(self._price_buffer[tick.symbol_id]) > self._rolling_price_window * 2:
                self._price_buffer[tick.symbol_id] = self._price_buffer[tick.symbol_id][
                    -self._rolling_price_window:
                ]

            if tick.symbol_id not in self._volume_buffer:
                self._volume_buffer[tick.symbol_id] = []
            self._volume_buffer[tick.symbol_id].append(float(tick.last_size))
            if len(self._volume_buffer[tick.symbol_id]) > self._rolling_volume_window * 2:
                self._volume_buffer[tick.symbol_id] = self._volume_buffer[tick.symbol_id][
                    -self._rolling_volume_window:
                ]

            self._last_capture_time[tick.symbol_id] = tick.capture_time_ns
        else:
            stats.total_rejected += 1
            for r in reasons:
                stats.rejections_by_reason[r.value] = (
                    stats.rejections_by_reason.get(r.value, 0) + 1
                )
            if stats.exceeds_alert_threshold:
                logger.warning(
                    "Symbol %d rejection rate %.4f%% exceeds threshold (%.4f%%)",
                    tick.symbol_id,
                    stats.rejection_rate * 100,
                    self._alert_rejection_rate * 100,
                )

        return QualityResult(tick=tick, accepted=accepted, rejection_reasons=reasons)

    def check_batch(self, ticks: list[Tick]) -> tuple[list[Tick], list[tuple[Tick, list[str]]]]:
        """Check a batch of ticks. Returns (accepted_ticks, rejected_with_reasons)."""
        accepted = []
        rejected = []
        for tick in ticks:
            result = self.check_tick(tick)
            if result.accepted:
                accepted.append(tick)
            else:
                rejected.append(
                    (tick, [r.value for r in result.rejection_reasons])
                )
        return accepted, rejected

    def check_dataframe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Check a DataFrame of tick records.

        Returns (accepted_df, rejected_df).
        rejected_df has an extra 'rejection_reason' column (semicolon-separated).
        """
        accepted_mask = np.ones(len(df), dtype=bool)
        rejection_reasons = [""] * len(df)

        for i in range(len(df)):
            row = df.iloc[i]
            tick = Tick(
                exchange_time_ns=int(row["exchange_time_ns"]),
                capture_time_ns=int(row["capture_time_ns"]),
                symbol_id=int(row["symbol_id"]),
                bid=float(row["bid"]),
                ask=float(row["ask"]),
                bid_size=int(row["bid_size"]),
                ask_size=int(row["ask_size"]),
                last_price=float(row["last_price"]),
                last_size=int(row["last_size"]),
                trade_condition=int(row.get("trade_condition", 0)),
            )
            result = self.check_tick(tick)
            if not result.accepted:
                accepted_mask[i] = False
                rejection_reasons[i] = ";".join(r.value for r in result.rejection_reasons)

        accepted_df = df[accepted_mask].reset_index(drop=True)
        rejected_df = df[~accepted_mask].copy()
        rejected_df["rejection_reason"] = [r for r, m in zip(rejection_reasons, ~accepted_mask) if m]
        rejected_df = rejected_df.reset_index(drop=True)

        return accepted_df, rejected_df

    def get_stats(self, symbol_id: int | None = None) -> QualityStats | dict[int, QualityStats]:
        """Get quality statistics for a symbol or all symbols."""
        if symbol_id is not None:
            return self._get_stats(symbol_id)
        return dict(self._stats)

    def reset(self) -> None:
        """Reset all rolling state and statistics."""
        self._price_buffer.clear()
        self._volume_buffer.clear()
        self._last_capture_time.clear()
        self._stats.clear()
