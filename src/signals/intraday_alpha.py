from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np

from src.backtester.types import OrderIntent, OrderType, Side
from src.regime.params import RegimeParams, get_regime_params
from src.signals.combiner import LEAD_LAG_PAIRS, lead_lag_impulse
from src.signals.tier1 import (
    signal_mean_reversion,
    signal_momentum,
    signal_ofi,
    signal_volume_anomaly,
)


def _clip(value: float, limit: float = 1.0) -> float:
    return float(np.clip(value, -limit, limit))


def _safe_std(values: list[float] | np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values))


@dataclass
class IntradayAlphaConfig:
    short_window: int = 20
    medium_window: int = 60
    long_window: int = 180
    pair_window: int = 80
    min_history: int = 15
    base_order_size: int = 10
    entry_threshold: float = 0.35
    exit_threshold: float = 0.10
    crisis_vpin_threshold: float = 0.90
    high_vol_multiplier: float = 1.75
    range_trend_multiplier: float = 0.30
    mean_reversion_scale: float = 2.0
    momentum_scale: float = 2.5
    pair_scale: float = 2.0
    lead_lag_weight: float = 0.40
    pair_weight: float = 0.30
    volume_weight: float = 0.30


class IntradayAlphaSleeve:
    """
    Orthogonal intraday sleeve intended to complement the slower v3/v4 book.

    It combines:
    - short-horizon mean reversion
    - cross-asset lead/lag impulses
    - simple cross-asset spread reversion
    - regime-aware weighting borrowed from the existing regime parameter stack

    The paper path uses explicit price/volume state. The backtester path uses
    MarketStateVector features so live and research share the same signal shape.
    """

    def __init__(
        self,
        symbol_map: Optional[dict[int, str]] = None,
        config: Optional[IntradayAlphaConfig] = None,
    ):
        self.config = config or IntradayAlphaConfig()
        self.symbol_map = dict(symbol_map or {})
        self._ticker_to_symbol = {ticker: sid for sid, ticker in self.symbol_map.items()}
        maxlen = max(
            self.config.short_window,
            self.config.medium_window,
            self.config.long_window,
            self.config.pair_window,
        ) + 5
        self._price_history: dict[int, Deque[tuple[int, float]]] = defaultdict(
            lambda: deque(maxlen=maxlen)
        )
        self._return_history: dict[int, Deque[float]] = defaultdict(
            lambda: deque(maxlen=maxlen)
        )
        self._volume_history: dict[int, Deque[float]] = defaultdict(
            lambda: deque(maxlen=maxlen)
        )
        self._spread_history: dict[tuple[int, int], Deque[float]] = defaultdict(
            lambda: deque(maxlen=maxlen)
        )
        self._latest_timestamp: dict[int, int] = {}
        self._latest_price: dict[int, float] = {}
        self._latest_msv_by_ticker: dict[str, object] = {}
        self.last_components: dict[int, dict[str, float | str]] = {}
        self.max_abs_lead_lag: dict[int, float] = defaultdict(float)

    def ticker_for(self, symbol_id: int) -> str:
        return self.symbol_map.get(symbol_id, str(symbol_id))

    def symbol_for_ticker(self, ticker: str) -> Optional[int]:
        return self._ticker_to_symbol.get(ticker)

    def _observe_tick(
        self,
        symbol_id: int,
        price: float,
        timestamp_ns: int,
        volume: float = 0.0,
    ) -> None:
        history = self._price_history[symbol_id]
        prev_price = history[-1][1] if history else None
        if history and history[-1][0] == timestamp_ns:
            history[-1] = (timestamp_ns, price)
        else:
            history.append((timestamp_ns, price))
            if prev_price is not None and prev_price > 0 and price > 0:
                self._return_history[symbol_id].append(float(np.log(price / prev_price)))

        if volume > 0:
            self._volume_history[symbol_id].append(float(volume))

        self._latest_price[symbol_id] = price
        self._latest_timestamp[symbol_id] = timestamp_ns

        for leader_sid, lagger_sid, _, _ in self._resolved_pairs_for_lagger(symbol_id):
            leader_price = self._latest_price.get(leader_sid)
            if leader_price is None or leader_price <= 0 or price <= 0:
                continue
            spread = float(np.log(price) - np.log(leader_price))
            self._spread_history[(leader_sid, lagger_sid)].append(spread)

    def _resolved_pairs_for_lagger(self, lagger_sid: int) -> list[tuple[int, int, float, float]]:
        lagger_ticker = self.ticker_for(lagger_sid)
        matches: list[tuple[int, int, float, float]] = []
        for leader_ticker, pair_lagger, tau_ms, threshold in LEAD_LAG_PAIRS:
            if pair_lagger != lagger_ticker:
                continue
            leader_sid = self.symbol_for_ticker(leader_ticker)
            if leader_sid is None:
                continue
            matches.append((leader_sid, lagger_sid, float(tau_ms), float(threshold)))
        return matches

    def _infer_paper_regime(self, symbol_id: int, current_vpin: float = 0.0) -> str:
        if current_vpin >= self.config.crisis_vpin_threshold:
            return "LIQUIDITY_CRISIS"

        rets = list(self._return_history[symbol_id])
        if len(rets) < self.config.min_history:
            return "LOW_VOL_TRENDING"

        short = np.array(rets[-self.config.short_window :], dtype=float)
        medium = np.array(rets[-self.config.medium_window :], dtype=float)
        long = np.array(rets[-self.config.long_window :], dtype=float)
        short_vol = max(float(np.std(short)), 1e-8)
        long_vol = max(float(np.std(long)), 1e-8)
        trend = abs(float(np.mean(medium)))

        if short_vol > self.config.high_vol_multiplier * long_vol:
            if trend > 0.5 * short_vol:
                return "HIGH_VOL_TRENDING"
            return "HIGH_VOL_CHAOTIC"

        if trend < self.config.range_trend_multiplier * long_vol:
            return "MEAN_REVERTING_RANGE"

        return "LOW_VOL_TRENDING"

    def _infer_msv_regime(self, msv) -> str:
        vpin = float(getattr(msv, "vpin", 0.0) or 0.0)
        if vpin >= self.config.crisis_vpin_threshold:
            return "LIQUIDITY_CRISIS"

        short_vol = max(float(getattr(msv, "vol_60s", 0.0) or 0.0), 1e-8)
        long_vol = max(float(getattr(msv, "vol_1d", 0.0) or 0.0), 1e-8)
        trend = abs(
            0.65 * float(getattr(msv, "ret_300s", 0.0) or 0.0)
            + 0.35 * float(getattr(msv, "ret_1800s", 0.0) or 0.0)
        )
        z20 = abs(float(getattr(msv, "zscore_20", 0.0) or 0.0))

        if short_vol > self.config.high_vol_multiplier * long_vol:
            if trend > 0.5 * short_vol:
                return "HIGH_VOL_TRENDING"
            return "HIGH_VOL_CHAOTIC"

        if trend < self.config.range_trend_multiplier * long_vol and z20 > 1.0:
            return "MEAN_REVERTING_RANGE"

        return "LOW_VOL_TRENDING"

    def _paper_mean_reversion(self, symbol_id: int) -> float:
        rets = list(self._return_history[symbol_id])
        if len(rets) < self.config.min_history:
            return 0.0
        window = np.array(rets[-self.config.short_window :], dtype=float)
        sigma = max(float(np.std(window)), 1e-8)
        zscore = (window[-1] - float(np.mean(window))) / sigma
        return _clip(-zscore / self.config.mean_reversion_scale)

    def _paper_momentum(self, symbol_id: int) -> float:
        rets = list(self._return_history[symbol_id])
        if len(rets) < self.config.min_history:
            return 0.0
        window = np.array(rets[-self.config.medium_window :], dtype=float)
        sigma = max(float(np.std(window)), 1e-8)
        trend = float(np.sum(window)) / (sigma * np.sqrt(len(window)))
        return _clip(trend / self.config.momentum_scale)

    def _paper_volume_anomaly(self, symbol_id: int) -> float:
        vols = list(self._volume_history[symbol_id])
        rets = list(self._return_history[symbol_id])
        if len(vols) < self.config.min_history or not rets:
            return 0.0
        lookback = vols[-self.config.short_window :]
        avg_volume = max(float(np.mean(lookback)), 1e-8)
        volume_ratio = vols[-1] / avg_volume
        direction = np.sign(rets[-1])
        magnitude = (volume_ratio - 1.0) / 2.0
        return _clip(direction * magnitude)

    def _paper_lead_lag(self, symbol_id: int) -> float:
        score = 0.0
        for leader_sid, _, tau_ms, threshold in self._resolved_pairs_for_lagger(symbol_id):
            leader_rets = list(self._return_history[leader_sid])
            if len(leader_rets) < self.config.min_history:
                continue
            leader_ts = self._latest_timestamp.get(leader_sid)
            lagger_ts = self._latest_timestamp.get(symbol_id)
            if leader_ts is None or lagger_ts is None:
                continue
            window = np.array(leader_rets[-self.config.short_window :], dtype=float)
            sigma = max(float(np.std(window)), 1e-8)
            leader_z = (window[-1] - float(np.mean(window))) / sigma
            elapsed_ms = max(0.0, (lagger_ts - leader_ts) / 1_000_000.0)
            score += lead_lag_impulse(leader_z, elapsed_ms, tau_ms, threshold=threshold)
        return _clip(score)

    def _paper_pair_reversion(self, symbol_id: int) -> float:
        score = 0.0
        for leader_sid, lagger_sid, _, _ in self._resolved_pairs_for_lagger(symbol_id):
            spreads = list(self._spread_history[(leader_sid, lagger_sid)])
            if len(spreads) < self.config.min_history:
                continue
            window = np.array(spreads[-self.config.pair_window :], dtype=float)
            sigma = max(float(np.std(window)), 1e-8)
            zscore = (window[-1] - float(np.mean(window))) / sigma
            score += _clip(-zscore / self.config.pair_scale)
        return _clip(score)

    def _combine(
        self,
        regime_label: str,
        params: RegimeParams,
        mean_reversion_score: float,
        momentum_score: float,
        cross_asset_score: float,
        volume_score: float,
    ) -> float:
        if params.execution_mode == "REDUCE_ONLY" or params.position_size_scalar <= 0:
            return 0.0

        mr_total = 0.7 * mean_reversion_score + 0.3 * cross_asset_score
        micro_total = (
            (1.0 - self.config.volume_weight) * cross_asset_score
            + self.config.volume_weight * volume_score
        )
        raw_score = (
            params.mean_reversion_weight * mr_total
            + params.momentum_weight * momentum_score
            + params.ofi_weight * micro_total
        )

        active = [
            x for x in (mr_total, momentum_score, micro_total)
            if abs(x) > 0.05
        ]
        if regime_label == "HIGH_VOL_CHAOTIC" and active:
            signs = {int(np.sign(x)) for x in active if abs(x) > 0.05}
            if len(signs) > 1:
                raw_score *= 0.5

        return _clip(raw_score * params.position_size_scalar)

    def score_paper(self, symbol_id: int, price: float, engine) -> float:
        tick = None
        if hasattr(engine, "latest_ticks"):
            tick = engine.latest_ticks.get(symbol_id)

        timestamp_ns = getattr(tick, "timestamp_ns", getattr(engine, "_tick_count", 0))
        volume = float(getattr(tick, "volume", 0.0) or 0.0)
        self._observe_tick(symbol_id, price, timestamp_ns, volume)

        regime_label = self._infer_paper_regime(
            symbol_id,
            current_vpin=float(getattr(engine.portfolio, "current_vpin", 0.0) or 0.0),
        )
        params = get_regime_params(regime_label)
        mr = self._paper_mean_reversion(symbol_id)
        mom = self._paper_momentum(symbol_id)
        leadlag = self._paper_lead_lag(symbol_id)
        pair_reversion = self._paper_pair_reversion(symbol_id)
        volume_anomaly = self._paper_volume_anomaly(symbol_id)
        cross_asset = _clip(
            self.config.lead_lag_weight * leadlag
            + self.config.pair_weight * pair_reversion
        )
        score = self._combine(
            regime_label,
            params,
            mean_reversion_score=mr,
            momentum_score=mom,
            cross_asset_score=cross_asset,
            volume_score=volume_anomaly,
        )
        self.last_components[symbol_id] = {
            "regime": regime_label,
            "mean_reversion": mr,
            "momentum": mom,
            "lead_lag": leadlag,
            "pair_reversion": pair_reversion,
            "volume": volume_anomaly,
            "score": score,
        }
        self.max_abs_lead_lag[symbol_id] = max(
            self.max_abs_lead_lag[symbol_id], abs(leadlag)
        )
        return score

    def score_msv(self, msv) -> float:
        symbol_id = int(getattr(msv, "symbol_id"))
        regime_label = self._infer_msv_regime(msv)
        params = get_regime_params(regime_label)
        mr = float(signal_mean_reversion(msv, threshold=1.5))
        mom = float(signal_momentum(msv, threshold=0.35))
        volume_anomaly = float(signal_volume_anomaly(msv, vol_threshold=1.8))
        ofi = float(signal_ofi(msv))

        ticker = self.ticker_for(symbol_id)
        leadlag = 0.0
        for leader_ticker, lagger_ticker, tau_ms, threshold in LEAD_LAG_PAIRS:
            if lagger_ticker != ticker:
                continue
            leader_msv = self._latest_msv_by_ticker.get(leader_ticker)
            if leader_msv is None:
                continue
            leader_z = float(getattr(leader_msv, "zscore_20", 0.0) or 0.0)
            leader_ts = int(getattr(leader_msv, "timestamp_ns", 0) or 0)
            lagger_ts = int(getattr(msv, "timestamp_ns", 0) or 0)
            elapsed_ms = max(0.0, (lagger_ts - leader_ts) / 1_000_000.0)
            leadlag += lead_lag_impulse(leader_z, elapsed_ms, tau_ms, threshold=threshold)
        leadlag = _clip(leadlag)

        cross_asset = _clip(0.65 * leadlag + 0.35 * ofi)
        score = self._combine(
            regime_label,
            params,
            mean_reversion_score=mr,
            momentum_score=mom,
            cross_asset_score=cross_asset,
            volume_score=volume_anomaly,
        )
        self._latest_msv_by_ticker[ticker] = msv
        self.last_components[symbol_id] = {
            "regime": regime_label,
            "mean_reversion": mr,
            "momentum": mom,
            "lead_lag": leadlag,
            "ofi": ofi,
            "volume": volume_anomaly,
            "score": score,
        }
        self.max_abs_lead_lag[symbol_id] = max(
            self.max_abs_lead_lag[symbol_id], abs(leadlag)
        )
        return score

    def build_backtester_signal(
        self,
        order_size: Optional[int] = None,
        entry_threshold: Optional[float] = None,
        exit_threshold: Optional[float] = None,
    ):
        size = int(order_size or self.config.base_order_size)
        entry = float(entry_threshold or self.config.entry_threshold)
        exit_ = float(exit_threshold or self.config.exit_threshold)

        def _signal_fn(msv, positions):
            score = self.score_msv(msv)
            pos = positions.get(msv.symbol_id)
            qty = int(pos.quantity) if pos is not None else 0

            if abs(score) < exit_ and qty != 0:
                return OrderIntent(
                    symbol_id=msv.symbol_id,
                    side=Side.SELL if qty > 0 else Side.BUY,
                    order_type=OrderType.MARKET,
                    size=abs(qty),
                    signal_time_ns=int(getattr(msv, "timestamp_ns", 0) or 0),
                    signal_strength=float(abs(score)),
                )

            if score >= entry and qty <= 0:
                return OrderIntent(
                    symbol_id=msv.symbol_id,
                    side=Side.BUY,
                    order_type=OrderType.MARKET,
                    size=size + max(0, -qty),
                    signal_time_ns=int(getattr(msv, "timestamp_ns", 0) or 0),
                    signal_strength=float(score),
                )

            if score <= -entry and qty >= 0:
                return OrderIntent(
                    symbol_id=msv.symbol_id,
                    side=Side.SELL,
                    order_type=OrderType.MARKET,
                    size=size + max(0, qty),
                    signal_time_ns=int(getattr(msv, "timestamp_ns", 0) or 0),
                    signal_strength=float(abs(score)),
                )

            return None

        return _signal_fn

    def __call__(self, symbol_id: int, price: float, engine) -> float:
        return self.score_paper(symbol_id, price, engine)
