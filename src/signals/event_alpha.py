from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.events import EventStore
from src.data.fundamentals import FundamentalsStore
from src.data.symbol_master import SymbolMaster


EVENT_TYPE_MULTIPLIER = {
    "earnings": 1.30,
    "guidance": 1.20,
    "estimate_revision": 1.10,
    "m&a": 1.00,
    "product": 0.90,
    "macro": 0.75,
    "news": 0.85,
}

EVENT_SOURCE_QUALITY = {
    "POLYGON_BENZINGA_NEWS": 1.00,
    "POLYGON_BENZINGA_GUIDANCE": 0.90,
    "POLYGON_BENZINGA_ANALYST": 0.90,
    "POLYGON_BENZINGA_EARNINGS": 0.85,
    "ALPHA_VANTAGE_NEWS": 0.85,
    "ALPHA_VANTAGE_EARNINGS": 0.80,
    "ALPHA_VANTAGE_EARNINGS_EST": 0.80,
    "ALPHA_VANTAGE_EARNINGS_CALENDAR": 0.55,
    # Yahoo is backup-only; keep quality low so it cannot dominate sleeve sizing.
    "YAHOO_EARNINGS_HISTORY": 0.20,
    "YAHOO_UPGRADE_DOWNGRADE": 0.25,
    "YAHOO_EARNINGS_TREND": 0.15,
    "csv_ingest": 0.80,
    "SEC_SUBMISSIONS": 0.45,
    "SEC_COMPANYFACTS": 0.25,
}


@dataclass(slots=True)
class EventAlphaConfig:
    event_target_weight: float = 0.14
    fundamental_half_life_days: float = 28.0
    sentiment_half_life_days: float = 5.0
    lookback_sentiment_days: int = 14
    min_sentiment_confidence: float = 0.10
    surprise_weight: float = 0.55
    revision_weight: float = 0.25
    sentiment_weight: float = 0.20
    gap_confirmation_weight: float = 0.10
    max_abs_strength: float = 1.5
    min_rank_count: int = 5
    coverage_floor: int = 8
    coverage_full: int = 60
    base_fundamental_quality: float = 0.35
    base_sentiment_quality: float = 0.35
    estimate_quality_bonus: float = 0.20
    revision_quality_bonus: float = 0.10


@dataclass(slots=True)
class EventAlphaBuildResult:
    composite: pd.DataFrame
    fundamental: pd.DataFrame
    sentiment: pd.DataFrame
    fundamental_coverage: int
    estimate_coverage: int
    revision_coverage: int
    sentiment_coverage: int
    total_events: int
    sentiment_quality_scale: float
    data_quality_scale: float
    suggested_weight: float


def _clip(value: float, limit: float = 1.0) -> float:
    return float(np.clip(value, -limit, limit))


def _to_aligned_timestamp(index: pd.Index, timestamp_ns: int) -> pd.Timestamp:
    ts = pd.to_datetime(int(timestamp_ns), unit="ns", utc=True)
    if getattr(index, "tz", None) is None:
        return ts.tz_localize(None)
    return ts.tz_convert(index.tz)


def _exp_decay(offset_days: int, half_life_days: float) -> float:
    if half_life_days <= 0:
        return 1.0 if offset_days == 0 else 0.0
    return float(np.exp(-math.log(2.0) * offset_days / half_life_days))


def _sparse_rank(df: pd.DataFrame, min_rank_count: int = 5) -> pd.DataFrame:
    ranked = pd.DataFrame(0.0, index=df.index, columns=df.columns)
    for dt, row in df.iterrows():
        active = row[row.abs() > 1e-12]
        if active.empty:
            continue
        if len(active) < min_rank_count:
            max_abs = float(active.abs().max())
            if max_abs > 1e-12:
                ranked.loc[dt, active.index] = active / max_abs
            continue
        pct = active.rank(pct=True, method="average")
        ranked.loc[dt, active.index] = (pct - 0.5) * 2.0
    return ranked.fillna(0.0)


def _estimate_revision_strength(
    df: pd.DataFrame,
    canonical_id: int,
    published_at_ns: int,
) -> float:
    cid = df[df["canonical_id"] == canonical_id]
    if cid.empty:
        return 0.0

    analyst_metric = cid[cid["metric_name"] == FundamentalsStore.METRIC_ANALYST_REVISION]
    if not analyst_metric.empty:
        hist = analyst_metric[analyst_metric["published_at_ns"] <= published_at_ns].sort_values("published_at_ns")
        if not hist.empty:
            return float(np.clip(hist.iloc[-1]["value"], -1.0, 1.0))

    ests = cid[cid["metric_name"] == FundamentalsStore.METRIC_ANALYST_EST_EPS]
    est_hist = ests[ests["published_at_ns"] <= published_at_ns].sort_values("published_at_ns")
    if len(est_hist) < 2:
        return 0.0
    prev_est = float(est_hist.iloc[-2]["value"])
    last_est = float(est_hist.iloc[-1]["value"])
    if abs(prev_est) <= 1e-12:
        return 0.0
    return float(np.clip((last_est - prev_est) / abs(prev_est), -1.0, 1.0))


def _prior_actual_growth(
    df: pd.DataFrame,
    metric_name: str,
    canonical_id: int,
    published_at_ns: int,
) -> float | None:
    hist = df[
        (df["canonical_id"] == canonical_id)
        & (df["metric_name"] == metric_name)
        & (df["published_at_ns"] < published_at_ns)
    ].sort_values("published_at_ns")
    if hist.empty:
        return None
    prev = float(hist.iloc[-1]["value"])
    if abs(prev) <= 1e-12:
        return None
    curr_hist = df[
        (df["canonical_id"] == canonical_id)
        & (df["metric_name"] == metric_name)
        & (df["published_at_ns"] == published_at_ns)
    ]
    if curr_hist.empty:
        return None
    curr = float(curr_hist.iloc[-1]["value"])
    return float(np.clip((curr - prev) / abs(prev), -2.0, 2.0))


def _build_fundamental_panel(
    prices: pd.DataFrame,
    equity_syms: list[str],
    fundamentals_path: str,
    symbol_master_path: str,
    config: EventAlphaConfig,
) -> tuple[pd.DataFrame, int, int, int]:
    panel = np.zeros((len(prices.index), len(equity_syms)), dtype=np.float32)
    if not (os.path.exists(fundamentals_path) and os.path.exists(symbol_master_path)):
        empty = pd.DataFrame(panel, index=prices.index, columns=equity_syms)
        return empty, 0, 0, 0

    store = FundamentalsStore(fundamentals_path)
    sm = SymbolMaster(symbol_master_path)
    covered: set[str] = set()
    estimate_covered: set[str] = set()
    revision_covered: set[str] = set()
    col_idx = {ticker: i for i, ticker in enumerate(equity_syms)}
    horizon = int(max(config.fundamental_half_life_days * 4, 20))
    decay = np.array(
        [_exp_decay(offset, config.fundamental_half_life_days) for offset in range(horizon)],
        dtype=np.float32,
    )

    try:
        df = store.to_dataframe()
        if df.empty:
            empty = pd.DataFrame(panel, index=prices.index, columns=equity_syms)
            return empty, 0, 0, 0

        metrics = {
            FundamentalsStore.METRIC_EPS,
            FundamentalsStore.METRIC_ANALYST_EST_EPS,
            FundamentalsStore.METRIC_REVENUE,
            FundamentalsStore.METRIC_ANALYST_EST_REVENUE,
            FundamentalsStore.METRIC_ANALYST_REVISION,
        }
        df = df[df["metric_name"].isin(metrics)].copy()
        if df.empty:
            empty = pd.DataFrame(panel, index=prices.index, columns=equity_syms)
            return empty, 0, 0, 0

        for canonical_id, grp in df.groupby("canonical_id"):
            actuals = grp[grp["metric_name"] == FundamentalsStore.METRIC_EPS].sort_values("published_at_ns")
            if actuals.empty:
                continue

            revenue_actuals = grp[grp["metric_name"] == FundamentalsStore.METRIC_REVENUE].sort_values("published_at_ns")
            revenue_growth_by_pub: dict[int, float] = {}
            prev_rev = None
            for rev in revenue_actuals.itertuples():
                rev_val = float(rev.value)
                if prev_rev is not None and abs(prev_rev) > 1e-12:
                    revenue_growth_by_pub[int(rev.published_at_ns)] = float(
                        np.clip((rev_val - prev_rev) / abs(prev_rev), -2.0, 2.0)
                    )
                prev_rev = rev_val

            analyst_revision_hist = grp[
                grp["metric_name"] == FundamentalsStore.METRIC_ANALYST_REVISION
            ][["published_at_ns", "value"]].sort_values("published_at_ns")
            analyst_est_hist = grp[
                grp["metric_name"] == FundamentalsStore.METRIC_ANALYST_EST_EPS
            ][["published_at_ns", "value"]].sort_values("published_at_ns")
            analyst_rev_est_hist = grp[
                grp["metric_name"] == FundamentalsStore.METRIC_ANALYST_EST_REVENUE
            ][["published_at_ns", "value"]].sort_values("published_at_ns")
            analyst_revision_ns = analyst_revision_hist["published_at_ns"].to_numpy(dtype=np.int64)
            analyst_revision_vals = analyst_revision_hist["value"].to_numpy(dtype=np.float64)
            analyst_est_ns = analyst_est_hist["published_at_ns"].to_numpy(dtype=np.int64)
            analyst_est_vals = analyst_est_hist["value"].to_numpy(dtype=np.float64)
            has_estimate_support = (
                not analyst_est_hist.empty
                or not analyst_rev_est_hist.empty
                or not analyst_revision_hist.empty
            )
            has_revision_support = (
                not analyst_revision_hist.empty
                or len(analyst_est_hist) >= 2
                or len(analyst_rev_est_hist) >= 2
            )
            prev_eps = None

            for actual in actuals.itertuples():
                published_at_ns = int(actual.published_at_ns)
                surprise = store.get_earnings_surprise(canonical_id, published_at_ns)
                if surprise is None or not np.isfinite(surprise):
                    actual_value = float(actual.value)
                    if prev_eps is not None and abs(prev_eps) > 1e-12:
                        surprise = float(np.clip((actual_value - prev_eps) / abs(prev_eps), -2.0, 2.0))
                    else:
                        surprise = None
                if surprise is None or not np.isfinite(surprise):
                    prev_eps = float(actual.value)
                    continue

                revenue_surprise = store.get_surprise(
                    canonical_id,
                    published_at_ns,
                    FundamentalsStore.METRIC_REVENUE,
                    FundamentalsStore.METRIC_ANALYST_EST_REVENUE,
                )
                if revenue_surprise is None or not np.isfinite(revenue_surprise):
                    revenue_surprise = revenue_growth_by_pub.get(published_at_ns)
                revenue_component = 0.0
                if revenue_surprise is not None and np.isfinite(revenue_surprise):
                    revenue_component = float(np.clip(1.5 * revenue_surprise, -1.0, 1.0))

                revision = 0.0
                rev_idx = np.searchsorted(analyst_revision_ns, published_at_ns, side="right") - 1
                if rev_idx >= 0 and len(analyst_revision_vals):
                    revision = float(np.clip(analyst_revision_vals[rev_idx], -1.0, 1.0))
                else:
                    est_idx = np.searchsorted(analyst_est_ns, published_at_ns, side="right")
                    if est_idx >= 2:
                        prev_est = float(analyst_est_vals[est_idx - 2])
                        last_est = float(analyst_est_vals[est_idx - 1])
                        if abs(prev_est) > 1e-12:
                            revision = float(np.clip((last_est - prev_est) / abs(prev_est), -1.0, 1.0))

                ticker = sm.get_ticker_at(canonical_id, published_at_ns)
                if ticker not in col_idx:
                    prev_eps = float(actual.value)
                    continue

                covered.add(ticker)
                if has_estimate_support:
                    estimate_covered.add(ticker)
                if has_revision_support:
                    revision_covered.add(ticker)
                strength = (
                    config.surprise_weight * float(np.clip(2.5 * surprise, -1.0, 1.0))
                    + config.revision_weight * revision
                    + config.gap_confirmation_weight * revenue_component
                )
                strength = float(np.clip(strength, -config.max_abs_strength, config.max_abs_strength))
                if abs(strength) < 1e-12:
                    prev_eps = float(actual.value)
                    continue

                start_dt = _to_aligned_timestamp(prices.index, published_at_ns)
                start_idx = prices.index.searchsorted(start_dt, side="left")
                if start_idx >= len(prices.index):
                    prev_eps = float(actual.value)
                    continue

                span = min(horizon, len(prices.index) - start_idx)
                panel[start_idx:start_idx + span, col_idx[ticker]] += strength * decay[:span]
                prev_eps = float(actual.value)
    finally:
        sm.close()
        store.close()

    return (
        pd.DataFrame(panel, index=prices.index, columns=equity_syms),
        len(covered),
        len(estimate_covered),
        len(revision_covered),
    )


def _build_sentiment_panel(
    prices: pd.DataFrame,
    equity_syms: list[str],
    event_store_path: str | None,
    symbol_master_path: str | None,
    config: EventAlphaConfig,
) -> tuple[pd.DataFrame, int, int, float]:
    panel = np.zeros((len(prices.index), len(equity_syms)), dtype=np.float32)
    if not event_store_path or not os.path.exists(event_store_path):
        return pd.DataFrame(panel, index=prices.index, columns=equity_syms), 0, 0, 0.0

    store = EventStore(event_store_path)
    sm = SymbolMaster(symbol_master_path) if symbol_master_path and os.path.exists(symbol_master_path) else None
    covered: set[str] = set()
    source_quality_by_ticker: dict[str, float] = {}
    total_events = 0
    col_idx = {ticker: i for i, ticker in enumerate(equity_syms)}
    horizon = int(max(config.sentiment_half_life_days * 5, 10))
    decay = np.array(
        [_exp_decay(offset, config.sentiment_half_life_days) for offset in range(horizon)],
        dtype=np.float32,
    )

    try:
        df = store.to_dataframe()
        if df.empty:
            return pd.DataFrame(panel, index=prices.index, columns=equity_syms), 0, 0, 0.0

        for event in df.itertuples():
            ticker = event.ticker
            if ticker not in col_idx and sm is not None:
                ticker = sm.get_ticker_at(int(event.canonical_id), int(event.published_at_ns))
            if ticker not in col_idx:
                continue

            confidence = float(np.clip(getattr(event, "confidence", 0.0), 0.0, 1.0))
            if confidence < config.min_sentiment_confidence:
                continue

            event_type = str(getattr(event, "event_type", "news")).lower()
            strength = (
                float(getattr(event, "sentiment_score", 0.0))
                * float(np.clip(getattr(event, "relevance", 1.0), 0.0, 2.0))
                * float(np.clip(getattr(event, "novelty", 1.0), 0.0, 2.0))
                * confidence
                * EVENT_TYPE_MULTIPLIER.get(event_type, 0.85)
            )
            strength = float(np.clip(strength, -config.max_abs_strength, config.max_abs_strength))
            if abs(strength) < 1e-12:
                continue

            covered.add(ticker)
            source_name = str(getattr(event, "source", "") or "")
            source_quality = EVENT_SOURCE_QUALITY.get(source_name, 0.60)
            source_quality_by_ticker[ticker] = max(source_quality_by_ticker.get(ticker, 0.0), source_quality)
            total_events += 1
            start_dt = _to_aligned_timestamp(prices.index, int(event.published_at_ns))
            start_idx = prices.index.searchsorted(start_dt, side="left")
            if start_idx >= len(prices.index):
                continue

            span = min(horizon, len(prices.index) - start_idx)
            panel[start_idx:start_idx + span, col_idx[ticker]] += strength * decay[:span]
    finally:
        if sm is not None:
            sm.close()
        store.close()

    avg_source_quality = float(np.mean(list(source_quality_by_ticker.values()))) if source_quality_by_ticker else 0.0
    return pd.DataFrame(panel, index=prices.index, columns=equity_syms), len(covered), total_events, avg_source_quality


def build_event_alpha_signal(
    prices: pd.DataFrame,
    equity_syms: list[str],
    fundamentals_path: str = "data/fundamentals.db",
    symbol_master_path: str = "data/symbol_master.db",
    event_store_path: str | None = "data/events.db",
    config: EventAlphaConfig | None = None,
) -> EventAlphaBuildResult:
    config = config or EventAlphaConfig()

    fundamental_raw, fundamental_coverage, estimate_coverage, revision_coverage = _build_fundamental_panel(
        prices, equity_syms, fundamentals_path, symbol_master_path, config
    )
    sentiment_raw, sentiment_coverage, total_events, sentiment_quality_scale = _build_sentiment_panel(
        prices, equity_syms, event_store_path, symbol_master_path, config
    )

    fundamental_ranked = _sparse_rank(fundamental_raw, min_rank_count=config.min_rank_count)
    sentiment_ranked = _sparse_rank(sentiment_raw, min_rank_count=config.min_rank_count)
    composite = (
        (1.0 - config.sentiment_weight) * fundamental_ranked
        + config.sentiment_weight * sentiment_ranked
    ).clip(-1.0, 1.0)

    coverage = max(fundamental_coverage, sentiment_coverage)
    coverage_scale = np.clip(
        (coverage - config.coverage_floor)
        / max(config.coverage_full - config.coverage_floor, 1),
        0.0,
        1.0,
    )
    quality_scale = 0.0
    if fundamental_coverage > 0:
        quality_scale += config.base_fundamental_quality
        quality_scale += config.estimate_quality_bonus * np.clip(
            estimate_coverage / max(fundamental_coverage, 1),
            0.0,
            1.0,
        )
        quality_scale += config.revision_quality_bonus * np.clip(
            revision_coverage / max(fundamental_coverage, 1),
            0.0,
            1.0,
        )
    if sentiment_coverage > 0:
        quality_scale += config.base_sentiment_quality * float(np.clip(sentiment_quality_scale, 0.0, 1.0))
    quality_scale = float(np.clip(quality_scale, 0.0, 1.0))
    support_scale = 1.0
    if fundamental_coverage > 0:
        estimate_ratio = estimate_coverage / max(fundamental_coverage, 1)
        revision_ratio = revision_coverage / max(fundamental_coverage, 1)
        support_scale = 0.40 + 0.60 * float(np.clip(max(estimate_ratio, revision_ratio) / 0.20, 0.0, 1.0))
    suggested_weight = float(config.event_target_weight * coverage_scale * quality_scale * support_scale)

    return EventAlphaBuildResult(
        composite=composite.fillna(0.0),
        fundamental=fundamental_ranked.fillna(0.0),
        sentiment=sentiment_ranked.fillna(0.0),
        fundamental_coverage=fundamental_coverage,
        estimate_coverage=estimate_coverage,
        revision_coverage=revision_coverage,
        sentiment_coverage=sentiment_coverage,
        total_events=total_events,
        sentiment_quality_scale=sentiment_quality_scale,
        data_quality_scale=quality_scale,
        suggested_weight=suggested_weight,
    )


class EventDrivenAlphaSleeve:
    def __init__(
        self,
        symbol_map: dict[int, str] | None = None,
        fundamentals_path: str | None = None,
        symbol_master_path: str | None = None,
        event_store_path: str | None = None,
        config: EventAlphaConfig | None = None,
    ):
        self.symbol_map = dict(symbol_map or {})
        self.config = config or EventAlphaConfig()
        self._fund = (
            FundamentalsStore(fundamentals_path)
            if fundamentals_path and os.path.exists(fundamentals_path)
            else None
        )
        self._sm = (
            SymbolMaster(symbol_master_path)
            if symbol_master_path and os.path.exists(symbol_master_path)
            else None
        )
        self._events = (
            EventStore(event_store_path)
            if event_store_path and os.path.exists(event_store_path)
            else None
        )
        self._fund_cache: dict[tuple[int, int], float] = {}
        self._event_cache: dict[tuple[int, int], float] = {}

    def close(self) -> None:
        if self._fund is not None:
            self._fund.close()
        if self._sm is not None:
            self._sm.close()
        if self._events is not None:
            self._events.close()

    def _canonical_id_for(self, symbol_id: int, as_of_ns: int) -> int | None:
        if self._sm is None:
            return symbol_id
        ticker = self.symbol_map.get(symbol_id)
        if ticker is None:
            return symbol_id
        return self._sm.resolve_ticker(ticker, as_of_ns)

    def _fundamental_score(self, canonical_id: int, as_of_ns: int) -> float:
        if self._fund is None:
            return 0.0
        day_bucket = int(as_of_ns // 86_400_000_000_000)
        key = (canonical_id, day_bucket)
        if key in self._fund_cache:
            return self._fund_cache[key]

        surprise = self._fund.get_earnings_surprise(canonical_id, as_of_ns)
        if surprise is None or not np.isfinite(surprise):
            self._fund_cache[key] = 0.0
            return 0.0

        est_hist = self._fund.get_history(
            canonical_id,
            FundamentalsStore.METRIC_ANALYST_EST_EPS,
            end_ns=as_of_ns,
        )
        revision = 0.0
        if len(est_hist) >= 2:
            prev_est = float(est_hist[-2].value)
            last_est = float(est_hist[-1].value)
            if abs(prev_est) > 1e-12:
                revision = (last_est - prev_est) / abs(prev_est)

        score = (
            self.config.surprise_weight * float(np.clip(2.5 * surprise, -1.0, 1.0))
            + self.config.revision_weight * float(np.clip(revision, -1.0, 1.0))
        )
        self._fund_cache[key] = _clip(score)
        return self._fund_cache[key]

    def _sentiment_score(self, canonical_id: int, ticker: str | None, as_of_ns: int) -> float:
        if self._events is None:
            return 0.0

        day_bucket = int(as_of_ns // 86_400_000_000_000)
        key = (canonical_id, day_bucket)
        if key in self._event_cache:
            return self._event_cache[key]

        lookback_ns = int(self.config.lookback_sentiment_days * 86_400_000_000_000)
        records = self._events.get_recent(canonical_id, as_of_ns, lookback_ns)
        if not records and ticker:
            records = self._events.get_recent_by_ticker(ticker, as_of_ns, lookback_ns)

        if not records:
            self._event_cache[key] = 0.0
            return 0.0

        score = 0.0
        for record in records:
            if record.confidence < self.config.min_sentiment_confidence:
                continue
            age_days = max(0.0, (as_of_ns - record.published_at_ns) / 86_400_000_000_000)
            strength = (
                record.sentiment_score
                * max(record.relevance, 0.0)
                * max(record.novelty, 0.0)
                * max(record.confidence, 0.0)
                * EVENT_TYPE_MULTIPLIER.get(record.event_type.lower(), 0.85)
                * _exp_decay(int(age_days), self.config.sentiment_half_life_days)
            )
            score += strength

        self._event_cache[key] = _clip(score)
        return self._event_cache[key]

    def score_paper(self, symbol_id: int, price: float, engine) -> float:
        tick = engine.latest_ticks.get(symbol_id)
        as_of_ns = int(tick.timestamp_ns) if tick is not None else time.time_ns()
        ticker = self.symbol_map.get(symbol_id)
        canonical_id = self._canonical_id_for(symbol_id, as_of_ns)
        if canonical_id is None:
            return 0.0

        event_score = self._fundamental_score(canonical_id, as_of_ns)
        sentiment_score = self._sentiment_score(canonical_id, ticker, as_of_ns)

        gap_confirmation = 0.0
        rets = engine.returns.get(symbol_id, [])
        if rets:
            recent = float(np.sum(rets[-5:]))
            if abs(recent) > 1e-12:
                gap_confirmation = np.sign(event_score + sentiment_score) * min(abs(recent) * 6.0, 1.0)

        score = (
            (1.0 - self.config.sentiment_weight - self.config.gap_confirmation_weight) * event_score
            + self.config.sentiment_weight * sentiment_score
            + self.config.gap_confirmation_weight * gap_confirmation
        )
        return _clip(score)
