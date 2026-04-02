from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import requests

from src.data.events import EventRecord, EventStore
from src.data.fundamentals import FundamentalRecord, FundamentalsStore
from src.data.ingest.raw_cache import RawDataCache, to_jsonable
from src.data.symbol_master import SymbolMaster

logger = logging.getLogger(__name__)

UTC = timezone.utc


def chunked(values: Sequence[str], size: int) -> Iterable[list[str]]:
    size = max(int(size), 1)
    for i in range(0, len(values), size):
        yield list(values[i:i + size])


def safe_float(value) -> float | None:
    if isinstance(value, dict):
        value = value.get("raw", value.get("fmt"))
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def safe_int(value) -> int | None:
    if isinstance(value, dict):
        value = value.get("raw", value.get("fmt"))
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def clip_score(value: float, scale: float = 1.0) -> float:
    return float(np.clip(value / max(scale, 1e-12), -1.0, 1.0))


def score_upgrade_downgrade(action: str | None, from_grade: str | None, to_grade: str | None) -> float:
    action_l = str(action or "").lower()
    from_l = str(from_grade or "").lower()
    to_l = str(to_grade or "").lower()
    score = 0.0

    if any(term in action_l for term in ("upgrade", "raise", "raised", "reiterate", "initiated")):
        score += 0.70
    if any(term in action_l for term in ("downgrade", "lower", "lowered", "cut", "suspended")):
        score -= 0.70

    if any(term in to_l for term in ("strong buy", "buy", "overweight", "outperform")):
        score += 0.25
    if any(term in to_l for term in ("strong sell", "sell", "underweight", "underperform")):
        score -= 0.25

    if any(term in from_l for term in ("strong buy", "buy", "overweight", "outperform")) and score < 0:
        score -= 0.10
    if any(term in from_l for term in ("strong sell", "sell", "underweight", "underperform")) and score > 0:
        score += 0.10

    if "hold" in to_l or "neutral" in to_l:
        score *= 0.75

    return float(np.clip(score, -1.0, 1.0))


def ns_from_epoch_seconds(value) -> int | None:
    sec = safe_int(value)
    if sec is None or sec <= 0:
        return None
    return int(sec * 1_000_000_000)


@dataclass(slots=True)
class YahooBackfillStats:
    events_inserted: int = 0
    event_duplicates: int = 0
    fundamentals_inserted: int = 0
    fundamental_duplicates: int = 0
    raw_cache_hits: int = 0
    raw_cache_writes: int = 0
    intraday_files_written: int = 0
    intraday_failed: int = 0


class YahooEventBackfiller:
    QUOTE_SUMMARY_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
    CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"

    def __init__(
        self,
        symbol_master_path: str,
        fundamentals_path: str,
        events_path: str,
        cache_dir: str | Path = "data/cache/pit",
        refresh_cache: bool = False,
        cache_only: bool = False,
        timeout_seconds: float = 20.0,
        sleep_seconds: float = 0.25,
    ):
        self.symbol_master_path = str(symbol_master_path)
        self.fundamentals_path = str(fundamentals_path)
        self.events_path = str(events_path)
        self.sm = SymbolMaster(symbol_master_path)
        self.fund = FundamentalsStore(fundamentals_path)
        self.events = EventStore(events_path)
        self.raw_cache = RawDataCache(cache_dir)
        self.refresh_cache = bool(refresh_cache)
        self.cache_only = bool(cache_only)
        self.timeout_seconds = float(timeout_seconds)
        self.sleep_seconds = float(max(sleep_seconds, 0.0))
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (compatible; one-brain-fund/1.0)",
                "Accept": "application/json,text/plain,*/*",
            }
        )
        self.stats = YahooBackfillStats()
        self._existing_event_keys = self._load_existing_event_keys()
        self._existing_fund_keys = self._load_existing_fund_keys()

    def close(self):
        self.session.close()
        self.sm.close()
        self.fund.close()
        self.events.close()

    def _sleep(self):
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)

    def _load_existing_event_keys(self) -> set[tuple]:
        if not Path(self.events_path).exists():
            return set()
        df = self.events.to_dataframe()
        if df.empty:
            return set()
        return {
            (
                int(row.canonical_id),
                int(row.published_at_ns),
                str(row.event_type),
                str(row.source),
                str(row.headline),
            )
            for row in df.itertuples()
        }

    def _load_existing_fund_keys(self) -> set[tuple]:
        if not Path(self.fundamentals_path).exists():
            return set()
        df = self.fund.to_dataframe()
        if df.empty:
            return set()
        return {
            (
                int(row.canonical_id),
                str(row.metric_name),
                int(row.published_at_ns),
                int(row.period_end_ns),
                str(row.source),
            )
            for row in df.itertuples()
        }

    def _resolve_canonical(self, ticker: str, as_of_ns: int | None = None) -> int | None:
        resolved = self.sm.resolve_ticker(ticker, int(as_of_ns or time.time_ns()))
        return int(resolved) if resolved is not None else None

    def _fetch_cached_json(self, namespace: str, key: str, fetcher) -> dict:
        cached = None if self.refresh_cache else self.raw_cache.read_json(namespace, key)
        if cached is not None:
            self.stats.raw_cache_hits += 1
            return cached
        if self.cache_only:
            return {}
        payload = fetcher() or {}
        self.raw_cache.write_json(namespace, key, to_jsonable(payload))
        self.stats.raw_cache_writes += 1
        return payload

    def _request_json(self, url: str, params: dict[str, str]) -> dict:
        response = self.session.get(url, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def fetch_quote_summary(self, ticker: str, modules: Sequence[str]) -> dict:
        modules_str = ",".join(modules)
        key = f"{ticker.lower()}::{modules_str}"
        namespace = "yahoo/quote_summary"
        return self._fetch_cached_json(
            namespace,
            key,
            lambda: self._request_json(
                self.QUOTE_SUMMARY_URL.format(ticker=ticker),
                {"modules": modules_str},
            ),
        )

    def fetch_intraday_chart(self, ticker: str, interval: str, range_name: str) -> dict:
        key = f"{ticker.lower()}::{interval}::{range_name}"
        namespace = "yahoo/chart"
        return self._fetch_cached_json(
            namespace,
            key,
            lambda: self._request_json(
                self.CHART_URL.format(ticker=ticker),
                {"interval": interval, "range": range_name, "events": "div,splits", "includePrePost": "true"},
            ),
        )

    def _add_fund_records(self, records: list[FundamentalRecord]) -> None:
        fresh: list[FundamentalRecord] = []
        for record in records:
            key = (
                int(record.canonical_id),
                str(record.metric_name),
                int(record.published_at_ns),
                int(record.period_end_ns),
                str(record.source),
            )
            if key in self._existing_fund_keys:
                self.stats.fundamental_duplicates += 1
                continue
            self._existing_fund_keys.add(key)
            fresh.append(record)
        if fresh:
            self.fund.add_records_batch(fresh)
            self.stats.fundamentals_inserted += len(fresh)

    def _add_event_records(self, records: list[EventRecord]) -> None:
        fresh: list[EventRecord] = []
        for record in records:
            key = (
                int(record.canonical_id),
                int(record.published_at_ns),
                str(record.event_type),
                str(record.source),
                str(record.headline),
            )
            if key in self._existing_event_keys:
                self.stats.event_duplicates += 1
                continue
            self._existing_event_keys.add(key)
            fresh.append(record)
        if fresh:
            self.events.add_records_batch(fresh)
            self.stats.events_inserted += len(fresh)

    def parse_quote_summary(
        self,
        ticker: str,
        payload: dict,
        *,
        as_of_ns: int | None = None,
    ) -> tuple[list[FundamentalRecord], list[EventRecord]]:
        as_of_ns = int(as_of_ns or time.time_ns())
        canonical_id = self._resolve_canonical(ticker, as_of_ns=as_of_ns)
        if canonical_id is None:
            return [], []

        result = (
            (((payload or {}).get("quoteSummary") or {}).get("result") or [None])[0]
            or {}
        )
        if not result:
            return [], []

        fund_records: list[FundamentalRecord] = []
        event_records: list[EventRecord] = []

        earnings_history = (
            ((result.get("earningsHistory") or {}).get("history")) or []
        )
        for item in earnings_history:
            published_at_ns = (
                ns_from_epoch_seconds(item.get("date"))
                or ns_from_epoch_seconds(item.get("quarter"))
                or ns_from_epoch_seconds(item.get("startDate"))
                or as_of_ns
            )
            period_end_ns = (
                ns_from_epoch_seconds(item.get("quarter"))
                or ns_from_epoch_seconds(item.get("startDate"))
                or published_at_ns
            )
            eps_actual = safe_float(item.get("epsActual"))
            eps_est = safe_float(item.get("epsEstimate"))
            surprise_pct = safe_float(item.get("surprisePercent"))
            surprise_score = clip_score(surprise_pct or 0.0, scale=20.0)
            confidence = float(np.clip(0.35 + 0.40 * abs(surprise_score), 0.20, 0.95))

            if eps_actual is not None:
                fund_records.append(
                    FundamentalRecord(
                        canonical_id=canonical_id,
                        metric_name=FundamentalsStore.METRIC_EPS,
                        value=float(eps_actual),
                        published_at_ns=published_at_ns,
                        period_end_ns=period_end_ns,
                        source="YAHOO_EARNINGS_HISTORY",
                    )
                )
            if eps_est is not None:
                fund_records.append(
                    FundamentalRecord(
                        canonical_id=canonical_id,
                        metric_name=FundamentalsStore.METRIC_ANALYST_EST_EPS,
                        value=float(eps_est),
                        published_at_ns=published_at_ns,
                        period_end_ns=period_end_ns,
                        source="YAHOO_EARNINGS_HISTORY",
                    )
                )
            if surprise_pct is not None:
                fund_records.append(
                    FundamentalRecord(
                        canonical_id=canonical_id,
                        metric_name=FundamentalsStore.METRIC_ANALYST_REVISION,
                        value=float(np.clip((surprise_pct or 0.0) / 100.0, -1.0, 1.0)),
                        published_at_ns=published_at_ns,
                        period_end_ns=period_end_ns,
                        source="YAHOO_EARNINGS_HISTORY",
                    )
                )

            direction = "beat" if surprise_score >= 0 else "miss"
            headline = f"{ticker} earnings {direction}: surprise {surprise_pct if surprise_pct is not None else 0:.2f}%"
            event_records.append(
                EventRecord(
                    canonical_id=canonical_id,
                    ticker=ticker,
                    published_at_ns=published_at_ns,
                    event_type="earnings",
                    source="YAHOO_EARNINGS_HISTORY",
                    headline=headline,
                    body="Yahoo quoteSummary earningsHistory",
                    sentiment_score=surprise_score,
                    relevance=1.25,
                    novelty=1.0,
                    confidence=confidence,
                    metadata=json.dumps(item, sort_keys=True),
                )
            )

        upgrade_hist = (
            ((result.get("upgradeDowngradeHistory") or {}).get("history")) or []
        )
        for item in upgrade_hist:
            published_at_ns = ns_from_epoch_seconds(item.get("epochGradeDate")) or as_of_ns
            action = str(item.get("action") or "")
            from_grade = str(item.get("fromGrade") or "")
            to_grade = str(item.get("toGrade") or "")
            firm = str(item.get("firm") or "").strip()
            score = score_upgrade_downgrade(action, from_grade, to_grade)
            confidence = float(np.clip(0.45 + 0.30 * abs(score), 0.25, 0.90))

            event_records.append(
                EventRecord(
                    canonical_id=canonical_id,
                    ticker=ticker,
                    published_at_ns=published_at_ns,
                    event_type="estimate_revision",
                    source="YAHOO_UPGRADE_DOWNGRADE",
                    headline=f"{ticker} {action or 'rating change'} {from_grade}->{to_grade} ({firm})".strip(),
                    body="Yahoo upgrade/downgrade history",
                    sentiment_score=score,
                    relevance=1.10,
                    novelty=1.0,
                    confidence=confidence,
                    metadata=json.dumps(item, sort_keys=True),
                )
            )
            fund_records.append(
                FundamentalRecord(
                    canonical_id=canonical_id,
                    metric_name=FundamentalsStore.METRIC_ANALYST_REVISION,
                    value=score,
                    published_at_ns=published_at_ns,
                    period_end_ns=published_at_ns,
                    source="YAHOO_UPGRADE_DOWNGRADE",
                )
            )

        earnings_trend = ((result.get("earningsTrend") or {}).get("trend")) or []
        for item in earnings_trend:
            eps_trend = item.get("epsTrend") or {}
            curr = safe_float(eps_trend.get("current"))
            prior = (
                safe_float(eps_trend.get("30daysAgo"))
                or safe_float(eps_trend.get("60daysAgo"))
                or safe_float(eps_trend.get("90daysAgo"))
            )
            if curr is None or prior in (None, 0):
                continue
            revision = float(np.clip((curr - prior) / abs(prior), -1.0, 1.0))
            end_date_ns = ns_from_epoch_seconds(item.get("endDate")) or as_of_ns
            fund_records.append(
                FundamentalRecord(
                    canonical_id=canonical_id,
                    metric_name=FundamentalsStore.METRIC_ANALYST_REVISION,
                    value=revision,
                    published_at_ns=as_of_ns,
                    period_end_ns=end_date_ns,
                    source="YAHOO_EARNINGS_TREND",
                )
            )

        return fund_records, event_records

    def backfill_quote_summary(
        self,
        tickers: Sequence[str],
        modules: Sequence[str] | None = None,
        chunk_size: int = 40,
    ) -> None:
        modules = list(
            modules
            or [
                "earningsHistory",
                "earningsTrend",
                "upgradeDowngradeHistory",
                "recommendationTrend",
                "calendarEvents",
            ]
        )
        for chunk in chunked(list(tickers), chunk_size):
            logger.info("  yahoo quote chunk: %s", ",".join(chunk[:6]) + ("..." if len(chunk) > 6 else ""))
            for ticker in chunk:
                try:
                    payload = self.fetch_quote_summary(ticker, modules)
                except Exception as exc:
                    logger.warning("Yahoo quote summary failed for %s: %s", ticker, exc)
                    continue
                fund_records, event_records = self.parse_quote_summary(ticker, payload)
                self._add_fund_records(fund_records)
                self._add_event_records(event_records)
                self._sleep()

    def parse_intraday_chart(self, payload: dict) -> pd.DataFrame:
        result = (((payload or {}).get("chart") or {}).get("result") or [None])[0] or {}
        if not result:
            return pd.DataFrame()
        timestamps = result.get("timestamp") or []
        indicators = (result.get("indicators") or {}).get("quote") or []
        if not timestamps or not indicators:
            return pd.DataFrame()
        quote = indicators[0] or {}
        rows = []
        for idx, ts in enumerate(timestamps):
            ts_ns = ns_from_epoch_seconds(ts)
            if ts_ns is None:
                continue
            o = safe_float((quote.get("open") or [None])[idx])
            h = safe_float((quote.get("high") or [None])[idx])
            l = safe_float((quote.get("low") or [None])[idx])
            c = safe_float((quote.get("close") or [None])[idx])
            v = safe_float((quote.get("volume") or [None])[idx]) or 0.0
            if c is None:
                continue
            if o is None:
                o = c
            if h is None:
                h = max(o, c)
            if l is None:
                l = min(o, c)
            rows.append(
                {
                    "timestamp_ns": int(ts_ns),
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": float(max(v, 0.0)),
                }
            )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df.sort_values("timestamp_ns").drop_duplicates(subset=["timestamp_ns"], keep="last").reset_index(drop=True)

    def backfill_intraday_bars(
        self,
        tickers: Sequence[str],
        *,
        cache_dir: str | Path,
        interval: str = "1m",
        range_name: str = "60d",
    ) -> dict[str, int]:
        target_dir = Path(cache_dir).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        written = failed = skipped = 0
        for ticker in tickers:
            cache_path = target_dir / f"{ticker}_{interval}.parquet"
            meta_path = target_dir / f"{ticker}_{interval}.meta.json"
            existing = pd.DataFrame()
            if cache_path.exists():
                try:
                    existing = pd.read_parquet(cache_path)
                except Exception:
                    existing = pd.DataFrame()
            try:
                payload = self.fetch_intraday_chart(ticker, interval=interval, range_name=range_name)
                incoming = self.parse_intraday_chart(payload)
            except Exception as exc:
                logger.warning("Yahoo intraday fetch failed for %s: %s", ticker, exc)
                failed += 1
                continue
            if incoming.empty:
                skipped += 1
                continue
            merged = pd.concat([existing, incoming], ignore_index=True) if not existing.empty else incoming
            merged = merged.sort_values("timestamp_ns").drop_duplicates(subset=["timestamp_ns"], keep="last").reset_index(drop=True)
            merged.to_parquet(cache_path, index=False)
            start_ts = pd.to_datetime(int(merged["timestamp_ns"].iloc[0]), unit="ns", utc=True).isoformat()
            end_ts = pd.to_datetime(int(merged["timestamp_ns"].iloc[-1]), unit="ns", utc=True).isoformat()
            meta = {
                "symbol": ticker,
                "interval": interval,
                "range": range_name,
                "rows": int(len(merged)),
                "start_ts": start_ts,
                "end_ts": end_ts,
                "source": "YAHOO_CHART",
                "updated_at": datetime.now(tz=UTC).isoformat(),
            }
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
            written += 1
            self._sleep()
        self.stats.intraday_files_written += written
        self.stats.intraday_failed += failed
        return {"written": written, "failed": failed, "skipped": skipped}

    def summary(self) -> dict[str, int]:
        return {
            "events_inserted": self.stats.events_inserted,
            "event_duplicates": self.stats.event_duplicates,
            "fundamentals_inserted": self.stats.fundamentals_inserted,
            "fundamental_duplicates": self.stats.fundamental_duplicates,
            "raw_cache_hits": self.stats.raw_cache_hits,
            "raw_cache_writes": self.stats.raw_cache_writes,
            "intraday_files_written": self.stats.intraday_files_written,
            "intraday_failed": self.stats.intraday_failed,
        }
