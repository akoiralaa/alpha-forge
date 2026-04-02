from __future__ import annotations

import csv
import io
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
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
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def clip_score(value: float, scale: float = 1.0) -> float:
    return float(np.clip(value / max(scale, 1e-12), -1.0, 1.0))


def parse_iso_date_ns(value: str | None, fallback_hour: int = 16) -> int | None:
    if not value:
        return None
    value = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%dT%H%M%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(value, fmt)
            if fmt == "%Y-%m-%d":
                dt = dt.replace(hour=fallback_hour, minute=0, second=0)
            dt = dt.replace(tzinfo=UTC)
            return int(dt.timestamp() * 1_000_000_000)
        except ValueError:
            continue
    return None


@dataclass(slots=True)
class AlphaVantageBackfillStats:
    events_inserted: int = 0
    event_duplicates: int = 0
    fundamentals_inserted: int = 0
    fundamental_duplicates: int = 0
    raw_cache_hits: int = 0
    raw_cache_misses: int = 0
    raw_cache_writes: int = 0
    network_requests: int = 0
    symbols_skipped: int = 0


class AlphaVantageEventBackfiller:
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(
        self,
        api_key: str,
        symbol_master_path: str,
        fundamentals_path: str,
        events_path: str,
        cache_dir: str | Path = "data/cache/pit",
        refresh_cache: bool = False,
        cache_only: bool = False,
        requests_per_minute: int = 5,
        timeout_seconds: float = 20.0,
    ):
        self.api_key = str(api_key or "").strip()
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required")
        self.symbol_master_path = str(symbol_master_path)
        self.fundamentals_path = str(fundamentals_path)
        self.events_path = str(events_path)
        self.sm = SymbolMaster(symbol_master_path)
        self.fund = FundamentalsStore(fundamentals_path)
        self.events = EventStore(events_path)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "one-brain-fund/1.0"})
        self.raw_cache = RawDataCache(cache_dir)
        self.refresh_cache = bool(refresh_cache)
        self.cache_only = bool(cache_only)
        self.sleep_seconds = 60.0 / max(int(requests_per_minute), 1)
        self.timeout_seconds = float(timeout_seconds)
        self.stats = AlphaVantageBackfillStats()
        self._existing_event_keys = self._load_existing_event_keys()
        self._existing_fund_keys = self._load_existing_fund_keys()
        self._last_fetch_made_request = False

    def close(self):
        self.session.close()
        self.sm.close()
        self.fund.close()
        self.events.close()

    def _sleep(self):
        time.sleep(self.sleep_seconds)

    def _maybe_sleep_after_fetch(self):
        if self._last_fetch_made_request:
            self._sleep()

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
        self._last_fetch_made_request = False
        cached = None if self.refresh_cache else self.raw_cache.read_json(namespace, key)
        if cached is not None:
            self.stats.raw_cache_hits += 1
            return cached
        self.stats.raw_cache_misses += 1
        if self.cache_only:
            return {}
        self._last_fetch_made_request = True
        self.stats.network_requests += 1
        payload = fetcher() or {}
        self.raw_cache.write_json(namespace, key, to_jsonable(payload))
        self.stats.raw_cache_writes += 1
        return payload

    def _fetch_cached_text(self, namespace: str, key: str, fetcher) -> str:
        self._last_fetch_made_request = False
        cached = None if self.refresh_cache else self.raw_cache.read_json(namespace, key)
        if cached is not None:
            self.stats.raw_cache_hits += 1
            return str(cached)
        self.stats.raw_cache_misses += 1
        if self.cache_only:
            return ""
        self._last_fetch_made_request = True
        self.stats.network_requests += 1
        payload = str(fetcher() or "")
        self.raw_cache.write_json(namespace, key, payload)
        self.stats.raw_cache_writes += 1
        return payload

    def _request_json(self, params: dict[str, str]) -> dict:
        response = self.session.get(self.BASE_URL, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def _request_text(self, params: dict[str, str]) -> str:
        response = self.session.get(self.BASE_URL, params=params, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.text

    def fetch_function_json(self, fn_name: str, ticker: str) -> dict:
        key = f"{fn_name}:{ticker.upper()}"
        return self._fetch_cached_json(
            "alpha_vantage/json",
            key,
            lambda: self._request_json(
                {"function": fn_name, "symbol": ticker, "apikey": self.api_key}
            ),
        )

    def fetch_news_sentiment(self, ticker: str, limit: int = 200) -> dict:
        key = f"NEWS_SENTIMENT:{ticker.upper()}:{limit}"
        return self._fetch_cached_json(
            "alpha_vantage/news",
            key,
            lambda: self._request_json(
                {"function": "NEWS_SENTIMENT", "tickers": ticker, "limit": str(limit), "apikey": self.api_key}
            ),
        )

    def fetch_earnings_calendar(self, ticker: str, horizon: str = "12month") -> str:
        key = f"EARNINGS_CALENDAR:{ticker.upper()}:{horizon}"
        return self._fetch_cached_text(
            "alpha_vantage/csv",
            key,
            lambda: self._request_text(
                {"function": "EARNINGS_CALENDAR", "symbol": ticker, "horizon": horizon, "apikey": self.api_key}
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

    def parse_earnings_payload(self, ticker: str, payload: dict) -> tuple[list[FundamentalRecord], list[EventRecord]]:
        if "quarterlyEarnings" not in payload:
            return [], []
        as_of_ns = time.time_ns()
        canonical_id = self._resolve_canonical(ticker, as_of_ns=as_of_ns)
        if canonical_id is None:
            return [], []
        fund_records: list[FundamentalRecord] = []
        event_records: list[EventRecord] = []
        for row in payload.get("quarterlyEarnings", []) or []:
            published_at_ns = parse_iso_date_ns(row.get("reportedDate"), fallback_hour=16) or as_of_ns
            period_end_ns = parse_iso_date_ns(row.get("fiscalDateEnding"), fallback_hour=16) or published_at_ns
            actual_eps = safe_float(row.get("reportedEPS"))
            est_eps = safe_float(row.get("estimatedEPS"))
            surprise_pct = safe_float(row.get("surprisePercentage"))
            score = clip_score(surprise_pct or 0.0, scale=20.0)
            confidence = float(np.clip(0.35 + 0.45 * abs(score), 0.25, 0.95))
            if actual_eps is not None:
                fund_records.append(
                    FundamentalRecord(
                        canonical_id=canonical_id,
                        metric_name=FundamentalsStore.METRIC_EPS,
                        value=actual_eps,
                        published_at_ns=published_at_ns,
                        period_end_ns=period_end_ns,
                        source="ALPHA_VANTAGE_EARNINGS",
                    )
                )
            if est_eps is not None:
                fund_records.append(
                    FundamentalRecord(
                        canonical_id=canonical_id,
                        metric_name=FundamentalsStore.METRIC_ANALYST_EST_EPS,
                        value=est_eps,
                        published_at_ns=published_at_ns,
                        period_end_ns=period_end_ns,
                        source="ALPHA_VANTAGE_EARNINGS",
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
                        source="ALPHA_VANTAGE_EARNINGS",
                    )
                )
            event_records.append(
                EventRecord(
                    canonical_id=canonical_id,
                    ticker=ticker,
                    published_at_ns=published_at_ns,
                    event_type="earnings",
                    source="ALPHA_VANTAGE_EARNINGS",
                    headline=f"{ticker} reported EPS surprise {surprise_pct if surprise_pct is not None else 0:.2f}%",
                    body="Alpha Vantage quarterly earnings",
                    sentiment_score=score,
                    relevance=1.2,
                    novelty=1.0,
                    confidence=confidence,
                    metadata=json.dumps(row, sort_keys=True),
                )
            )
        return fund_records, event_records

    def parse_estimates_payload(self, ticker: str, payload: dict, as_of_ns: int | None = None) -> tuple[list[FundamentalRecord], list[EventRecord]]:
        if "estimates" not in payload:
            return [], []
        as_of_ns = int(as_of_ns or time.time_ns())
        canonical_id = self._resolve_canonical(ticker, as_of_ns=as_of_ns)
        if canonical_id is None:
            return [], []
        fund_records: list[FundamentalRecord] = []
        event_records: list[EventRecord] = []
        for row in payload.get("estimates", []) or []:
            period_end_ns = parse_iso_date_ns(row.get("date"), fallback_hour=16) or as_of_ns
            avg_est = safe_float(row.get("eps_estimate_average"))
            prev_30 = safe_float(row.get("eps_estimate_average_30_days_ago"))
            up_30 = safe_float(row.get("eps_estimate_revision_up_trailing_30_days")) or 0.0
            down_30 = safe_float(row.get("eps_estimate_revision_down_trailing_30_days")) or 0.0
            rev_ratio = 0.0
            if avg_est is not None and prev_30 not in (None, 0):
                rev_ratio = float(np.clip((avg_est - prev_30) / abs(prev_30), -1.0, 1.0))
            revision_balance = float(np.clip((up_30 - down_30) / max(up_30 + down_30, 1.0), -1.0, 1.0))
            revision_score = float(np.clip(0.7 * rev_ratio + 0.3 * revision_balance, -1.0, 1.0))
            if avg_est is not None:
                fund_records.append(
                    FundamentalRecord(
                        canonical_id=canonical_id,
                        metric_name=FundamentalsStore.METRIC_ANALYST_EST_EPS,
                        value=avg_est,
                        published_at_ns=as_of_ns,
                        period_end_ns=period_end_ns,
                        source="ALPHA_VANTAGE_EARNINGS_EST",
                    )
                )
            if abs(revision_score) > 1e-12:
                fund_records.append(
                    FundamentalRecord(
                        canonical_id=canonical_id,
                        metric_name=FundamentalsStore.METRIC_ANALYST_REVISION,
                        value=revision_score,
                        published_at_ns=as_of_ns,
                        period_end_ns=period_end_ns,
                        source="ALPHA_VANTAGE_EARNINGS_EST",
                    )
                )
                event_records.append(
                    EventRecord(
                        canonical_id=canonical_id,
                        ticker=ticker,
                        published_at_ns=as_of_ns,
                        event_type="estimate_revision",
                        source="ALPHA_VANTAGE_EARNINGS_EST",
                        headline=f"{ticker} estimate revision for {row.get('horizon', 'horizon')} ({row.get('date', '')})",
                        body="Alpha Vantage earnings estimates/revisions",
                        sentiment_score=revision_score,
                        relevance=1.1,
                        novelty=0.7,
                        confidence=float(np.clip(0.30 + 0.50 * abs(revision_score), 0.20, 0.90)),
                        metadata=json.dumps(row, sort_keys=True),
                    )
                )
        return fund_records, event_records

    def parse_news_payload(self, ticker: str, payload: dict) -> list[EventRecord]:
        feed = payload.get("feed")
        if not isinstance(feed, list):
            return []
        as_of_ns = time.time_ns()
        canonical_id = self._resolve_canonical(ticker, as_of_ns=as_of_ns)
        if canonical_id is None:
            return []
        records: list[EventRecord] = []
        for row in feed:
            published_at_ns = parse_iso_date_ns(row.get("time_published"), fallback_hour=12) or as_of_ns
            overall = safe_float(row.get("overall_sentiment_score")) or 0.0
            ticker_score = overall
            ticker_conf = 0.35
            for ts_row in row.get("ticker_sentiment", []) or []:
                if str(ts_row.get("ticker", "")).upper() == ticker.upper():
                    ticker_score = safe_float(ts_row.get("ticker_sentiment_score")) or overall
                    ticker_conf = safe_float(ts_row.get("relevance_score")) or ticker_conf
                    break
            headline = str(row.get("title") or "").strip()
            if not headline:
                continue
            records.append(
                EventRecord(
                    canonical_id=canonical_id,
                    ticker=ticker,
                    published_at_ns=published_at_ns,
                    event_type="news",
                    source="ALPHA_VANTAGE_NEWS",
                    headline=headline[:500],
                    body=str(row.get("summary") or "")[:2000],
                    sentiment_score=float(np.clip(ticker_score, -1.0, 1.0)),
                    relevance=float(np.clip(ticker_conf, 0.1, 2.0)),
                    novelty=1.0,
                    confidence=float(np.clip(0.20 + 0.60 * ticker_conf, 0.15, 0.95)),
                    metadata=json.dumps(row, sort_keys=True),
                )
            )
        return records

    def parse_earnings_calendar_csv(self, ticker: str, csv_text: str) -> list[EventRecord]:
        if not csv_text.strip():
            return []
        as_of_ns = time.time_ns()
        canonical_id = self._resolve_canonical(ticker, as_of_ns=as_of_ns)
        if canonical_id is None:
            return []
        out: list[EventRecord] = []
        reader = csv.DictReader(io.StringIO(csv_text))
        for row in reader:
            report_ns = parse_iso_date_ns(row.get("reportDate"), fallback_hour=16)
            if report_ns is None:
                continue
            estimate = safe_float(row.get("estimate"))
            tod = str(row.get("timeOfTheDay") or "").lower()
            time_label = "bmo" if "pre" in tod else ("amc" if "post" in tod else "regular")
            out.append(
                EventRecord(
                    canonical_id=canonical_id,
                    ticker=ticker,
                    published_at_ns=as_of_ns,
                    event_type="earnings",
                    source="ALPHA_VANTAGE_EARNINGS_CALENDAR",
                    headline=f"{ticker} scheduled earnings on {row.get('reportDate')} ({time_label})",
                    body=f"Expected EPS {estimate if estimate is not None else 'n/a'} {row.get('currency', '')}".strip(),
                    sentiment_score=0.0,
                    relevance=0.65,
                    novelty=0.4,
                    confidence=0.55,
                    metadata=json.dumps(row, sort_keys=True),
                )
            )
        return out

    def backfill(
        self,
        tickers: Sequence[str],
        *,
        include_news: bool = True,
        include_calendar: bool = True,
        chunk_size: int = 25,
    ) -> None:
        for chunk in chunked(list(tickers), chunk_size):
            logger.info("  alpha vantage chunk: %s", ",".join(chunk[:6]) + ("..." if len(chunk) > 6 else ""))
            for ticker in chunk:
                event_records: list[EventRecord] = []
                fund_records: list[FundamentalRecord] = []
                try:
                    earnings_payload = self.fetch_function_json("EARNINGS", ticker)
                    fr, er = self.parse_earnings_payload(ticker, earnings_payload)
                    fund_records.extend(fr)
                    event_records.extend(er)
                except Exception as exc:
                    logger.warning("Alpha Vantage EARNINGS failed for %s: %s", ticker, exc)
                finally:
                    self._maybe_sleep_after_fetch()

                try:
                    estimate_payload = self.fetch_function_json("EARNINGS_ESTIMATES", ticker)
                    fr, er = self.parse_estimates_payload(ticker, estimate_payload)
                    fund_records.extend(fr)
                    event_records.extend(er)
                except Exception as exc:
                    logger.warning("Alpha Vantage EARNINGS_ESTIMATES failed for %s: %s", ticker, exc)
                finally:
                    self._maybe_sleep_after_fetch()

                if include_calendar:
                    try:
                        csv_text = self.fetch_earnings_calendar(ticker, horizon="12month")
                        event_records.extend(self.parse_earnings_calendar_csv(ticker, csv_text))
                    except Exception as exc:
                        logger.warning("Alpha Vantage EARNINGS_CALENDAR failed for %s: %s", ticker, exc)
                    finally:
                        self._maybe_sleep_after_fetch()

                if include_news:
                    try:
                        news_payload = self.fetch_news_sentiment(ticker, limit=200)
                        event_records.extend(self.parse_news_payload(ticker, news_payload))
                    except Exception as exc:
                        logger.warning("Alpha Vantage NEWS_SENTIMENT failed for %s: %s", ticker, exc)
                    finally:
                        self._maybe_sleep_after_fetch()

                if not fund_records and not event_records:
                    self.stats.symbols_skipped += 1
                self._add_fund_records(fund_records)
                self._add_event_records(event_records)

    def summary(self) -> dict[str, int]:
        return {
            "events_inserted": self.stats.events_inserted,
            "event_duplicates": self.stats.event_duplicates,
            "fundamentals_inserted": self.stats.fundamentals_inserted,
            "fundamental_duplicates": self.stats.fundamental_duplicates,
            "raw_cache_hits": self.stats.raw_cache_hits,
            "raw_cache_misses": self.stats.raw_cache_misses,
            "raw_cache_writes": self.stats.raw_cache_writes,
            "network_requests": self.stats.network_requests,
            "symbols_skipped": self.stats.symbols_skipped,
        }
