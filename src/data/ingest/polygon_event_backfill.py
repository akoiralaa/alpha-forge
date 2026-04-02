from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from polygon import RESTClient

from src.data.events import EventRecord, EventStore
from src.data.fundamentals import FundamentalRecord, FundamentalsStore
from src.data.ingest.base import AssetClass, date_to_ns
from src.data.ingest.raw_cache import RawDataCache, to_jsonable
from src.data.symbol_master import SymbolMaster
from src.signals.sentiment import build_sentiment_model

logger = logging.getLogger(__name__)

NY_TZ = ZoneInfo("America/New_York")
UTC = timezone.utc

POSITIVE_ACTIONS = {
    "upgrade", "raised", "raise", "reiterate", "initiates", "initiate",
    "resume", "resumed", "outperform", "overweight", "buy", "strong buy",
}
NEGATIVE_ACTIONS = {
    "downgrade", "downgraded", "lowered", "lower", "underperform",
    "underweight", "sell", "strong sell", "cut",
}


@dataclass(slots=True)
class BackfillStats:
    events_inserted: int = 0
    event_duplicates: int = 0
    fundamentals_inserted: int = 0
    fundamental_duplicates: int = 0
    raw_cache_hits: int = 0
    raw_cache_writes: int = 0


def chunked(values: Sequence[str], size: int) -> Iterable[list[str]]:
    size = max(int(size), 1)
    for i in range(0, len(values), size):
        yield list(values[i:i + size])


def parse_market_datetime_ns(date_value: str | None, time_hint: str | None = None) -> int | None:
    if not date_value:
        return None

    hint = (time_hint or "").strip().lower()
    hour = 16
    minute = 5

    if hint in {"bmo", "before market open", "before open", "pre-market"}:
        hour, minute = 8, 0
    elif hint in {"amc", "after market close", "after close", "post-market"}:
        hour, minute = 16, 30
    else:
        cleaned = hint.replace(".", ":")
        for fmt in ("%H:%M", "%I:%M%p", "%I%p"):
            try:
                parsed = datetime.strptime(cleaned.replace(" ", ""), fmt)
                hour, minute = parsed.hour, parsed.minute
                break
            except ValueError:
                continue

    ts = pd.Timestamp(f"{date_value} {hour:02d}:{minute:02d}", tz=NY_TZ)
    return int(ts.tz_convert("UTC").value)


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


def clamp_score(value: float, scale: float = 1.0) -> float:
    return float(np.clip(value / scale, -1.0, 1.0))


def row_value(record, field: str, default=None):
    if isinstance(record, dict):
        return record.get(field, default)
    return getattr(record, field, default)


def score_earnings_surprise(
    eps_surprise_percent: float | None,
    revenue_surprise_percent: float | None,
) -> tuple[float, float]:
    eps_component = clamp_score(safe_float(eps_surprise_percent) or 0.0, scale=15.0)
    rev_component = clamp_score(safe_float(revenue_surprise_percent) or 0.0, scale=10.0)
    score = float(np.clip(0.7 * eps_component + 0.3 * rev_component, -1.0, 1.0))
    confidence = float(np.clip(0.45 + 0.30 * abs(score), 0.25, 0.95))
    return score, confidence


def score_guidance_change(guidance) -> tuple[float, float]:
    cur_eps = safe_float(row_value(guidance, "estimated_eps_guidance"))
    prev_min_eps = safe_float(row_value(guidance, "previous_min_eps_guidance"))
    prev_max_eps = safe_float(row_value(guidance, "previous_max_eps_guidance"))
    prev_eps_mid = None
    if prev_min_eps is not None or prev_max_eps is not None:
        vals = [v for v in [prev_min_eps, prev_max_eps] if v is not None]
        prev_eps_mid = float(np.mean(vals)) if vals else None

    cur_rev = safe_float(row_value(guidance, "estimated_revenue_guidance"))
    prev_min_rev = safe_float(row_value(guidance, "previous_min_revenue_guidance"))
    prev_max_rev = safe_float(row_value(guidance, "previous_max_revenue_guidance"))
    prev_rev_mid = None
    if prev_min_rev is not None or prev_max_rev is not None:
        vals = [v for v in [prev_min_rev, prev_max_rev] if v is not None]
        prev_rev_mid = float(np.mean(vals)) if vals else None

    eps_delta = 0.0
    if cur_eps is not None and prev_eps_mid not in (None, 0):
        eps_delta = clamp_score((cur_eps - prev_eps_mid) / abs(prev_eps_mid), scale=0.10)

    rev_delta = 0.0
    if cur_rev is not None and prev_rev_mid not in (None, 0):
        rev_delta = clamp_score((cur_rev - prev_rev_mid) / abs(prev_rev_mid), scale=0.08)

    positioning = str(row_value(guidance, "positioning", "") or "").lower()
    if "raise" in positioning or "above" in positioning:
        eps_delta = max(eps_delta, 0.35)
    elif "cut" in positioning or "below" in positioning or "lower" in positioning:
        eps_delta = min(eps_delta, -0.35)

    score = float(np.clip(0.65 * eps_delta + 0.35 * rev_delta, -1.0, 1.0))
    confidence = float(np.clip(0.40 + 0.35 * abs(score), 0.20, 0.90))
    return score, confidence


def score_analyst_revision(insight) -> float:
    rating_action = str(row_value(insight, "rating_action", "") or "").lower()
    rating = str(row_value(insight, "rating", "") or "").lower()
    text = f"{rating_action} {rating} {row_value(insight, 'insight', '') or ''}".lower()

    score = 0.0
    if any(term in text for term in POSITIVE_ACTIONS):
        score += 0.55
    if any(term in text for term in NEGATIVE_ACTIONS):
        score -= 0.55

    price_target = safe_float(row_value(insight, "price_target"))
    if price_target is not None:
        score += 0.10 if price_target > 0 else 0.0

    return float(np.clip(score, -1.0, 1.0))


def build_active_equity_tickers(symbol_master_path: str | Path, as_of_ns: int | None = None) -> list[str]:
    as_of_ns = int(as_of_ns or time.time_ns())
    sm = SymbolMaster(symbol_master_path)
    try:
        instruments = sm.get_active_instruments(as_of_ns, asset_class=AssetClass.EQUITY)
        tickers = sorted({inst.ticker for inst in instruments})
        return tickers
    finally:
        sm.close()


def ensure_equity_universe_in_symbol_master(
    symbol_master_path: str | Path,
    tickers: Sequence[str],
    valid_from_ns: int | None = None,
) -> int:
    valid_from_ns = int(valid_from_ns or date_to_ns(2000, 1, 1))
    as_of_ns = time.time_ns()
    sm = SymbolMaster(symbol_master_path)
    inserted = 0
    try:
        for ticker in tickers:
            if sm.resolve_ticker(ticker, as_of_ns) is not None:
                continue
            sm.add_instrument(
                exchange="SMART",
                ticker=ticker,
                valid_from_ns=valid_from_ns,
                asset_class=AssetClass.EQUITY,
                currency="USD",
            )
            inserted += 1
    finally:
        sm.close()
    return inserted


class PolygonEventBackfiller:
    def __init__(
        self,
        api_key: str,
        symbol_master_path: str,
        fundamentals_path: str,
        events_path: str,
        rate_limit_per_minute: int = 5,
        prefer_hf: bool = False,
        hf_model_name: str = "ProsusAI/finbert",
        cache_dir: str | Path = "data/cache/pit",
        refresh_cache: bool = False,
        cache_only: bool = False,
    ):
        self.client = RESTClient(api_key=api_key)
        self.symbol_master_path = str(symbol_master_path)
        self.fundamentals_path = str(fundamentals_path)
        self.events_path = str(events_path)
        self.rate_limit_sleep = 60.0 / max(rate_limit_per_minute, 1)
        self.sm = SymbolMaster(symbol_master_path)
        self.fund = FundamentalsStore(fundamentals_path)
        self.events = EventStore(events_path)
        self.sentiment_model = build_sentiment_model(prefer_hf=prefer_hf, model_name=hf_model_name)
        self.stats = BackfillStats()
        self.raw_cache = RawDataCache(cache_dir)
        self.refresh_cache = bool(refresh_cache)
        self.cache_only = bool(cache_only)
        self._existing_event_keys = self._load_existing_event_keys()
        self._existing_fund_keys = self._load_existing_fund_keys()

    def close(self):
        self.sm.close()
        self.fund.close()
        self.events.close()

    def _sleep(self):
        time.sleep(self.rate_limit_sleep)

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

    def _add_event_records(self, records: list[EventRecord]) -> None:
        if not records:
            return
        fresh = []
        for record in records:
            key = (
                record.canonical_id,
                record.published_at_ns,
                record.event_type,
                record.source,
                record.headline,
            )
            if key in self._existing_event_keys:
                self.stats.event_duplicates += 1
                continue
            self._existing_event_keys.add(key)
            fresh.append(record)
        if fresh:
            self.events.add_records_batch(fresh)
            self.stats.events_inserted += len(fresh)

    def _add_fundamental_records(self, records: list[FundamentalRecord]) -> None:
        if not records:
            return
        fresh = []
        for record in records:
            key = (
                record.canonical_id,
                record.metric_name,
                record.published_at_ns,
                record.period_end_ns,
                record.source,
            )
            if key in self._existing_fund_keys:
                self.stats.fundamental_duplicates += 1
                continue
            self._existing_fund_keys.add(key)
            fresh.append(record)
        if fresh:
            self.fund.add_records_batch(fresh)
            self.stats.fundamentals_inserted += len(fresh)

    def _resolve_ticker(self, ticker: str | None, published_at_ns: int | None) -> tuple[int | None, str | None]:
        if not ticker or published_at_ns is None:
            return None, None
        resolved = self.sm.resolve_ticker(str(ticker), int(published_at_ns))
        return resolved, ticker if resolved is not None else None

    def _polygon_chunk_key(self, chunk: Sequence[str], start_date: str, end_date: str) -> str:
        return f"{start_date}__{end_date}__{'-'.join(sorted(str(t).upper() for t in chunk))}"

    def _fetch_cached_polygon_rows(self, namespace: str, cache_key: str, fetcher) -> list[dict]:
        def _do_fetch():
            rows = list(fetcher())
            self._sleep()
            return to_jsonable(rows)

        payload, cache_hit = self.raw_cache.get_or_fetch_json(
            namespace,
            cache_key,
            _do_fetch,
            refresh=self.refresh_cache,
            cache_only=self.cache_only,
        )
        if cache_hit:
            self.stats.raw_cache_hits += 1
        else:
            self.stats.raw_cache_writes += 1
        if isinstance(payload, dict) and "rows" in payload:
            return list(payload["rows"] or [])
        return list(payload or [])

    def backfill_earnings(self, tickers: Sequence[str], start_date: str, end_date: str, chunk_size: int = 25) -> None:
        logger.info("Backfilling Benzinga earnings for %d tickers", len(tickers))
        for chunk in chunked(list(tickers), chunk_size):
            logger.info("  earnings chunk: %s", ",".join(chunk[:4]) + ("..." if len(chunk) > 4 else ""))
            try:
                rows = self._fetch_cached_polygon_rows(
                    "polygon/benzinga_earnings",
                    self._polygon_chunk_key(chunk, start_date, end_date),
                    lambda: self.client.list_benzinga_earnings(
                        ticker_any_of=chunk,
                        date_gte=start_date,
                        date_lte=end_date,
                        limit=1000,
                        sort="date",
                    ),
                )
            except Exception as exc:
                logger.warning("Benzinga earnings fetch failed for chunk %s: %s", chunk[:4], exc)
                continue
            event_records: list[EventRecord] = []
            fund_records: list[FundamentalRecord] = []
            for earning in rows:
                published_at_ns = parse_market_datetime_ns(row_value(earning, "date"), row_value(earning, "time"))
                canonical_id, ticker = self._resolve_ticker(row_value(earning, "ticker"), published_at_ns)
                if canonical_id is None or ticker is None or published_at_ns is None:
                    continue

                period_end_ns = published_at_ns
                actual_eps = safe_float(row_value(earning, "actual_eps"))
                est_eps = safe_float(row_value(earning, "estimated_eps"))
                actual_revenue = safe_float(row_value(earning, "actual_revenue"))
                est_revenue = safe_float(row_value(earning, "estimated_revenue"))
                surprise_score, confidence = score_earnings_surprise(
                    row_value(earning, "eps_surprise_percent"),
                    row_value(earning, "revenue_surprise_percent"),
                )

                if actual_eps is not None:
                    fund_records.append(FundamentalRecord(
                        canonical_id=canonical_id,
                        metric_name=FundamentalsStore.METRIC_EPS,
                        value=actual_eps,
                        published_at_ns=published_at_ns,
                        period_end_ns=period_end_ns,
                        source="POLYGON_BENZINGA_EARNINGS",
                    ))
                if est_eps is not None:
                    fund_records.append(FundamentalRecord(
                        canonical_id=canonical_id,
                        metric_name=FundamentalsStore.METRIC_ANALYST_EST_EPS,
                        value=est_eps,
                        published_at_ns=published_at_ns,
                        period_end_ns=period_end_ns,
                        source="POLYGON_BENZINGA_EARNINGS",
                    ))
                if actual_revenue is not None:
                    fund_records.append(FundamentalRecord(
                        canonical_id=canonical_id,
                        metric_name=FundamentalsStore.METRIC_REVENUE,
                        value=actual_revenue,
                        published_at_ns=published_at_ns,
                        period_end_ns=period_end_ns,
                        source="POLYGON_BENZINGA_EARNINGS",
                    ))
                if est_revenue is not None:
                    fund_records.append(FundamentalRecord(
                        canonical_id=canonical_id,
                        metric_name=FundamentalsStore.METRIC_ANALYST_EST_REVENUE,
                        value=est_revenue,
                        published_at_ns=published_at_ns,
                        period_end_ns=period_end_ns,
                        source="POLYGON_BENZINGA_EARNINGS",
                    ))

                metadata = {
                    "fiscal_year": row_value(earning, "fiscal_year"),
                    "fiscal_period": row_value(earning, "fiscal_period"),
                    "eps_surprise_percent": safe_float(row_value(earning, "eps_surprise_percent")),
                    "revenue_surprise_percent": safe_float(row_value(earning, "revenue_surprise_percent")),
                    "importance": row_value(earning, "importance"),
                }
                headline = (
                    f"{ticker} earnings: EPS {actual_eps if actual_eps is not None else 'n/a'} "
                    f"vs est {est_eps if est_eps is not None else 'n/a'}"
                )
                event_records.append(EventRecord(
                    canonical_id=canonical_id,
                    ticker=ticker,
                    published_at_ns=published_at_ns,
                    event_type="earnings",
                    source="POLYGON_BENZINGA_EARNINGS",
                    headline=headline,
                    body=str(row_value(earning, "notes", "") or ""),
                    sentiment_score=surprise_score,
                    relevance=1.25,
                    novelty=1.15,
                    confidence=confidence,
                    metadata=json.dumps(metadata, sort_keys=True),
                ))
            self._add_fundamental_records(fund_records)
            self._add_event_records(event_records)

    def backfill_guidance(self, tickers: Sequence[str], start_date: str, end_date: str, chunk_size: int = 25) -> None:
        logger.info("Backfilling Benzinga guidance for %d tickers", len(tickers))
        for chunk in chunked(list(tickers), chunk_size):
            try:
                rows = self._fetch_cached_polygon_rows(
                    "polygon/benzinga_guidance",
                    self._polygon_chunk_key(chunk, start_date, end_date),
                    lambda: self.client.list_benzinga_guidance(
                        ticker_any_of=chunk,
                        date_gte=start_date,
                        date_lte=end_date,
                        limit=1000,
                        sort="date",
                    ),
                )
            except Exception as exc:
                logger.warning("Benzinga guidance fetch failed for chunk %s: %s", chunk[:4], exc)
                continue
            event_records: list[EventRecord] = []
            for guidance in rows:
                published_at_ns = parse_market_datetime_ns(row_value(guidance, "date"), row_value(guidance, "time"))
                canonical_id, ticker = self._resolve_ticker(row_value(guidance, "ticker"), published_at_ns)
                if canonical_id is None or ticker is None or published_at_ns is None:
                    continue
                score, confidence = score_guidance_change(guidance)
                metadata = {
                    "positioning": row_value(guidance, "positioning"),
                    "estimated_eps_guidance": safe_float(row_value(guidance, "estimated_eps_guidance")),
                    "estimated_revenue_guidance": safe_float(row_value(guidance, "estimated_revenue_guidance")),
                    "previous_min_eps_guidance": safe_float(row_value(guidance, "previous_min_eps_guidance")),
                    "previous_max_eps_guidance": safe_float(row_value(guidance, "previous_max_eps_guidance")),
                    "previous_min_revenue_guidance": safe_float(row_value(guidance, "previous_min_revenue_guidance")),
                    "previous_max_revenue_guidance": safe_float(row_value(guidance, "previous_max_revenue_guidance")),
                }
                headline = f"{ticker} guidance update"
                event_records.append(EventRecord(
                    canonical_id=canonical_id,
                    ticker=ticker,
                    published_at_ns=published_at_ns,
                    event_type="guidance",
                    source="POLYGON_BENZINGA_GUIDANCE",
                    headline=headline,
                    body=str(row_value(guidance, "notes", "") or ""),
                    sentiment_score=score,
                    relevance=1.10,
                    novelty=1.00,
                    confidence=confidence,
                    metadata=json.dumps(metadata, sort_keys=True),
                ))
            self._add_event_records(event_records)

    def backfill_analyst_insights(
        self,
        tickers: Sequence[str],
        start_date: str,
        end_date: str,
        chunk_size: int = 25,
    ) -> None:
        logger.info("Backfilling Benzinga analyst insights for %d tickers", len(tickers))
        for chunk in chunked(list(tickers), chunk_size):
            try:
                rows = self._fetch_cached_polygon_rows(
                    "polygon/benzinga_analyst",
                    self._polygon_chunk_key(chunk, start_date, end_date),
                    lambda: self.client.list_benzinga_analyst_insights(
                        ticker_any_of=chunk,
                        date_gte=start_date,
                        date_lte=end_date,
                        limit=1000,
                        sort="date",
                    ),
                )
            except Exception as exc:
                logger.warning("Benzinga analyst fetch failed for chunk %s: %s", chunk[:4], exc)
                continue
            event_records: list[EventRecord] = []
            fund_records: list[FundamentalRecord] = []
            for insight in rows:
                published_at_ns = parse_market_datetime_ns(row_value(insight, "date"))
                canonical_id, ticker = self._resolve_ticker(row_value(insight, "ticker"), published_at_ns)
                if canonical_id is None or ticker is None or published_at_ns is None:
                    continue
                revision_score = score_analyst_revision(insight)
                confidence = float(np.clip(0.35 + 0.40 * abs(revision_score), 0.15, 0.90))
                fund_records.append(FundamentalRecord(
                    canonical_id=canonical_id,
                    metric_name=FundamentalsStore.METRIC_ANALYST_REVISION,
                    value=revision_score,
                    published_at_ns=published_at_ns,
                    period_end_ns=published_at_ns,
                    source="POLYGON_BENZINGA_ANALYST",
                ))
                metadata = {
                    "rating": row_value(insight, "rating"),
                    "rating_action": row_value(insight, "rating_action"),
                    "price_target": safe_float(row_value(insight, "price_target")),
                    "firm": row_value(insight, "firm"),
                }
                headline = f"{ticker} analyst insight: {row_value(insight, 'rating_action', '') or 'update'}"
                event_records.append(EventRecord(
                    canonical_id=canonical_id,
                    ticker=ticker,
                    published_at_ns=published_at_ns,
                    event_type="estimate_revision",
                    source="POLYGON_BENZINGA_ANALYST",
                    headline=headline,
                    body=str(row_value(insight, "insight", "") or ""),
                    sentiment_score=revision_score,
                    relevance=0.95,
                    novelty=0.90,
                    confidence=confidence,
                    metadata=json.dumps(metadata, sort_keys=True),
                ))
            self._add_fundamental_records(fund_records)
            self._add_event_records(event_records)

    def backfill_news(
        self,
        tickers: Sequence[str],
        start_date: str,
        end_date: str,
        chunk_size: int = 25,
    ) -> None:
        logger.info("Backfilling Benzinga news for %d tickers", len(tickers))
        for chunk in chunked(list(tickers), chunk_size):
            try:
                rows = self._fetch_cached_polygon_rows(
                    "polygon/benzinga_news",
                    self._polygon_chunk_key(chunk, start_date, end_date),
                    lambda: self.client.list_benzinga_news_v2(
                        tickers_any_of=chunk,
                        published_gte=start_date,
                        published_lte=end_date,
                        limit=1000,
                        sort="published",
                    ),
                )
            except Exception as exc:
                logger.warning("Benzinga news fetch failed for chunk %s: %s", chunk[:4], exc)
                continue
            event_records: list[EventRecord] = []
            for item in rows:
                published = row_value(item, "published")
                if not published:
                    continue
                published_ts = pd.Timestamp(published)
                if published_ts.tzinfo is None:
                    published_ts = published_ts.tz_localize("UTC")
                else:
                    published_ts = published_ts.tz_convert("UTC")
                published_at_ns = int(published_ts.value)
                body = str(row_value(item, "body", "") or row_value(item, "teaser", "") or "")
                title = str(row_value(item, "title", "") or "")
                sentiment = self.sentiment_model.score_text(f"{title}. {body}".strip())
                mapped_tickers = [
                    str(t) for t in (row_value(item, "tickers", None) or []) if str(t) in chunk
                ]
                for ticker in mapped_tickers:
                    canonical_id, resolved_ticker = self._resolve_ticker(ticker, published_at_ns)
                    if canonical_id is None or resolved_ticker is None:
                        continue
                    metadata = {
                        "author": row_value(item, "author"),
                        "url": row_value(item, "url"),
                        "channels": row_value(item, "channels"),
                        "tags": row_value(item, "tags"),
                        "model_name": sentiment.model_name,
                        "label": sentiment.label,
                    }
                    event_records.append(EventRecord(
                        canonical_id=canonical_id,
                        ticker=resolved_ticker,
                        published_at_ns=published_at_ns,
                        event_type="news",
                        source="POLYGON_BENZINGA_NEWS",
                        headline=title or f"{resolved_ticker} news item",
                        body=body,
                        sentiment_score=sentiment.score,
                        relevance=1.0,
                        novelty=1.0,
                        confidence=sentiment.confidence,
                        metadata=json.dumps(metadata, sort_keys=True),
                    ))
            self._add_event_records(event_records)

    def backfill_income_statements(
        self,
        tickers: Sequence[str],
        start_date: str,
        end_date: str,
        chunk_size: int = 25,
    ) -> None:
        logger.info("Backfilling Polygon income statements for %d tickers", len(tickers))
        for chunk in chunked(list(tickers), chunk_size):
            try:
                rows = self._fetch_cached_polygon_rows(
                    "polygon/financials_income_statements",
                    self._polygon_chunk_key(chunk, start_date, end_date),
                    lambda: self.client.list_financials_income_statements(
                        tickers_any_of=chunk,
                        filing_date_gte=start_date,
                        filing_date_lte=end_date,
                        limit=1000,
                        sort="filing_date",
                    ),
                )
            except Exception as exc:
                logger.warning("Polygon financials fetch failed for chunk %s: %s", chunk[:4], exc)
                continue
            fund_records: list[FundamentalRecord] = []
            for stmt in rows:
                ticker_list = row_value(stmt, "tickers", None) or []
                ticker = str(ticker_list[0]) if ticker_list else None
                filing_date = row_value(stmt, "filing_date")
                period_end = row_value(stmt, "period_end")
                if not ticker or not filing_date:
                    continue
                published_at_ns = parse_market_datetime_ns(filing_date)
                period_end_ns = parse_market_datetime_ns(period_end) or published_at_ns
                canonical_id, resolved_ticker = self._resolve_ticker(ticker, published_at_ns)
                if canonical_id is None or resolved_ticker is None or published_at_ns is None:
                    continue
                eps = safe_float(row_value(stmt, "diluted_earnings_per_share"))
                if eps is None:
                    eps = safe_float(row_value(stmt, "basic_earnings_per_share"))
                revenue = safe_float(row_value(stmt, "revenue"))
                if eps is not None:
                    fund_records.append(FundamentalRecord(
                        canonical_id=canonical_id,
                        metric_name=FundamentalsStore.METRIC_EPS,
                        value=eps,
                        published_at_ns=published_at_ns,
                        period_end_ns=period_end_ns,
                        source="POLYGON_FINANCIALS",
                    ))
                if revenue is not None:
                    fund_records.append(FundamentalRecord(
                        canonical_id=canonical_id,
                        metric_name=FundamentalsStore.METRIC_REVENUE,
                        value=revenue,
                        published_at_ns=published_at_ns,
                        period_end_ns=period_end_ns,
                        source="POLYGON_FINANCIALS",
                    ))
            self._add_fundamental_records(fund_records)

    def summary(self) -> dict[str, int]:
        return {
            "events_inserted": self.stats.events_inserted,
            "event_duplicates": self.stats.event_duplicates,
            "fundamentals_inserted": self.stats.fundamentals_inserted,
            "fundamental_duplicates": self.stats.fundamental_duplicates,
            "raw_cache_hits": self.stats.raw_cache_hits,
            "raw_cache_writes": self.stats.raw_cache_writes,
        }
