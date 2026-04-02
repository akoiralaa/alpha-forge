from __future__ import annotations

import json
import logging
import time
import gzip
import zlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from src.data.events import EventRecord, EventStore
from src.data.fundamentals import FundamentalRecord, FundamentalsStore
from src.data.ingest.polygon_event_backfill import parse_market_datetime_ns
from src.data.ingest.raw_cache import RawDataCache
from src.data.symbol_master import SymbolMaster
from src.signals.sentiment import build_sentiment_model

logger = logging.getLogger(__name__)

SEC_BASE = "https://data.sec.gov"
TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"

EPS_TAGS = [
    "EarningsPerShareDiluted",
    "EarningsPerShareBasicAndDiluted",
    "EarningsPerShareBasic",
]
REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "Revenues",
]
QUARTERLY_FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A", "20-F", "20-F/A"}
SUBMISSION_EVENT_FORMS = {
    "8-K", "8-K/A", "10-Q", "10-Q/A", "10-K", "10-K/A", "6-K", "6-K/A", "20-F", "20-F/A",
}

POSITIVE_ITEM_HINTS = {"2.02", "8.01"}
NEGATIVE_ITEM_HINTS = {"2.04", "2.05", "3.01", "4.02"}


@dataclass(slots=True)
class SecBackfillStats:
    events_inserted: int = 0
    event_duplicates: int = 0
    fundamentals_inserted: int = 0
    fundamental_duplicates: int = 0
    raw_cache_hits: int = 0
    raw_cache_writes: int = 0


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _period_end_ns(value: str | None, published_at_ns: int) -> int:
    return parse_market_datetime_ns(value) or published_at_ns


def _chunked(values: Sequence[str], size: int) -> Iterable[list[str]]:
    size = max(int(size), 1)
    for i in range(0, len(values), size):
        yield list(values[i:i + size])


def _choose_unit_block(facts: dict, tags: list[str], units: list[str]) -> list[dict]:
    for tag in tags:
        node = ((facts.get("us-gaap") or {}).get(tag) or {}).get("units") or {}
        for unit in units:
            if unit in node:
                return list(node[unit] or [])
    return []


def _recent_to_rows(recent: dict | None) -> list[dict]:
    recent = recent or {}
    if not recent:
        return []
    keys = [str(k) for k in recent.keys()]
    if not keys:
        return []
    size = max((len(recent.get(k) or []) for k in keys), default=0)
    rows: list[dict] = []
    for idx in range(size):
        row = {}
        for key in keys:
            values = recent.get(key) or []
            row[key] = values[idx] if idx < len(values) else None
        rows.append(row)
    return rows


def _normalize_item_tokens(raw_items: str | None) -> list[str]:
    if not raw_items:
        return []
    text = str(raw_items).replace("Item", "").replace("item", "")
    tokens = []
    for token in text.replace(",", " ").split():
        token = token.strip().strip(".")
        if token:
            tokens.append(token)
    return tokens


def _sec_event_type(form: str, items: list[str], headline: str) -> str:
    form = str(form or "").upper()
    headline_l = headline.lower()
    item_set = set(items)
    if form in {"10-Q", "10-K", "10-Q/A", "10-K/A", "20-F", "20-F/A"}:
        return "earnings"
    if "2.02" in item_set:
        return "earnings"
    if "8.01" in item_set:
        if any(term in headline_l for term in ("guidance", "outlook", "forecast")):
            return "guidance"
        return "news"
    if any(item in item_set for item in ("1.01", "2.01")):
        return "m&a"
    if any(item in item_set for item in ("7.01", "9.01")):
        return "news"
    return "news"


def _sec_filing_score(form: str, items: list[str], headline: str) -> tuple[float, float]:
    headline_l = headline.lower()
    score = 0.0
    if any(item in POSITIVE_ITEM_HINTS for item in items):
        score += 0.15
    if any(item in NEGATIVE_ITEM_HINTS for item in items):
        score -= 0.20
    if "guidance" in headline_l or "outlook" in headline_l:
        score += 0.15
    if "earnings" in headline_l or "results" in headline_l:
        score += 0.10
    if "investigation" in headline_l or "restatement" in headline_l:
        score -= 0.25
    if form.upper().startswith("8-K"):
        confidence = 0.55
    elif form.upper().startswith("10-") or form.upper().startswith("20-"):
        confidence = 0.65
    else:
        confidence = 0.45
    return float(np.clip(score, -0.35, 0.35)), confidence


class SecCompanyFactsBackfiller:
    def __init__(
        self,
        user_agent: str,
        symbol_master_path: str,
        fundamentals_path: str,
        events_path: str,
        request_sleep: float = 0.2,
        cache_dir: str | Path = "data/cache/pit",
        refresh_cache: bool = False,
        cache_only: bool = False,
        prefer_hf: bool = False,
        hf_model_name: str = "ProsusAI/finbert",
    ):
        self.user_agent = user_agent
        self.request_sleep = float(max(request_sleep, 0.05))
        self.symbol_master_path = str(symbol_master_path)
        self.fundamentals_path = str(fundamentals_path)
        self.events_path = str(events_path)
        self.sm = SymbolMaster(symbol_master_path)
        self.fund = FundamentalsStore(fundamentals_path)
        self.events = EventStore(events_path)
        self.stats = SecBackfillStats()
        self.raw_cache = RawDataCache(cache_dir)
        self.refresh_cache = bool(refresh_cache)
        self.cache_only = bool(cache_only)
        self.sentiment_model = build_sentiment_model(prefer_hf=prefer_hf, model_name=hf_model_name)
        self._existing_event_keys = self._load_existing_event_keys()
        self._existing_fund_keys = self._load_existing_fund_keys()
        self._ticker_map = self._fetch_ticker_cik_map()

    def close(self):
        self.sm.close()
        self.fund.close()
        self.events.close()

    def _fetch_json_from_source(self, url: str) -> dict:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json,text/json,*/*",
            "Accept-Encoding": "gzip, deflate",
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = resp.read()
            encoding = (resp.headers.get("Content-Encoding") or "").lower()
        if encoding == "gzip" or payload[:2] == b"\x1f\x8b":
            payload = gzip.decompress(payload)
        elif encoding == "deflate":
            payload = zlib.decompress(payload)
        time.sleep(self.request_sleep)
        return json.loads(payload.decode("utf-8"))

    def _fetch_json(self, url: str, cache_key: str | None = None) -> dict:
        namespace = "sec/http_json"
        cache_key = cache_key or url
        payload, cache_hit = self.raw_cache.get_or_fetch_json(
            namespace,
            cache_key,
            lambda: self._fetch_json_from_source(url),
            refresh=self.refresh_cache,
            cache_only=self.cache_only,
        )
        if cache_hit:
            self.stats.raw_cache_hits += 1
        else:
            self.stats.raw_cache_writes += 1
        return dict(payload)

    def _fetch_ticker_cik_map(self) -> dict[str, str]:
        raw = self._fetch_json(TICKER_CIK_URL, cache_key="ticker_cik_map")
        out: dict[str, str] = {}
        for value in raw.values():
            ticker = str(value.get("ticker", "")).upper()
            cik = str(value.get("cik_str", "")).zfill(10)
            if ticker and cik:
                out[ticker] = cik
        return out

    def _load_existing_event_keys(self) -> set[tuple]:
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

    def _iter_submission_rows(self, ticker: str, cik: str) -> list[dict]:
        submissions = self._fetch_json(
            f"{SEC_BASE}/submissions/CIK{cik}.json",
            cache_key=f"submissions_{ticker.upper()}_{cik}",
        )
        filings = submissions.get("filings") or {}
        rows = _recent_to_rows(filings.get("recent") or {})
        for extra in filings.get("files") or []:
            name = str(extra.get("name", "")).strip()
            if not name:
                continue
            try:
                older = self._fetch_json(
                    f"{SEC_BASE}/submissions/{name}",
                    cache_key=f"submissions_file_{name}",
                )
            except Exception as exc:
                logger.warning("SEC submissions file fetch failed for %s/%s: %s", ticker, name, exc)
                continue
            rows.extend(_recent_to_rows(older))
        return rows

    def _build_submission_events(
        self,
        ticker: str,
        canonical_id: int,
        cik: str,
    ) -> list[EventRecord]:
        try:
            rows = self._iter_submission_rows(ticker, cik)
        except Exception as exc:
            logger.warning("SEC submissions fetch failed for %s: %s", ticker, exc)
            return []

        event_records: list[EventRecord] = []
        seen: set[tuple[int, str, str]] = set()
        for row in rows:
            form = str(row.get("form", "") or "").upper()
            if form not in SUBMISSION_EVENT_FORMS:
                continue

            filed = row.get("filingDate") or row.get("reportDate")
            published_at_ns = parse_market_datetime_ns(filed)
            if published_at_ns is None:
                continue

            items = _normalize_item_tokens(row.get("items"))
            primary_desc = str(row.get("primaryDocDescription", "") or "").strip()
            primary_doc = str(row.get("primaryDocument", "") or "").strip()
            accession = str(row.get("accessionNumber", "") or "").strip()
            headline = primary_desc or f"{ticker} filed {form}"
            if primary_doc and primary_doc.lower() not in headline.lower():
                headline = f"{headline} ({primary_doc})"

            dedupe_key = (published_at_ns, form, headline)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            heuristic_score, heuristic_conf = _sec_filing_score(form, items, headline)
            sentiment = self.sentiment_model.score_text(f"{headline}. {' '.join(items)}")
            score = float(np.clip(0.35 * heuristic_score + 0.65 * sentiment.score, -1.0, 1.0))
            confidence = float(np.clip(max(heuristic_conf, sentiment.confidence), 0.15, 0.95))
            event_type = _sec_event_type(form, items, headline)
            event_records.append(EventRecord(
                canonical_id=canonical_id,
                ticker=ticker,
                published_at_ns=published_at_ns,
                event_type=event_type,
                source="SEC_SUBMISSIONS",
                headline=headline,
                body=f"SEC {form} filing {accession}".strip(),
                sentiment_score=score,
                relevance=1.05 if form.startswith("8-K") else 0.90,
                novelty=1.00 if form.startswith("8-K") else 0.85,
                confidence=confidence,
                metadata=json.dumps({
                    "form": form,
                    "items": items,
                    "accession_number": accession,
                    "primary_document": primary_doc,
                    "filing_date": filed,
                    "source": "submissions",
                    "model_name": sentiment.model_name,
                    "label": sentiment.label,
                }, sort_keys=True),
            ))
        return event_records

    def backfill(self, tickers: Sequence[str]) -> None:
        logger.info("Backfilling SEC company facts for %d tickers", len(tickers))
        for chunk in _chunked(list(tickers), 25):
            logger.info("  sec chunk: %s", ",".join(chunk[:4]) + ("..." if len(chunk) > 4 else ""))
            for ticker in chunk:
                cik = self._ticker_map.get(ticker.upper())
                if not cik:
                    continue
                try:
                    facts = self._fetch_json(
                        f"{SEC_BASE}/api/xbrl/companyfacts/CIK{cik}.json",
                        cache_key=f"companyfacts_{ticker.upper()}_{cik}",
                    )
                except Exception as exc:
                    logger.warning("SEC facts fetch failed for %s: %s", ticker, exc)
                    continue

                eps_rows = _choose_unit_block(facts.get("facts") or {}, EPS_TAGS, ["USD/shares", "USD/share"])
                rev_rows = _choose_unit_block(facts.get("facts") or {}, REVENUE_TAGS, ["USD"])
                if not eps_rows and not rev_rows:
                    continue

                sample_rows = eps_rows or rev_rows
                form_filtered = [row for row in sample_rows if str(row.get("form", "")).upper() in QUARTERLY_FORMS]
                sample_rows = form_filtered or sample_rows

                published_pairs: set[tuple[int, int]] = set()
                event_records: list[EventRecord] = []
                fund_records: list[FundamentalRecord] = []

                def _resolve() -> tuple[int | None, str | None]:
                    for row in sample_rows:
                        published_at_ns = parse_market_datetime_ns(row.get("filed"))
                        if published_at_ns is None:
                            continue
                        cid = self.sm.resolve_ticker(ticker, published_at_ns)
                        if cid is not None:
                            return cid, ticker
                    return None, None

                canonical_id, resolved_ticker = _resolve()
                if canonical_id is None or resolved_ticker is None:
                    continue

                for row in eps_rows:
                    if str(row.get("form", "")).upper() not in QUARTERLY_FORMS:
                        continue
                    value = _safe_float(row.get("val"))
                    published_at_ns = parse_market_datetime_ns(row.get("filed"))
                    if value is None or published_at_ns is None:
                        continue
                    period_end_ns = _period_end_ns(row.get("end"), published_at_ns)
                    fund_records.append(FundamentalRecord(
                        canonical_id=canonical_id,
                        metric_name=FundamentalsStore.METRIC_EPS,
                        value=value,
                        published_at_ns=published_at_ns,
                        period_end_ns=period_end_ns,
                        source="SEC_COMPANYFACTS",
                    ))
                    published_pairs.add((published_at_ns, period_end_ns))

                for row in rev_rows:
                    if str(row.get("form", "")).upper() not in QUARTERLY_FORMS:
                        continue
                    value = _safe_float(row.get("val"))
                    published_at_ns = parse_market_datetime_ns(row.get("filed"))
                    if value is None or published_at_ns is None:
                        continue
                    period_end_ns = _period_end_ns(row.get("end"), published_at_ns)
                    fund_records.append(FundamentalRecord(
                        canonical_id=canonical_id,
                        metric_name=FundamentalsStore.METRIC_REVENUE,
                        value=value,
                        published_at_ns=published_at_ns,
                        period_end_ns=period_end_ns,
                        source="SEC_COMPANYFACTS",
                    ))
                    published_pairs.add((published_at_ns, period_end_ns))

                for published_at_ns, period_end_ns in sorted(published_pairs):
                    event_records.append(EventRecord(
                        canonical_id=canonical_id,
                        ticker=resolved_ticker,
                        published_at_ns=published_at_ns,
                        event_type="earnings",
                        source="SEC_COMPANYFACTS",
                        headline=f"{resolved_ticker} SEC earnings filing",
                        body="SEC company facts filing event",
                        sentiment_score=0.0,
                        relevance=0.85,
                        novelty=0.80,
                        confidence=0.60,
                        metadata=json.dumps({
                            "period_end_ns": period_end_ns,
                            "source": "companyfacts",
                        }, sort_keys=True),
                    ))

                self._add_fundamental_records(fund_records)
                self._add_event_records(event_records)
                self._add_event_records(self._build_submission_events(resolved_ticker, canonical_id, cik))

    def summary(self) -> dict[str, int]:
        return {
            "events_inserted": self.stats.events_inserted,
            "event_duplicates": self.stats.event_duplicates,
            "fundamentals_inserted": self.stats.fundamentals_inserted,
            "fundamental_duplicates": self.stats.fundamental_duplicates,
            "raw_cache_hits": self.stats.raw_cache_hits,
            "raw_cache_writes": self.stats.raw_cache_writes,
        }
