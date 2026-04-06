from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from src.data.events import EventRecord, EventStore
from src.data.ingest.raw_cache import RawDataCache, to_jsonable
from src.signals.sentiment import build_sentiment_model

UTC = timezone.utc


def _parse_since_ns(value: str | None) -> int | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    ts = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return int(ts.timestamp() * 1_000_000_000)


@dataclass(slots=True)
class FinBertBackfillStats:
    processed: int = 0
    skipped: int = 0
    inserted: int = 0
    duplicates: int = 0
    raw_cache_hits: int = 0
    raw_cache_misses: int = 0
    raw_cache_writes: int = 0


class FinBertEventBackfiller:
    ENRICHED_SOURCE = "FINBERT_ENRICHED"

    def __init__(
        self,
        events_path: str,
        *,
        cache_dir: str | Path = "data/cache/pit",
        model_name: str = "ProsusAI/finbert",
        prefer_hf: bool = True,
        allow_lexicon_fallback: bool = False,
        refresh_cache: bool = False,
        cache_only: bool = False,
    ):
        self.events_path = str(events_path)
        self.events = EventStore(events_path)
        self.raw_cache = RawDataCache(cache_dir)
        self.refresh_cache = bool(refresh_cache)
        self.cache_only = bool(cache_only)
        self.stats = FinBertBackfillStats()
        self.model = build_sentiment_model(prefer_hf=prefer_hf, model_name=model_name)
        if not allow_lexicon_fallback and getattr(self.model, "model_name", "") != model_name:
            raise RuntimeError(
                f"FinBERT model '{model_name}' unavailable; resolved '{getattr(self.model, 'model_name', 'unknown')}'. "
                "Install transformers+torch or pass --allow-lexicon-fallback."
            )
        self._existing_keys = self._load_existing_keys()

    def close(self) -> None:
        self.events.close()

    def _load_existing_keys(self) -> set[tuple[int, int, str, str, str]]:
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
            if str(row.source) == self.ENRICHED_SOURCE
        }

    def _cache_key(self, row) -> str:
        return (
            f"{int(row.canonical_id)}:"
            f"{int(row.published_at_ns)}:"
            f"{str(row.ticker).upper()}:"
            f"{str(row.event_type).lower()}:"
            f"{str(row.headline).strip()[:200]}"
        )

    def _score_row(self, row) -> dict | None:
        namespace = "finbert/scores"
        key = self._cache_key(row)
        cached = None if self.refresh_cache else self.raw_cache.read_json(namespace, key)
        if cached is not None:
            self.stats.raw_cache_hits += 1
            return cached

        self.stats.raw_cache_misses += 1
        if self.cache_only:
            return None

        text = f"{str(row.headline or '').strip()}. {str(row.body or '').strip()}".strip()
        if not text or text == ".":
            return None
        score = self.model.score_text(text[:2048])
        payload = {
            "score": float(score.score),
            "confidence": float(score.confidence),
            "label": str(score.label),
            "model_name": str(score.model_name),
        }
        self.raw_cache.write_json(namespace, key, to_jsonable(payload))
        self.stats.raw_cache_writes += 1
        return payload

    def backfill(
        self,
        *,
        source_filter: Sequence[str] | None = None,
        event_types: Sequence[str] | None = None,
        since: str | None = None,
        limit: int = 0,
        min_confidence: float = 0.05,
    ) -> None:
        df = self.events.to_dataframe()
        if df.empty:
            return

        filtered = df.copy()
        filtered = filtered[filtered["source"] != self.ENRICHED_SOURCE]
        filtered = filtered[filtered["headline"].fillna("").str.len() > 0]
        since_ns = _parse_since_ns(since)
        if since_ns is not None:
            filtered = filtered[filtered["published_at_ns"] >= since_ns]
        if source_filter:
            src_set = {s.strip() for s in source_filter if str(s).strip()}
            filtered = filtered[filtered["source"].isin(src_set)]
        if event_types:
            evt_set = {s.strip().lower() for s in event_types if str(s).strip()}
            filtered = filtered[filtered["event_type"].str.lower().isin(evt_set)]
        filtered = filtered.sort_values("published_at_ns")
        if limit and limit > 0:
            filtered = filtered.tail(int(limit))
        if filtered.empty:
            return

        out: list[EventRecord] = []
        for row in filtered.itertuples(index=False):
            self.stats.processed += 1
            score_payload = self._score_row(row)
            if score_payload is None:
                self.stats.skipped += 1
                continue

            confidence = float(score_payload.get("confidence", 0.0))
            if confidence < float(min_confidence):
                self.stats.skipped += 1
                continue

            key = (
                int(row.canonical_id),
                int(row.published_at_ns),
                str(row.event_type),
                self.ENRICHED_SOURCE,
                str(row.headline),
            )
            if key in self._existing_keys:
                self.stats.duplicates += 1
                continue
            self._existing_keys.add(key)

            metadata = {
                "orig_source": str(row.source),
                "orig_confidence": float(getattr(row, "confidence", 0.0) or 0.0),
                "model_name": str(score_payload.get("model_name", "")),
                "model_label": str(score_payload.get("label", "")),
            }
            out.append(
                EventRecord(
                    canonical_id=int(row.canonical_id),
                    ticker=str(row.ticker),
                    published_at_ns=int(row.published_at_ns),
                    event_type=str(row.event_type),
                    source=self.ENRICHED_SOURCE,
                    headline=str(row.headline),
                    body=str(row.body or ""),
                    sentiment_score=float(score_payload.get("score", 0.0)),
                    relevance=float(getattr(row, "relevance", 1.0) or 1.0),
                    novelty=float(getattr(row, "novelty", 1.0) or 1.0),
                    confidence=confidence,
                    metadata=json.dumps(metadata, sort_keys=True),
                )
            )

        if out:
            self.events.add_records_batch(out)
            self.stats.inserted += len(out)

    def summary(self) -> dict[str, int | str]:
        return {
            "model": str(getattr(self.model, "model_name", "unknown")),
            "processed": self.stats.processed,
            "skipped": self.stats.skipped,
            "inserted": self.stats.inserted,
            "duplicates": self.stats.duplicates,
            "raw_cache_hits": self.stats.raw_cache_hits,
            "raw_cache_misses": self.stats.raw_cache_misses,
            "raw_cache_writes": self.stats.raw_cache_writes,
        }

