#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ingest.raw_cache import RawDataCache, to_jsonable


FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
DEFAULT_SERIES = "T10Y2Y,UNRATE,FEDFUNDS,DGS10,DGS2,CPIAUCSL"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill lightweight FRED macro cache for v8 overlay")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--api-key", default="")
    p.add_argument("--series", default=DEFAULT_SERIES, help="Comma-separated FRED series IDs")
    p.add_argument("--start-date", default="1990-01-01")
    p.add_argument("--end-date", default=date.today().isoformat())
    p.add_argument("--cache-dir", default="~/.alphaforge/cache/macro")
    p.add_argument("--merged-file", default="~/.alphaforge/cache/macro/fred_daily.parquet")
    p.add_argument("--raw-cache-dir", default="data/cache/pit")
    p.add_argument("--requests-per-minute", type=int, default=60)
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--cache-only", action="store_true")
    p.add_argument("--timeout-seconds", type=float, default=20.0)
    return p.parse_args()


def _parse_series(text: str) -> list[str]:
    return [s.strip().upper() for s in str(text).split(",") if s.strip()]


def _safe_float(text) -> float | None:
    if text is None:
        return None
    value = str(text).strip()
    if value in {"", ".", "nan", "None"}:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _fetch_series_payload(
    session: requests.Session,
    raw_cache: RawDataCache,
    *,
    api_key: str,
    series_id: str,
    start_date: str,
    end_date: str,
    timeout_seconds: float,
    refresh_cache: bool,
    cache_only: bool,
) -> tuple[dict, bool]:
    key = f"{series_id}:{start_date}:{end_date}"
    if not refresh_cache:
        cached = raw_cache.read_json("fred/json", key)
        if cached is not None:
            return dict(cached), True
    if cache_only:
        return {}, False

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
        "sort_order": "asc",
    }
    response = session.get(FRED_URL, params=params, timeout=float(timeout_seconds))
    response.raise_for_status()
    payload = response.json()
    raw_cache.write_json("fred/json", key, to_jsonable(payload))
    return payload, False


def _payload_to_series_df(payload: dict) -> pd.DataFrame:
    observations = list(payload.get("observations", []) or [])
    rows: list[tuple[pd.Timestamp, float]] = []
    for obs in observations:
        dt = pd.to_datetime(obs.get("date"), errors="coerce")
        val = _safe_float(obs.get("value"))
        if pd.isna(dt) or val is None:
            continue
        rows.append((pd.Timestamp(dt).normalize(), float(val)))
    if not rows:
        return pd.DataFrame(columns=["date", "value"])
    df = pd.DataFrame(rows, columns=["date", "value"]).drop_duplicates(subset=["date"], keep="last")
    return df.sort_values("date").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    api_key = str(args.api_key or os.getenv("FRED_API_KEY", "")).strip()
    if not api_key and not args.cache_only:
        raise RuntimeError("FRED_API_KEY not set. Add it to .env or pass --api-key.")
    if not api_key and args.cache_only:
        api_key = "cache-only"

    cache_dir = Path(os.path.expanduser(args.cache_dir)).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    merged_file = Path(os.path.expanduser(args.merged_file)).resolve()
    merged_file.parent.mkdir(parents=True, exist_ok=True)
    raw_cache = RawDataCache(args.raw_cache_dir)

    session = requests.Session()
    session.headers.update({"User-Agent": "one-brain-fund/1.0"})
    sleep_seconds = 60.0 / max(int(args.requests_per_minute), 1)

    stats = {
        "series_requested": 0,
        "series_written": 0,
        "raw_cache_hits": 0,
        "raw_cache_misses": 0,
        "raw_cache_writes": 0,
        "network_requests": 0,
        "errors": [],
    }
    frames: dict[str, pd.DataFrame] = {}
    for sid in _parse_series(args.series):
        stats["series_requested"] += 1
        try:
            payload, from_cache = _fetch_series_payload(
                session,
                raw_cache,
                api_key=api_key,
                series_id=sid,
                start_date=args.start_date,
                end_date=args.end_date,
                timeout_seconds=args.timeout_seconds,
                refresh_cache=bool(args.refresh_cache),
                cache_only=bool(args.cache_only),
            )
            if from_cache:
                stats["raw_cache_hits"] += 1
            else:
                stats["raw_cache_misses"] += 1
                if payload:
                    stats["raw_cache_writes"] += 1
                    stats["network_requests"] += 1
            df = _payload_to_series_df(payload)
            if df.empty:
                continue
            out_file = cache_dir / f"{sid}.parquet"
            df.to_parquet(out_file, index=False)
            stats["series_written"] += 1
            frames[sid] = df
            if not from_cache and not args.cache_only:
                time.sleep(sleep_seconds)
        except Exception as exc:
            stats["errors"].append(f"{sid}: {exc}")

    # Include any already-cached per-series parquet files not re-fetched this run.
    for p in sorted(cache_dir.glob("*.parquet")):
        sid = p.stem.upper()
        if sid in frames:
            continue
        try:
            df = pd.read_parquet(p)
            if not df.empty and {"date", "value"}.issubset(set(df.columns)):
                frames[sid] = df[["date", "value"]].copy()
        except Exception:
            continue

    if frames:
        min_dt = min(pd.to_datetime(df["date"]).min() for df in frames.values())
        max_dt = max(pd.to_datetime(df["date"]).max() for df in frames.values())
        daily_idx = pd.date_range(pd.Timestamp(min_dt).normalize(), pd.Timestamp(max_dt).normalize(), freq="D")
        merged = pd.DataFrame(index=daily_idx)
        for sid, df in sorted(frames.items()):
            s = (
                df.assign(date=pd.to_datetime(df["date"]))
                .set_index("date")["value"]
                .sort_index()
            )
            merged[sid] = s.reindex(daily_idx).ffill()
        merged.index.name = "date"
        merged.to_parquet(merged_file)
        stats["merged_file"] = str(merged_file)
        stats["merged_rows"] = int(len(merged))
        stats["merged_cols"] = int(merged.shape[1])
    else:
        stats["merged_file"] = str(merged_file)
        stats["merged_rows"] = 0
        stats["merged_cols"] = 0

    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
