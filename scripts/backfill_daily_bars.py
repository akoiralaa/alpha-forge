#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ingest.base import AssetClass
from src.data.ingest.daily_bar_cache import (
    ASSET_CLASS_MAP,
    cache_end_date,
    cache_is_stale,
    load_universe_static,
    merge_bar_frames,
    write_cache_metadata,
)
from src.data.ingest.data_manager import build_data_manager_from_env


DEFAULT_CACHE_DIR = Path("~/.one_brain_fund/cache/bars").expanduser()


def parse_args():
    p = argparse.ArgumentParser(description="Backfill and refresh local daily bar parquet cache")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--config", default="config/sp500_universe.yaml")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--stale-days", type=int, default=1)
    p.add_argument("--incremental-buffer-days", type=int, default=10)
    p.add_argument("--full-refresh", action="store_true")
    p.add_argument("--cache-only", action="store_true")
    p.add_argument("--symbols", default="", help="Comma-separated subset")
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--provider", default="", choices=["", "ibkr", "polygon", "alpaca"])
    return p.parse_args()


def provider_fetch(dm, provider_name: str, symbol: str, asset_class: AssetClass, start_ns: int, end_ns: int) -> pd.DataFrame:
    provider = dm._providers.get(provider_name)  # thin operational script; uses existing routing object
    if provider is None or not provider.is_connected:
        return pd.DataFrame()
    if start_ns > 0:
        df = provider.get_historical_bars(symbol, asset_class, start_ns, end_ns, bar_size="1day")
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["timestamp_ns"], unit="ns", utc=True)
        df = df[["date", "open", "high", "low", "close", "volume"]].copy()
        df.loc[df["volume"] <= 0, "volume"] = 100
    else:
        df = dm._fetch_daily_from_provider(provider, provider_name, symbol, asset_class)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], utc=True)
    if start_ns:
        start_dt = pd.to_datetime(start_ns, unit="ns", utc=True)
        df = df[df["date"] >= start_dt]
    if end_ns:
        end_dt = pd.to_datetime(end_ns, unit="ns", utc=True)
        df = df[df["date"] <= end_dt]
    return df.reset_index(drop=True)


def fetch_incremental(dm, symbol: str, asset_class: AssetClass, provider_name: str, start_ns: int, end_ns: int) -> tuple[pd.DataFrame, str]:
    if provider_name:
        return provider_fetch(dm, provider_name, symbol, asset_class, start_ns, end_ns), provider_name
    df = dm.get_historical_bars(symbol, asset_class, start_ns, end_ns, bar_size="1day")
    if df.empty:
        return pd.DataFrame(), ""
    df["date"] = pd.to_datetime(df["timestamp_ns"], unit="ns", utc=True)
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df.loc[df["volume"] <= 0, "volume"] = 100
    return df.reset_index(drop=True), "incremental"


def main():
    args = parse_args()
    load_dotenv(args.env_file)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    universe = load_universe_static(args.config)
    if args.symbols.strip():
        selected = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        universe = [(sym, at) for sym, at in universe if sym.upper() in selected]
    if args.max_symbols > 0:
        universe = universe[: args.max_symbols]
    if not universe:
        raise RuntimeError("No instruments resolved for daily bar backfill")

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.now(timezone.utc)
    end_ns = int(now_utc.timestamp() * 1_000_000_000)
    stale_threshold = pd.Timestamp(now_utc).tz_convert("UTC").normalize()

    dm = build_data_manager_from_env()
    if not args.cache_only:
        results = dm.connect_all()
        connected = [name for name, ok in results.items() if ok]
        logging.info("Connected providers: %s", ", ".join(connected) if connected else "none")

    stats = {"written": 0, "skipped": 0, "failed": 0}
    try:
        for idx, (symbol, asset_type) in enumerate(universe, 1):
            cache_path = cache_dir / f"{symbol}_{asset_type}.parquet"
            meta_path = cache_dir / f"{symbol}_{asset_type}.meta.json"
            logging.info("[%03d/%03d] %s (%s)", idx, len(universe), symbol, asset_type)

            existing = pd.DataFrame()
            if cache_path.exists():
                try:
                    existing = pd.read_parquet(cache_path)
                    existing["date"] = pd.to_datetime(existing["date"], utc=True)
                except Exception as exc:
                    logging.warning("  cache read failed for %s: %s", symbol, exc)
                    existing = pd.DataFrame()

            if args.cache_only:
                if existing.empty:
                    logging.warning("  cache miss")
                    stats["failed"] += 1
                else:
                    logging.info("  cache ok: %d rows", len(existing))
                    stats["skipped"] += 1
                continue

            asset_class = ASSET_CLASS_MAP[asset_type]
            source = args.provider or ""
            fresh_enough = (not args.full_refresh) and (not existing.empty) and (not cache_is_stale(existing, stale_days=args.stale_days, as_of=stale_threshold))
            if fresh_enough:
                logging.info("  skip fresh cache: %d rows through %s", len(existing), cache_end_date(existing).date())
                stats["skipped"] += 1
                continue

            if args.full_refresh or existing.empty:
                if source:
                    fetched = provider_fetch(dm, source, symbol, asset_class, 0, end_ns)
                    provider_used = source
                else:
                    fetched = dm.fetch_daily_bars(symbol, asset_class)
                    provider_used = "auto"
            else:
                last_cached = cache_end_date(existing)
                start_dt = (last_cached - pd.Timedelta(days=max(args.incremental_buffer_days, 1))).to_pydatetime()
                start_ns = int(start_dt.timestamp() * 1_000_000_000)
                fetched, provider_used = fetch_incremental(dm, symbol, asset_class, source, start_ns, end_ns)

            if fetched.empty:
                logging.warning("  no data returned")
                stats["failed"] += 1
                continue

            merged = merge_bar_frames(existing, fetched)
            merged.to_parquet(cache_path, index=False)
            start_date = str(pd.to_datetime(merged["date"], utc=True).iloc[0].date()) if len(merged) else None
            end_date = str(pd.to_datetime(merged["date"], utc=True).iloc[-1].date()) if len(merged) else None
            write_cache_metadata(
                meta_path,
                symbol=symbol,
                asset_type=asset_type,
                provider=provider_used,
                rows=len(merged),
                start_date=start_date,
                end_date=end_date,
            )
            logging.info("  wrote %d rows (%s -> %s)", len(merged), start_date, end_date)
            stats["written"] += 1
    finally:
        dm.disconnect_all()

    print(stats)


if __name__ == "__main__":
    main()
