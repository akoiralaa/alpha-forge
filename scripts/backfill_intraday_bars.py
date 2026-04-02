#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ingest.base import AssetClass
from src.data.ingest.daily_bar_cache import ASSET_CLASS_MAP, load_universe_static
from src.data.ingest.data_manager import build_data_manager_from_env


DEFAULT_CACHE_DIR = Path("data/cache/intraday")


def parse_args():
    p = argparse.ArgumentParser(description="Backfill and refresh local intraday bar parquet cache")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--config", default="config/sp500_universe.yaml")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--symbols", default="", help="Comma-separated subset")
    p.add_argument("--max-symbols", type=int, default=0)
    p.add_argument("--interval", default="1m", choices=["1m", "5m", "15m", "1hour"])
    p.add_argument("--days-back", type=int, default=45)
    p.add_argument("--stale-hours", type=int, default=8)
    p.add_argument("--incremental-buffer-hours", type=int, default=24)
    p.add_argument("--provider", default="", choices=["", "ibkr", "polygon", "alpaca"])
    p.add_argument("--full-refresh", action="store_true")
    p.add_argument("--cache-only", action="store_true")
    return p.parse_args()


def _to_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "timestamp_ns" not in out.columns:
        return pd.DataFrame()
    keep = [c for c in ["timestamp_ns", "open", "high", "low", "close", "volume", "vwap"] if c in out.columns]
    out = out[keep].copy()
    out["timestamp_ns"] = out["timestamp_ns"].astype("int64")
    for col in [c for c in ["open", "high", "low", "close", "volume", "vwap"] if c in out.columns]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["timestamp_ns", "close"])
    return out.sort_values("timestamp_ns").drop_duplicates(subset=["timestamp_ns"], keep="last").reset_index(drop=True)


def _latest_ts_ns(df: pd.DataFrame) -> int | None:
    if df is None or df.empty or "timestamp_ns" not in df.columns:
        return None
    return int(df["timestamp_ns"].iloc[-1])


def provider_fetch(dm, provider_name: str, symbol: str, asset_class: AssetClass, start_ns: int, end_ns: int, interval: str) -> pd.DataFrame:
    provider = dm._providers.get(provider_name)  # operational script, narrow usage
    if provider is None or not provider.is_connected:
        return pd.DataFrame()
    try:
        return _to_frame(provider.get_historical_bars(symbol, asset_class, start_ns, end_ns, bar_size=interval))
    except Exception:
        return pd.DataFrame()


def fetch_intraday(dm, symbol: str, asset_class: AssetClass, start_ns: int, end_ns: int, interval: str, provider_name: str) -> tuple[pd.DataFrame, str]:
    if provider_name:
        return provider_fetch(dm, provider_name, symbol, asset_class, start_ns, end_ns, interval), provider_name
    try:
        df = dm.get_historical_bars(symbol, asset_class, start_ns, end_ns, bar_size=interval)
    except Exception:
        return pd.DataFrame(), ""
    return _to_frame(df), "auto"


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
        raise RuntimeError("No instruments resolved for intraday backfill")

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(tz=timezone.utc)
    end_ns = int(now.timestamp() * 1_000_000_000)
    stale_ns = int((now - timedelta(hours=max(args.stale_hours, 0))).timestamp() * 1_000_000_000)
    full_start_ns = int((now - timedelta(days=max(args.days_back, 1))).timestamp() * 1_000_000_000)
    buffer_ns = int(max(args.incremental_buffer_hours, 1) * 3600 * 1_000_000_000)

    dm = build_data_manager_from_env()
    if not args.cache_only:
        results = dm.connect_all()
        connected = [name for name, ok in results.items() if ok]
        logging.info("Connected providers: %s", ", ".join(connected) if connected else "none")

    stats = {"written": 0, "skipped": 0, "failed": 0}
    try:
        for idx, (symbol, asset_type) in enumerate(universe, 1):
            asset_class = ASSET_CLASS_MAP.get(asset_type)
            if asset_class is None:
                logging.warning("[%03d/%03d] %s (%s) unknown asset class", idx, len(universe), symbol, asset_type)
                stats["failed"] += 1
                continue
            cache_path = cache_dir / f"{symbol}_{asset_type}_{args.interval}.parquet"
            meta_path = cache_dir / f"{symbol}_{asset_type}_{args.interval}.meta.json"
            logging.info("[%03d/%03d] %s (%s)", idx, len(universe), symbol, asset_type)

            existing = pd.DataFrame()
            if cache_path.exists():
                try:
                    existing = _to_frame(pd.read_parquet(cache_path))
                except Exception as exc:
                    logging.warning("  cache read failed: %s", exc)
                    existing = pd.DataFrame()

            if args.cache_only:
                if existing.empty:
                    logging.warning("  cache miss")
                    stats["failed"] += 1
                else:
                    logging.info("  cache ok: %d rows", len(existing))
                    stats["skipped"] += 1
                continue

            if (not args.full_refresh) and (not existing.empty):
                last_ns = _latest_ts_ns(existing)
                if last_ns is not None and last_ns >= stale_ns:
                    logging.info("  skip fresh cache (%d rows)", len(existing))
                    stats["skipped"] += 1
                    continue
                start_ns = max((last_ns or full_start_ns) - buffer_ns, full_start_ns)
            else:
                start_ns = full_start_ns

            incoming, provider_used = fetch_intraday(
                dm,
                symbol=symbol,
                asset_class=asset_class,
                start_ns=start_ns,
                end_ns=end_ns,
                interval=args.interval,
                provider_name=args.provider,
            )
            if incoming.empty:
                logging.warning("  no data returned")
                stats["failed"] += 1
                continue

            merged = pd.concat([existing, incoming], ignore_index=True) if not existing.empty else incoming
            merged = _to_frame(merged)
            merged.to_parquet(cache_path, index=False)

            start_ts = pd.to_datetime(int(merged["timestamp_ns"].iloc[0]), unit="ns", utc=True).isoformat()
            end_ts = pd.to_datetime(int(merged["timestamp_ns"].iloc[-1]), unit="ns", utc=True).isoformat()
            payload = {
                "symbol": symbol,
                "asset_type": asset_type,
                "interval": args.interval,
                "rows": int(len(merged)),
                "start_ts": start_ts,
                "end_ts": end_ts,
                "provider": provider_used,
                "updated_at": datetime.now(tz=timezone.utc).isoformat(),
            }
            meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            logging.info("  wrote %d rows (%s -> %s)", len(merged), start_ts, end_ts)
            stats["written"] += 1
    finally:
        dm.disconnect_all()

    print(stats)


if __name__ == "__main__":
    main()
