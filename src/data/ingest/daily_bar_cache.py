from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yaml

from src.data.ingest.base import AssetClass


ASSET_CLASS_MAP = {
    "ETF": AssetClass.ETF,
    "EQUITY": AssetClass.EQUITY,
    "FUTURE": AssetClass.FUTURE,
    "COMMODITY": AssetClass.COMMODITY,
    "BOND": AssetClass.BOND,
    "FX": AssetClass.FX,
    "VOLATILITY": AssetClass.VOLATILITY,
}

UNIVERSE_KEYS = [
    ("sector_etfs", "ETF"),
    ("equities", "EQUITY"),
    ("equity_index_futures", "FUTURE"),
    ("commodity_futures", "COMMODITY"),
    ("fixed_income_futures", "BOND"),
    ("fx_pairs", "FX"),
    ("vix_futures", "VOLATILITY"),
]


def load_universe_static(config_path: str | Path) -> list[tuple[str, str]]:
    with Path(config_path).open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    instruments = cfg.get("instruments", {}) or {}
    universe: list[tuple[str, str]] = []
    for key, asset_type in UNIVERSE_KEYS:
        for symbol in instruments.get(key, []) or []:
            universe.append((str(symbol), asset_type))
    return universe


def merge_bar_frames(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    frames = [df.copy() for df in (existing, incoming) if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    merged = pd.concat(frames, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"], utc=True)
    merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return merged.reset_index(drop=True)


def cache_end_date(df: pd.DataFrame) -> pd.Timestamp | None:
    if df is None or df.empty or "date" not in df.columns:
        return None
    dates = pd.to_datetime(df["date"], utc=True)
    if len(dates) == 0:
        return None
    return pd.Timestamp(dates.iloc[-1]).tz_convert("UTC").normalize()


def cache_is_stale(df: pd.DataFrame, *, stale_days: int, as_of: pd.Timestamp | None = None) -> bool:
    end_date = cache_end_date(df)
    if end_date is None:
        return True
    as_of = (as_of or pd.Timestamp.utcnow()).tz_convert("UTC").normalize()
    return end_date < (as_of - pd.Timedelta(days=max(int(stale_days), 0)))


def write_cache_metadata(
    metadata_path: str | Path,
    *,
    symbol: str,
    asset_type: str,
    provider: str,
    rows: int,
    start_date: str | None,
    end_date: str | None,
) -> None:
    payload = {
        "symbol": symbol,
        "asset_type": asset_type,
        "provider": provider,
        "rows": int(rows),
        "start_date": start_date,
        "end_date": end_date,
        "updated_at": pd.Timestamp.utcnow().isoformat(),
    }
    path = Path(metadata_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "ASSET_CLASS_MAP",
    "load_universe_static",
    "merge_bar_frames",
    "cache_end_date",
    "cache_is_stale",
    "write_cache_metadata",
]
