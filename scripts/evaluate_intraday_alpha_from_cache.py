#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paper.engine import PaperConfig, PaperTick, PaperTradingEngine
from src.signals.intraday_alpha import IntradayAlphaSleeve


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate intraday alpha sleeve from cached intraday bars")
    p.add_argument("--cache-dir", default="data/cache/intraday")
    p.add_argument("--symbols", default="QQQ,AAPL")
    p.add_argument("--interval", default="5m")
    p.add_argument("--max-rows-per-symbol", type=int, default=30_000)
    return p.parse_args()


def find_cache_file(cache_dir: Path, symbol: str, interval: str) -> Path | None:
    candidates = sorted(cache_dir.glob(f"{symbol}_*_{interval}.parquet"))
    if candidates:
        return candidates[0]
    candidates = sorted(cache_dir.glob(f"{symbol}_{interval}.parquet"))
    if candidates:
        return candidates[0]
    return None


def load_ticks_for_symbol(path: Path, symbol_id: int, max_rows: int) -> list[PaperTick]:
    df = pd.read_parquet(path)
    if "timestamp_ns" not in df.columns or "close" not in df.columns:
        return []
    df = df.sort_values("timestamp_ns").tail(max_rows)
    if "volume" not in df.columns:
        df["volume"] = 0
    ticks: list[PaperTick] = []
    for row in df.itertuples():
        price = float(row.close)
        if not math.isfinite(price) or price <= 0:
            continue
        volume = int(max(float(getattr(row, "volume", 0.0) or 0.0), 0.0))
        ticks.append(
            PaperTick(
                symbol_id=symbol_id,
                price=price,
                volume=volume,
                timestamp_ns=int(row.timestamp_ns),
            )
        )
    return ticks


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise RuntimeError("No symbols provided")

    symbol_map: dict[int, str] = {}
    all_ticks: list[PaperTick] = []
    for idx, symbol in enumerate(symbols, start=1):
        path = find_cache_file(cache_dir, symbol, args.interval)
        if path is None:
            raise FileNotFoundError(f"No intraday cache found for {symbol} in {cache_dir}")
        symbol_map[idx] = symbol
        all_ticks.extend(load_ticks_for_symbol(path, idx, args.max_rows_per_symbol))

    if not all_ticks:
        raise RuntimeError("No ticks loaded from cache")
    all_ticks.sort(key=lambda t: (t.timestamp_ns, t.symbol_id))

    engine = PaperTradingEngine(
        PaperConfig(
            signal_threshold=0.03,
            reconciliation_interval_ticks=500,
            risk_budget_per_position=0.0005,
            kelly_fraction=0.05,
            max_position_pct_nav=0.10,
        )
    )
    sleeve = IntradayAlphaSleeve(symbol_map=symbol_map)
    engine.set_signal_function(sleeve)
    stats = engine.run_session(all_ticks)

    t0 = min(t.timestamp_ns for t in all_ticks)
    t1 = max(t.timestamp_ns for t in all_ticks)
    years = max((t1 - t0) / 1_000_000_000 / 86400.0 / 365.25, 1e-6)
    gross_return = (stats.final_nav / engine.config.initial_nav) - 1.0
    cagr = (stats.final_nav / engine.config.initial_nav) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    print(f"Symbols: {','.join(symbols)} | Interval: {args.interval} | Ticks: {len(all_ticks):,}")
    print(f"Window: {pd.to_datetime(t0, unit='ns', utc=True)} -> {pd.to_datetime(t1, unit='ns', utc=True)} ({years:.2f} years)")
    print(f"Orders submitted/filled: {stats.orders_submitted}/{stats.orders_filled}")
    print(f"Final NAV: ${stats.final_nav:,.2f} | PnL: ${stats.total_pnl:,.2f}")
    print(f"Gross return: {gross_return:+.2%} | CAGR (window-scaled): {cagr:+.2%}")
    print(f"Max DD: {-stats.max_drawdown_pct:.2%}")


if __name__ == "__main__":
    main()
