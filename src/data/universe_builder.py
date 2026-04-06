"""
Dynamic universe builder for AlphaForge.

Pulls all active US equities from Polygon.io (primary) and/or Alpaca (fallback),
applies liquidity filters (ADV, price, market cap), and merges with static
futures/FX/commodities from the YAML config to produce the full tradeable universe.

Output format: list[tuple[str, str]] — (symbol, asset_type) — matching the
backtest's load_universe() contract.

Rate-limit aware: uses bulk endpoints (snapshots, list_tickers) to minimize
API calls. Results are cached locally with a configurable staleness window.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_CACHE_DIR = Path("~/.one_brain_fund/cache").expanduser()
DEFAULT_CACHE_FILE = "universe_cache.json"
DEFAULT_STALENESS_HOURS = 24

# Polygon free tier: 5 requests/minute
POLYGON_FREE_TIER_RPM = 5


@dataclass(slots=True)
class LiquidityFilter:
    """Thresholds for equity inclusion in the tradeable universe."""

    adv_min_usd: float = 10_000_000.0       # 20-day average daily volume in USD
    price_min: float = 5.0                    # minimum last price
    market_cap_min: float = 500_000_000.0     # minimum market capitalization


@dataclass(slots=True)
class UniverseBuilderConfig:
    """All knobs for the universe builder."""

    polygon_api_key: str = ""
    alpaca_api_key: str = ""
    alpaca_api_secret: str = ""
    polygon_rate_limit_rpm: int = POLYGON_FREE_TIER_RPM
    liquidity: LiquidityFilter = field(default_factory=LiquidityFilter)
    cache_dir: Path = DEFAULT_CACHE_DIR
    cache_file: str = DEFAULT_CACHE_FILE
    staleness_hours: int = DEFAULT_STALENESS_HOURS
    yaml_config_path: str = ""


def config_from_env(yaml_config_path: str = "") -> UniverseBuilderConfig:
    """Build config from environment variables."""
    return UniverseBuilderConfig(
        polygon_api_key=os.environ.get("POLYGON_API_KEY", ""),
        alpaca_api_key=os.environ.get("ALPACA_API_KEY", ""),
        alpaca_api_secret=os.environ.get("ALPACA_API_SECRET", ""),
        polygon_rate_limit_rpm=int(os.environ.get("POLYGON_RATE_LIMIT_PER_MIN", "5")),
        yaml_config_path=yaml_config_path,
    )


def config_from_yaml(yaml_path: str) -> UniverseBuilderConfig:
    """Build config from YAML data_layer.yaml, supplemented by env vars."""
    cfg = config_from_env(yaml_config_path=yaml_path)
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    filters = raw.get("universe_filters", {})
    cfg.liquidity = LiquidityFilter(
        adv_min_usd=float(filters.get("equity_adv_min_usd", 10_000_000)),
        price_min=float(filters.get("equity_price_min", 5.0)),
        market_cap_min=float(filters.get("equity_market_cap_min", 500_000_000)),
    )
    return cfg


# ── Cache ─────────────────────────────────────────────────────────────────────

def _cache_path(cfg: UniverseBuilderConfig) -> Path:
    return cfg.cache_dir / cfg.cache_file


def _load_cache(cfg: UniverseBuilderConfig) -> Optional[list[tuple[str, str]]]:
    """Return cached equity universe if fresh, else None."""
    path = _cache_path(cfg)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Cache unreadable (%s), will rebuild: %s", path, e)
        return None

    ts_str = data.get("timestamp_utc", "")
    if not ts_str:
        return None

    cached_time = datetime.fromisoformat(ts_str)
    age_hours = (datetime.now(tz=timezone.utc) - cached_time).total_seconds() / 3600.0

    if age_hours > cfg.staleness_hours:
        logger.info(
            "Cache is %.1f hours old (limit %d), will rebuild.",
            age_hours,
            cfg.staleness_hours,
        )
        return None

    equities = [tuple(pair) for pair in data.get("equities", [])]
    logger.info(
        "Loaded %d equities from cache (%.1f hours old).", len(equities), age_hours
    )
    return equities


def _save_cache(cfg: UniverseBuilderConfig, equities: list[tuple[str, str]]) -> None:
    """Persist equity universe to local JSON."""
    path = _cache_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "count": len(equities),
        "equities": equities,
    }
    path.write_text(json.dumps(data, indent=2))
    logger.info("Cached %d equities to %s", len(equities), path)


# ── Polygon: pull all active tickers ──────────────────────────────────────────

def _polygon_rate_sleep(rpm: int) -> None:
    """Sleep enough to stay under the per-minute rate limit."""
    time.sleep(60.0 / max(rpm, 1))


def _fetch_all_tickers_polygon(api_key: str, rpm: int) -> list[dict]:
    """
    Pull every active US stock ticker from Polygon via list_tickers.

    The client's list_tickers auto-paginates (returns an iterator that fetches
    successive pages). We set limit=1000 (the maximum) to minimize page count.
    Each page is one API call.
    """
    try:
        from polygon import RESTClient
    except ImportError:
        raise ImportError("polygon-api-client is required: pip install polygon-api-client")

    client = RESTClient(api_key=api_key)

    print("[universe] Fetching all active US stock tickers from Polygon.io ...")
    tickers: list[dict] = []
    page = 0

    # list_tickers returns a lazy iterator that auto-paginates.
    # Each yielded item is a Ticker object; each page consumes one API call.
    for t in client.list_tickers(
        market="stocks",
        active=True,
        limit=1000,
        sort="ticker",
        order="asc",
    ):
        tickers.append({
            "symbol": t.ticker,
            "name": getattr(t, "name", ""),
            "type": getattr(t, "type", ""),
            "primary_exchange": getattr(t, "primary_exchange", ""),
            "currency_name": getattr(t, "currency_name", ""),
            "locale": getattr(t, "locale", ""),
        })

        # Progress reporting every 1000 tickers
        if len(tickers) % 1000 == 0:
            print(f"  ... {len(tickers)} tickers enumerated so far")

    print(f"[universe] Polygon returned {len(tickers)} active stock tickers total.")
    return tickers


def _filter_us_common_stock(tickers: list[dict]) -> list[dict]:
    """
    Pre-filter to US common stocks only.

    Removes:
    - Non-US tickers (locale != 'us' or currency != 'usd')
    - Warrants, units, rights, preferred shares, ADRs of non-US companies
    - Tickers with special suffixes (e.g., .WS for warrants, .U for units)
    - Blank/weird tickers
    """
    kept = []
    for t in tickers:
        sym = t["symbol"]
        if not sym or len(sym) > 5:
            # Most US common stocks are 1-5 chars; 6+ usually warrants/units
            # Exception: some valid tickers like GOOGL are 5 chars
            # We keep up to 5 and let liquidity filter handle the rest
            continue

        # Polygon ticker types: CS (common stock), ETF, ADRC, etc.
        # We want CS (common stock) and ETF
        ticker_type = t.get("type", "").upper()
        if ticker_type and ticker_type not in ("CS", "ETF"):
            continue

        # Must be US locale and USD denominated
        locale = t.get("locale", "").lower()
        currency = t.get("currency_name", "").lower()
        if locale and locale != "us":
            continue
        if currency and currency != "usd":
            continue

        kept.append(t)

    removed = len(tickers) - len(kept)
    print(f"[universe] Pre-filter: {len(kept)} US common stocks/ETFs kept, {removed} removed (warrants, units, non-US, etc.)")
    return kept


# ── Polygon: snapshot-based liquidity filter ──────────────────────────────────

def _fetch_snapshots_polygon(
    api_key: str,
    rpm: int,
) -> dict[str, dict]:
    """
    Fetch snapshots for ALL stocks in a single API call.

    Polygon's get_snapshot_all("stocks") returns the full market in one response.
    This is the most rate-limit-friendly way to get price and volume data.
    Returns {ticker: {price, volume, prev_close, prev_volume}}.

    NOTE: Snapshot data is only populated during/after market hours.
    Before 4am EST it may be empty.
    """
    try:
        from polygon import RESTClient
    except ImportError:
        raise ImportError("polygon-api-client is required: pip install polygon-api-client")

    client = RESTClient(api_key=api_key)

    print("[universe] Fetching market snapshots from Polygon (single bulk call) ...")
    try:
        snapshots = client.get_snapshot_all("stocks")
    except Exception as e:
        logger.error("Polygon snapshot fetch failed: %s", e)
        print(f"[universe] WARNING: Snapshot fetch failed: {e}")
        return {}

    result: dict[str, dict] = {}
    for snap in snapshots:
        ticker = snap.ticker
        if not ticker:
            continue

        # Extract price and volume from the day agg and prev_day agg
        day = snap.day
        prev = snap.prev_day

        day_close = day.close if day and day.close else 0.0
        day_volume = day.volume if day and day.volume else 0.0
        day_vwap = day.vwap if day and day.vwap else 0.0

        prev_close = prev.close if prev and prev.close else 0.0
        prev_volume = prev.volume if prev and prev.volume else 0.0
        prev_vwap = prev.vwap if prev and prev.vwap else 0.0

        # Use the best available price: today's close, or prev day's close
        price = day_close if day_close > 0 else prev_close

        # Dollar volume: use VWAP * volume for accuracy, fall back to close * volume
        if day_vwap > 0 and day_volume > 0:
            dollar_volume = day_vwap * day_volume
        elif price > 0 and day_volume > 0:
            dollar_volume = price * day_volume
        elif prev_vwap > 0 and prev_volume > 0:
            dollar_volume = prev_vwap * prev_volume
        elif prev_close > 0 and prev_volume > 0:
            dollar_volume = prev_close * prev_volume
        else:
            dollar_volume = 0.0

        result[ticker] = {
            "price": price,
            "dollar_volume": dollar_volume,
            "day_volume": day_volume,
            "prev_volume": prev_volume,
        }

    print(f"[universe] Snapshots received for {len(result)} tickers.")
    _polygon_rate_sleep(rpm)
    return result


def _fetch_market_caps_polygon(
    api_key: str,
    rpm: int,
    symbols: list[str],
) -> dict[str, float]:
    """
    Fetch market cap via get_ticker_details for each symbol.

    This is the expensive part -- one API call per ticker. To stay under the
    free-tier rate limit, we batch with sleeps. For large universes this can
    take a long time, so we only call this for tickers that already pass the
    price + ADV filters (typically ~1000-2000 out of ~7000).
    """
    try:
        from polygon import RESTClient
    except ImportError:
        raise ImportError("polygon-api-client is required: pip install polygon-api-client")

    client = RESTClient(api_key=api_key)
    sleep_seconds = 60.0 / max(rpm, 1)

    print(f"[universe] Fetching market caps for {len(symbols)} tickers "
          f"(~{len(symbols) * sleep_seconds / 60:.0f} min at {rpm} req/min) ...")

    caps: dict[str, float] = {}
    errors = 0
    for i, sym in enumerate(symbols):
        try:
            details = client.get_ticker_details(sym)
            mcap = details.market_cap if details.market_cap else 0.0
            caps[sym] = mcap
        except Exception as e:
            logger.debug("get_ticker_details failed for %s: %s", sym, e)
            errors += 1
            caps[sym] = 0.0

        # Rate limit
        if i < len(symbols) - 1:
            time.sleep(sleep_seconds)

        # Progress every 50 tickers
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(symbols)} market caps fetched ({errors} errors)")

    print(f"[universe] Market caps fetched: {len(caps)} total, {errors} errors.")
    return caps


# ── Alpaca: fallback ticker list ──────────────────────────────────────────────

def _fetch_all_tickers_alpaca(api_key: str, api_secret: str) -> list[dict]:
    """
    Pull all active, tradable US stock tickers from Alpaca.

    Alpaca returns the full asset list in a single call (no pagination needed).
    This is a good fallback if Polygon is unavailable.
    """
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetStatus
    except ImportError:
        raise ImportError("alpaca-py is required: pip install alpaca-py")

    print("[universe] Fetching all active US stock tickers from Alpaca ...")
    client = TradingClient(api_key=api_key, secret_key=api_secret)

    request = GetAssetsRequest(status=AssetStatus.ACTIVE)
    assets = client.get_all_assets(filter=request)

    tickers = []
    for a in assets:
        # Only US equities that are tradable
        if not a.tradable:
            continue
        if str(a.asset_class).upper() not in ("US_EQUITY", "ASSETCLASS.US_EQUITY"):
            continue

        sym = a.symbol
        if not sym or len(sym) > 5:
            continue

        tickers.append({
            "symbol": sym,
            "name": a.name or "",
            "exchange": str(a.exchange) if a.exchange else "",
            "shortable": a.shortable,
            "fractionable": a.fractionable,
        })

    print(f"[universe] Alpaca returned {len(tickers)} active, tradable US equities.")
    return tickers


# ── Core: build dynamic equity universe ───────────────────────────────────────

def _build_dynamic_equities(cfg: UniverseBuilderConfig) -> list[tuple[str, str]]:
    """
    Build the dynamic equity universe:
      1. Pull all active US tickers (Polygon primary, Alpaca fallback)
      2. Pre-filter to common stocks
      3. Apply price + ADV filter via Polygon snapshots
      4. Apply market cap filter via Polygon ticker details
      5. Return [(symbol, "EQUITY"), ...] for stocks and [(symbol, "ETF"), ...] for ETFs

    Strategy to minimize API calls:
      - list_tickers: auto-paginated, ~8 pages for ~8000 tickers
      - get_snapshot_all: 1 call returns all tickers' price/volume
      - get_ticker_details: 1 call per ticker, but only for post-snapshot survivors
    """
    # Step 1: Get the raw ticker list
    raw_tickers: list[dict] = []
    ticker_source = "none"

    if cfg.polygon_api_key:
        try:
            raw_tickers = _fetch_all_tickers_polygon(
                cfg.polygon_api_key, cfg.polygon_rate_limit_rpm
            )
            ticker_source = "polygon"
        except Exception as e:
            logger.error("Polygon ticker fetch failed: %s", e)
            print(f"[universe] Polygon ticker fetch failed: {e}")

    if not raw_tickers and cfg.alpaca_api_key and cfg.alpaca_api_secret:
        try:
            raw_tickers = _fetch_all_tickers_alpaca(
                cfg.alpaca_api_key, cfg.alpaca_api_secret
            )
            ticker_source = "alpaca"
        except Exception as e:
            logger.error("Alpaca ticker fetch failed: %s", e)
            print(f"[universe] Alpaca ticker fetch failed: {e}")

    if not raw_tickers:
        print("[universe] ERROR: No tickers available from any provider.")
        return []

    # Step 2: Pre-filter to US common stocks (only for Polygon, Alpaca already filtered)
    if ticker_source == "polygon":
        filtered_tickers = _filter_us_common_stock(raw_tickers)
    else:
        filtered_tickers = raw_tickers

    symbols = [t["symbol"] for t in filtered_tickers]
    symbol_set = set(symbols)

    # Build a lookup for ticker type (Polygon provides CS vs ETF)
    ticker_type_map: dict[str, str] = {}
    for t in filtered_tickers:
        tt = t.get("type", "").upper()
        if tt == "ETF":
            ticker_type_map[t["symbol"]] = "ETF"
        else:
            ticker_type_map[t["symbol"]] = "EQUITY"

    # Step 3: Fetch snapshots for price + ADV screening
    snapshots: dict[str, dict] = {}
    if cfg.polygon_api_key:
        try:
            snapshots = _fetch_snapshots_polygon(
                cfg.polygon_api_key, cfg.polygon_rate_limit_rpm
            )
        except Exception as e:
            logger.error("Polygon snapshot fetch failed: %s", e)
            print(f"[universe] Snapshot fetch failed: {e}")

    if not snapshots:
        print("[universe] WARNING: No snapshot data available. Cannot apply price/ADV filters.")
        print("[universe] Returning all tickers without liquidity filtering.")
        return [(sym, ticker_type_map.get(sym, "EQUITY")) for sym in symbols]

    # Step 4: Apply price and ADV filters using snapshot data
    # Dollar volume from a single day's snapshot is a noisy proxy for 20-day ADV.
    # We use it as a first pass: if a single day's dollar volume is below the ADV
    # threshold, the 20-day average is almost certainly below too. This eliminates
    # the vast majority of illiquid names and avoids per-ticker API calls for them.
    price_adv_survivors: list[str] = []
    price_rejected = 0
    adv_rejected = 0
    no_data = 0

    for sym in symbols:
        snap = snapshots.get(sym)
        if snap is None:
            no_data += 1
            continue

        price = snap["price"]
        dollar_volume = snap["dollar_volume"]

        if price < cfg.liquidity.price_min:
            price_rejected += 1
            continue

        if dollar_volume < cfg.liquidity.adv_min_usd:
            adv_rejected += 1
            continue

        price_adv_survivors.append(sym)

    print(
        f"[universe] Price/ADV filter: {len(price_adv_survivors)} pass, "
        f"{price_rejected} below ${cfg.liquidity.price_min} price, "
        f"{adv_rejected} below ${cfg.liquidity.adv_min_usd/1e6:.0f}M ADV, "
        f"{no_data} no snapshot data."
    )

    # Step 5: Apply market cap filter
    # This requires per-ticker API calls, so we only do it for survivors.
    if cfg.polygon_api_key and cfg.liquidity.market_cap_min > 0:
        market_caps = _fetch_market_caps_polygon(
            cfg.polygon_api_key,
            cfg.polygon_rate_limit_rpm,
            price_adv_survivors,
        )

        final_symbols: list[str] = []
        mcap_rejected = 0
        for sym in price_adv_survivors:
            mcap = market_caps.get(sym, 0.0)
            if mcap >= cfg.liquidity.market_cap_min:
                final_symbols.append(sym)
            else:
                mcap_rejected += 1

        print(
            f"[universe] Market cap filter: {len(final_symbols)} pass, "
            f"{mcap_rejected} below ${cfg.liquidity.market_cap_min/1e9:.1f}B."
        )
    else:
        # No market cap data available — pass all price/ADV survivors
        final_symbols = price_adv_survivors
        print("[universe] Skipping market cap filter (no Polygon key or threshold is 0).")

    # Step 6: Assign asset types
    equities: list[tuple[str, str]] = []
    etf_count = 0
    equity_count = 0
    for sym in sorted(final_symbols):
        atype = ticker_type_map.get(sym, "EQUITY")
        equities.append((sym, atype))
        if atype == "ETF":
            etf_count += 1
        else:
            equity_count += 1

    print(
        f"[universe] Dynamic universe: {len(equities)} instruments "
        f"({equity_count} equities, {etf_count} ETFs)."
    )
    return equities


# ── Static instruments from YAML ──────────────────────────────────────────────

_YAML_SECTION_TO_ASSET_TYPE = {
    "sector_etfs": "ETF",
    "equity_index_futures": "FUTURE",
    "commodity_futures": "COMMODITY",
    "fixed_income_futures": "BOND",
    "fx_pairs": "FX",
    "vix_futures": "VOLATILITY",
}


def _load_static_instruments(yaml_path: str) -> list[tuple[str, str]]:
    """Load the static (non-equity) instruments from the YAML config."""
    if not yaml_path or not os.path.exists(yaml_path):
        logger.warning("No YAML config at %s — returning empty static universe.", yaml_path)
        return []

    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    instruments = raw.get("instruments", {})
    static: list[tuple[str, str]] = []

    for section, atype in _YAML_SECTION_TO_ASSET_TYPE.items():
        for sym in instruments.get(section, []):
            static.append((sym, atype))

    print(f"[universe] Static instruments from YAML: {len(static)}")
    return static


# ── Public API ────────────────────────────────────────────────────────────────

def build_equity_universe(
    cfg: Optional[UniverseBuilderConfig] = None,
    force_refresh: bool = False,
) -> list[tuple[str, str]]:
    """
    Build the dynamic equity universe with caching.

    Returns list of (symbol, asset_type) tuples where asset_type is "EQUITY" or "ETF".

    Args:
        cfg: Configuration. If None, built from environment variables.
        force_refresh: If True, ignore cache and rebuild from scratch.
    """
    if cfg is None:
        cfg = config_from_env()

    # Check cache first
    if not force_refresh:
        cached = _load_cache(cfg)
        if cached is not None:
            return cached

    # Build from providers
    equities = _build_dynamic_equities(cfg)

    # Cache the results
    if equities:
        _save_cache(cfg, equities)

    return equities


def build_full_universe(
    cfg: Optional[UniverseBuilderConfig] = None,
    yaml_config_path: str = "",
    force_refresh: bool = False,
) -> list[tuple[str, str]]:
    """
    Build the complete tradeable universe:
      - Dynamic equities/ETFs from Polygon/Alpaca (with liquidity filters)
      - Static futures/FX/commodities/bonds/vol from the YAML config

    Returns list of (symbol, asset_type) tuples compatible with backtest's load_universe().

    Args:
        cfg: Configuration. If None, built from environment variables.
        yaml_config_path: Path to data_layer.yaml. Overrides cfg.yaml_config_path if provided.
        force_refresh: If True, ignore cache and rebuild equities from scratch.
    """
    if cfg is None:
        cfg = config_from_env(yaml_config_path=yaml_config_path)

    if yaml_config_path:
        cfg.yaml_config_path = yaml_config_path
        # Also load filters from YAML if available
        try:
            with open(yaml_config_path) as f:
                raw = yaml.safe_load(f)
            filters = raw.get("universe_filters", {})
            cfg.liquidity = LiquidityFilter(
                adv_min_usd=float(filters.get("equity_adv_min_usd", cfg.liquidity.adv_min_usd)),
                price_min=float(filters.get("equity_price_min", cfg.liquidity.price_min)),
                market_cap_min=float(filters.get("equity_market_cap_min", cfg.liquidity.market_cap_min)),
            )
        except Exception as e:
            logger.warning("Could not load filters from YAML: %s", e)

    print("=" * 70)
    print("  ONE BRAIN FUND — Dynamic Universe Builder")
    print("=" * 70)
    print(f"  Filters: ADV >= ${cfg.liquidity.adv_min_usd/1e6:.0f}M | "
          f"Price >= ${cfg.liquidity.price_min:.0f} | "
          f"MCap >= ${cfg.liquidity.market_cap_min/1e9:.1f}B")
    print(f"  Polygon key: {'set' if cfg.polygon_api_key else 'MISSING'} | "
          f"Alpaca key: {'set' if cfg.alpaca_api_key else 'MISSING'}")
    print("=" * 70)

    # 1. Dynamic equities
    equities = build_equity_universe(cfg=cfg, force_refresh=force_refresh)

    # 2. Static instruments from YAML
    static = _load_static_instruments(cfg.yaml_config_path)

    # 3. Merge, avoiding duplicates (static takes precedence for asset type
    #    in case an ETF like SPY appears in both dynamic and static lists)
    static_symbols = {sym for sym, _ in static}
    merged: list[tuple[str, str]] = list(static)
    for sym, atype in equities:
        if sym not in static_symbols:
            merged.append((sym, atype))

    # Summary
    type_counts: dict[str, int] = {}
    for _, atype in merged:
        type_counts[atype] = type_counts.get(atype, 0) + 1

    print()
    print("=" * 70)
    print(f"  FINAL UNIVERSE: {len(merged)} instruments")
    for atype in sorted(type_counts.keys()):
        print(f"    {atype:12s}: {type_counts[atype]:>5d}")
    print("=" * 70)

    return merged


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    """Run universe builder from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="AlphaForge — Dynamic Universe Builder")
    parser.add_argument(
        "--config",
        type=str,
        default="config/data_layer.yaml",
        help="Path to data_layer.yaml",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cache and rebuild from scratch",
    )
    parser.add_argument(
        "--adv-min",
        type=float,
        default=None,
        help="Override: minimum 20-day ADV in USD (e.g. 10000000)",
    )
    parser.add_argument(
        "--price-min",
        type=float,
        default=None,
        help="Override: minimum price (e.g. 5.0)",
    )
    parser.add_argument(
        "--mcap-min",
        type=float,
        default=None,
        help="Override: minimum market cap in USD (e.g. 500000000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write universe to this JSON file",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = config_from_yaml(args.config) if os.path.exists(args.config) else config_from_env()
    cfg.yaml_config_path = args.config

    # Apply CLI overrides
    if args.adv_min is not None:
        cfg.liquidity.adv_min_usd = args.adv_min
    if args.price_min is not None:
        cfg.liquidity.price_min = args.price_min
    if args.mcap_min is not None:
        cfg.liquidity.market_cap_min = args.mcap_min

    universe = build_full_universe(cfg=cfg, force_refresh=args.force_refresh)

    # Optionally write to file
    if args.output:
        output_data = {
            "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
            "count": len(universe),
            "universe": universe,
        }
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"\nWrote universe to {args.output}")

    # Print first/last few
    print(f"\nFirst 10: {universe[:10]}")
    print(f"Last 10:  {universe[-10:]}")


if __name__ == "__main__":
    main()
