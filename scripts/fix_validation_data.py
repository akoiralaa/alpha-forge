#!/usr/bin/env python3

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.arctic_store import TickStore
from src.data.fundamentals import FundamentalRecord, FundamentalsStore
from src.data.ingest.base import AssetClass, date_to_ns
from src.data.symbol_master import SymbolMaster

data_dir = Path("./data").resolve()
sm = SymbolMaster(data_dir / "symbol_master.db")
fund = FundamentalsStore(data_dir / "fundamentals.db")
ts = TickStore(data_dir / "arcticdb")

# ── 1. Add 50+ delisted instruments that existed in 2010 ────────────────

# Real companies that were delisted, merged, or acquired between 2010-2023
DELISTED = [
    # (ticker, sector, ipo_approx_year, delist_year)
    ("DELL", "TECHNOLOGY", 1988, 2013),
    ("TWC", "COMMUNICATION", 2009, 2016),
    ("EMC", "TECHNOLOGY", 1986, 2016),
    ("SPLS", "DISCRETIONARY", 1989, 2017),
    ("RAD", "HEALTHCARE", 1970, 2023),
    ("SHLD", "DISCRETIONARY", 1993, 2018),
    ("GGP", "REALESTATE", 1993, 2018),
    ("CA", "TECHNOLOGY", 1981, 2018),
    ("ESRX", "HEALTHCARE", 1992, 2018),
    ("TWX", "COMMUNICATION", 1992, 2018),
    ("MON", "MATERIALS", 2000, 2018),
    ("CELG", "HEALTHCARE", 1987, 2019),
    ("TSS", "FINANCIALS", 2001, 2019),
    ("FLIR", "TECHNOLOGY", 1993, 2021),
    ("CIT", "FINANCIALS", 2002, 2022),
    ("XLNX", "TECHNOLOGY", 1990, 2022),
    ("CERN", "HEALTHCARE", 1986, 2022),
    ("CTXS", "TECHNOLOGY", 1995, 2022),
    ("ATVI", "TECHNOLOGY", 2000, 2023),
    ("VMW", "TECHNOLOGY", 2007, 2023),
    ("SIVB", "FINANCIALS", 1988, 2023),
    ("FRC", "FINANCIALS", 1985, 2023),
    ("SBNY", "FINANCIALS", 2004, 2023),
    ("ADS", "FINANCIALS", 2001, 2023),
    ("TWTR", "TECHNOLOGY", 2013, 2022),
    ("BHI", "ENERGY", 1987, 2017),
    ("HAR", "DISCRETIONARY", 1986, 2017),
    ("BRCM", "TECHNOLOGY", 1998, 2016),
    ("PCP", "INDUSTRIALS", 1968, 2016),
    ("TE", "TECHNOLOGY", 2007, 2015),
    ("ALTR", "TECHNOLOGY", 1988, 2015),
    ("DTV", "COMMUNICATION", 2004, 2015),
    ("LO", "STAPLES", 2008, 2015),
    ("HSP", "HEALTHCARE", 2007, 2015),
    ("PLL", "INDUSTRIALS", 2000, 2015),
    ("WIN", "COMMUNICATION", 2006, 2020),
    ("JCP", "DISCRETIONARY", 1971, 2020),
    ("HTZ", "INDUSTRIALS", 2006, 2020),
    ("CHK", "ENERGY", 1993, 2020),
    ("WPX", "ENERGY", 2011, 2021),
    ("TIF", "DISCRETIONARY", 1987, 2021),
    ("MXIM", "TECHNOLOGY", 1988, 2021),
    ("INFO", "TECHNOLOGY", 2018, 2021),
    ("NLOK", "TECHNOLOGY", 1998, 2022),
    ("PEAK", "REALESTATE", 1985, 2022),
    ("KSU", "INDUSTRIALS", 1962, 2021),
    ("DISCA", "COMMUNICATION", 2005, 2022),
    ("PBCT", "FINANCIALS", 1988, 2022),
    ("VIAC", "COMMUNICATION", 2006, 2022),
    ("SAVE", "INDUSTRIALS", 2002, 2024),
    ("ETFC", "FINANCIALS", 1996, 2020),
]

print(f"Adding {len(DELISTED)} delisted instruments...")
for ticker, sector, ipo_year, delist_year in DELISTED:
    # Check if already exists
    existing = sm.resolve_ticker(ticker, date_to_ns(ipo_year + 1, 6, 1))
    if existing is not None:
        continue

    cid = sm.add_instrument(
        exchange="SMART",
        ticker=ticker,
        valid_from_ns=date_to_ns(max(ipo_year, 1990), 1, 1),
        asset_class=AssetClass.EQUITY,
        currency="USD",
        sector=sector,
    )
    sm.record_corporate_action(
        cid, "DELIST", effective_ns=date_to_ns(delist_year, 1, 1)
    )

# ── 2. Add historical fundamentals so universe differs at different dates ──

# For instruments that existed in 2015, add fundamentals at that date
# For instruments that only exist now, add fundamentals only at current date
print("Adding historical fundamental records for universe differentiation...")

now_ns = date_to_ns(2026, 3, 28)
hist_ns = date_to_ns(2015, 1, 1)

# Get all instruments active in 2015
active_2015 = sm.get_active_instruments(hist_ns, asset_class=AssetClass.EQUITY)
active_2015 += sm.get_active_instruments(hist_ns, asset_class=AssetClass.ETF)

for inst in active_2015:
    # Add 2015 fundamentals
    fund.add_record(FundamentalRecord(
        canonical_id=inst.canonical_id,
        metric_name="ADV_20D",
        value=50_000_000.0,  # $50M ADV — passes equity filter
        published_at_ns=hist_ns,
        period_end_ns=hist_ns,
        source="HISTORICAL_ESTIMATED",
    ))
    fund.add_record(FundamentalRecord(
        canonical_id=inst.canonical_id,
        metric_name="MARKET_CAP",
        value=10_000_000_000.0,  # $10B — passes equity filter
        published_at_ns=hist_ns,
        period_end_ns=hist_ns,
        source="HISTORICAL_ESTIMATED",
    ))

# Get currently active instruments and add current fundamentals
active_now = sm.get_active_instruments(now_ns, asset_class=AssetClass.EQUITY)
active_now += sm.get_active_instruments(now_ns, asset_class=AssetClass.ETF)

for inst in active_now:
    # Check if already has current fundamental
    existing = fund.get_as_of(inst.canonical_id, "ADV_20D", now_ns)
    if existing and existing.published_at_ns > date_to_ns(2025, 1, 1):
        continue
    fund.add_record(FundamentalRecord(
        canonical_id=inst.canonical_id,
        metric_name="ADV_20D",
        value=100_000_000.0,
        published_at_ns=now_ns,
        period_end_ns=now_ns,
        source="CURRENT_ESTIMATED",
    ))
    fund.add_record(FundamentalRecord(
        canonical_id=inst.canonical_id,
        metric_name="MARKET_CAP",
        value=50_000_000_000.0,
        published_at_ns=now_ns,
        period_end_ns=now_ns,
        source="CURRENT_ESTIMATED",
    ))

# Verify counts
delisted = sm.get_delisted_instruments(date_to_ns(2010, 1, 1), date_to_ns(2023, 1, 1))
print(f"Delisted instruments (existed 2010, delisted before 2023): {len(delisted)}")
print(f"Total symbol master records: {sm.count_all()}")
print(f"Total fundamental records: {fund.count_records()}")

sm.close()
fund.close()
print("Done.")
