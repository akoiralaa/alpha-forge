"""Shared fixtures for Phase 1 tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.data.arctic_store import TickStore
from src.data.fundamentals import FundamentalsStore
from src.data.ingest.base import AssetClass, Tick, date_to_ns
from src.data.quality_pipeline import DataQualityPipeline
from src.data.symbol_master import SymbolMaster


@pytest.fixture
def tmp_dir(tmp_path):
    """Temporary directory for test data."""
    return tmp_path


@pytest.fixture
def tick_store(tmp_dir):
    """Fresh ArcticDB tick store."""
    return TickStore(tmp_dir / "arcticdb")


@pytest.fixture
def symbol_master(tmp_dir):
    """Fresh symbol master database."""
    sm = SymbolMaster(tmp_dir / "symbol_master.db")
    yield sm
    sm.close()


@pytest.fixture
def fundamentals(tmp_dir):
    """Fresh fundamentals store."""
    f = FundamentalsStore(tmp_dir / "fundamentals.db")
    yield f
    f.close()


@pytest.fixture
def quality_pipeline():
    """Fresh data quality pipeline."""
    return DataQualityPipeline()


def make_tick(
    symbol_id: int = 1,
    exchange_time_ns: int = 1_000_000_000_000,
    capture_offset_ns: int = 100_000,
    bid: float = 100.0,
    ask: float = 100.05,
    last_price: float = 100.02,
    last_size: int = 50,
    bid_size: int = 100,
    ask_size: int = 100,
) -> Tick:
    """Helper to create a valid tick with sensible defaults."""
    return Tick(
        exchange_time_ns=exchange_time_ns,
        capture_time_ns=exchange_time_ns + capture_offset_ns,
        symbol_id=symbol_id,
        bid=bid,
        ask=ask,
        bid_size=bid_size,
        ask_size=ask_size,
        last_price=last_price,
        last_size=last_size,
    )


def make_tick_series(
    symbol_id: int = 1,
    n: int = 100,
    start_ns: int = 1_000_000_000_000,
    interval_ns: int = 1_000_000,
    base_price: float = 100.0,
    spread: float = 0.05,
) -> list[Tick]:
    """Generate a series of valid ticks for testing."""
    import numpy as np

    rng = np.random.default_rng(42)
    ticks = []
    price = base_price

    for i in range(n):
        price += rng.normal(0, 0.01)
        price = max(price, 1.0)  # Keep positive
        exchange_ns = start_ns + i * interval_ns
        ticks.append(Tick(
            exchange_time_ns=exchange_ns,
            capture_time_ns=exchange_ns + 100_000,
            symbol_id=symbol_id,
            bid=round(price - spread / 2, 4),
            ask=round(price + spread / 2, 4),
            bid_size=100 + rng.integers(-20, 20),
            ask_size=100 + rng.integers(-20, 20),
            last_price=round(price, 4),
            last_size=int(50 + rng.integers(-10, 10)),
        ))

    return ticks
