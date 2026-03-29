
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Ensure build output is importable
build_dir = Path(__file__).resolve().parent.parent.parent / "src" / "cpp" / "build"
sys.path.insert(0, str(build_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from engine import (
    FeatureEngine,
    FeatureEngineConfig,
    Tick,
    WelfordAccumulator,
    CircularBufferDouble,
)

@pytest.fixture
def default_config():
    cfg = FeatureEngineConfig()
    cfg.warmup_ticks = 10
    return cfg

@pytest.fixture
def engine(default_config):
    return FeatureEngine(default_config)

def make_tick(
    symbol_id: int = 1,
    ts_ns: int = 0,
    price: float = 100.0,
    size: int = 100,
    spread: float = 0.02,
) -> Tick:
    t = Tick()
    t.symbol_id = symbol_id
    t.exchange_time_ns = ts_ns
    t.capture_time_ns = ts_ns + 1000
    t.bid = price - spread / 2.0
    t.ask = price + spread / 2.0
    t.bid_size = 500
    t.ask_size = 500
    t.last_price = price
    t.last_size = size
    return t
