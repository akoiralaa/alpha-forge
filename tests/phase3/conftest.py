
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

build_dir = Path(__file__).resolve().parent.parent.parent / "src" / "cpp" / "build"
sys.path.insert(0, str(build_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from engine import FeatureEngine, FeatureEngineConfig, Tick

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

def generate_tick_series(
    n: int = 1000,
    symbol_id: int = 1,
    start_price: float = 100.0,
    drift: float = 0.0,
    vol: float = 0.001,
    spread: float = 0.02,
    seed: int = 42,
) -> list[Tick]:
    rng = np.random.default_rng(seed)
    prices = [start_price]
    for i in range(1, n):
        ret = drift + vol * rng.standard_normal()
        prices.append(prices[-1] * math.exp(ret))

    ticks = []
    for i, price in enumerate(prices):
        ticks.append(make_tick(
            symbol_id=symbol_id,
            ts_ns=i * 1_000_000_000,
            price=price,
            size=100 + int(rng.integers(0, 500)),
            spread=spread,
        ))
    return ticks

@pytest.fixture
def default_engine():
    cfg = FeatureEngineConfig()
    cfg.warmup_ticks = 50
    return FeatureEngine(cfg)

@pytest.fixture
def tick_series():
    return generate_tick_series(n=2000, seed=42)
