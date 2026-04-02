from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.paper.engine import PaperConfig, PaperTick, PaperTradingEngine
from src.signals.intraday_alpha import IntradayAlphaSleeve


def _generate_lead_lag_ticks(
    leader_sid: int,
    lagger_sid: int,
    n: int = 120,
    seed: int = 42,
) -> list[PaperTick]:
    rng = np.random.default_rng(seed)
    leader_price = 100.0
    lagger_price = 100.0
    ticks: list[PaperTick] = []

    for i in range(n):
        leader_ret = 0.002 * math.sin(i / 8.0) + rng.normal(0.0, 0.003)
        leader_price *= math.exp(leader_ret)
        ticks.append(PaperTick(
            symbol_id=leader_sid,
            price=leader_price,
            volume=1800 + int(rng.integers(0, 300)),
            timestamp_ns=i * 1_000_000_000,
        ))

        # Lagger prints before it has fully incorporated the leader move.
        ticks.append(PaperTick(
            symbol_id=lagger_sid,
            price=lagger_price,
            volume=1500 + int(rng.integers(0, 250)),
            timestamp_ns=i * 1_000_000_000 + 100_000_000,
        ))
        lagger_price *= math.exp(0.65 * leader_ret + rng.normal(0.0, 0.0015))

    return ticks


class TestIntradayAlphaPaper:
    def test_intraday_sleeve_generates_live_orders(self):
        engine = PaperTradingEngine(
            PaperConfig(
                signal_threshold=0.03,
                reconciliation_interval_ticks=50,
                risk_budget_per_position=0.0005,
                kelly_fraction=0.05,
                max_position_pct_nav=0.10,
            )
        )
        sleeve = IntradayAlphaSleeve({1: "QQQ", 2: "AAPL"})
        engine.set_signal_function(sleeve)

        ticks = _generate_lead_lag_ticks(1, 2, n=150)
        stats = engine.run_session(ticks)

        assert stats.orders_submitted > 0
        assert stats.orders_filled > 0
        assert 2 in sleeve.last_components
        assert sleeve.max_abs_lead_lag[2] > 0
