from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.backtester.types import Position, Side
from src.signals.intraday_alpha import IntradayAlphaSleeve
from tests.phase4.conftest import make_msv


class TestIntradayAlphaSleeve:
    def test_liquidity_crisis_blocks_new_signal(self):
        sleeve = IntradayAlphaSleeve({1: "ES"})
        msv = make_msv(
            symbol_id=1,
            timestamp_ns=1_000,
            vpin=0.95,
            zscore_20=-3.0,
            zscore_100=-2.0,
            ret_60s=0.8,
            ret_300s=0.5,
        )
        score = sleeve.score_msv(msv)
        assert score == 0.0
        assert sleeve.last_components[1]["regime"] == "LIQUIDITY_CRISIS"

    def test_lead_lag_supports_lagger(self):
        sleeve = IntradayAlphaSleeve({1: "QQQ", 2: "AAPL"})
        leader = make_msv(
            symbol_id=1,
            timestamp_ns=1_000_000_000,
            zscore_20=2.8,
            ret_60s=0.7,
            ret_300s=0.5,
            ret_1800s=0.4,
            vol_60s=0.01,
            vol_1d=0.02,
        )
        lagger = make_msv(
            symbol_id=2,
            timestamp_ns=1_100_000_000,
            zscore_20=0.1,
            ret_60s=0.0,
            ret_300s=0.0,
            ret_1800s=0.0,
            vol_60s=0.01,
            vol_1d=0.02,
        )
        sleeve.score_msv(leader)
        score = sleeve.score_msv(lagger)
        assert sleeve.last_components[2]["lead_lag"] > 0
        assert score > 0

    def test_backtester_adapter_enters_and_exits(self):
        sleeve = IntradayAlphaSleeve({1: "ES"})
        signal_fn = sleeve.build_backtester_signal(
            order_size=10,
            entry_threshold=0.05,
            exit_threshold=0.02,
        )

        entry_msv = make_msv(
            symbol_id=1,
            timestamp_ns=1,
            zscore_20=-3.0,
            zscore_100=-2.0,
            ret_60s=0.4,
            ret_300s=0.3,
            ret_1800s=0.2,
            ofi=0.3,
            volume_ratio_20=3.0,
            ret_1s=0.02,
            vol_60s=0.01,
            vol_1d=0.02,
        )
        order = signal_fn(entry_msv, {})
        assert order is not None
        assert order.side == Side.BUY
        assert order.size == 10

        positions = {1: Position(symbol_id=1, quantity=10, avg_price=100.0)}
        exit_msv = make_msv(
            symbol_id=1,
            timestamp_ns=2,
            zscore_20=0.0,
            zscore_100=0.0,
            ret_60s=0.0,
            ret_300s=0.0,
            ret_1800s=0.0,
            ofi=0.0,
            volume_ratio_20=1.0,
            vol_60s=0.01,
            vol_1d=0.02,
        )
        flatten = signal_fn(exit_msv, positions)
        assert flatten is not None
        assert flatten.side == Side.SELL
        assert flatten.size == 10
