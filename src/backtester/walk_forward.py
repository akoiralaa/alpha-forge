
from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Callable, Optional

import numpy as np

from src.backtester.engine import Backtester
from src.backtester.types import (
    BacktestResult,
    LockBox,
    StressScenario,
    WalkForwardResult,
)

def walk_forward(
    run_backtest_fn: Callable[[date, date], BacktestResult],
    train_fn: Callable[[date, date], None],
    start_date: date,
    end_date: date,
    train_window_days: int = 252,
    test_window_days: int = 63,
    embargo_days: int = 30,
    step_days: int = 21,
) -> list[WalkForwardResult]:
    results = []
    t = start_date

    while True:
        train_end = t + timedelta(days=train_window_days)
        test_start = train_end + timedelta(days=embargo_days)
        test_end = test_start + timedelta(days=test_window_days)

        if test_end > end_date:
            break

        # Train
        train_fn(t, train_end)

        # Test (no peeking into embargo or train period)
        result = run_backtest_fn(test_start, test_end)

        results.append(WalkForwardResult(
            train_range=(t, train_end),
            embargo_range=(train_end, test_start),
            test_range=(test_start, test_end),
            sharpe=result.sharpe,
            max_drawdown=result.max_drawdown,
            n_trades=result.n_trades,
        ))

        t += timedelta(days=step_days)

    return results

def verify_embargo(
    embargo_days: int,
    max_feature_lookback_days: int,
) -> bool:
    return embargo_days >= max_feature_lookback_days

def monte_carlo_permutation_test(
    real_sharpe: float,
    run_backtest_fn: Callable[[], BacktestResult],
    n_permutations: int = 1000,
    seed: int = 42,
) -> tuple[bool, float]:
    rng = np.random.default_rng(seed)
    perm_sharpes = []

    for _ in range(n_permutations):
        result = run_backtest_fn()
        perm_sharpes.append(result.sharpe)

    p95 = float(np.percentile(perm_sharpes, 95))
    passed = real_sharpe > p95
    return passed, p95

# ── Stress test scenarios ────────────────────────────────────

def apply_stress_scenario(
    ticks: list,
    scenario: StressScenario,
    rng: np.random.Generator = None,
) -> list:
    if rng is None:
        rng = np.random.default_rng(42)

    # Import here to avoid circular
    try:
        from engine import Tick
    except ImportError:
        from _onebrain_cpp import Tick

    if scenario == StressScenario.LIQUIDITY_BLACKOUT:
        # Widen spreads 5x for random 30-second windows
        modified = []
        blackout_starts = sorted(rng.choice(len(ticks), size=min(10, len(ticks)), replace=False))
        blackout_set = set()
        for start_idx in blackout_starts:
            if start_idx < len(ticks):
                start_ts = ticks[start_idx].capture_time_ns
                end_ts = start_ts + 30_000_000_000  # 30 seconds
                for j in range(start_idx, min(start_idx + 500, len(ticks))):
                    if ticks[j].capture_time_ns <= end_ts:
                        blackout_set.add(j)

        for i, tick in enumerate(ticks):
            t = Tick()
            t.symbol_id = tick.symbol_id
            t.exchange_time_ns = tick.exchange_time_ns
            t.capture_time_ns = tick.capture_time_ns
            t.last_price = tick.last_price
            t.last_size = tick.last_size
            t.bid_size = tick.bid_size
            t.ask_size = tick.ask_size
            if i in blackout_set:
                mid = (tick.bid + tick.ask) / 2.0
                spread = tick.ask - tick.bid
                t.bid = mid - spread * 2.5
                t.ask = mid + spread * 2.5
            else:
                t.bid = tick.bid
                t.ask = tick.ask
            modified.append(t)
        return modified

    elif scenario == StressScenario.GAP_DOWN_10PCT:
        # 10% gap at first tick, zero depth for 100ms
        modified = []
        gap_applied = False
        gap_end_ns = 0
        for tick in ticks:
            t = Tick()
            t.symbol_id = tick.symbol_id
            t.exchange_time_ns = tick.exchange_time_ns
            t.capture_time_ns = tick.capture_time_ns
            t.bid_size = tick.bid_size
            t.ask_size = tick.ask_size
            t.last_size = tick.last_size

            if not gap_applied:
                # Apply 10% gap
                t.bid = tick.bid * 0.90
                t.ask = tick.ask * 0.90
                t.last_price = tick.last_price * 0.90
                gap_applied = True
                gap_end_ns = tick.capture_time_ns + 100_000_000  # 100ms
            elif tick.capture_time_ns < gap_end_ns:
                t.bid = tick.bid * 0.90
                t.ask = tick.ask * 0.90
                t.last_price = tick.last_price * 0.90
                t.bid_size = 0
                t.ask_size = 0
            else:
                t.bid = tick.bid * 0.92  # partial recovery
                t.ask = tick.ask * 0.92
                t.last_price = tick.last_price * 0.92
            modified.append(t)
        return modified

    elif scenario == StressScenario.FEED_OUTAGE_5S:
        # Remove ticks for 5 seconds in the middle, then resume with gap
        if len(ticks) < 10:
            return ticks
        mid_idx = len(ticks) // 2
        outage_start_ns = ticks[mid_idx].capture_time_ns
        outage_end_ns = outage_start_ns + 5_000_000_000  # 5 seconds
        modified = []
        for tick in ticks:
            if outage_start_ns <= tick.capture_time_ns < outage_end_ns:
                continue  # drop tick
            t = Tick()
            t.symbol_id = tick.symbol_id
            t.exchange_time_ns = tick.exchange_time_ns
            t.capture_time_ns = tick.capture_time_ns
            t.bid = tick.bid
            t.ask = tick.ask
            t.bid_size = tick.bid_size
            t.ask_size = tick.ask_size
            t.last_price = tick.last_price
            t.last_size = tick.last_size
            if tick.capture_time_ns >= outage_end_ns and modified:
                # Gap: price jumps 2%
                t.bid *= 0.98
                t.ask *= 0.98
                t.last_price *= 0.98
            modified.append(t)
        return modified

    elif scenario == StressScenario.CORRELATION_CRISIS:
        # All symbols move together: replace prices with avg + small noise
        if not ticks:
            return ticks
        # Group by timestamp, correlate
        modified = []
        prev_avg = None
        for tick in ticks:
            t = Tick()
            t.symbol_id = tick.symbol_id
            t.exchange_time_ns = tick.exchange_time_ns
            t.capture_time_ns = tick.capture_time_ns
            t.bid_size = tick.bid_size
            t.ask_size = tick.ask_size
            t.last_size = tick.last_size
            # Shift price toward a common mean with 0.95 correlation
            base = tick.last_price
            noise = rng.normal(0, base * 0.001)
            shock = -base * 0.03  # 3% correlated shock
            t.last_price = base + 0.95 * shock + 0.05 * noise
            t.bid = t.last_price - (tick.ask - tick.bid) / 2
            t.ask = t.last_price + (tick.ask - tick.bid) / 2
            modified.append(t)
        return modified

    # For historical scenarios (CRISIS_2008, etc.), return ticks unchanged
    # The caller loads the appropriate historical data
    return ticks

def run_stress_tests(
    backtester: Backtester,
    base_ticks: list,
    max_drawdown_pct: float = 0.20,
    seed: int = 42,
) -> dict[str, tuple[bool, float]]:
    rng = np.random.default_rng(seed)
    results = {}

    # Only run synthetic scenarios that don't require historical data
    synthetic_scenarios = [
        StressScenario.LIQUIDITY_BLACKOUT,
        StressScenario.GAP_DOWN_10PCT,
        StressScenario.FEED_OUTAGE_5S,
        StressScenario.CORRELATION_CRISIS,
    ]

    for scenario in synthetic_scenarios:
        stressed_ticks = apply_stress_scenario(base_ticks, scenario, rng)
        result = backtester.run(stressed_ticks)
        dd = result.max_drawdown_pct
        passed = dd < max_drawdown_pct
        results[scenario.value] = (passed, dd)

    return results
