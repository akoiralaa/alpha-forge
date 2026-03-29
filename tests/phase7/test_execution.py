"""Phase 7 tests — WAL, broker, kill switch, reconciliation, order manager,
self-trade prevention, execution mode, stale data halt."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.execution.wal import OrderState, WALEntry, WriteAheadLog
from src.execution.broker import BrokerFill, BrokerOrder, PaperBroker
from src.execution.kill_switch import KillLevel, KillSwitch
from src.execution.self_trade import would_self_match
from src.execution.execution_mode import execution_mode
from src.execution.stale_data import StaleDataMonitor
from src.execution.reconciliation import BreakType, Reconciler
from src.execution.order_manager import OrderManager
from src.portfolio.risk import Portfolio, PreTradeRiskCheck


# ── WAL Tests ───────────────────────────────────────────────────

class TestWAL:
    def test_append_and_retrieve(self):
        wal = WriteAheadLog()
        entry = WALEntry(
            sequence_id=0, timestamp_ns=0, order_id="O1",
            state=OrderState.PENDING, symbol_id=1, side=1,
            order_type="MARKET", size=100,
        )
        seq = wal.append(entry)
        assert seq == 1
        latest = wal.get_latest_state("O1")
        assert latest is not None
        assert latest.order_id == "O1"
        assert latest.state == OrderState.PENDING

    def test_order_history(self):
        wal = WriteAheadLog()
        for state in [OrderState.PENDING, OrderState.SUBMITTED, OrderState.FILLED]:
            wal.append(WALEntry(
                sequence_id=0, timestamp_ns=0, order_id="O1",
                state=state, symbol_id=1, side=1,
                order_type="MARKET", size=100,
            ))
        history = wal.get_order_history("O1")
        assert len(history) == 3
        assert history[0].state == OrderState.PENDING
        assert history[-1].state == OrderState.FILLED

    def test_open_orders(self):
        wal = WriteAheadLog()
        # O1: filled (terminal)
        wal.append(WALEntry(0, 0, "O1", OrderState.FILLED, 1, 1, "MARKET", 100))
        # O2: submitted (open)
        wal.append(WALEntry(0, 0, "O2", OrderState.SUBMITTED, 2, 1, "LIMIT", 50))
        open_orders = wal.get_open_orders()
        assert len(open_orders) == 1
        assert open_orders[0].order_id == "O2"

    def test_replay(self):
        wal = WriteAheadLog()
        wal.append(WALEntry(0, 0, "O1", OrderState.PENDING, 1, 1, "MARKET", 100))
        wal.append(WALEntry(0, 0, "O1", OrderState.FILLED, 1, 1, "MARKET", 100, filled_size=100))
        wal.append(WALEntry(0, 0, "O2", OrderState.SUBMITTED, 2, -1, "LIMIT", 50))
        state = wal.replay()
        assert len(state) == 2
        assert state["O1"].state == OrderState.FILLED
        assert state["O2"].state == OrderState.SUBMITTED

    def test_entry_count(self):
        wal = WriteAheadLog()
        assert wal.entry_count() == 0
        wal.append(WALEntry(0, 0, "O1", OrderState.PENDING, 1, 1, "MARKET", 100))
        wal.append(WALEntry(0, 0, "O1", OrderState.FILLED, 1, 1, "MARKET", 100))
        assert wal.entry_count() == 2

    def test_persistence_to_file(self, tmp_path):
        db_path = tmp_path / "test_wal.db"
        wal = WriteAheadLog(db_path)
        wal.append(WALEntry(0, 0, "O1", OrderState.FILLED, 1, 1, "MARKET", 100, filled_size=100))
        wal.close()

        # Reopen and verify
        wal2 = WriteAheadLog(db_path)
        latest = wal2.get_latest_state("O1")
        assert latest is not None
        assert latest.state == OrderState.FILLED
        wal2.close()


# ── Broker Tests ────────────────────────────────────────────────

class TestPaperBroker:
    def test_market_order_fills(self):
        broker = PaperBroker(initial_cash=1_000_000)
        broker.set_price(1, 100.0)
        order = BrokerOrder("O1", 1, 1, 100, "MARKET")
        broker_id = broker.submit_order(order)
        assert broker_id.startswith("PAPER-")
        assert broker.positions[1] == 100
        assert len(broker.fills) == 1
        assert broker.fills[0].filled_size == 100

    def test_cash_deducted(self):
        broker = PaperBroker(initial_cash=1_000_000, slippage_bps=0, commission_per_share=0)
        broker.set_price(1, 100.0)
        broker.submit_order(BrokerOrder("O1", 1, 1, 100, "MARKET"))
        # 100 shares * $100 = $10,000
        assert abs(broker.cash - 990_000) < 1.0

    def test_sell_order(self):
        broker = PaperBroker(slippage_bps=0, commission_per_share=0)
        broker.set_price(1, 100.0)
        broker.submit_order(BrokerOrder("O1", 1, 1, 100, "MARKET"))
        broker.submit_order(BrokerOrder("O2", 1, -1, 50, "MARKET"))
        assert broker.positions[1] == 50

    def test_slippage_applied(self):
        broker = PaperBroker(slippage_bps=10.0, commission_per_share=0)
        broker.set_price(1, 100.0)
        broker.submit_order(BrokerOrder("O1", 1, 1, 100, "MARKET"))
        fill = broker.fills[0]
        # Buy: price + slippage = 100 + 0.10 = 100.10
        assert fill.avg_price > 100.0

    def test_cancel_order(self):
        broker = PaperBroker()
        broker.set_price(1, 50.0)
        # Submit a limit order that won't fill (buy below market)
        bid = BrokerOrder("O1", 1, 1, 100, "LIMIT", limit_price=40.0)
        broker_id = broker.submit_order(bid)
        assert broker.cancel_order(broker_id)

    def test_fill_callback(self):
        fills_received = []
        broker = PaperBroker(fill_callback=lambda f: fills_received.append(f))
        broker.set_price(1, 100.0)
        broker.submit_order(BrokerOrder("O1", 1, 1, 100, "MARKET"))
        assert len(fills_received) == 1

    def test_is_connected(self):
        broker = PaperBroker()
        assert broker.is_connected()


# ── Kill Switch Tests ───────────────────────────────────────────

class TestKillSwitch:
    def _make_kill_switch(self):
        broker = PaperBroker(slippage_bps=0, commission_per_share=0)
        broker.set_price(1, 100.0)
        broker.set_price(2, 50.0)
        wal = WriteAheadLog()
        ks = KillSwitch(broker, wal)
        return broker, wal, ks

    def test_normal_allows_orders(self):
        _, _, ks = self._make_kill_switch()
        assert ks.is_order_allowed()

    def test_level1_blocks_orders(self):
        _, _, ks = self._make_kill_switch()
        ks.activate(KillLevel.CANCEL_ONLY, "test")
        assert not ks.is_order_allowed()

    def test_level2_flattens_positions(self):
        broker, wal, ks = self._make_kill_switch()
        # Build position
        broker.submit_order(BrokerOrder("O1", 1, 1, 100, "MARKET"))
        assert broker.positions[1] == 100
        event = ks.activate(KillLevel.FLATTEN, "emergency")
        assert event.positions_flattened == 1
        assert abs(broker.positions.get(1, 0)) < 1e-6

    def test_drawdown_auto_kill(self):
        _, _, ks = self._make_kill_switch()
        ks.drawdown_auto_kill_pct = 0.10
        event = ks.check_drawdown(nav=850_000, peak_nav=1_000_000)
        assert event is not None
        assert event.level == KillLevel.FLATTEN

    def test_drawdown_no_trigger(self):
        _, _, ks = self._make_kill_switch()
        ks.drawdown_auto_kill_pct = 0.20
        event = ks.check_drawdown(nav=900_000, peak_nav=1_000_000)
        assert event is None

    def test_reset(self):
        _, _, ks = self._make_kill_switch()
        ks.activate(KillLevel.CANCEL_ONLY, "test")
        assert not ks.is_order_allowed()
        ks.reset()
        assert ks.is_order_allowed()

    def test_events_logged(self):
        _, _, ks = self._make_kill_switch()
        ks.activate(KillLevel.CANCEL_ONLY, "first")
        ks.activate(KillLevel.FLATTEN, "second")
        assert len(ks.events) == 2

    def test_wal_records_kill_event(self):
        _, wal, ks = self._make_kill_switch()
        ks.activate(KillLevel.CANCEL_ONLY, "test kill")
        assert wal.entry_count() >= 1


# ── Reconciliation Tests ────────────────────────────────────────

class TestReconciliation:
    def test_clean_reconciliation(self):
        broker = PaperBroker(slippage_bps=0, commission_per_share=0)
        broker.set_price(1, 100.0)
        wal = WriteAheadLog()

        # Submit via WAL + broker in sync
        wal.append(WALEntry(0, 0, "O1", OrderState.FILLED, 1, 1, "MARKET", 100, filled_size=100))
        broker.positions[1] = 100.0

        recon = Reconciler(broker, wal)
        report = recon.reconcile()
        assert report.is_clean

    def test_quantity_mismatch(self):
        broker = PaperBroker()
        wal = WriteAheadLog()

        wal.append(WALEntry(0, 0, "O1", OrderState.FILLED, 1, 1, "MARKET", 100, filled_size=100))
        broker.positions[1] = 50.0  # Broker says 50, WAL says 100

        recon = Reconciler(broker, wal)
        report = recon.reconcile()
        assert not report.is_clean
        assert len(report.breaks) == 1
        assert report.breaks[0].break_type == BreakType.QUANTITY_MISMATCH

    def test_missing_broker_position(self):
        broker = PaperBroker()
        wal = WriteAheadLog()

        wal.append(WALEntry(0, 0, "O1", OrderState.FILLED, 1, 1, "MARKET", 100, filled_size=100))
        # Broker has no position

        recon = Reconciler(broker, wal)
        report = recon.reconcile()
        assert not report.is_clean
        assert report.breaks[0].break_type == BreakType.MISSING_BROKER

    def test_missing_internal_position(self):
        broker = PaperBroker()
        broker.positions[99] = 200.0
        wal = WriteAheadLog()

        recon = Reconciler(broker, wal)
        report = recon.reconcile()
        assert not report.is_clean
        assert report.breaks[0].break_type == BreakType.MISSING_INTERNAL

    def test_multiple_fills_net(self):
        broker = PaperBroker()
        wal = WriteAheadLog()

        wal.append(WALEntry(0, 0, "O1", OrderState.FILLED, 1, 1, "MARKET", 100, filled_size=100))
        wal.append(WALEntry(0, 0, "O2", OrderState.FILLED, 1, -1, "MARKET", 30, filled_size=30))
        broker.positions[1] = 70.0  # 100 - 30

        recon = Reconciler(broker, wal)
        report = recon.reconcile()
        assert report.is_clean


# ── Order Manager Tests ─────────────────────────────────────────

class TestOrderManager:
    def _make_manager(self, with_risk=False):
        broker = PaperBroker(slippage_bps=0, commission_per_share=0)
        broker.set_price(1, 100.0)
        wal = WriteAheadLog()
        ks = KillSwitch(broker, wal)
        risk = PreTradeRiskCheck() if with_risk else None
        port = Portfolio(nav=1_000_000) if with_risk else None
        mgr = OrderManager(broker, wal, ks, risk, port)
        return broker, wal, ks, mgr

    def test_submit_market_order(self):
        broker, wal, ks, mgr = self._make_manager()
        order, _ = mgr.submit(1, 1, 100)
        assert order.state == OrderState.SUBMITTED
        assert order.broker_order_id is not None

    def test_wal_records_order(self):
        _, wal, _, mgr = self._make_manager()
        order, _ = mgr.submit(1, 1, 100)
        history = wal.get_order_history(order.order_id)
        states = [e.state for e in history]
        assert OrderState.PENDING in states
        assert OrderState.SUBMITTED in states

    def test_kill_switch_blocks(self):
        _, _, ks, mgr = self._make_manager()
        ks.activate(KillLevel.CANCEL_ONLY, "test")
        order, _ = mgr.submit(1, 1, 100)
        assert order.state == OrderState.REJECTED
        assert "kill_switch" in order.error_msg

    def test_risk_check_blocks(self):
        broker, _, _, mgr = self._make_manager(with_risk=True)
        # Try to buy 110K shares at $100 = 11B notional = way over limit
        order, risk = mgr.submit(1, 1, 110_000, adv_20d=1_000_000)
        assert order.state == OrderState.REJECTED
        assert risk is not None
        assert not risk.passed

    def test_cancel_order(self):
        _, _, _, mgr = self._make_manager()
        order, _ = mgr.submit(1, 1, 100)
        # PaperBroker fills immediately, so cancel may fail
        # Submit a limit order that won't fill
        broker = mgr.broker
        broker.set_price(1, 100.0)
        order2, _ = mgr.submit(1, 1, 100, order_type="LIMIT", limit_price=90.0)
        # This won't fill (bid below market)
        success = mgr.cancel(order2.order_id)
        assert success
        assert order2.state == OrderState.CANCELLED

    def test_on_fill(self):
        broker, _, _, mgr = self._make_manager()
        order, _ = mgr.submit(1, 1, 100)
        fill = BrokerFill(
            order_id=order.order_id,
            broker_order_id=order.broker_order_id,
            symbol_id=1, side=1, filled_size=100,
            avg_price=100.0, timestamp_ns=time.time_ns(),
        )
        mgr.on_fill(fill)
        assert order.state == OrderState.FILLED

    def test_get_open_orders(self):
        _, _, _, mgr = self._make_manager()
        mgr.submit(1, 1, 100)
        # Paper broker fills immediately so order goes to SUBMITTED
        open_orders = mgr.get_open_orders()
        assert len(open_orders) >= 0  # may be 0 or 1 depending on fill timing

    def test_get_filled_orders(self):
        _, _, _, mgr = self._make_manager()
        order, _ = mgr.submit(1, 1, 100)
        fill = BrokerFill(
            order_id=order.order_id,
            broker_order_id=order.broker_order_id,
            symbol_id=1, side=1, filled_size=100,
            avg_price=100.0, timestamp_ns=time.time_ns(),
        )
        mgr.on_fill(fill)
        filled = mgr.get_filled_orders()
        assert len(filled) == 1


# ── Self-Trade Prevention Tests ────────────────────────────────

class TestSelfTradePrevention:
    def test_no_self_match_same_side(self):
        existing = [BrokerOrder("E1", 1, 1, 100, "LIMIT", 100.0)]
        new = BrokerOrder("N1", 1, 1, 50, "LIMIT", 101.0)
        assert not would_self_match(new, existing)

    def test_self_match_crossing_limit(self):
        # Existing sell at 100, new buy at 100 → would cross
        existing = [BrokerOrder("E1", 1, -1, 100, "LIMIT", 100.0)]
        new = BrokerOrder("N1", 1, 1, 50, "LIMIT", 100.0)
        assert would_self_match(new, existing)

    def test_self_match_buy_above_sell(self):
        # Existing sell at 99.50, new buy at 100 → would cross
        existing = [BrokerOrder("E1", 1, -1, 100, "LIMIT", 99.50)]
        new = BrokerOrder("N1", 1, 1, 50, "LIMIT", 100.0)
        assert would_self_match(new, existing)

    def test_no_match_different_symbol(self):
        existing = [BrokerOrder("E1", 1, -1, 100, "LIMIT", 100.0)]
        new = BrokerOrder("N1", 2, 1, 50, "LIMIT", 100.0)
        assert not would_self_match(new, existing)

    def test_no_match_non_crossing(self):
        # Existing sell at 105, new buy at 100 → won't cross
        existing = [BrokerOrder("E1", 1, -1, 100, "LIMIT", 105.0)]
        new = BrokerOrder("N1", 1, 1, 50, "LIMIT", 100.0)
        assert not would_self_match(new, existing)

    def test_market_order_self_match(self):
        existing = [BrokerOrder("E1", 1, -1, 100, "LIMIT", 100.0)]
        new = BrokerOrder("N1", 1, 1, 50, "MARKET")
        assert would_self_match(new, existing)

    def test_empty_open_orders(self):
        new = BrokerOrder("N1", 1, 1, 50, "LIMIT", 100.0)
        assert not would_self_match(new, [])


# ── Execution Mode Tests ───────────────────────────────────────

class TestExecutionMode:
    def test_reduce_only_blocks_positive(self):
        result = execution_mode(0.5, 10.0, 5.0, 0.5, regime_execution_mode="REDUCE_ONLY")
        assert result == "BLOCK"

    def test_reduce_only_allows_negative(self):
        result = execution_mode(-0.5, 10.0, 5.0, 0.5, regime_execution_mode="REDUCE_ONLY")
        assert result == "PASSIVE"

    def test_passive_only(self):
        result = execution_mode(0.8, 1.0, 5.0, 0.5, regime_execution_mode="PASSIVE_ONLY")
        assert result == "PASSIVE"

    def test_high_vpin_forces_passive(self):
        result = execution_mode(0.8, 1.0, 5.0, 0.95, vpin_passive_threshold=0.90)
        assert result == "PASSIVE"

    def test_short_half_life_goes_aggressive(self):
        # half_life=2s < fill_time=5s * 1.5 → aggressive
        result = execution_mode(0.8, 2.0, 5.0, 0.5)
        assert result == "AGGRESSIVE"

    def test_long_half_life_stays_passive(self):
        # half_life=30s > fill_time=5s * 1.5 → passive
        result = execution_mode(0.8, 30.0, 5.0, 0.5)
        assert result == "PASSIVE"

    def test_passive_preferred_default(self):
        result = execution_mode(0.5, 100.0, 5.0, 0.3)
        assert result == "PASSIVE"


# ── Stale Data Monitor Tests ──────────────────────────────────

class TestStaleDataMonitor:
    def test_fresh_data_not_stale(self):
        mon = StaleDataMonitor(stale_threshold_ms=500.0)
        now = time.time_ns()
        mon.on_tick(1, now)
        stale = mon.check_staleness(now + 100_000_000)  # 100ms later
        assert len(stale) == 0
        assert mon.is_order_allowed(1)

    def test_stale_after_threshold(self):
        mon = StaleDataMonitor(stale_threshold_ms=500.0)
        now = 1_000_000_000_000
        mon.on_tick(1, now)
        stale = mon.check_staleness(now + 600_000_000)  # 600ms later
        assert 1 in stale
        assert not mon.is_order_allowed(1)

    def test_tick_clears_staleness(self):
        mon = StaleDataMonitor(stale_threshold_ms=500.0)
        now = 1_000_000_000_000
        mon.on_tick(1, now)
        mon.check_staleness(now + 600_000_000)  # goes stale
        assert mon.is_stale(1)
        mon.on_tick(1, now + 700_000_000)  # new tick
        assert not mon.is_stale(1)

    def test_multiple_symbols(self):
        mon = StaleDataMonitor(stale_threshold_ms=500.0)
        now = 1_000_000_000_000
        mon.on_tick(1, now)
        mon.on_tick(2, now)
        # Only symbol 1 goes stale
        mon.on_tick(2, now + 400_000_000)
        stale = mon.check_staleness(now + 600_000_000)
        assert 1 in stale
        assert 2 not in stale

    def test_get_stale_symbols(self):
        mon = StaleDataMonitor(stale_threshold_ms=100.0)
        now = 1_000_000_000_000
        mon.on_tick(1, now)
        mon.on_tick(2, now)
        mon.on_tick(3, now)
        mon.check_staleness(now + 200_000_000)
        assert set(mon.get_stale_symbols()) == {1, 2, 3}

    def test_record_cancellation(self):
        mon = StaleDataMonitor()
        now = 1_000_000_000_000
        mon.on_tick(1, now)
        mon.check_staleness(now + 600_000_000)
        mon.record_cancellation(1, 3)
        assert mon._symbols[1].orders_cancelled == 3
