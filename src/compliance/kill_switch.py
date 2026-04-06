"""
AlphaForge — Compliance & Kill Switch Layer
============================================
Hard boundary logic that sits between signal generation and order submission.
Every check here is a gate: fail any one and the order is rejected with a reason code.

This layer answers the PM question: "What stops your bot from buying the entire
exchange if Polygon sends a bad print?"

Checks implemented
──────────────────
1. BAD_PRINT_FILTER       Price deviates > N% from rolling anchor — reject stale/erroneous ticks
2. MAX_ORDER_NOTIONAL     Single ticket cannot exceed a hard dollar cap
3. MAX_POSITION_NOTIONAL  Resulting position cannot exceed per-symbol NAV cap
4. WASH_TRADE_GUARD       Cannot submit a buy if an open sell exists for the same symbol (and vice versa)
5. EVENT_BLACKOUT         Hard flatten / no-new-orders window around FOMC, NFP, earnings
6. DRAWDOWN_HALT          Portfolio-level drawdown exceeds emergency threshold → halt all new orders
7. ADVERSE_SELECTION_TAX  Passive limit fills assumed to face immediate 0.5× spread adverse move —
                          applied as a cost adjustment to expected edge before order submission

Usage
─────
    from src.compliance.kill_switch import KillSwitch, OrderIntent

    ks = KillSwitch(nav=10_000_000, max_order_notional=500_000)
    intent = OrderIntent(symbol="AAPL", side="buy", quantity=100, limit_price=185.00,
                         order_type="limit", is_passive=True)
    result = ks.validate(intent, current_nav=9_800_000, portfolio_drawdown=-0.04)
    if not result.approved:
        log.warning(f"Order rejected: {result.reason}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd


# ── data classes ─────────────────────────────────────────────────────────────

@dataclass
class OrderIntent:
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float                    # shares / contracts
    limit_price: float                 # proposed execution price
    order_type: Literal["market", "limit"] = "limit"
    is_passive: bool = True            # passive = at-the-queue; aggressive = cross-spread
    estimated_spread_bps: float = 10.0 # bid-ask spread estimate in bps


@dataclass
class ValidationResult:
    approved: bool
    reason: str = ""
    adjusted_edge_bps: float = 0.0    # edge after adverse-selection tax
    warnings: list[str] = field(default_factory=list)


@dataclass
class PriceAnchor:
    """Rolling price reference used by bad-print filter."""
    symbol: str
    anchor_price: float
    updated_at: datetime
    n_observations: int = 1


# ── kill switch ───────────────────────────────────────────────────────────────

class KillSwitch:
    """
    Stateful compliance gate. Instantiate once per session; call validate() per order.

    Parameters
    ──────────
    nav                     Starting NAV (updated on each validate call)
    max_order_notional      Hard cap per single order ticket ($)
    max_position_pct        Max resulting position size as % of NAV
    bad_print_pct           Reject if price deviates > this % from anchor
    drawdown_halt_threshold Portfolio DD that triggers full trading halt
    adverse_selection_bps   Assumed adverse move after passive fill (bps)
    """

    def __init__(
        self,
        nav: float = 10_000_000,
        max_order_notional: float = 500_000,
        max_position_pct: float = 0.06,
        bad_print_pct: float = 0.03,
        drawdown_halt_threshold: float = -0.25,
        adverse_selection_bps: float = 5.0,   # 0.5 × 10 bps typical spread
    ):
        self.nav = nav
        self.max_order_notional = max_order_notional
        self.max_position_pct = max_position_pct
        self.bad_print_pct = bad_print_pct
        self.drawdown_halt_threshold = drawdown_halt_threshold
        self.adverse_selection_bps = adverse_selection_bps

        self._price_anchors: dict[str, PriceAnchor] = {}
        self._open_orders: dict[str, list[OrderIntent]] = {}  # symbol → [pending orders]
        self._halted: bool = False
        self._halt_reason: str = ""

    # ── public API ────────────────────────────────────────────────────────────

    def validate(
        self,
        intent: OrderIntent,
        current_nav: float,
        portfolio_drawdown: float,
        event_blackout: bool = False,
        open_position_notional: float = 0.0,
    ) -> ValidationResult:
        """
        Run all compliance checks. Returns ValidationResult with approved=True/False.
        Checks are ordered from cheapest to most expensive computation.
        """
        self.nav = current_nav
        warnings: list[str] = []

        # 1. Hard halt — portfolio DD or manual halt
        if self._halted:
            return ValidationResult(False, f"TRADING_HALTED: {self._halt_reason}")

        if portfolio_drawdown <= self.drawdown_halt_threshold:
            self._halted = True
            self._halt_reason = (
                f"portfolio DD {portfolio_drawdown:.1%} breached "
                f"halt threshold {self.drawdown_halt_threshold:.1%}"
            )
            return ValidationResult(False, f"DRAWDOWN_HALT: {self._halt_reason}")

        # 2. Event blackout — no new orders around binary events
        if event_blackout:
            return ValidationResult(
                False,
                "EVENT_BLACKOUT: hard shutdown active around scheduled macro event "
                "(FOMC/NFP/earnings). Historical correlations may not hold. "
                "Await regime stabilisation before re-entry."
            )

        # 3. Bad print filter
        anchor_check = self._check_bad_print(intent)
        if not anchor_check.approved:
            return anchor_check

        # 4. Order notional cap
        order_notional = intent.quantity * intent.limit_price
        if order_notional > self.max_order_notional:
            return ValidationResult(
                False,
                f"MAX_ORDER_NOTIONAL: ${order_notional:,.0f} exceeds "
                f"cap ${self.max_order_notional:,.0f}"
            )

        # 5. Position size cap
        resulting_notional = open_position_notional + order_notional
        max_notional = self.nav * self.max_position_pct
        if resulting_notional > max_notional:
            return ValidationResult(
                False,
                f"MAX_POSITION_NOTIONAL: resulting ${resulting_notional:,.0f} "
                f"exceeds {self.max_position_pct:.0%} NAV cap (${max_notional:,.0f})"
            )

        # 6. Wash trade guard
        wash_check = self._check_wash_trade(intent)
        if not wash_check.approved:
            return wash_check

        # 7. Adverse selection tax on passive limits
        adjusted_edge_bps = 0.0
        if intent.order_type == "limit" and intent.is_passive:
            adjusted_edge_bps = -self.adverse_selection_bps
            warnings.append(
                f"ADVERSE_SELECTION: passive limit faces estimated "
                f"{self.adverse_selection_bps:.1f} bps adverse move on fill. "
                f"Ensure signal edge > {self.adverse_selection_bps:.1f} bps to justify."
            )

        # Register order as pending
        self._open_orders.setdefault(intent.symbol, []).append(intent)

        return ValidationResult(
            approved=True,
            reason="OK",
            adjusted_edge_bps=adjusted_edge_bps,
            warnings=warnings,
        )

    def update_price_anchor(self, symbol: str, price: float) -> None:
        """Call this every time a new quote/bar arrives for a symbol."""
        if symbol not in self._price_anchors:
            self._price_anchors[symbol] = PriceAnchor(
                symbol=symbol, anchor_price=price, updated_at=datetime.utcnow()
            )
        else:
            anchor = self._price_anchors[symbol]
            # Exponential moving anchor — weight 10% new price, 90% existing
            anchor.anchor_price = 0.90 * anchor.anchor_price + 0.10 * price
            anchor.updated_at = datetime.utcnow()
            anchor.n_observations += 1

    def confirm_fill(self, symbol: str, side: str) -> None:
        """Remove order from pending queue after confirmed fill."""
        if symbol in self._open_orders:
            self._open_orders[symbol] = [
                o for o in self._open_orders[symbol] if o.side != side
            ]

    def manual_halt(self, reason: str) -> None:
        """PM or risk desk can trigger a hard halt at any time."""
        self._halted = True
        self._halt_reason = f"MANUAL: {reason}"

    def resume(self) -> None:
        """Clear halt state. Requires explicit resume — never auto-resumes."""
        self._halted = False
        self._halt_reason = ""

    @property
    def is_halted(self) -> bool:
        return self._halted

    # ── private checks ────────────────────────────────────────────────────────

    def _check_bad_print(self, intent: OrderIntent) -> ValidationResult:
        """
        Reject if price deviates more than bad_print_pct from the rolling anchor.
        Catches erroneous ticks (SPY at $0.01, AAPL at $10,000).
        """
        symbol = intent.symbol
        price = intent.limit_price

        if price <= 0:
            return ValidationResult(False, f"BAD_PRINT: {symbol} price <= 0 ({price})")

        if symbol not in self._price_anchors:
            # First observation — seed anchor and pass
            self.update_price_anchor(symbol, price)
            return ValidationResult(True, "OK")

        anchor = self._price_anchors[symbol].anchor_price
        if anchor <= 0:
            self.update_price_anchor(symbol, price)
            return ValidationResult(True, "OK")

        deviation = abs(price - anchor) / anchor
        if deviation > self.bad_print_pct:
            return ValidationResult(
                False,
                f"BAD_PRINT: {symbol} price ${price:.2f} deviates "
                f"{deviation:.1%} from anchor ${anchor:.2f} "
                f"(limit {self.bad_print_pct:.0%})"
            )

        self.update_price_anchor(symbol, price)
        return ValidationResult(True, "OK")

    def _check_wash_trade(self, intent: OrderIntent) -> ValidationResult:
        """
        Reject if there is already a pending order on the opposite side for the same symbol.
        Prevents self-crossing and wash trade violations.
        """
        pending = self._open_orders.get(intent.symbol, [])
        opposite = "sell" if intent.side == "buy" else "buy"
        for order in pending:
            if order.side == opposite:
                return ValidationResult(
                    False,
                    f"WASH_TRADE: {intent.symbol} has open {opposite} order "
                    f"(qty={order.quantity} @ ${order.limit_price:.2f}). "
                    f"Cancel existing order before submitting opposing side."
                )
        return ValidationResult(True, "OK")
