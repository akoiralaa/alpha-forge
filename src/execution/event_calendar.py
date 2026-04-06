"""
AlphaForge — Event Calendar & Blackout Engine
===============================================
Hard shutdown logic around scheduled macro events where historical
correlations break down and the model's edge is undefined.

The trader's answer: "Your Stat-Arb model shows a 3σ entry on ETH/BTC,
but the move is driven by a massive hack on an ETH-based bridge.
Do you take the trade?"

Answer: No. The model assumes cointegration based on historical correlation,
but a structural break (hack, regulatory shock, liquidity crisis) invalidates
the regime. Engage EVENT_BLACKOUT and wait for regime stabilisation.

Blackout windows
────────────────
FOMC decisions      : Day of + 1 day after  (rate decisions move vol term structure)
NFP (Jobs report)   : Day of only           (intraday vol spike, correlations unstable)
CPI/PCE             : Day of only
Earnings (per sym)  : 1 day before + day of (gap risk, binary event)
Custom events       : Inject ad-hoc via add_custom_blackout()

Blackout does not mean flat — it means:
  - No NEW position initiations
  - Existing positions held (closing orders still permitted)
  - Re-entry allowed once is_blackout() returns False
"""

from __future__ import annotations

from datetime import date, timedelta


# ── hardcoded FOMC decision dates (most recent 3 years) ──────────────────────
# Source: federalreserve.gov/monetarypolicy/fomccalendars.htm
# Update annually. Only decision day (not meeting start day).

_FOMC_DATES: set[date] = {
    # 2023
    date(2023, 2, 1), date(2023, 3, 22), date(2023, 5, 3),
    date(2023, 6, 14), date(2023, 7, 26), date(2023, 9, 20),
    date(2023, 11, 1), date(2023, 12, 13),
    # 2024
    date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1),
    date(2024, 6, 12), date(2024, 7, 31), date(2024, 9, 18),
    date(2024, 11, 7), date(2024, 12, 18),
    # 2025
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7),
    date(2025, 6, 18), date(2025, 7, 30), date(2025, 9, 17),
    date(2025, 11, 5), date(2025, 12, 17),
    # 2026
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29),
    date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
    date(2026, 11, 4), date(2026, 12, 16),
}

# NFP = first Friday of each month (approximation — verify against BLS calendar)
# CPI = typically 2nd or 3rd week of month — add manually or fetch from BLS
_NFP_DATES: set[date] = {
    # 2024
    date(2024, 1, 5), date(2024, 2, 2), date(2024, 3, 8),
    date(2024, 4, 5), date(2024, 5, 3), date(2024, 6, 7),
    date(2024, 7, 5), date(2024, 8, 2), date(2024, 9, 6),
    date(2024, 10, 4), date(2024, 11, 1), date(2024, 12, 6),
    # 2025
    date(2025, 1, 10), date(2025, 2, 7), date(2025, 3, 7),
    date(2025, 4, 4), date(2025, 5, 2), date(2025, 6, 6),
    date(2025, 7, 3), date(2025, 8, 1), date(2025, 9, 5),
    date(2025, 10, 3), date(2025, 11, 7), date(2025, 12, 5),
    # 2026
    date(2026, 1, 9), date(2026, 2, 6), date(2026, 3, 6),
    date(2026, 4, 3),
}


class EventCalendar:
    """
    Tracks scheduled macro events and per-symbol earnings blackout windows.

    Usage
    ─────
        cal = EventCalendar()
        cal.add_earnings_date("AAPL", date(2025, 5, 1))

        if cal.is_blackout(today):
            kill_switch.manual_halt("EVENT_BLACKOUT")

        active = cal.active_events(today)
        # → ["FOMC_DECISION", "EARNINGS:AAPL"]
    """

    def __init__(
        self,
        fomc_buffer_days_after: int = 1,
        nfp_buffer_days_after: int = 0,
        earnings_buffer_days_before: int = 1,
    ):
        self.fomc_buffer_after = fomc_buffer_days_after
        self.nfp_buffer_after = nfp_buffer_days_after
        self.earnings_buffer_before = earnings_buffer_days_before

        self._earnings: dict[str, set[date]] = {}   # symbol → {earnings_dates}
        self._custom: dict[date, str] = {}           # date → reason

    # ── public API ────────────────────────────────────────────────────────────

    def add_earnings_date(self, symbol: str, earnings_date: date) -> None:
        self._earnings.setdefault(symbol.upper(), set()).add(earnings_date)

    def add_custom_blackout(self, dt: date, reason: str) -> None:
        """
        Inject ad-hoc event: exchange outage, regulatory halt, known structural break.
        Example: add_custom_blackout(date(2022, 5, 12), "UST depeg / LUNA collapse")
        """
        self._custom[dt] = reason

    def is_blackout(self, dt: date, symbol: str | None = None) -> bool:
        """
        Returns True if dt falls within any blackout window.
        If symbol provided, also checks earnings blackout for that symbol.
        """
        return len(self.active_events(dt, symbol)) > 0

    def active_events(self, dt: date, symbol: str | None = None) -> list[str]:
        """Return list of active event names on this date."""
        events: list[str] = []

        # FOMC: decision day + N days after
        for fomc_day in _FOMC_DATES:
            if fomc_day <= dt <= fomc_day + timedelta(days=self.fomc_buffer_after):
                events.append("FOMC_DECISION")
                break

        # NFP: release day + N days after
        for nfp_day in _NFP_DATES:
            if nfp_day <= dt <= nfp_day + timedelta(days=self.nfp_buffer_after):
                events.append("NFP_RELEASE")
                break

        # Custom blackouts
        if dt in self._custom:
            events.append(f"CUSTOM:{self._custom[dt]}")

        # Per-symbol earnings
        if symbol:
            sym = symbol.upper()
            for edate in self._earnings.get(sym, set()):
                window_start = edate - timedelta(days=self.earnings_buffer_before)
                if window_start <= dt <= edate:
                    events.append(f"EARNINGS:{sym}")
                    break

        return events

    def blackout_series(self, index: "pd.DatetimeIndex") -> "pd.Series":
        """
        Vectorised: return a boolean Series aligned to a DatetimeIndex.
        True = blackout day, False = clear to trade.
        Useful for masking backtest weight updates on event days.
        """
        import pandas as pd
        result = pd.Series(False, index=index)
        for dt_ts in index:
            dt = dt_ts.date() if hasattr(dt_ts, "date") else dt_ts
            result.loc[dt_ts] = self.is_blackout(dt)
        return result

    def next_clear_date(self, from_date: date) -> date:
        """Return the first trading day after the current blackout window."""
        dt = from_date
        for _ in range(30):
            dt += timedelta(days=1)
            if dt.weekday() < 5 and not self.is_blackout(dt):
                return dt
        return from_date + timedelta(days=30)  # fallback
