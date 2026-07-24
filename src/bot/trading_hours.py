"""Trading hours guard for the live EODHD universe.

All schedules are expressed in UTC.  IG follows fixed UTC schedules; they do NOT
shift with BST/DST for most instruments (the London-time appearance of shifting is
because IG quotes UTC open/close times that happen to look like London clock times
in winter).

Asset categories and their IG spread-bet trading windows (UTC) — the symbol
set for each category is derived from EODHD_UNIVERSE, not hardcoded here:

  FOREX             — 24/5, Sun 21:00 – Fri 21:00, no daily break
  METALS (XAU/USD, XAG/USD)
                    — 24/5, Sun 23:00 – Fri 22:00, daily maintenance 22:00–23:00 UTC
  US_EQUITY (single-name shares)
                    — Mon–Fri 14:30–21:00 UTC (NYSE cash hours only),
                      excluding NYSE full-day holidays (see _US_EQUITY_HOLIDAYS)

Reference: IG dealing hours pages.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from bot.data.eodhd_symbols import EODHD_UNIVERSE

# ---------------------------------------------------------------------------
# Asset classification
# ---------------------------------------------------------------------------

# Derived from EODHD_UNIVERSE (the live universe's single source of truth) by
# asset_class, so a universe change can't drift out of sync with a
# hand-maintained parallel list the way the old hardcoded frozensets did.
# is_market_open fails open for unknown symbols (24/7), so a pair missing
# from EODHD_UNIVERSE is treated as tradeable on weekends — pinned by
# tests/test_trading_hours.py::TestUniverseCoverage.
_FOREX = frozenset(s.bot_key for s in EODHD_UNIVERSE.values() if s.asset_class == "forex")

# Metals are IG-native since 2026-06-19 (IGCandleLSFeed streams IG spot 24/5),
# so the lenient 24/5 window here is correct for both entries and closes.
_METALS = frozenset(s.bot_key for s in EODHD_UNIVERSE.values() if s.asset_class == "metal")

# US single-name shares on the IG US-share DFB; NYSE RTH window.
_US_EQUITY = frozenset(s.bot_key for s in EODHD_UNIVERSE.values() if s.asset_class == "equity")

# NYSE full-day closures (UTC dates).  The US-share/ETF instruments on IG follow
# the NYSE calendar, and EODHD sources their ticks from US venues — so on these
# days no candles form AND the IG market is shut, meaning entries can't fire and
# positions can't be closed.  Without this table the weekday session check
# (_us_equity_open) would report the market *open* on a holiday (e.g. Juneteenth
# 2026-06-19, a Friday), wasting top-K slots on un-fillable US names.
#
# Full-day closures only — NYSE half-days (early 18:00 ET close, e.g. day after
# Thanksgiving) are intentionally NOT modelled; an entry near the open is still
# valid and the cost of a late half-day entry is small versus a full closure.
#
# ⚠️ MAINTENANCE: floating holidays shift each year — extend this table annually.
# Years absent here fall through to the plain weekday session (fail-open), which
# re-introduces the holiday bug, so keep at least the current + next year listed.
_US_EQUITY_HOLIDAYS = frozenset(
    [
        # 2026
        date(2026, 1, 1),  # New Year's Day
        date(2026, 1, 19),  # Martin Luther King Jr. Day
        date(2026, 2, 16),  # Washington's Birthday (Presidents' Day)
        date(2026, 4, 3),  # Good Friday
        date(2026, 5, 25),  # Memorial Day
        date(2026, 6, 19),  # Juneteenth
        date(2026, 7, 3),  # Independence Day (observed; Jul 4 is Sat)
        date(2026, 9, 7),  # Labor Day
        date(2026, 11, 26),  # Thanksgiving Day
        date(2026, 12, 25),  # Christmas Day
        # 2027
        date(2027, 1, 1),  # New Year's Day
        date(2027, 1, 18),  # Martin Luther King Jr. Day
        date(2027, 2, 15),  # Washington's Birthday (Presidents' Day)
        date(2027, 3, 26),  # Good Friday
        date(2027, 5, 31),  # Memorial Day
        date(2027, 6, 18),  # Juneteenth (observed; Jun 19 is Sat)
        date(2027, 7, 5),  # Independence Day (observed; Jul 4 is Sun)
        date(2027, 9, 6),  # Labor Day
        date(2027, 11, 25),  # Thanksgiving Day
        date(2027, 12, 24),  # Christmas Day (observed; Dec 25 is Sat)
    ]
)


def is_us_equity_holiday(dt: datetime) -> bool:
    """True if *dt* falls on a known NYSE full-day closure (see table caveats)."""
    return dt.date() in _US_EQUITY_HOLIDAYS


def _weekday(dt: datetime) -> int:
    """Return ISO weekday: Mon=1 … Sun=7."""
    return dt.isoweekday()


def is_market_open(symbol: str, now: datetime | None = None) -> bool:
    """Return True if the IG spread-bet market for *symbol* is open right now.

    This is the *lenient* check — it reports the market as open whenever the
    underlying instrument is technically tradeable.  Use ``is_safe_for_entry``
    for the stricter pre-entry check that adds buffers around IG's daily
    funding/maintenance windows (when entries often hit ``MARKET_CLOSED_WITH_EDITS``
    or are silently rolled by IG demo).

    Args:
        symbol: Canonical bot symbol (bot_key), e.g. ``"EUR/USD"``, ``"F"``.
        now:    UTC datetime to check (defaults to ``datetime.now(UTC)``).

    Returns:
        True if tradeable, False if market is closed or in maintenance.
    """
    if now is None:
        now = datetime.now(UTC)
    elif now.tzinfo is None:
        raise ValueError("'now' must be timezone-aware")

    if symbol in _FOREX:
        return _forex_open(now)
    if symbol in _METALS:
        return _metals_open(now)
    if symbol in _US_EQUITY:
        return _us_equity_open(now)
    # Unknown symbol — assume tradeable (fail open so we don't silently block new assets)
    return True


# ---------------------------------------------------------------------------
# Market categorisation + human-readable schedules (for the dashboard reference)
# ---------------------------------------------------------------------------

# Stable category key → (display label, human schedule string in UTC).  The
# schedule strings mirror the windows enforced by the per-category helpers
# below; keep them in sync if a window ever changes.
MARKET_CATEGORIES: dict[str, tuple[str, str]] = {
    "forex": ("Forex", "24/5 · Sun 21:00 – Fri 21:00 UTC"),
    "metals": ("Metals", "24/5 · Sun 23:00 – Fri 22:00 UTC (daily maint. 22:00–23:00)"),
    "us_equity": ("US shares", "Mon–Fri 14:30–21:00 UTC"),
}


def market_category(symbol: str) -> str:
    """Return the market-category key (see :data:`MARKET_CATEGORIES`) for *symbol*.

    Unknown symbols fall into ``"other"`` (treated as always-open by
    :func:`is_market_open`).
    """
    if symbol in _FOREX:
        return "forex"
    if symbol in _METALS:
        return "metals"
    if symbol in _US_EQUITY:
        return "us_equity"
    return "other"


def seconds_until_open(symbol: str, now: datetime | None = None) -> int | None:
    """Seconds until *symbol*'s market next opens, or ``0`` if it is open now.

    Uses the lenient :func:`is_market_open` gate (the "is the market currently
    closed" question the dashboard countdown answers).  Scans forward at
    minute resolution — every schedule transition here is minute-aligned — over
    a 7-day horizon (longer than any weekend/maintenance gap).  Returns ``None``
    if no open is found within the horizon (should not happen for known assets).
    """
    if now is None:
        now = datetime.now(UTC)
    elif now.tzinfo is None:
        raise ValueError("'now' must be timezone-aware")
    if is_market_open(symbol, now):
        return 0
    probe = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    horizon = now + timedelta(days=7)
    while probe <= horizon:
        if is_market_open(symbol, probe):
            return int((probe - now).total_seconds())
        probe += timedelta(minutes=1)
    return None


_BAR_MS = 3_600_000  # 1h bar in milliseconds
# Look-back horizon when hunting for the last in-session hour: long enough to
# clear a weekend adjacent to a holiday (e.g. Fri holiday + Sat/Sun) without
# ever looping unbounded on a genuinely never-open symbol.
_MAX_BACKFILL_LOOKBACK_HOURS = 8 * 24


def last_expected_closed_bar_ms(symbol: str, now: datetime | None = None) -> int | None:
    """UTC epoch-ms open-timestamp of the most recent 1h bar that has *fully
    closed* and fell within an open-market hour for *symbol*.

    A 1h bar is stamped at its open and confirms at its close (the next ``:00``),
    so the newest closed bar opens at ``floor(now, 1h) - 1h``.  Walking back from
    there, this returns the first hour the market was open (lenient
    :func:`is_market_open`, checked at the bar's *open* — so the trailing bar
    before a session close is included and the maintenance/closed hours are
    skipped).  That is the newest candle a healthy feed must already hold.

    Feeds compare a symbol's newest buffered candle against this to detect a
    silently-stalled feed independently of buffer *depth* — the count-based skip
    alone lets a full-but-stale buffer skip the repair backfill (2026-07-05 gap).

    Returns ``None`` only if no open-market hour is found within an 8-day
    look-back — a safety fallback for unknown/never-open symbols; for known
    assets a weekend/holiday gap always resolves well inside the horizon.
    """
    if now is None:
        now = datetime.now(UTC)
    elif now.tzinfo is None:
        raise ValueError("'now' must be timezone-aware")

    # Open ts of the last fully-closed 1h bar (the current hour's bar is still
    # forming), then walk back to the most recent hour the market was open.
    bar_open = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    for _ in range(_MAX_BACKFILL_LOOKBACK_HOURS):
        if is_market_open(symbol, bar_open):
            return int(bar_open.timestamp() * 1000)
        bar_open -= timedelta(hours=1)
    return None


def is_safe_for_entry(symbol: str, now: datetime | None = None) -> bool:
    """Stricter variant of :func:`is_market_open` for the *entry* path.

    Adds buffers around IG's daily reconciliation/funding windows where
    positions may be unclosable or silently rolled on demo:

      * **Forex** — block 21:50–22:10 UTC (the 22:00 UTC daily funding tick;
        positions opened here have hit ``MARKET_CLOSED_WITH_EDITS`` and silent
        server-side closes during the funding flip).
      * **Metals** — extend the existing 22:00–23:00 UTC maintenance break
        to 21:55–23:05 UTC.

    Closes (stop-loss, TP, signal exits) still use the lenient
    ``is_market_open`` — a stuck position should always be allowed to attempt
    a close.  The deferred-close path catches ``MarketClosedError`` from the
    REST endpoint when the window is hit.
    """
    if not is_market_open(symbol, now):
        return False
    if now is None:
        now = datetime.now(UTC)
    hm = now.hour * 60 + now.minute
    # 21:50–22:10 UTC forex funding window; 21:55–23:05 UTC widened metals buffer.
    if symbol in _FOREX and 21 * 60 + 50 <= hm < 22 * 60 + 10:
        return False
    return not (symbol in _METALS and 21 * 60 + 55 <= hm <= 23 * 60 + 5)


def in_equity_mark_blackout(now: datetime | None = None) -> bool:
    """True during the daily IG rollover/maintenance window where account-level
    equity marks are unreliable.

    Covers the union of the FX 22:00 funding tick (21:50–22:10) and the metals
    22:00–23:00 maintenance break (widened to 23:05) → **21:50–23:05 UTC**.  IG's
    gold/silver and FX marks go stale/wide here; a single bad mark spiked the
    account equity and tripped the RED drawdown breaker on 2026-06-05.  The
    drawdown breaker freezes during this window — and since new entries are
    already blocked by ``is_safe_for_entry`` across the same period, freezing the
    entry-halt costs nothing.
    """
    if now is None:
        now = datetime.now(UTC)
    hm = now.hour * 60 + now.minute
    return 21 * 60 + 50 <= hm <= 23 * 60 + 5


# ---------------------------------------------------------------------------
# Per-category schedule helpers
# ---------------------------------------------------------------------------


def _forex_open(dt: datetime) -> bool:
    """24/5: Sun 21:00 UTC – Fri 21:00 UTC, no daily break."""
    wd = _weekday(dt)
    h = dt.hour
    m = dt.minute
    hm = h * 60 + m  # minutes since midnight UTC

    if wd == 6:  # Saturday — always closed
        return False
    if wd == 7:  # Sunday — opens at 21:00
        return hm >= 21 * 60
    if wd == 5:  # Friday — closes at 21:00
        return hm < 21 * 60
    return True  # Mon–Thu always open


def _metals_open(dt: datetime) -> bool:
    """24/5: Sun 23:00 – Fri 22:00 UTC, with daily 22:00–23:00 UTC maintenance."""
    wd = _weekday(dt)
    h = dt.hour
    m = dt.minute
    hm = h * 60 + m

    if wd == 6:  # Saturday — closed
        return False
    if wd == 7:  # Sunday — opens at 23:00
        return hm >= 23 * 60
    if wd == 5:  # Friday — closes at 22:00
        return hm < 22 * 60
    # Mon–Thu: open except 22:00–23:00 maintenance window
    return not (22 * 60 <= hm < 23 * 60)


def _us_equity_open(dt: datetime) -> bool:
    """NYSE cash session: Mon–Fri 14:30–21:00 UTC, excluding full-day holidays."""
    wd = _weekday(dt)
    if wd >= 6:  # Sat or Sun
        return False
    if is_us_equity_holiday(dt):
        return False
    hm = dt.hour * 60 + dt.minute
    return 14 * 60 + 30 <= hm < 21 * 60
