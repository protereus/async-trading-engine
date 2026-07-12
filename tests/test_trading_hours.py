"""Tests for trading_hours.is_market_open()."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from bot.trading_hours import (
    MARKET_CATEGORIES,
    is_market_open,
    is_safe_for_entry,
    is_us_equity_holiday,
    last_expected_closed_bar_ms,
    market_category,
    seconds_until_open,
)


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def utc(weekday_iso: int, hour: int, minute: int = 0) -> datetime:
    """Build a UTC datetime for a given ISO weekday (Mon=1..Sun=7) and time.

    Uses a known reference week: Mon=2026-04-13 (April 13, 2026).
    """
    base_monday = datetime(2026, 4, 13, tzinfo=UTC)
    delta_days = weekday_iso - 1
    from datetime import timedelta

    return base_monday.replace(hour=hour, minute=minute) + timedelta(days=delta_days)


# ---------------------------------------------------------------------------
# Forex (24/5 Sun 21:00 – Fri 21:00, no daily break)
# ---------------------------------------------------------------------------


class TestForex:
    def test_monday_open(self) -> None:
        assert is_market_open("EUR/USD", utc(1, 12)) is True

    def test_thursday_midnight(self) -> None:
        assert is_market_open("GBP/JPY", utc(4, 0)) is True

    def test_friday_before_close(self) -> None:
        assert is_market_open("USD/JPY", utc(5, 20, 59)) is True

    def test_friday_at_close(self) -> None:
        assert is_market_open("USD/JPY", utc(5, 21, 0)) is False

    def test_saturday_closed(self) -> None:
        assert is_market_open("EUR/USD", utc(6, 12)) is False

    def test_sunday_before_open(self) -> None:
        assert is_market_open("EUR/USD", utc(7, 20, 59)) is False

    def test_sunday_at_open(self) -> None:
        assert is_market_open("EUR/USD", utc(7, 21, 0)) is True

    def test_various_forex_pairs(self) -> None:
        t = utc(2, 10)  # Tuesday 10:00
        for sym in ["GBP/USD", "USD/CAD", "AUD/JPY", "EUR/AUD", "NZD/USD", "EUR/JPY"]:
            assert is_market_open(sym, t) is True


# ---------------------------------------------------------------------------
# Metals (XAU/USD, XAG/USD) — 24/5 Sun 23:00 – Fri 22:00, daily break 22:00-23:00
# ---------------------------------------------------------------------------


class TestMetals:
    def test_monday_daytime(self) -> None:
        assert is_market_open("XAU/USD", utc(1, 12)) is True

    def test_daily_maintenance_window(self) -> None:
        # 22:00 UTC — closed for maintenance
        assert is_market_open("XAU/USD", utc(2, 22, 0)) is False
        assert is_market_open("XAG/USD", utc(3, 22, 30)) is False

    def test_after_maintenance(self) -> None:
        assert is_market_open("XAU/USD", utc(2, 23, 0)) is True

    def test_saturday_closed(self) -> None:
        assert is_market_open("XAG/USD", utc(6, 12)) is False

    def test_sunday_before_open(self) -> None:
        assert is_market_open("XAU/USD", utc(7, 22, 59)) is False

    def test_sunday_at_open(self) -> None:
        assert is_market_open("XAU/USD", utc(7, 23, 0)) is True

    def test_friday_before_close(self) -> None:
        assert is_market_open("XAG/USD", utc(5, 21, 59)) is True

    def test_friday_at_close(self) -> None:
        assert is_market_open("XAG/USD", utc(5, 22, 0)) is False


# ---------------------------------------------------------------------------
# US Equity (14 single-name shares) — Mon–Fri 14:30–21:00 UTC
# ---------------------------------------------------------------------------


class TestUSEquity:
    def test_weekday_during_session(self) -> None:
        assert is_market_open("F", utc(2, 16)) is True
        assert is_market_open("T", utc(3, 18)) is True
        assert is_market_open("PFE", utc(4, 14, 30)) is True
        assert is_market_open("VZ", utc(1, 20, 59)) is True
        assert is_market_open("XOM", utc(5, 15)) is True

    def test_weekday_before_open(self) -> None:
        assert is_market_open("F", utc(2, 14, 29)) is False
        assert is_market_open("XOM", utc(1, 10)) is False

    def test_weekday_at_close(self) -> None:
        assert is_market_open("T", utc(3, 21, 0)) is False

    def test_weekend_closed(self) -> None:
        assert is_market_open("F", utc(6, 16)) is False
        assert is_market_open("XOM", utc(7, 16)) is False

    def test_eodhd_equities_follow_nyse_session(self) -> None:
        # EODHD migration: the 14 US single-name shares use the same RTH window.
        for sym in ("F", "XOM", "BAC", "PFE", "INTC", "T"):
            assert is_market_open(sym, utc(2, 16)) is True  # mid-session
            assert is_market_open(sym, utc(2, 14, 29)) is False  # pre-open
            assert is_market_open(sym, utc(6, 16)) is False  # weekend

    def test_eodhd_silver_uses_metals_window(self) -> None:
        # XAG/USD (SLV-sourced) follows the 24/5 metals window like XAU/USD.
        assert is_market_open("XAG/USD", utc(1, 12)) is True
        assert is_market_open("XAG/USD", utc(2, 22, 0)) is False  # maintenance


class TestUSEquityHolidays:
    # Juneteenth 2026-06-19 is a Friday inside the 14:30–21:00 session window;
    # without the holiday table the weekday check would report it open.
    JUNETEENTH_2026 = datetime(2026, 6, 19, 18, 0, tzinfo=UTC)
    NORMAL_WED_2026 = datetime(2026, 6, 17, 18, 0, tzinfo=UTC)

    def test_us_share_closed_on_holiday(self) -> None:
        for sym in ("XOM", "F", "BMY", "PFE"):
            assert is_market_open(sym, self.JUNETEENTH_2026) is False
            assert is_safe_for_entry(sym, self.JUNETEENTH_2026) is False

    def test_us_share_open_on_normal_weekday(self) -> None:
        assert is_market_open("XOM", self.NORMAL_WED_2026) is True
        assert is_safe_for_entry("XOM", self.NORMAL_WED_2026) is True

    def test_forex_unaffected_by_us_holiday(self) -> None:
        # FX does not follow the NYSE calendar — must stay open on a US holiday.
        assert is_market_open("EUR/USD", self.JUNETEENTH_2026) is True

    def test_is_us_equity_holiday_helper(self) -> None:
        assert is_us_equity_holiday(self.JUNETEENTH_2026) is True
        assert is_us_equity_holiday(self.NORMAL_WED_2026) is False
        # Observed-date shifts: Independence Day 2026 lands on Fri Jul 3.
        assert is_us_equity_holiday(datetime(2026, 7, 3, 16, tzinfo=UTC)) is True
        assert is_us_equity_holiday(datetime(2026, 7, 4, 16, tzinfo=UTC)) is False


# ---------------------------------------------------------------------------
# Unknown symbol — fail open
# ---------------------------------------------------------------------------


class TestUnknown:
    def test_unknown_symbol_returns_true(self) -> None:
        assert is_market_open("UNKNOWN/SYM", utc(2, 12)) is True

    def test_crypto_symbols_fail_open(self) -> None:
        # BTC/USD and ETH/USD removed from universe; they are now "unknown" — fail open
        assert is_market_open("BTC/USD", utc(6, 12)) is True  # would be closed if still classified
        assert is_market_open("ETH/USD", utc(6, 12)) is True


# ---------------------------------------------------------------------------
# Timezone-aware requirement
# ---------------------------------------------------------------------------


class TestTimezone:
    def test_naive_datetime_raises(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            is_market_open("EUR/USD", datetime(2026, 4, 14, 12, 0))  # naive


# ---------------------------------------------------------------------------
# is_safe_for_entry — funding/maintenance buffer around 22:00 UTC
# ---------------------------------------------------------------------------


class TestSafeForEntryForex:
    def test_outside_buffer_allows_entry(self) -> None:
        assert is_safe_for_entry("EUR/USD", utc(2, 14)) is True
        assert is_safe_for_entry("GBP/USD", utc(2, 21, 49)) is True
        assert is_safe_for_entry("USD/JPY", utc(2, 22, 10)) is True

    def test_inside_funding_window_blocks_entry(self) -> None:
        # 21:50 UTC — funding-window start
        assert is_safe_for_entry("EUR/USD", utc(2, 21, 50)) is False
        # 22:00 UTC — peak of the funding tick
        assert is_safe_for_entry("GBP/USD", utc(2, 22, 0)) is False
        # 22:09 UTC — last minute of the window
        assert is_safe_for_entry("USD/JPY", utc(2, 22, 9)) is False

    def test_closed_market_still_returns_false(self) -> None:
        # Saturday — market closed, entry obviously blocked
        assert is_safe_for_entry("EUR/USD", utc(6, 12)) is False


class TestSafeForEntryMetals:
    def test_inside_widened_maintenance_blocks(self) -> None:
        # 21:55 UTC — 5 min before normal 22:00 break
        assert is_safe_for_entry("XAU/USD", utc(2, 21, 55)) is False
        # 23:05 UTC — 5 min after normal 23:00 reopen
        assert is_safe_for_entry("XAG/USD", utc(2, 23, 5)) is False
        # 22:30 UTC — already blocked by is_market_open
        assert is_safe_for_entry("XAU/USD", utc(2, 22, 30)) is False

    def test_outside_widened_maintenance_allows(self) -> None:
        assert is_safe_for_entry("XAU/USD", utc(2, 21, 54)) is True
        # 23:06 UTC — just past widened buffer
        assert is_safe_for_entry("XAG/USD", utc(2, 23, 6)) is True


class TestSafeForEntryUSEquity:
    def test_inside_session_unaffected(self) -> None:
        # US equity has no funding-window buffer; just defers to is_market_open
        assert is_safe_for_entry("F", utc(2, 16)) is True

    def test_closed_session_blocks(self) -> None:
        assert is_safe_for_entry("F", utc(2, 22, 0)) is False


# ---------------------------------------------------------------------------
# Market categorisation + closed-countdown (dashboard reference)
# ---------------------------------------------------------------------------


class TestMarketCategory:
    def test_known_symbols(self) -> None:
        assert market_category("EUR/USD") == "forex"
        assert market_category("XAU/USD") == "metals"
        assert market_category("F") == "us_equity"
        assert market_category("XAG/USD") == "metals"

    def test_unknown_symbol_is_other(self) -> None:
        assert market_category("NOPE") == "other"

    def test_every_category_has_label_and_schedule(self) -> None:
        for label, schedule in MARKET_CATEGORIES.values():
            assert label and schedule


class TestSecondsUntilOpen:
    def test_open_market_returns_zero(self) -> None:
        assert seconds_until_open("EUR/USD", utc(3, 12)) == 0

    def test_forex_closed_saturday_counts_to_sunday_open(self) -> None:
        # Sat 12:00 → forex reopens Sun 21:00 UTC == 33h.
        assert seconds_until_open("EUR/USD", utc(6, 12)) == 33 * 3600

    def test_metals_closed_saturday_counts_to_sunday_2300(self) -> None:
        # Sat 12:00 → metals reopen Sun 23:00 UTC == 35h.
        assert seconds_until_open("XAU/USD", utc(6, 12)) == 35 * 3600

    def test_us_equity_after_close_counts_to_next_session(self) -> None:
        # Wed 21:00 (just closed) → next open Thu 14:30 UTC == 17h30m.
        assert seconds_until_open("F", utc(3, 21, 0)) == (17 * 60 + 30) * 60


# ---------------------------------------------------------------------------
# last_expected_closed_bar_ms — newest 1h bar a healthy feed must already hold
# ---------------------------------------------------------------------------


class TestLastExpectedClosedBar:
    def test_forex_midweek_is_prior_hour(self) -> None:
        # Thu 12:30 → last closed 1h bar opened at 11:00 (closes 12:00 ≤ now).
        assert last_expected_closed_bar_ms("EUR/USD", utc(4, 12, 30)) == _ms(utc(4, 11, 0))

    def test_forex_on_the_hour_is_two_hours_back(self) -> None:
        # Thu 12:00 exactly → the 11:00 bar closed at 12:00, so it is expected.
        assert last_expected_closed_bar_ms("EUR/USD", utc(4, 12, 0)) == _ms(utc(4, 11, 0))

    def test_forex_sunday_reopen_incident(self) -> None:
        # The 2026-07-05 case: Sun 23:53 → the 22:00 bar (open 21:00 reopen) has
        # closed, so a healthy feed must hold the Sun 22:00 bar.
        assert last_expected_closed_bar_ms("EUR/USD", utc(7, 23, 53)) == _ms(utc(7, 22, 0))

    def test_forex_just_after_reopen_walks_back_to_friday(self) -> None:
        # Sun 21:30 → the 20:00 bar is pre-reopen (closed market); newest real
        # bar is Friday's last (opened 20:00, closed 21:00).
        assert last_expected_closed_bar_ms("EUR/USD", utc(7, 21, 30)) == _ms(utc(5, 20, 0))

    def test_forex_saturday_walks_back_to_friday_last_bar(self) -> None:
        assert last_expected_closed_bar_ms("EUR/USD", utc(6, 12, 0)) == _ms(utc(5, 20, 0))

    def test_us_equity_in_session(self) -> None:
        # Wed 16:30 → the 15:00 bar has closed within RTH.
        assert last_expected_closed_bar_ms("F", utc(3, 16, 30)) == _ms(utc(3, 15, 0))

    def test_us_equity_after_close_is_last_rth_bar(self) -> None:
        # Wed 22:00 → 21:00 bar is post-close; last RTH bar opened 20:00.
        assert last_expected_closed_bar_ms("F", utc(3, 22, 0)) == _ms(utc(3, 20, 0))

    def test_us_equity_just_after_open_has_no_new_bar(self) -> None:
        # Wed 14:45 → no in-session hourly bar has closed yet today; newest real
        # bar is Tuesday's last RTH bar (opened 20:00).
        assert last_expected_closed_bar_ms("F", utc(3, 14, 45)) == _ms(utc(2, 20, 0))

    def test_us_equity_holiday_walks_back_a_day(self) -> None:
        # Juneteenth 2026-06-19 (Fri holiday) 16:00 → walk back to Thu 06-18 20:00.
        friday = datetime(2026, 6, 19, 16, 0, tzinfo=UTC)
        expected = datetime(2026, 6, 18, 20, 0, tzinfo=UTC)
        assert last_expected_closed_bar_ms("F", friday) == _ms(expected)

    def test_metals_skips_maintenance_hour(self) -> None:
        # Wed 23:30 → the 22:00 bar falls in the 22:00–23:00 maintenance break
        # (no bar), so the last expected bar opened at 21:00.
        assert last_expected_closed_bar_ms("XAU/USD", utc(3, 23, 30)) == _ms(utc(3, 21, 0))

    def test_unknown_symbol_is_always_open(self) -> None:
        # Fail-open: unknown symbols report the plain prior hour.
        assert last_expected_closed_bar_ms("FOO/BAR", utc(6, 3, 30)) == _ms(utc(6, 2, 0))

    def test_naive_now_raises(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            last_expected_closed_bar_ms("EUR/USD", datetime(2026, 4, 16, 12, 0))  # noqa: DTZ001
