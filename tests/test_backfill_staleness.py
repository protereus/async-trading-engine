"""Freshness-aware startup backfill.

Covers the shared ``needs_backfill`` predicate (depth OR freshness) and its
wiring into ``EODHDFeed._backfill_below_threshold`` — the repair path for a
deep-but-stale buffer after a silent feed drop (the 2026-07-05 weekend gap).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from bot.config import BotConfig
from bot.core.event_bus import EventBus
from bot.core.models import Candle
from bot.data.backfill import needs_backfill
from bot.data.eodhd_feed import EODHDFeed
from bot.data.store import DataStore
from bot.trading_hours import last_expected_closed_bar_ms

_BAR_MS = 3_600_000
SYM = "EUR/USD"
# Thursday 12:30 UTC — mid-session for FX, so the market is unambiguously open.
NOW = datetime(2026, 4, 16, 12, 30, tzinfo=UTC)


def _candle(symbol: str, ts_ms: int) -> Candle:
    return Candle(
        symbol=symbol,
        timestamp=ts_ms,
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        volume=0.0,
        is_confirmed=True,
    )


def _fill(store: DataStore, symbol: str, n: int, latest_ms: int) -> None:
    """Add *n* ascending 1h candles ending at *latest_ms*."""
    for i in range(n):
        store.add_candle(_candle(symbol, latest_ms - (n - 1 - i) * _BAR_MS))


# ---------------------------------------------------------------------------
# needs_backfill predicate
# ---------------------------------------------------------------------------


class TestNeedsBackfill:
    def test_empty_store_triggers(self) -> None:
        store = DataStore(buffer_size=100)
        assert needs_backfill(store, SYM, threshold=5, now=NOW) is True

    def test_depth_below_threshold_triggers(self) -> None:
        store = DataStore(buffer_size=100)
        _fill(store, SYM, 2, last_expected_closed_bar_ms(SYM, NOW) or 0)
        assert needs_backfill(store, SYM, threshold=5, now=NOW) is True

    def test_deep_and_fresh_skips(self) -> None:
        store = DataStore(buffer_size=100)
        _fill(store, SYM, 10, last_expected_closed_bar_ms(SYM, NOW) or 0)
        assert needs_backfill(store, SYM, threshold=5, now=NOW) is False

    def test_deep_but_stale_triggers(self) -> None:
        store = DataStore(buffer_size=100)
        last = last_expected_closed_bar_ms(SYM, NOW) or 0
        _fill(store, SYM, 10, last - 3 * _BAR_MS)  # buffer deep, tail 3 bars behind
        assert needs_backfill(store, SYM, threshold=5, now=NOW) is True

    def test_newer_than_expected_is_fresh(self) -> None:
        # A mid-hour partial bar (newer than the last closed bar) is not stale.
        store = DataStore(buffer_size=100)
        last = last_expected_closed_bar_ms(SYM, NOW) or 0
        _fill(store, SYM, 10, last + _BAR_MS)
        assert needs_backfill(store, SYM, threshold=5, now=NOW) is False

    def test_weekend_complete_buffer_is_fresh(self) -> None:
        # Sat restart, buffer holds Friday's last bar → no needless refetch.
        sat = datetime(2026, 4, 18, 12, 0, tzinfo=UTC)
        store = DataStore(buffer_size=100)
        _fill(store, SYM, 10, last_expected_closed_bar_ms(SYM, sat) or 0)
        assert needs_backfill(store, SYM, threshold=5, now=sat) is False

    def test_weekend_missing_friday_bars_triggers(self) -> None:
        # Even on the weekend, a buffer missing the tail of Friday's session is
        # repaired (freshness compares against the last in-session bar).
        sat = datetime(2026, 4, 18, 12, 0, tzinfo=UTC)
        store = DataStore(buffer_size=100)
        last = last_expected_closed_bar_ms(SYM, sat) or 0
        _fill(store, SYM, 10, last - 2 * _BAR_MS)
        assert needs_backfill(store, SYM, threshold=5, now=sat) is True

    def test_none_last_expected_falls_back_to_depth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # When no bar is expected (helper returns None), only depth decides.
        monkeypatch.setattr("bot.data.backfill.last_expected_closed_bar_ms", lambda *_: None)
        store = DataStore(buffer_size=100)
        _fill(store, SYM, 10, int(NOW.timestamp() * 1000) - 999 * _BAR_MS)  # deep but ancient
        assert needs_backfill(store, SYM, threshold=5, now=NOW) is False
        assert needs_backfill(store, SYM, threshold=50, now=NOW) is True  # depth still fires


# ---------------------------------------------------------------------------
# EODHDFeed wiring — _backfill_below_threshold refetches stale symbols
# ---------------------------------------------------------------------------


def _one_symbol_feed(bot_key: str) -> tuple[EODHDFeed, DataStore]:
    config = BotConfig(
        candle_exchange="eodhd", eodhd_api="k", candle_buffer_size=50, kronos_context_bars=2
    )
    store = DataStore(buffer_size=50)
    feed = EODHDFeed(store, EventBus(), config)
    feed._symbols = [s for s in feed._symbols if s.bot_key == bot_key]
    assert feed._symbols, f"{bot_key} not in EODHD universe"
    return feed, store


def _patch_fetch(monkeypatch: pytest.MonkeyPatch, feed: EODHDFeed) -> list[str]:
    fetched: list[str] = []

    async def fake_fetch(sym: object) -> list[dict[str, object]]:
        fetched.append(sym.bot_key)  # type: ignore[attr-defined]
        return []

    monkeypatch.setattr(feed, "_fetch_intraday", fake_fetch)
    return fetched


@pytest.mark.asyncio
async def test_backfill_refetches_deep_but_stale_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    feed, store = _one_symbol_feed(SYM)
    # Buffer well above threshold (2) but a month stale → freshness must fire.
    old = int((datetime.now(UTC) - timedelta(days=30)).timestamp() * 1000)
    _fill(store, SYM, 5, old)
    fetched = _patch_fetch(monkeypatch, feed)

    await feed._backfill_below_threshold()

    assert fetched == [SYM]


@pytest.mark.asyncio
async def test_backfill_skips_deep_and_fresh_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    feed, store = _one_symbol_feed(SYM)
    # Latest bar stamped "now" is newer than any last-closed bar → fresh.
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    _fill(store, SYM, 5, now_ms)
    fetched = _patch_fetch(monkeypatch, feed)

    await feed._backfill_below_threshold()

    assert fetched == []
