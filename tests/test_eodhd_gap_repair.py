"""Post-close gap repair.

A restart drops the hour bar that straddles it: the aggregator discards the
mid-forming partial bucket, and EODHD REST can't serve the bar until the
provider finalises it.  These tests pin the repair pipeline:

- the aggregator's ``drop_callback`` fires on a partial-bucket drop,
- ``EODHDFeed`` queues the dropped (symbol, hour) and, at the :10 repair slot,
  re-fetches it via REST, lands it in the DB, and reloads the store buffer
  (``DataStore.add_candle`` drops out-of-order bars, so append can't work),
- unavailable bars stay queued, dead hours are pruned, stale entries expire,
- a startup seed covers the boundary-straddling restart the callback misses.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from bot.config import BotConfig
from bot.core.event_bus import EventBus
from bot.core.models import Candle
from bot.data.candle_db import CandleDB
from bot.data.eodhd_feed import _REPAIR_MAX_AGE_MS, _REPAIR_SEED_LOOKBACK_HOURS, EODHDFeed
from bot.data.ig_candle_aggregator import IGCandleAggregator
from bot.data.store import DataStore

HOUR_MS = 3_600_000

# An hour-aligned base timestamp within the repair-expiry window of "now".
NOW_H = (int(time.time() * 1000) // HOUR_MS) * HOUR_MS


def _candle(symbol: str, ts: int, price: float = 1.0) -> Candle:
    return Candle(
        symbol=symbol,
        timestamp=ts,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=0.0,
        is_confirmed=True,
    )


def _bar(ts_ms: int, price: float = 1.5) -> dict[str, object]:
    return {
        "timestamp": ts_ms // 1000,
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "volume": None,
    }


def _make_feed(
    tmp_path: Path, with_db: bool = True
) -> tuple[EODHDFeed, DataStore, CandleDB | None]:
    # Buffer must exceed _REPAIR_SEED_LOOKBACK_HOURS so a filled seed window
    # isn't evicted from the deque and mistaken for missing hours.
    config = BotConfig(candle_exchange="eodhd", eodhd_api="k", candle_buffer_size=48)
    store = DataStore(buffer_size=48)
    cdb: CandleDB | None = None
    if with_db:
        cdb = CandleDB(db_path=str(tmp_path / "test_candles.db"))
        cdb.init_db()
    feed = EODHDFeed(store, EventBus(), config, candle_db=cdb)
    return feed, store, cdb


# ---------------------------------------------------------------------------
# Aggregator drop callback
# ---------------------------------------------------------------------------


class TestAggregatorDropCallback:
    def test_fires_on_partial_bucket_drop(self) -> None:
        drops: list[tuple[str, int]] = []
        agg = IGCandleAggregator(
            emit_callback=lambda c: None, drop_callback=lambda s, h: drops.append((s, h))
        )
        h0 = NOW_H - 2 * HOUR_MS
        # First tick lands mid-hour (minute 24) → bucket ineligible.
        agg.ingest_tick("EUR/USD", h0 + 24 * 60_000, 1.1, market_open=True)
        # Rollover into the next hour drops the partial bucket.
        agg.ingest_tick("EUR/USD", h0 + HOUR_MS, 1.2, market_open=True)
        assert drops == [("EUR/USD", h0)]

    def test_not_fired_on_eligible_emit(self) -> None:
        drops: list[tuple[str, int]] = []
        emitted: list[Candle] = []
        agg = IGCandleAggregator(
            emit_callback=emitted.append, drop_callback=lambda s, h: drops.append((s, h))
        )
        h0 = NOW_H - 2 * HOUR_MS
        agg.ingest_tick("EUR/USD", h0, 1.1, market_open=True)  # minute 0 → eligible
        agg.ingest_tick("EUR/USD", h0 + HOUR_MS, 1.2, market_open=True)
        assert emitted and emitted[0].timestamp == h0
        assert drops == []

    def test_default_none_callback_is_safe(self) -> None:
        agg = IGCandleAggregator(emit_callback=lambda c: None)
        h0 = NOW_H - 2 * HOUR_MS
        agg.ingest_tick("EUR/USD", h0 + 24 * 60_000, 1.1, market_open=True)
        agg.ingest_tick("EUR/USD", h0 + HOUR_MS, 1.2, market_open=True)  # no raise


# ---------------------------------------------------------------------------
# DataStore.replace_candles
# ---------------------------------------------------------------------------


class TestReplaceCandles:
    def test_replaces_buffer_making_repaired_bar_visible(self) -> None:
        store = DataStore(buffer_size=10)
        h = NOW_H
        store.add_candle(_candle("EUR/USD", h - 3 * HOUR_MS))
        store.add_candle(_candle("EUR/USD", h - HOUR_MS))
        # The repaired (older) bar cannot be appended…
        store.add_candle(_candle("EUR/USD", h - 2 * HOUR_MS))
        assert store.get_candle_count("EUR/USD") == 2
        # …but a reload replaces the buffer wholesale.
        full = [_candle("EUR/USD", h - k * HOUR_MS) for k in (3, 2, 1)]
        store.replace_candles("EUR/USD", full)
        assert [c.timestamp for c in store.get_candles("EUR/USD")] == [
            h - 3 * HOUR_MS,
            h - 2 * HOUR_MS,
            h - HOUR_MS,
        ]

    def test_respects_buffer_size(self) -> None:
        store = DataStore(buffer_size=2)
        h = NOW_H
        store.replace_candles("EUR/USD", [_candle("EUR/USD", h - k * HOUR_MS) for k in (3, 2, 1)])
        assert store.get_candle_count("EUR/USD") == 2
        assert store.get_latest_candle("EUR/USD").timestamp == h - HOUR_MS


# ---------------------------------------------------------------------------
# EODHDFeed — recording drops
# ---------------------------------------------------------------------------


class TestRecordDroppedBucket:
    def test_aggregator_drop_lands_on_repair_list(self, tmp_path: Path) -> None:
        feed, _, _ = _make_feed(tmp_path)
        h0 = NOW_H - 2 * HOUR_MS
        feed._aggregator.ingest_tick("EUR/USD", h0 + 24 * 60_000, 1.1, market_open=True)
        feed._aggregator.ingest_tick("EUR/USD", h0 + HOUR_MS, 1.2, market_open=True)
        assert feed._pending_repairs == {"EUR/USD": {h0}}

    def test_noop_without_candle_db(self, tmp_path: Path) -> None:
        feed, _, _ = _make_feed(tmp_path, with_db=False)
        feed._record_dropped_bucket("EUR/USD", NOW_H - HOUR_MS)
        assert feed._pending_repairs == {}


# ---------------------------------------------------------------------------
# EODHDFeed — repair pass
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRepairPending:
    async def test_repairs_bar_into_db_and_reloads_store(self, tmp_path: Path) -> None:
        feed, store, cdb = _make_feed(tmp_path)
        assert cdb is not None
        h = NOW_H - HOUR_MS  # hour to repair
        # Store + DB hold the surrounding history; h itself is the gap.
        store_candles = [_candle("EUR/USD", h - 2 * HOUR_MS), _candle("EUR/USD", h + HOUR_MS)]
        for c in store_candles:
            store.add_candle(c)
            cdb.insert_candle(c)
        feed._pending_repairs = {"EUR/USD": {h}}

        seen_from_ts: list[int] = []

        async def fake_fetch(sym: object, from_ts: int | None = None) -> list[dict[str, object]]:
            seen_from_ts.append(from_ts)
            return [_bar(h)]

        feed._fetch_intraday = fake_fetch  # type: ignore[method-assign]
        await feed._repair_pending()

        # Fetch window starts one hour before the pending bar (boundary insurance).
        assert seen_from_ts == [h // 1000 - 3600]
        # Repaired bar landed in the DB…
        assert any(c.timestamp == h for c in cdb.get_candles("EUR/USD", limit=10))
        # …and the store buffer was reloaded so the strategy sees it in order.
        assert [c.timestamp for c in store.get_candles("EUR/USD")] == [
            h - 2 * HOUR_MS,
            h,
            h + HOUR_MS,
        ]
        assert feed._pending_repairs == {}

    async def test_unserved_bar_stays_pending(self, tmp_path: Path) -> None:
        feed, store, _ = _make_feed(tmp_path)
        h = NOW_H - HOUR_MS

        async def fake_fetch(sym: object, from_ts: int | None = None) -> list[dict[str, object]]:
            return []  # provider hasn't finalised the bar yet

        feed._fetch_intraday = fake_fetch  # type: ignore[method-assign]
        feed._pending_repairs = {"EUR/USD": {h}}
        await feed._repair_pending()
        assert feed._pending_repairs == {"EUR/USD": {h}}

    async def test_dead_hour_pruned_when_provider_finalised_past_it(self, tmp_path: Path) -> None:
        feed, store, cdb = _make_feed(tmp_path)
        assert cdb is not None
        h = NOW_H - 3 * HOUR_MS

        async def fake_fetch(sym: object, from_ts: int | None = None) -> list[dict[str, object]]:
            return [_bar(h + HOUR_MS)]  # later bar exists, h itself never will

        feed._fetch_intraday = fake_fetch  # type: ignore[method-assign]
        feed._pending_repairs = {"EUR/USD": {h}}
        await feed._repair_pending()
        assert feed._pending_repairs == {}
        assert not any(c.timestamp == h for c in cdb.get_candles("EUR/USD", limit=10))

    async def test_expired_hours_dropped_without_fetch(self, tmp_path: Path) -> None:
        feed, _, _ = _make_feed(tmp_path)
        stale = NOW_H - _REPAIR_MAX_AGE_MS - 2 * HOUR_MS
        fetches: list[int] = []

        async def fake_fetch(sym: object, from_ts: int | None = None) -> list[dict[str, object]]:
            fetches.append(1)
            return []

        feed._fetch_intraday = fake_fetch  # type: ignore[method-assign]
        feed._pending_repairs = {"EUR/USD": {stale}}
        await feed._repair_pending()
        assert feed._pending_repairs == {}
        assert fetches == []

    async def test_null_ohlc_bar_treated_as_dead_hour(self, tmp_path: Path) -> None:
        feed, _, cdb = _make_feed(tmp_path)
        assert cdb is not None
        h = NOW_H - 3 * HOUR_MS
        null_bar = {"timestamp": h // 1000, "open": None, "high": None, "low": None, "close": None}

        async def fake_fetch(sym: object, from_ts: int | None = None) -> list[dict[str, object]]:
            return [null_bar, _bar(h + HOUR_MS)]

        feed._fetch_intraday = fake_fetch  # type: ignore[method-assign]
        feed._pending_repairs = {"EUR/USD": {h}}
        await feed._repair_pending()
        assert feed._pending_repairs == {}
        assert not any(c.timestamp == h for c in cdb.get_candles("EUR/USD", limit=10))


# ---------------------------------------------------------------------------
# EODHDFeed — startup seeding (boundary-straddle restarts + lost interior holes)
# ---------------------------------------------------------------------------


def _fill_window(
    store: DataStore, expected: int, skip: set[int] | None = None, symbol: str = "EUR/USD"
) -> None:
    """Populate *store* with every hour of the seed window except *skip*."""
    skip = skip or set()
    start = expected - (_REPAIR_SEED_LOOKBACK_HOURS - 1) * HOUR_MS
    for ts in range(start, expected + 1, HOUR_MS):
        if ts not in skip:
            store.add_candle(_candle(symbol, ts))


class TestSeedRepairsFromFreshness:
    @pytest.fixture(autouse=True)
    def _always_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # NOW_H-derived hours may fall on a real weekend; keep tests
        # deterministic by defaulting the market-open gate to True.
        monkeypatch.setattr("bot.data.eodhd_feed.is_market_open", lambda key, now=None: True)

    def test_seeds_missing_tail(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Boundary-straddle restart: the newest 2 closed hours never formed."""
        feed, store, _ = _make_feed(tmp_path)
        expected = NOW_H - HOUR_MS
        _fill_window(store, expected, skip={expected - HOUR_MS, expected})
        monkeypatch.setattr("bot.data.eodhd_feed.last_expected_closed_bar_ms", lambda key: expected)
        feed._seed_repairs_from_freshness()
        assert feed._pending_repairs["EUR/USD"] == {expected - HOUR_MS, expected}

    def test_seeds_interior_hole(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A queued repair lost to a restart re-seeds as an interior hole."""
        feed, store, _ = _make_feed(tmp_path)
        expected = NOW_H - HOUR_MS
        hole = expected - 5 * HOUR_MS
        _fill_window(store, expected, skip={hole})
        monkeypatch.setattr("bot.data.eodhd_feed.last_expected_closed_bar_ms", lambda key: expected)
        feed._seed_repairs_from_freshness()
        assert feed._pending_repairs["EUR/USD"] == {hole}

    def test_hole_outside_lookback_not_seeded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        feed, store, _ = _make_feed(tmp_path)
        expected = NOW_H - HOUR_MS
        old_hole = expected - (_REPAIR_SEED_LOOKBACK_HOURS + 5) * HOUR_MS
        store.add_candle(_candle("EUR/USD", old_hole - HOUR_MS))
        store.add_candle(_candle("EUR/USD", old_hole + HOUR_MS))
        _fill_window(store, expected)
        monkeypatch.setattr("bot.data.eodhd_feed.last_expected_closed_bar_ms", lambda key: expected)
        feed._seed_repairs_from_freshness()
        assert feed._pending_repairs == {}

    def test_market_closed_hours_not_seeded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        feed, store, _ = _make_feed(tmp_path)
        expected = NOW_H - HOUR_MS
        hole = expected - 5 * HOUR_MS
        _fill_window(store, expected, skip={hole})
        monkeypatch.setattr("bot.data.eodhd_feed.last_expected_closed_bar_ms", lambda key: expected)
        monkeypatch.setattr(
            "bot.data.eodhd_feed.is_market_open",
            lambda key, now=None: int(now.timestamp() * 1000) != hole,
        )
        feed._seed_repairs_from_freshness()
        assert feed._pending_repairs == {}

    def test_fresh_symbol_not_seeded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        feed, store, _ = _make_feed(tmp_path)
        expected = NOW_H - HOUR_MS
        _fill_window(store, expected)
        monkeypatch.setattr("bot.data.eodhd_feed.last_expected_closed_bar_ms", lambda key: expected)
        feed._seed_repairs_from_freshness()
        assert feed._pending_repairs == {}

    def test_noop_without_candle_db(self, tmp_path: Path) -> None:
        feed, store, _ = _make_feed(tmp_path, with_db=False)
        store.add_candle(_candle("EUR/USD", NOW_H - 10 * HOUR_MS))
        feed._seed_repairs_from_freshness()
        assert feed._pending_repairs == {}
