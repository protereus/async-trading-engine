"""Tests for IGCandleLSFeed — the D1 LS-wiring layer that routes
1-minute IG chart updates through the IGCandleAggregator into the
candle store + DB + EVENT_NEW_CANDLE.

Mocks the Lightstreamer SDK and IGClient — no live network calls.
Covers the dispatch path (parse + mid + UTM correction + market-open
gate) and the aggregator-callback path (store / DB / event emission).
The aggregator's bucket math is tested in ``test_ig_candle_aggregator``;
here we just verify the wiring around it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from bot.config import BotConfig
from bot.core.event_bus import EVENT_NEW_CANDLE, EventBus
from bot.core.models import Candle
from bot.data.ig_candle_feed import IGCandleLSFeed, _epic_to_canonical_map, _utm_to_utc_ms
from bot.data.store import DataStore

# 2026-05-29 14:00:00 UTC = 1780041600000 ms — same anchor as the aggregator tests.
_HOUR_START_UTC_MS = 1_780_041_600_000


def _make_feed(symbols: dict[str, str] | None = None) -> IGCandleLSFeed:
    """Build a feed with mocked dependencies and a custom EPIC→symbol map."""
    store = DataStore(buffer_size=10)
    bus = EventBus()
    # Low kronos_context_bars so the gap-aware backfill's "buffered" threshold is
    # reachable with tiny fixtures (real default is 400).
    config = BotConfig(twelve_data_api="t", candle_buffer_size=10, kronos_context_bars=2)
    candle_db = MagicMock()
    candle_db.insert_candle = MagicMock()

    client = MagicMock()
    client._cst = "CST"
    client._xst = "XST"
    client._account_id = "Z6ACA2"
    client._ls_endpoint = "https://push.lightstreamer.com"
    client.refresh_session = AsyncMock()

    feed = IGCandleLSFeed(client, store, bus, config, candle_db=candle_db)
    # Override the EPIC map (the real one reads IG_NATIVE_CANDLE_SYMBOLS, which is
    # empty at D1; tests need a non-empty map to exercise the dispatch path).
    if symbols is not None:
        feed._epic_to_symbol = symbols
    return feed


# ---------------------------------------------------------------------------
# UTM correction helper (IG emits London-local epoch ms, not real UTC)
# ---------------------------------------------------------------------------


class TestUtmCorrection:
    def test_empty_input_returns_zero(self) -> None:
        assert _utm_to_utc_ms("") == 0

    def test_malformed_input_returns_zero(self) -> None:
        assert _utm_to_utc_ms("not-a-number") == 0

    def test_london_offset_applied(self) -> None:
        """For a moment when London = UTC+1 (BST), the corrected UTC ms
        is the input + 3_600_000.  Verify against a deterministic London
        time using zoneinfo."""
        # Pick a known BST moment so we control the offset
        london_local_dt = datetime(2026, 5, 29, 15, 0, 0, tzinfo=ZoneInfo("Europe/London"))
        # That moment in real UTC
        expected_utc_ms = int(london_local_dt.astimezone(UTC).timestamp() * 1000)
        # IG would emit the London-local epoch ms (NOT the UTC one).  Reconstruct
        # the "raw" UTM value by stripping the offset from the real UTC ms:
        offset = ZoneInfo("Europe/London").utcoffset(london_local_dt.replace(tzinfo=None))
        offset_ms = int(offset.total_seconds() * 1000) if offset else 0
        raw_utm = str(expected_utc_ms - offset_ms)
        assert _utm_to_utc_ms(raw_utm) == expected_utc_ms


# ---------------------------------------------------------------------------
# Reverse-lookup helper
# ---------------------------------------------------------------------------


class TestEpicToCanonicalMap:
    def test_set_contains_metals(self) -> None:
        """Since 2026-06-19 the IG-native set is the two metals (XAU/XAG),
        sourced from IG spot 24/5 instead of the US-session-only GLD/SLV ETFs."""
        from bot.data.ig_candle_aggregator import IG_NATIVE_CANDLE_SYMBOLS

        assert frozenset({"XAU/USD", "XAG/USD"}) == IG_NATIVE_CANDLE_SYMBOLS
        # Reverse map resolves each metal to its IG spot EPIC (via the EODHD map).
        mapping = _epic_to_canonical_map()
        assert mapping == {
            "CS.D.USCGC.TODAY.IP": "XAU/USD",
            "CS.D.USCSI.TODAY.IP": "XAG/USD",
        }

    def test_populated_set_yields_inverted_map(self) -> None:
        """The reverse lookup is SYMBOL_EPIC_MAP ∩ IG_NATIVE_CANDLE_SYMBOLS."""
        with patch("bot.data.ig_candle_feed.IG_NATIVE_CANDLE_SYMBOLS", frozenset({"XAU/USD"})):
            mapping = _epic_to_canonical_map()
        assert mapping == {"CS.D.USCGC.TODAY.IP": "XAU/USD"}


# ---------------------------------------------------------------------------
# Dispatch — parse a chart update and feed the aggregator
# ---------------------------------------------------------------------------


class TestDispatch:
    def _msg(
        self,
        *,
        item_name: str = "CHART:CC.D.CL.USS.IP:1MINUTE",
        utm: str = "0",
        bid_close: str = "8800.0",
        ofr_close: str = "8802.0",
        ltv: str = "10",
        cons_end: str = "1",
    ) -> dict:
        return {
            "item": item_name,
            "fields": {
                "UTM": utm,
                "BID_CLOSE": bid_close,
                "OFR_CLOSE": ofr_close,
                "LTV": ltv,
                "CONS_END": cons_end,
            },
        }

    def test_dispatch_feeds_aggregator_with_mid_and_market_open(self) -> None:
        feed = _make_feed({"CC.D.CL.USS.IP": "USO"})
        feed._aggregator = MagicMock()
        # Simulate an in-BST UTM: pick a known offset and reconstruct the raw value
        offset_ms = 3_600_000  # +01:00 in BST
        raw_utm = str(_HOUR_START_UTC_MS - offset_ms)
        with patch("bot.data.ig_candle_feed.is_market_open", return_value=True):
            feed._dispatch(self._msg(item_name="CHART:CC.D.CL.USS.IP:1MINUTE", utm=raw_utm))
        feed._aggregator.ingest_tick.assert_called_once()
        args, kwargs = feed._aggregator.ingest_tick.call_args
        assert args[0] == "USO"  # canonical symbol, not the EPIC
        # UTM corrected to real UTC — verify it's within one summer-offset of the
        # raw input (the exact offset depends on what BST resolves to right now).
        assert args[1] > int(raw_utm)  # corrected ts > raw
        assert args[2] == pytest.approx((8800.0 + 8802.0) / 2.0)
        assert kwargs["market_open"] is True
        assert kwargs["ltv"] == 10.0

    def test_dispatch_drops_unknown_epic(self) -> None:
        feed = _make_feed({"CC.D.CL.USS.IP": "USO"})
        feed._aggregator = MagicMock()
        feed._dispatch(self._msg(item_name="CHART:CS.D.UNKNOWN.IP:1MINUTE"))
        feed._aggregator.ingest_tick.assert_not_called()

    def test_dispatch_drops_non_positive_prices(self) -> None:
        feed = _make_feed({"CC.D.CL.USS.IP": "USO"})
        feed._aggregator = MagicMock()
        feed._dispatch(self._msg(bid_close="0", ofr_close="8800.0"))
        feed._dispatch(self._msg(bid_close="8800.0", ofr_close=""))
        feed._aggregator.ingest_tick.assert_not_called()

    def test_dispatch_drops_missing_utm(self) -> None:
        """Without a valid UTM we'd have to guess the bar boundary from
        wall-clock — and would mis-bucket around the hour boundary itself.
        Safer to drop the frame."""
        feed = _make_feed({"CC.D.CL.USS.IP": "USO"})
        feed._aggregator = MagicMock()
        feed._dispatch(self._msg(utm=""))
        feed._aggregator.ingest_tick.assert_not_called()

    def test_dispatch_drops_malformed_item_name(self) -> None:
        feed = _make_feed({"CC.D.CL.USS.IP": "USO"})
        feed._aggregator = MagicMock()
        feed._dispatch({"item": "INVALID", "fields": {}})
        feed._aggregator.ingest_tick.assert_not_called()

    def test_dispatch_passes_through_market_closed_flag(self) -> None:
        """Whatever is_market_open returns is what the aggregator gets —
        the gate logic lives inside the aggregator (probe finding: LS
        emits 24/7 regardless of underlying market state)."""
        feed = _make_feed({"CC.D.CL.USS.IP": "USO"})
        feed._aggregator = MagicMock()
        raw_utm = str(_HOUR_START_UTC_MS - 3_600_000)
        with patch("bot.data.ig_candle_feed.is_market_open", return_value=False):
            feed._dispatch(self._msg(utm=raw_utm))
        feed._aggregator.ingest_tick.assert_called_once()
        assert feed._aggregator.ingest_tick.call_args.kwargs["market_open"] is False


# ---------------------------------------------------------------------------
# emit_callback — what happens when the aggregator finalises a candle
# ---------------------------------------------------------------------------


class TestEmitCallback:
    def _candle(self) -> Candle:
        return Candle(
            symbol="USO",
            timestamp=_HOUR_START_UTC_MS,
            open=88.0,
            high=88.5,
            low=87.8,
            close=88.3,
            volume=120.0,
            is_confirmed=True,
        )

    def test_writes_to_store_and_db(self) -> None:
        feed = _make_feed({"CC.D.CL.USS.IP": "USO"})
        feed._loop = None  # exercise the loop=None branch (event emit skipped)
        candle = self._candle()
        feed._emit_aggregated_candle(candle)
        assert feed._store.get_latest_candle("USO") == candle
        feed._candle_db.insert_candle.assert_called_once_with(candle)

    def test_db_failure_does_not_break_store_write(self) -> None:
        """A DB write hiccup must not lose the in-memory candle that the
        strategy reads from next rerank."""
        feed = _make_feed({"CC.D.CL.USS.IP": "USO"})
        feed._loop = None
        feed._candle_db.insert_candle.side_effect = RuntimeError("disk full")
        candle = self._candle()
        # Must not raise
        feed._emit_aggregated_candle(candle)
        assert feed._store.get_latest_candle("USO") == candle

    @pytest.mark.asyncio
    async def test_event_new_candle_emitted_with_canonical_key(self) -> None:
        """The bus event must carry the canonical-symbol Candle, since the
        strategy subscribes by canonical symbol (not by EPIC)."""
        import asyncio

        feed = _make_feed({"CC.D.CL.USS.IP": "USO"})
        feed._loop = asyncio.get_running_loop()

        received: list[Candle] = []

        async def on_candle(data) -> None:
            received.append(data)

        feed._event_bus.subscribe(EVENT_NEW_CANDLE, on_candle)

        feed._emit_aggregated_candle(self._candle())
        # The emit_callback is sync — it schedules the bus emission via
        # loop.create_task.  Drain the scheduled task (one yield gets it
        # started, a second lets gather()-ed subscribers complete).
        for _ in range(3):
            await asyncio.sleep(0)
        assert len(received) == 1
        assert received[0].symbol == "USO"
        assert received[0].timestamp == _HOUR_START_UTC_MS


# ---------------------------------------------------------------------------
# Run lifecycle — dormant when set is empty
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# D3 startup migration — wipe pre-cutover rows, REST-backfill IG-level bars
# ---------------------------------------------------------------------------


def _ig_bars(symbol: str, n: int, base_close: float = 4466.0) -> list[Candle]:
    """n ascending IG-level hourly bars for *symbol* (close ≈ base_close)."""
    return [
        Candle(
            symbol=symbol,
            timestamp=_HOUR_START_UTC_MS + i * 3_600_000,
            open=base_close + i,
            high=base_close + 10 + i,
            low=base_close - 10 + i,
            close=base_close + i,
            volume=10.0,
            is_confirmed=True,
        )
        for i in range(n)
    ]


class TestBackfillIfNeeded:
    """Coverage for the gap-aware startup warm-up (replaces the old
    delete-first-every-restart ``_migrate_and_backfill``): a one-time
    wrong-basis wipe, REST backfill only for below-threshold symbols, then a
    DB→store warm pass.  ``kronos_context_bars`` is 2 in the test config so a
    2-bar DB counts as 'buffered'."""

    @pytest.mark.asyncio
    async def test_buffered_symbol_skips_backfill_and_wipe(self) -> None:
        """A symbol already at/above threshold with matching basis: no REST
        backfill, no wipe — only the 1-bar basis probe + a DB→store warm.
        This is the every-restart steady state, and the whole point: no
        allowance-burning full re-fetch."""
        feed = _make_feed({"CS.D.USCGC.TODAY.IP": "XAU/USD"})
        stored = _ig_bars("XAU/USD", 2, base_close=4466.0)  # >= threshold(2), IG-basis
        feed._candle_db.get_candles = MagicMock(return_value=stored)
        feed._candle_db.delete_candles_for_symbol = MagicMock(return_value=0)
        feed._candle_db.insert_candles = MagicMock()
        # Probe returns a fresh IG bar at the same basis (ratio ≈ 1 → no wipe).
        probe = _ig_bars("XAU/USD", 1, base_close=4470.0)
        with patch(
            "bot.data.ig_candle_feed.fetch_ig_hourly_backfill",
            new=AsyncMock(return_value=probe),
        ) as mock_fetch:
            await feed._backfill_if_needed()

        # Only the 1-bar basis probe hit IG — no full backfill fetch.
        assert mock_fetch.call_count == 1
        assert mock_fetch.call_args.kwargs.get("limit") == 1
        feed._candle_db.delete_candles_for_symbol.assert_not_called()
        feed._candle_db.insert_candles.assert_not_called()
        # Store warmed from the DB rows.
        assert feed._store.get_latest_candle("XAU/USD") == stored[-1]

    @pytest.mark.asyncio
    async def test_cold_symbol_backfills_without_wipe(self) -> None:
        """Empty DB (cold start): no rows to mis-scale, so no probe/wipe — just
        a REST backfill that lands in the DB."""
        feed = _make_feed({"CS.D.USCGC.TODAY.IP": "XAU/USD"})
        feed._candle_db.get_candles = MagicMock(return_value=[])  # nothing stored
        feed._candle_db.delete_candles_for_symbol = MagicMock(return_value=0)
        feed._candle_db.insert_candles = MagicMock()
        bars = _ig_bars("XAU/USD", 3)
        with patch(
            "bot.data.ig_candle_feed.fetch_ig_hourly_backfill",
            new=AsyncMock(return_value=bars),
        ) as mock_fetch:
            await feed._backfill_if_needed()

        feed._candle_db.delete_candles_for_symbol.assert_not_called()
        # Full backfill fetched at candle_buffer_size and inserted.
        assert mock_fetch.call_args.kwargs.get("limit") == feed._config.candle_buffer_size
        feed._candle_db.insert_candles.assert_called_once_with(bars)

    @pytest.mark.asyncio
    async def test_wrong_basis_triggers_one_time_wipe(self) -> None:
        """Stored rows ~110× off the fresh IG bar (old SLV-ETF basis) → wipe,
        then re-seed at IG scale."""
        feed = _make_feed({"CS.D.USCSI.TODAY.IP": "XAG/USD"})
        # Stored at the old ETF basis (~68); fresh IG silver level ~7456.
        etf_rows = _ig_bars("XAG/USD", 2, base_close=68.0)
        ig_bars = _ig_bars("XAG/USD", 3, base_close=7456.0)
        feed._candle_db.get_candles = MagicMock(return_value=etf_rows)
        feed._candle_db.delete_candles_for_symbol = MagicMock(return_value=2)
        feed._candle_db.insert_candles = MagicMock()
        # First call = probe (limit 1, IG basis); subsequent = full backfill.
        with patch(
            "bot.data.ig_candle_feed.fetch_ig_hourly_backfill",
            new=AsyncMock(side_effect=[ig_bars[:1], ig_bars]),
        ):
            await feed._backfill_if_needed()
        feed._candle_db.delete_candles_for_symbol.assert_called_once_with("XAG/USD")

    @pytest.mark.asyncio
    async def test_matching_basis_does_not_wipe(self) -> None:
        """Stored rows at IG basis (ratio ≈ 1) → no wipe even when above
        threshold."""
        feed = _make_feed({"CS.D.USCGC.TODAY.IP": "XAU/USD"})
        feed._candle_db.get_candles = MagicMock(return_value=_ig_bars("XAU/USD", 2, 4466.0))
        feed._candle_db.delete_candles_for_symbol = MagicMock(return_value=0)
        with patch(
            "bot.data.ig_candle_feed.fetch_ig_hourly_backfill",
            new=AsyncMock(return_value=_ig_bars("XAU/USD", 1, 4466.0)),
        ):
            await feed._backfill_if_needed()
        feed._candle_db.delete_candles_for_symbol.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_candle_db_loads_store_directly(self) -> None:
        """With ``candle_db`` None, the backfill bars load straight into the
        in-memory store (no DB to warm from)."""
        feed = _make_feed({"CS.D.USCGC.TODAY.IP": "XAU/USD"})
        feed._candle_db = None
        bars = _ig_bars("XAU/USD", 3)
        with patch(
            "bot.data.ig_candle_feed.fetch_ig_hourly_backfill",
            new=AsyncMock(return_value=bars),
        ):
            await feed._backfill_if_needed()
        assert feed._store.get_latest_candle("XAU/USD") == bars[-1]

    @pytest.mark.asyncio
    async def test_empty_backfill_does_not_raise(self) -> None:
        """REST returns [] for a needy symbol: no insert, no crash — live LS
        ticks will warm it instead."""
        feed = _make_feed({"CS.D.USCGC.TODAY.IP": "XAU/USD"})
        feed._candle_db.get_candles = MagicMock(return_value=[])
        feed._candle_db.insert_candles = MagicMock()
        with patch(
            "bot.data.ig_candle_feed.fetch_ig_hourly_backfill",
            new=AsyncMock(return_value=[]),
        ):
            await feed._backfill_if_needed()  # must not raise
        feed._candle_db.insert_candles.assert_not_called()
        assert feed._store.get_latest_candle("XAU/USD") is None


class TestRunDormant:
    @pytest.mark.asyncio
    async def test_run_no_ops_when_set_empty(self) -> None:
        """The D1 default: IG_NATIVE_CANDLE_SYMBOLS is empty, the feed
        should construct fine but ``run()`` must wait quietly without
        opening any LS connection (the task supervisor expects long-lived
        tasks; returning early would look like a crash)."""
        import asyncio

        feed = _make_feed({})  # empty epic map

        # Run for a short window, then cancel — should sit in the wait
        # without raising.
        import contextlib

        task = asyncio.create_task(feed.run())
        await asyncio.sleep(0.05)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # No LS client should have been constructed
        assert feed._ls_client is None
