"""Tests for IGFeed — Lightstreamer bridge and candle parsing logic."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.core.event_bus import EVENT_NEW_CANDLE, EVENT_ORDER_FILLED, EventBus
from bot.core.models import Candle, OrderSide, OrderStatus
from bot.data.ig_feed import IGFeed
from bot.data.ig_feed_handlers import TickValidator, _mid, _safe_float
from bot.data.ig_ls_listeners import _ConnectionListener, _LSQueueListener

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(epics: list[str] | None = None) -> MagicMock:
    cfg = MagicMock()
    cfg.ig_epics = epics or ["CS.D.AVXUSD.TODAY.IP"]
    cfg.candle_timeframe = "1m"
    cfg.candle_buffer_size = 100
    return cfg


def _make_client(cst: str = "CST123", xst: str = "XST456", account_id: str = "Z6ACA2") -> MagicMock:
    client = MagicMock()
    client._cst = cst
    client._xst = xst
    client._account_id = account_id
    client._ls_endpoint = "https://push.lightstreamer.com"
    # Public accessors the feed's Lightstreamer connectors use (mirror the
    # private session tokens above).
    client.ls_password = f"CST-{cst}|XST-{xst}"
    client.account_id = account_id
    client.lightstreamer_endpoint = "https://push.lightstreamer.com"
    client.datapoints_remaining = 9_000
    client.fetch_ohlcv = AsyncMock(return_value=[])
    return client


def _make_feed(epics: list[str] | None = None) -> tuple[IGFeed, MagicMock, EventBus, MagicMock]:
    client = _make_client()
    store = MagicMock()
    store.add_candle = MagicMock()
    bus = EventBus()
    config = _make_config(epics)
    candle_db = MagicMock()
    candle_db.get_candles = MagicMock(return_value=[])
    candle_db.get_earliest_timestamp = MagicMock(return_value=None)
    candle_db.insert_candle = MagicMock()
    feed = IGFeed(client, store, bus, config, candle_db)
    return feed, client, bus, candle_db


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_valid_string(self) -> None:
        assert _safe_float("1.5") == 1.5

    def test_valid_float(self) -> None:
        assert _safe_float(2.0) == 2.0

    def test_none_returns_zero(self) -> None:
        assert _safe_float(None) == 0.0

    def test_invalid_string_returns_zero(self) -> None:
        assert _safe_float("N/A") == 0.0


class TestMid:
    def test_both_present(self) -> None:
        assert _mid(1.0, 2.0) == 1.5

    def test_only_bid(self) -> None:
        assert _mid(1.0, 0.0) == 1.0

    def test_only_offer(self) -> None:
        assert _mid(0.0, 2.0) == 2.0

    def test_both_zero(self) -> None:
        assert _mid(0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# IGFeed construction
# ---------------------------------------------------------------------------


class TestIGFeedConstruction:
    def test_epics_loaded_from_config(self) -> None:
        feed, *_ = _make_feed(["CS.D.AVXUSD.TODAY.IP", "CS.D.BITCOIN.TODAY.IP"])
        assert feed._epics == ["CS.D.AVXUSD.TODAY.IP", "CS.D.BITCOIN.TODAY.IP"]

    def test_ls_password_format(self) -> None:
        feed, client, *_ = _make_feed()
        assert feed._conn.ls_password() == "CST-CST123|XST-XST456"

    def test_closing_flag_initially_false(self) -> None:
        feed, *_ = _make_feed()
        assert feed._closing is False


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------


class TestBackfill:
    @pytest.mark.asyncio
    async def test_backfill_stores_rest_candles(self) -> None:
        feed, client, bus, candle_db = _make_feed()
        candle = Candle(
            timestamp=1_000_000,
            open=30.0,
            high=31.0,
            low=29.0,
            close=30.5,
            volume=100.0,
            symbol="CS.D.AVXUSD.TODAY.IP",
            is_confirmed=True,
        )
        client.fetch_ohlcv = AsyncMock(return_value=[candle])

        await feed._backfill("CS.D.AVXUSD.TODAY.IP", count=10)

        feed._store.add_candle.assert_called_with(candle)
        candle_db.insert_candle.assert_called_with(candle)

    @pytest.mark.asyncio
    async def test_backfill_skips_rest_if_db_full(self) -> None:
        feed, client, bus, candle_db = _make_feed()
        db_candles = [
            Candle(
                timestamp=i * 60_000,
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.0,
                volume=1.0,
                symbol="CS.D.AVXUSD.TODAY.IP",
                is_confirmed=True,
            )
            for i in range(10)
        ]
        candle_db.get_candles = MagicMock(return_value=db_candles)

        await feed._backfill("CS.D.AVXUSD.TODAY.IP", count=10)

        client.fetch_ohlcv.assert_not_called()
        assert feed._store.add_candle.call_count == 10

    @pytest.mark.asyncio
    async def test_backfill_handles_rest_error_gracefully(self) -> None:
        feed, client, bus, candle_db = _make_feed()
        client.fetch_ohlcv = AsyncMock(side_effect=Exception("REST failed"))

        # Should not raise
        await feed._backfill("CS.D.AVXUSD.TODAY.IP", count=10)


# ---------------------------------------------------------------------------
# Chart update handling
# ---------------------------------------------------------------------------


class TestHandleChartUpdate:
    @pytest.mark.asyncio
    async def test_unconfirmed_candle_stored_not_emitted(self) -> None:
        feed, client, bus, _ = _make_feed()
        received: list[Any] = []
        bus.subscribe(EVENT_NEW_CANDLE, lambda c: received.append(c))

        update = {
            "type": "chart",
            "item": "CHART:CS.D.AVXUSD.TODAY.IP:1MINUTE",
            "fields": {
                "UTM": "1700000000000",
                "BID_OPEN": "30.0",
                "BID_HIGH": "31.0",
                "BID_LOW": "29.0",
                "BID_CLOSE": "30.5",
                "OFR_OPEN": "30.1",
                "OFR_HIGH": "31.1",
                "OFR_LOW": "29.1",
                "OFR_CLOSE": "30.6",
                "LTV": "100",
                "CONS_END": "0",
            },
        }
        await feed._handlers.handle_chart_update(update)

        feed._store.add_candle.assert_called_once()
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_confirmed_candle_stored_and_emitted(self) -> None:
        feed, client, bus, candle_db = _make_feed()
        received: list[Any] = []
        bus.subscribe(EVENT_NEW_CANDLE, lambda c: received.append(c))

        update = {
            "type": "chart",
            "item": "CHART:CS.D.AVXUSD.TODAY.IP:1MINUTE",
            "fields": {
                "UTM": "1700000000000",
                "BID_OPEN": "30.0",
                "BID_HIGH": "31.0",
                "BID_LOW": "29.0",
                "BID_CLOSE": "30.5",
                "OFR_OPEN": "30.1",
                "OFR_HIGH": "31.1",
                "OFR_LOW": "29.1",
                "OFR_CLOSE": "30.6",
                "LTV": "100",
                "CONS_END": "1",
            },
        }
        await feed._handlers.handle_chart_update(update)

        assert len(received) == 1
        candle: Candle = received[0]
        assert candle.symbol == "CS.D.AVXUSD.TODAY.IP"
        assert candle.is_confirmed is True
        assert candle.close == pytest.approx(30.55, abs=0.01)  # mid of 30.5 and 30.6
        candle_db.insert_candle.assert_called_once()

    @pytest.mark.asyncio
    async def test_duplicate_confirmed_candle_not_re_emitted(self) -> None:
        feed, client, bus, _ = _make_feed()
        received: list[Any] = []
        bus.subscribe(EVENT_NEW_CANDLE, lambda c: received.append(c))

        update = {
            "type": "chart",
            "item": "CHART:CS.D.AVXUSD.TODAY.IP:1MINUTE",
            "fields": {
                "UTM": "1700000000000",
                "BID_OPEN": "30.0",
                "BID_HIGH": "31.0",
                "BID_LOW": "29.0",
                "BID_CLOSE": "30.5",
                "OFR_OPEN": "30.1",
                "OFR_HIGH": "31.1",
                "OFR_LOW": "29.1",
                "OFR_CLOSE": "30.6",
                "LTV": "100",
                "CONS_END": "1",
            },
        }
        await feed._handlers.handle_chart_update(update)
        await feed._handlers.handle_chart_update(update)  # same timestamp — should be deduped

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_zero_close_skipped(self) -> None:
        feed, *_ = _make_feed()
        update = {
            "type": "chart",
            "item": "CHART:CS.D.AVXUSD.TODAY.IP:1MINUTE",
            "fields": {
                "UTM": "1700000000000",
                "BID_CLOSE": None,
                "OFR_CLOSE": None,
                "CONS_END": "1",
            },
        }
        await feed._handlers.handle_chart_update(update)
        feed._store.add_candle.assert_not_called()

    @pytest.mark.asyncio
    async def test_malformed_item_name_ignored(self) -> None:
        feed, *_ = _make_feed()
        update = {"type": "chart", "item": "BADITEM", "fields": {"BID_CLOSE": "1.0"}}
        # Should not raise
        await feed._handlers.handle_chart_update(update)


# ---------------------------------------------------------------------------
# Trade update handling
# ---------------------------------------------------------------------------


class TestHandleTradeUpdate:
    @pytest.mark.asyncio
    async def test_accepted_fill_emits_event(self) -> None:
        import json as _json

        feed, client, bus, _ = _make_feed()
        received: list[Any] = []
        bus.subscribe(EVENT_ORDER_FILLED, lambda r: received.append(r))

        confirms = {
            "dealStatus": "ACCEPTED",
            "dealId": "D123",
            "dealReference": "REF456",
            "epic": "CS.D.AVXUSD.TODAY.IP",
            "direction": "BUY",
            "size": 2.0,
            "level": 30.5,
        }
        update = {
            "type": "trade",
            "fields": {"CONFIRMS": _json.dumps(confirms)},
        }
        await feed._handlers.handle_trade_update(update)

        assert len(received) == 1
        result = received[0]
        assert result.order_id == "D123"
        assert result.side == OrderSide.BUY
        assert result.status == OrderStatus.FILLED
        assert result.average_price == pytest.approx(30.5)

    @pytest.mark.asyncio
    async def test_rejected_fill_not_emitted(self) -> None:
        import json as _json

        feed, client, bus, _ = _make_feed()
        received: list[Any] = []
        bus.subscribe(EVENT_ORDER_FILLED, lambda r: received.append(r))

        confirms = {"dealStatus": "REJECTED", "reason": "MARKET_CLOSED"}
        update = {"type": "trade", "fields": {"CONFIRMS": _json.dumps(confirms)}}
        await feed._handlers.handle_trade_update(update)

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_missing_confirms_field_ignored(self) -> None:
        feed, *_ = _make_feed()
        update = {"type": "trade", "fields": {"OPU": "something"}}
        # Should not raise
        await feed._handlers.handle_trade_update(update)

    @pytest.mark.asyncio
    async def test_invalid_json_confirms_logged_not_raised(self) -> None:
        feed, *_ = _make_feed()
        update = {"type": "trade", "fields": {"CONFIRMS": "not-json"}}
        # Should not raise
        await feed._handlers.handle_trade_update(update)


# ---------------------------------------------------------------------------
# Account update handling
# ---------------------------------------------------------------------------


class TestHandleAccountUpdate:
    @pytest.mark.asyncio
    async def test_account_update_emits_event(self) -> None:
        """Every LS ACCOUNT push must publish an AccountUpdate on the bus so
        RiskManager can recompute the margin circuit-breaker state."""
        from bot.core.event_bus import EVENT_ACCOUNT_UPDATE
        from bot.core.models import AccountUpdate

        feed, _client, bus, _db = _make_feed()
        received: list[Any] = []

        async def _capture(payload: Any) -> None:
            received.append(payload)

        bus.subscribe(EVENT_ACCOUNT_UPDATE, _capture)

        update = {
            "type": "account",
            "fields": {
                "EQUITY": "5000.00",
                "MARGIN": "1500.00",
                "AVAILABLE_TO_DEAL": "3500.00",
                "PNL": "50.00",
            },
        }
        await feed._handlers.handle_account_update(update)

        assert len(received) == 1
        payload = received[0]
        assert isinstance(payload, AccountUpdate)
        assert payload.equity == 5000.0
        assert payload.margin_required == 1500.0
        assert payload.available_to_deal == 3500.0
        assert payload.unrealised_pnl == 50.0

    @pytest.mark.asyncio
    async def test_account_update_skips_empty_frame(self) -> None:
        """LS sends heartbeat frames with empty bodies — we must ignore them so
        the bus doesn't get spammed with zero-equity snapshots."""
        from bot.core.event_bus import EVENT_ACCOUNT_UPDATE

        feed, _client, bus, _db = _make_feed()
        received: list[Any] = []

        async def _capture(payload: Any) -> None:
            received.append(payload)

        bus.subscribe(EVENT_ACCOUNT_UPDATE, _capture)
        await feed._handlers.handle_account_update({"type": "account", "fields": {}})

        assert received == []


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestIGFeedClose:
    @pytest.mark.asyncio
    async def test_close_sets_closing_flag(self) -> None:
        feed, *_ = _make_feed()
        mock_ls = MagicMock()
        feed._ls_client = mock_ls
        await feed.close()
        assert feed._closing is True
        mock_ls.disconnect.assert_called_once()


# ---------------------------------------------------------------------------
# Heartbeat-driven self-healing reconnect (IG_LIVE_RISK_REFERENCE.md §3).
# IG_LIVE_RISK_REFERENCE.md §3 — silent listener-thread death detection.
# ---------------------------------------------------------------------------


@pytest.mark.preflight
class TestHeartbeat:
    @pytest.mark.asyncio
    async def test_force_full_reconnect_runs_spec_sequence(self) -> None:
        """Heartbeat-triggered recovery must: disconnect LS → REST re-auth →
        re-establish LS (which re-subscribes every channel)."""
        import time as _time

        feed, client, *_ = _make_feed()
        feed._loop = __import__("asyncio").get_running_loop()
        mock_ls = MagicMock()
        feed._ls_client = mock_ls
        client.refresh_session = AsyncMock()

        # Patch _connect_lightstreamer so we don't actually instantiate the SDK
        connect = MagicMock()
        feed._conn.connect_lightstreamer = connect  # type: ignore[method-assign]

        # Pretend startup happened a while ago so any grace window is irrelevant
        feed._started_at = _time.monotonic() - 999

        await feed._conn.force_full_reconnect("test")

        mock_ls.disconnect.assert_called_once()
        assert feed._ls_client is None
        client.refresh_session.assert_awaited_once()
        connect.assert_called_once()
        # Baseline reset so heartbeat doesn't immediately re-trip
        assert feed._last_tick_ts == 0.0

    @pytest.mark.asyncio
    async def test_force_full_reconnect_skipped_when_already_scheduled(self) -> None:
        """Concurrent triggers must coalesce — the SDK can fire DISCONNECTED
        and the heartbeat can trip simultaneously."""
        feed, client, *_ = _make_feed()
        feed._reconnect_scheduled = True
        client.refresh_session = AsyncMock()

        await feed._conn.force_full_reconnect("test")

        client.refresh_session.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_force_full_reconnect_no_op_when_closing(self) -> None:
        feed, client, *_ = _make_feed()
        feed._closing = True
        client.refresh_session = AsyncMock()

        await feed._conn.force_full_reconnect("test")

        client.refresh_session.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_force_full_reconnect_aborts_when_reauth_fails(self) -> None:
        """If REST re-auth fails the loop must not try to rebuild LS with stale
        tokens — it just returns and the next heartbeat cycle retries."""
        feed, client, *_ = _make_feed()
        feed._ls_client = MagicMock()
        client.refresh_session = AsyncMock(side_effect=RuntimeError("network down"))
        connect = MagicMock()
        feed._conn.connect_lightstreamer = connect  # type: ignore[method-assign]

        await feed._conn.force_full_reconnect("test")

        client.refresh_session.assert_awaited_once()
        connect.assert_not_called()

    @pytest.mark.asyncio
    async def test_heartbeat_loop_triggers_reconnect_on_stale_ticks(self) -> None:
        """When the last-tick timestamp is older than the threshold, the loop
        must call ``_force_full_reconnect``."""
        import asyncio as _asyncio
        import time as _time

        feed, *_ = _make_feed()
        feed._loop = _asyncio.get_running_loop()
        feed._started_at = _time.monotonic() - 999  # past grace window
        feed._last_tick_ts = _time.monotonic() - 60  # very stale

        force_reconnect = AsyncMock()
        feed._conn.force_full_reconnect = force_reconnect  # type: ignore[method-assign]

        from unittest.mock import patch

        with patch("bot.data.ig_ls_connection._HEARTBEAT_CHECK_INTERVAL_S", 0.001):
            task = _asyncio.create_task(feed._conn.heartbeat_loop())
            await _asyncio.sleep(0.05)
            feed._closing = True
            task.cancel()
            import contextlib as _ctx

            with _ctx.suppress(_asyncio.CancelledError):
                await task

        force_reconnect.assert_awaited()

    @pytest.mark.asyncio
    async def test_heartbeat_loop_respects_initial_grace_window(self) -> None:
        """No tick yet + we just started → don't force-reconnect."""
        import asyncio as _asyncio
        import time as _time

        feed, *_ = _make_feed()
        feed._loop = _asyncio.get_running_loop()
        feed._started_at = _time.monotonic()  # just started
        feed._last_tick_ts = 0.0  # no ticks yet

        force_reconnect = AsyncMock()
        feed._conn.force_full_reconnect = force_reconnect  # type: ignore[method-assign]

        from unittest.mock import patch

        with patch("bot.data.ig_ls_connection._HEARTBEAT_CHECK_INTERVAL_S", 0.001):
            task = _asyncio.create_task(feed._conn.heartbeat_loop())
            await _asyncio.sleep(0.05)
            feed._closing = True
            task.cancel()
            import contextlib as _ctx

            with _ctx.suppress(_asyncio.CancelledError):
                await task

        force_reconnect.assert_not_awaited()

    def test_listener_updates_heartbeat_timestamp(self) -> None:
        """Any tick on any subscription must bump the heartbeat baseline —
        proof that the SDK listener thread is still alive."""
        import asyncio as _asyncio
        import time as _time

        # _LSQueueListener already imported at module top

        feed, *_ = _make_feed()
        feed._last_tick_ts = 0.0
        queue: _asyncio.Queue[dict[str, Any]] = _asyncio.Queue()
        loop = MagicMock()
        loop.call_soon_threadsafe = MagicMock()
        listener = _LSQueueListener(queue, loop, "chart", feed)

        update = MagicMock()
        update.getFields = MagicMock(return_value={"BID_CLOSE": "1.10"})
        update.getItemName = MagicMock(return_value="CHART:EURUSD:1MINUTE")

        before = _time.monotonic()
        listener.onItemUpdate(update)
        after = _time.monotonic()

        assert before <= feed._last_tick_ts <= after

    def test_adapter_set_error_triggers_backoff_path(self) -> None:
        """``Cause: 2`` from the LS server must arm the long-backoff path so
        we don't pummel the endpoint."""
        # _ConnectionListener already imported at module top

        feed, *_ = _make_feed()
        feed._loop = MagicMock()
        feed._loop.call_soon_threadsafe = MagicMock()
        listener = _ConnectionListener(feed)

        listener.onServerError(2, "Requested Adapter Set not available")

        assert feed._adapter_set_error is True
        feed._loop.call_soon_threadsafe.assert_called_once()


# ---------------------------------------------------------------------------
# Preflight #1: after a forced reconnect, every MERGE + DISTINCT channel
# is re-subscribed.  IG_LIVE_RISK_REFERENCE.md §3.2: server-side subs do
# not survive reconnect.  The existing TestHeartbeat block proves that
# _connect_lightstreamer is *called* after teardown; this block proves
# every channel actually gets a new Subscription.
# ---------------------------------------------------------------------------


@pytest.mark.preflight
class TestReconnectResubscribe:
    @pytest.mark.asyncio
    async def test_force_reconnect_resubscribes_chart_trade_and_account(self) -> None:
        """After ``_force_full_reconnect`` the fresh LS client must receive
        a CHART (MERGE) sub per EPIC, a TRADE (DISTINCT) sub, and an
        ACCOUNT (MERGE) sub.  We mock the SDK ``ls.LightstreamerClient``
        and ``ls.Subscription`` constructors and assert what the feed
        actually issued."""
        import asyncio as _asyncio
        import time as _time
        from unittest.mock import patch

        epics = ["CS.D.AVXUSD.TODAY.IP", "CS.D.BITCOIN.TODAY.IP"]
        feed, client, *_ = _make_feed(epics)
        feed._loop = _asyncio.get_running_loop()
        feed._ls_client = MagicMock()
        client.refresh_session = AsyncMock()
        feed._started_at = _time.monotonic() - 999  # past grace window

        # Capture every Subscription() construction + every .subscribe()
        # call on the fresh client.
        new_ls_client = MagicMock()
        subscription_constructions: list[tuple[str, list[str]]] = []

        def _fake_subscription(mode: str, items: list[str], fields: list[str]) -> MagicMock:
            subscription_constructions.append((mode, list(items)))
            sub = MagicMock()
            return sub

        # Patch the LS SDK symbols used inside _connect_lightstreamer.
        with (
            patch("bot.data.ig_feed.ls.LightstreamerClient", return_value=new_ls_client),
            patch("bot.data.ig_feed.ls.Subscription", side_effect=_fake_subscription),
        ):
            await feed._conn.force_full_reconnect("test_resubscribe")

        # Three subscriptions placed against the fresh client.
        assert new_ls_client.subscribe.call_count == 3, (
            f"Expected 3 subscribes after reconnect, got {new_ls_client.subscribe.call_count}"
        )

        # Each channel's mode + item-prefix is correct.
        modes_and_items = subscription_constructions
        assert len(modes_and_items) == 3

        chart = next((m for m in modes_and_items if m[1] and m[1][0].startswith("CHART:")), None)
        trade = next((m for m in modes_and_items if m[1] and m[1][0].startswith("TRADE:")), None)
        account = next(
            (m for m in modes_and_items if m[1] and m[1][0].startswith("ACCOUNT:")), None
        )

        assert chart is not None, "CHART subscription missing after reconnect"
        assert chart[0] == "MERGE"
        # One CHART item per EPIC — covers the full universe being restored.
        assert {item.split(":")[1] for item in chart[1]} == set(epics)

        assert trade is not None, "TRADE subscription missing after reconnect"
        assert trade[0] == "DISTINCT"

        assert account is not None, "ACCOUNT subscription missing after reconnect"
        assert account[0] == "MERGE"


# ---------------------------------------------------------------------------
# Tick outlier rejection (TickValidator)
# IG_LIVE_RISK_REFERENCE.md §2.3
# ---------------------------------------------------------------------------


class TestTickValidator:
    def test_zero_or_negative_rejected(self) -> None:
        # TickValidator already imported at module top

        tv = TickValidator()
        assert tv.accept("EUR/USD", 0.0) is False
        assert tv.accept("EUR/USD", -1.10) is False

    def test_nan_or_inf_rejected(self) -> None:
        # TickValidator already imported at module top

        tv = TickValidator()
        assert tv.accept("EUR/USD", float("nan")) is False
        assert tv.accept("EUR/USD", float("inf")) is False

    def test_first_tick_accepted_to_prime(self) -> None:
        # TickValidator already imported at module top

        tv = TickValidator()
        assert tv.accept("EUR/USD", 1.1000) is True

    def test_accepts_normal_drift_during_prime_window(self) -> None:
        """Until the rolling window has ``min_prime`` returns, every (positive,
        finite) tick must be accepted — there's no σ baseline yet."""
        # TickValidator already imported at module top

        tv = TickValidator(min_prime=20)
        price = 1.1000
        for _ in range(15):
            price *= 1.0001  # tiny upward drift
            assert tv.accept("EUR/USD", price) is True

    def test_rejects_extreme_outlier_after_primer(self) -> None:
        """Once primed, a price step many σ away from the rolling mean is
        rejected — e.g. a tick that doubles in one bar after a flat history."""
        # TickValidator already imported at module top

        tv = TickValidator(min_prime=20, n_sigma=6.0, suspend_ticks=5)
        # Prime with tiny noise around 1.10
        import random

        random.seed(42)
        price = 1.10
        for _ in range(50):
            price += random.gauss(0, 0.00002)  # ~0.002 % noise
            assert tv.accept("EUR/USD", price) is True

        # Now jam in a 5 % jump — well beyond 6σ of bp-scale noise
        assert tv.accept("EUR/USD", price * 1.05) is False

    def test_suspends_epic_for_n_ticks_after_reject(self) -> None:
        # TickValidator already imported at module top

        tv = TickValidator(min_prime=20, n_sigma=6.0, suspend_ticks=5)
        # Prime
        import random

        random.seed(7)
        price = 100.0
        for _ in range(30):
            price *= 1 + random.gauss(0, 0.0001)
            tv.accept("XAU/USD", price)

        # Outlier triggers suspension
        assert tv.accept("XAU/USD", price * 1.10) is False

        # Next 4 ticks rejected even at sane values (cool-down)
        for i in range(4):
            assert tv.accept("XAU/USD", price * (1 + i * 1e-5)) is False, f"tick {i}"

        # After cool-down expires, accepts again
        assert tv.accept("XAU/USD", price) is True

    def test_epics_are_independent(self) -> None:
        """A rejection on EUR/USD must not suspend XAU/USD."""
        # TickValidator already imported at module top

        tv = TickValidator(min_prime=10, n_sigma=6.0)
        import random

        random.seed(1)
        # Prime both
        p_e, p_g = 1.10, 4500.0
        for _ in range(20):
            p_e *= 1 + random.gauss(0, 0.0001)
            p_g *= 1 + random.gauss(0, 0.0001)
            tv.accept("EUR/USD", p_e)
            tv.accept("XAU/USD", p_g)

        # Reject EUR/USD outlier
        assert tv.accept("EUR/USD", p_e * 1.20) is False
        # XAU/USD still accepts normal ticks
        assert tv.accept("XAU/USD", p_g) is True

    def test_reset_drops_state_for_named_epic(self) -> None:
        # TickValidator already imported at module top

        tv = TickValidator()
        tv.accept("EUR/USD", 1.10)
        tv.accept("XAU/USD", 4500.0)

        tv.reset("EUR/USD")
        assert "EUR/USD" not in tv._last_price
        assert "XAU/USD" in tv._last_price  # unaffected

    def test_reset_all(self) -> None:
        # TickValidator already imported at module top

        tv = TickValidator()
        tv.accept("EUR/USD", 1.10)
        tv.accept("XAU/USD", 4500.0)

        tv.reset()
        assert tv._last_price == {}
        assert tv._returns == {}


class TestSpreadMonitorIntegration:
    """IGFeed records the bid-ask spread on every *confirmed* candle
    close.  Intra-candle ticks don't add samples (keeps the rolling window
    at 1-min cadence)."""

    @pytest.mark.asyncio
    async def test_records_spread_on_confirmed_close(self) -> None:
        feed, *_ = _make_feed(["CS.D.EURUSD.MINI.IP"])

        # Prime the tick validator so the outlier filter accepts our test tick
        import random

        random.seed(0)
        price = 1.10
        for _ in range(30):
            price *= 1 + random.gauss(0, 0.0001)
            feed._tick_validator.accept("CS.D.EURUSD.MINI.IP", price)

        update = {
            "type": "chart",
            "item": "CHART:CS.D.EURUSD.MINI.IP:1MINUTE",
            "fields": {
                "BID_OPEN": "1.10000",
                "BID_HIGH": "1.10010",
                "BID_LOW": "1.09990",
                "BID_CLOSE": "1.10000",
                "OFR_OPEN": "1.10003",
                "OFR_HIGH": "1.10013",
                "OFR_LOW": "1.09993",
                "OFR_CLOSE": "1.10003",  # spread = 0.00003
                "UTM": str(1_700_000_000_000),
                "CONS_END": "1",  # confirmed close
                "LTV": "10",
            },
        }
        await feed._handlers.handle_chart_update(update)

        assert feed._spread_monitor.sample_count("CS.D.EURUSD.MINI.IP") == 1
        latest = feed._spread_monitor.latest_spread("CS.D.EURUSD.MINI.IP")
        assert latest == pytest.approx(0.00003, abs=1e-7)

    @pytest.mark.asyncio
    async def test_does_not_record_on_unconfirmed_tick(self) -> None:
        """CONS_END != "1" → still building the candle → no spread sample yet."""
        feed, *_ = _make_feed(["CS.D.EURUSD.MINI.IP"])

        import random

        random.seed(0)
        price = 1.10
        for _ in range(30):
            price *= 1 + random.gauss(0, 0.0001)
            feed._tick_validator.accept("CS.D.EURUSD.MINI.IP", price)

        update = {
            "type": "chart",
            "item": "CHART:CS.D.EURUSD.MINI.IP:1MINUTE",
            "fields": {
                "BID_OPEN": "1.10000",
                "BID_HIGH": "1.10010",
                "BID_LOW": "1.09990",
                "BID_CLOSE": "1.10000",
                "OFR_OPEN": "1.10003",
                "OFR_HIGH": "1.10013",
                "OFR_LOW": "1.09993",
                "OFR_CLOSE": "1.10003",
                "UTM": str(1_700_000_000_000),
                "CONS_END": "0",
                "LTV": "10",
            },
        }
        await feed._handlers.handle_chart_update(update)

        assert feed._spread_monitor.sample_count("CS.D.EURUSD.MINI.IP") == 0

    def test_feed_exposes_spread_monitor_property(self) -> None:
        feed, *_ = _make_feed()
        from bot.risk.spread_monitor import SpreadMonitor

        assert isinstance(feed.spread_monitor, SpreadMonitor)


class TestHandleChartUpdateRejectsOutliers:
    @pytest.mark.asyncio
    async def test_outlier_skips_store_and_event(self) -> None:
        """When the validator rejects a tick, _handle_chart_update must not
        add the candle to the store and must not emit EVENT_NEW_CANDLE."""
        from bot.core.event_bus import EVENT_NEW_CANDLE

        feed, _client, bus, _db = _make_feed(["CS.D.EURUSD.MINI.IP"])

        # Prime the validator at ~1.10
        import random

        random.seed(0)
        price = 1.10
        for _ in range(30):
            price *= 1 + random.gauss(0, 0.0001)
            feed._tick_validator.accept("CS.D.EURUSD.MINI.IP", price)

        # Now feed a wildly off-mid update through _handle_chart_update
        received: list[Any] = []

        async def _capture(c: Any) -> None:
            received.append(c)

        bus.subscribe(EVENT_NEW_CANDLE, _capture)

        outlier_update = {
            "type": "chart",
            "item": "CHART:CS.D.EURUSD.MINI.IP:1MINUTE",
            "fields": {
                "BID_OPEN": str(price),
                "BID_HIGH": str(price),
                "BID_LOW": str(price),
                "BID_CLOSE": str(price * 1.20),  # 20 % jump
                "OFR_OPEN": str(price),
                "OFR_HIGH": str(price),
                "OFR_LOW": str(price),
                "OFR_CLOSE": str(price * 1.20),
                "UTM": str(1_700_000_000_000),
                "CONS_END": "1",
                "LTV": "10",
            },
        }
        await feed._handlers.handle_chart_update(outlier_update)

        # The store should NOT have received the candle; no event emitted.
        feed._store.add_candle.assert_not_called()
        assert received == []
