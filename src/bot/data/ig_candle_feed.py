"""Lightstreamer candle feed for IG-native symbols (D1 LS wiring of
).

Standalone, minimal LS subscriber whose only job is to route 1-minute
CHART updates for the symbols in ``IG_NATIVE_CANDLE_SYMBOLS`` into the
``IGCandleAggregator``, then write the hourly Candles the aggregator
emits to the same ``DataStore`` + ``CandleDB`` the strategy reads, plus
fire ``EVENT_NEW_CANDLE`` keyed by the canonical candle symbol.

Why a separate class (not an IGFeed extension)
----------------------------------------------
``IGFeed`` does a lot — REST backfill, tick validation, spread monitor,
TRADE + ACCOUNT channels — and isn't constructed at all in
``candle_exchange='twelvedata'`` mode.  Bolting an "aggregator mode"
onto it would either duplicate state machines or expose a confusing
hybrid surface.  This class shares only what's actually shared (the
``IGClient`` for tokens, ``DataStore`` / ``CandleDB`` / ``EventBus`` for
write targets) and stays under ~200 LOC.

V1 scope
--------
- LS CHART:{epic}:1MINUTE subscription for IG-native EPICs only.
- Mid-of-bid/offer as the per-minute price into the aggregator.
- IG UTM is London-local epoch ms (BST=UTC+1 in summer per IG Labs);
  same correction the existing IGFeed applies before constructing
  Candles.
- Market-open gate via ``bot.trading_hours.is_market_open`` per the
  probe finding (LS emits 24/7 regardless of underlying market).
- Reconnect on bare DISCONNECTED: tear down, refresh session, re-subscribe.
  No active tick-gap heartbeat in V1; rely on the SDK's own retry plus
  this once-per-disconnect rebuild.

``IG_NATIVE_CANDLE_SYMBOLS`` is populated with the two metals (XAU/XAG)
since 2026-06-19, so this feed is active for spot gold/silver (24/5);
it stays a no-op whenever that set is empty.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import lightstreamer.client as ls

from bot.core.event_bus import EVENT_NEW_CANDLE

# EODHD is the active universe/EPIC map (the legacy twelve_data_feed map only
# serves the CANDLE_EXCHANGE=twelvedata rollback path).  IG-native candle
# symbols are EODHD bot_keys (XAU/USD, XAG/USD), so their EPICs must resolve
# through the EODHD map — CS.D.USCGC.TODAY.IP / CS.D.USCSI.TODAY.IP.
from bot.data.eodhd_symbols import SYMBOL_EPIC_MAP
from bot.data.ig_candle_aggregator import IG_NATIVE_CANDLE_SYMBOLS, IGCandleAggregator
from bot.trading_hours import is_market_open

if TYPE_CHECKING:
    from bot.config import BotConfig
    from bot.core.event_bus import EventBus
    from bot.core.models import Candle
    from bot.data.candle_db import CandleDB
    from bot.data.store import DataStore
    from bot.execution.ig_client import IGClient
    from bot.risk.spread_monitor import SpreadMonitor

logger = logging.getLogger(__name__)

_LONDON = ZoneInfo("Europe/London")

_CHART_FIELDS = [
    "UTM",
    "BID_CLOSE",
    "OFR_CLOSE",
    "LTV",
    "CONS_END",
]

# Backoff for the rebuild-on-disconnect loop.  Kept simple — the LS SDK
# also retries internally with its own DISCONNECTED:WILL-RETRY /
# TRYING-RECOVERY states; we only step in when the SDK gives up.
_RECONNECT_DELAY_S = 5.0


class _ChartListener:
    """SDK-thread → asyncio bridge.  Marshals each chart update onto the
    main event loop's queue so the rest of the class can stay async."""

    def __init__(
        self,
        queue: asyncio.Queue[dict[str, Any]],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._queue = queue
        self._loop = loop

    def onItemUpdate(self, update: Any) -> None:  # noqa: N802
        try:
            item_name = update.getItemName() or ""
            fields = dict(update.getFields() or {})
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait,
                {"item": item_name, "fields": fields},
            )
        except Exception:
            logger.exception("IGCandleLSFeed listener error")

    def onSubscription(self) -> None:  # noqa: N802
        logger.info("IGCandleLSFeed CHART subscription active")

    def onSubscriptionError(self, code: int, message: str) -> None:  # noqa: N802
        logger.error("IGCandleLSFeed subscription error %d: %s", code, message)


class _StatusListener:
    """Logs LS connection state; flips a flag on bare DISCONNECTED so the
    drain loop can rebuild on the next quiet tick."""

    def __init__(self, feed: IGCandleLSFeed) -> None:
        self._feed = feed

    def onStatusChange(self, status: str) -> None:  # noqa: N802
        logger.info("IGCandleLSFeed LS status: %s", status)
        # Only react to bare DISCONNECTED (SDK gave up) — leave the
        # WILL-RETRY / TRYING-RECOVERY paths to the SDK's own logic.
        if status == "DISCONNECTED":
            self._feed._needs_reconnect = True

    def onServerError(self, code: int, message: str) -> None:  # noqa: N802
        logger.error("IGCandleLSFeed LS server error %d: %s", code, message)


def _epic_to_canonical_map() -> dict[str, str]:
    """Reverse-lookup table for the IG-native subset only.

    Built on every feed startup from ``SYMBOL_EPIC_MAP`` ∩
    ``IG_NATIVE_CANDLE_SYMBOLS``.  Returns empty when the set is empty
    (the D1 default), which makes the feed a no-op until D3 populates
    the set.
    """
    return {SYMBOL_EPIC_MAP[sym]: sym for sym in IG_NATIVE_CANDLE_SYMBOLS if sym in SYMBOL_EPIC_MAP}


def _utm_to_utc_ms(utm_str: str) -> int:
    """Convert IG's London-local UTM string to a real UTC epoch ms.

    Matches the correction in IGFeed._handle_chart_update — IG Labs has
    confirmed UTM is reported in London local time, not UTC.  In BST
    (summer) that's a +1 h shift to get to real UTC.  Returns 0 if the
    input can't be parsed (caller treats as missing).
    """
    if not utm_str:
        return 0
    try:
        local_ms = int(utm_str)
    except (TypeError, ValueError):
        return 0
    utc_offset = datetime.now(_LONDON).utcoffset()
    london_offset_ms = int(utc_offset.total_seconds() * 1000) if utc_offset is not None else 0
    return local_ms + london_offset_ms


class IGCandleLSFeed:
    """Run the LS subscription for IG-native candle symbols and route
    each tick through the IGCandleAggregator into store / DB / event bus.

    No-op when ``IG_NATIVE_CANDLE_SYMBOLS`` is empty — construction
    succeeds but ``run()`` returns immediately.  This makes the wiring
    safe to instantiate unconditionally; D3 just adds symbols to the
    set to activate it.
    """

    def __init__(
        self,
        client: IGClient,
        store: DataStore,
        event_bus: EventBus,
        config: BotConfig,
        candle_db: CandleDB | None = None,
        spread_monitor: SpreadMonitor | None = None,
    ) -> None:
        self._client = client
        self._store = store
        self._event_bus = event_bus
        self._config = config
        self._candle_db = candle_db
        self._spread_monitor = spread_monitor
        # Last observed (bid, ask) per symbol from LS ticks; sampled
        # into spread_monitor once per confirmed candle close (see
        # _emit_aggregated_candle), matching spread_monitor.py's documented
        # once-per-candle sampling cadence.
        self._last_bid_ask: dict[str, tuple[float, float]] = {}

        self._epic_to_symbol: dict[str, str] = _epic_to_canonical_map()
        self._aggregator = IGCandleAggregator(emit_callback=self._emit_aggregated_candle)

        self._ls_client: ls.LightstreamerClient | None = None
        self._update_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closing = False
        self._needs_reconnect = False

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Connect LS, drain the update queue, rebuild on disconnect."""
        if not self._epic_to_symbol:
            logger.info(
                "IGCandleLSFeed: no IG-native candle symbols configured — "
                "feed is dormant (this is the D1 default; D3 cutover activates)"
            )
            # Sleep forever so the task doesn't immediately return and confuse
            # the supervisor.  Cancel propagates from the bot's shutdown_event.
            await asyncio.Event().wait()
            return

        self._loop = asyncio.get_running_loop()
        self._closing = False

        logger.info(
            "IGCandleLSFeed starting: %d EPIC(s) %s",
            len(self._epic_to_symbol),
            list(self._epic_to_symbol.keys()),
        )

        # Gap-aware warm-up before opening LS.  The old path deleted every row
        # and re-fetched the full ``candle_buffer_size`` per symbol on EVERY
        # restart (thousands of datapoints against IG's 10k/week historical
        # allowance).  Now we: (1) correct any wrong-basis rows once, (2) warm
        # the in-memory store from the DB, (3) REST-backfill only symbols still
        # short of the Kronos context window.  A buffered symbol costs just a
        # 1-bar basis probe; live LS ticks keep it fresh.
        await self._backfill_if_needed()

        try:
            self._connect()
            while not self._closing:
                if self._needs_reconnect:
                    await self._rebuild()
                try:
                    msg = await asyncio.wait_for(self._update_queue.get(), timeout=5.0)
                except TimeoutError:
                    continue
                self._dispatch(msg)
        except asyncio.CancelledError:
            logger.info("IGCandleLSFeed: cancelled")
            raise
        except Exception:
            logger.exception("IGCandleLSFeed: unhandled error in run()")
            raise
        finally:
            await self.close()

    async def close(self) -> None:
        """Disconnect LS and stop the drain loop."""
        self._closing = True
        if self._ls_client is not None:
            try:
                self._ls_client.disconnect()
            except Exception:
                logger.exception("IGCandleLSFeed: disconnect failed")
            self._ls_client = None

    # ------------------------------------------------------------------
    # Internals: gap-aware warm-up + REST backfill
    # ------------------------------------------------------------------

    async def _backfill_if_needed(self) -> None:
        """Gap-aware startup warm-up (replaces the old delete-first-every-restart).

        Ordering matters:
          1. ``_wipe_wrong_basis_rows`` — one-time correction of EODHD-scaled
             metal rows (DB only).
          2. ``_backfill_below_threshold`` — REST-fetch (DB only) for symbols
             still short of the Kronos context window.
          3. ``_warm_store_from_db`` — load the now-correct, complete DB history
             into the empty in-memory store in one clean ascending pass (avoids
             ``DataStore.add_candle`` rejecting out-of-order backfill bars).
        """
        await self._wipe_wrong_basis_rows()
        await self._backfill_below_threshold()
        self._warm_store_from_db()

    async def _wipe_wrong_basis_rows(self) -> None:
        """One-time basis correction.  Before 2026-06-19 metals were EODHD
        GLD/SLV-ETF-scaled (~10×/110× off the IG spot level); because
        ``insert_candles`` is ``INSERT OR IGNORE`` those stale rows would shadow
        fresh IG-level bars and silently corrupt Kronos.  Probe one fresh IG bar
        and compare to the newest stored close; wipe the symbol only on a clear
        basis mismatch (ratio outside [0.5, 2.0]).  No-op once the store is
        IG-native (ratio ≈ 1).  Conservative: never wipes on a probe failure or
        non-positive values.  Costs 1 datapoint per stored symbol per restart.
        """
        if self._candle_db is None:
            return
        from bot.data.ig_history import fetch_ig_hourly_backfill

        for epic, symbol in self._epic_to_symbol.items():
            existing = self._candle_db.get_candles(symbol, limit=1)
            if not existing:
                continue  # nothing stored — cold start, nothing to mis-scale
            probe = await fetch_ig_hourly_backfill(self._client, symbol, epic, limit=1)
            if not probe:
                logger.warning(
                    "IGCandleLSFeed: basis probe for %s returned no bars — "
                    "skipping wipe (live ticks will warm)",
                    symbol,
                )
                continue
            stored_close = existing[-1].close
            ig_close = probe[-1].close
            if stored_close <= 0 or ig_close <= 0:
                continue
            ratio = stored_close / ig_close
            if 0.5 <= ratio <= 2.0:
                continue  # basis matches — already IG-native
            removed = self._candle_db.delete_candles_for_symbol(symbol)
            logger.warning(
                "IGCandleLSFeed: %s rows were wrong-basis (stored=%.4f vs IG=%.4f, "
                "ratio=%.3f) — wiped %d row(s), will re-seed at IG scale",
                symbol,
                stored_close,
                ig_close,
                ratio,
                removed,
            )

    async def _backfill_below_threshold(self) -> None:
        """REST-backfill (DB only) symbols still short of the Kronos context
        window.  Already-buffered symbols are skipped — that removes the
        every-restart full re-fetch and its allowance burn; live LS ticks keep
        buffered symbols fresh.  Threshold is ``kronos_context_bars`` (the
        strategy's real requirement), not ``candle_buffer_size`` (which the live
        feed fills over time), mirroring ``EODHDFeed._backfill_below_threshold``.
        """
        from bot.data.ig_history import fetch_ig_hourly_backfill

        threshold = self._config.kronos_context_bars
        limit = self._config.candle_buffer_size

        def _have(symbol: str) -> int:
            if self._candle_db is not None:
                return len(self._candle_db.get_candles(symbol, limit=threshold))
            return self._store.get_candle_count(symbol)

        needy = [(e, s) for e, s in self._epic_to_symbol.items() if _have(s) < threshold]
        if not needy:
            logger.info(
                "IGCandleLSFeed: all symbols already buffered (>=%d bars) — skipping REST backfill",
                threshold,
            )
            return
        for epic, symbol in needy:
            bars = await fetch_ig_hourly_backfill(self._client, symbol, epic, limit=limit)
            if not bars:
                logger.warning(
                    "IGCandleLSFeed: backfill returned 0 bars for %s — "
                    "store will warm from live LS ticks instead",
                    symbol,
                )
                continue
            if self._candle_db is not None:
                try:
                    self._candle_db.insert_candles(bars)
                except Exception:
                    logger.exception(
                        "IGCandleLSFeed: backfill DB insert failed for %s "
                        "(store will warm from live LS ticks instead)",
                        symbol,
                    )
            else:
                # No DB — load bars straight into the store (ascending, empty buf).
                for bar in bars:
                    self._store.add_candle(bar)
            logger.info(
                "IGCandleLSFeed: backfilled %s — %d hourly bars (oldest=%d, latest=%d)",
                symbol,
                len(bars),
                bars[0].timestamp,
                bars[-1].timestamp,
            )

    def _warm_store_from_db(self) -> None:
        """Pre-populate the in-memory store from the local CandleDB.  EODHD
        excludes metals (they're IG-native), so this feed owns their warm-up —
        mirrors ``EODHDFeed._warm_store_from_db``.  A single ascending pass into
        the freshly-constructed (empty) buffer keeps ``add_candle`` happy.
        """
        if self._candle_db is None:
            return
        limit = self._config.candle_buffer_size
        total = 0
        for symbol in self._epic_to_symbol.values():
            for c in self._candle_db.get_candles(symbol, limit=limit):
                self._store.add_candle(c)
                total += 1
        logger.info(
            "IGCandleLSFeed: warmed store from DB — %d candles across %d symbols",
            total,
            len(self._epic_to_symbol),
        )

    # ------------------------------------------------------------------
    # Internals: LS connection
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Create the LS client + the single CHART subscription."""
        if self._loop is None:
            raise RuntimeError("IGCandleLSFeed._loop unset — call run() first")

        ls_client = ls.LightstreamerClient(self._client.lightstreamer_endpoint, "DEFAULT")
        ls_client.connectionDetails.setUser(self._client.account_id)
        ls_client.connectionDetails.setPassword(self._client.ls_password)
        ls_client.addListener(_StatusListener(self))

        ls_client.connect()
        self._ls_client = ls_client

        chart_items = [f"CHART:{epic}:1MINUTE" for epic in self._epic_to_symbol]
        chart_sub = ls.Subscription("MERGE", chart_items, _CHART_FIELDS)
        chart_sub.setDataAdapter("DEFAULT")
        chart_sub.setRequestedSnapshot("yes")
        chart_sub.addListener(_ChartListener(self._update_queue, self._loop))
        ls_client.subscribe(chart_sub)

        self._needs_reconnect = False
        logger.info(
            "IGCandleLSFeed subscribed: %s",
            [f"CHART:{epic}:1MINUTE" for epic in self._epic_to_symbol],
        )

    async def _rebuild(self) -> None:
        """Tear down + reconnect on bare DISCONNECTED.

        Refreshes the IG session so the LS password (CST/XST) is current
        — those tokens rotate on session keep-alive and a stale pair is
        a common cause of repeated DISCONNECT bounces.
        """
        logger.warning("IGCandleLSFeed: LS dropped — rebuilding in %.0fs", _RECONNECT_DELAY_S)
        await asyncio.sleep(_RECONNECT_DELAY_S)
        if self._ls_client is not None:
            try:
                self._ls_client.disconnect()
            except Exception:
                logger.exception("IGCandleLSFeed: tear-down disconnect failed")
            self._ls_client = None
        try:
            await self._client.refresh_session()
        except Exception:
            logger.exception("IGCandleLSFeed: session refresh failed — retrying next loop")
            return
        self._connect()

    # ------------------------------------------------------------------
    # Internals: dispatch + aggregator callback
    # ------------------------------------------------------------------

    def _dispatch(self, msg: dict[str, Any]) -> None:
        """Parse one chart update and feed the aggregator."""
        item_name = msg.get("item", "")
        fields = msg.get("fields", {})
        parts = item_name.split(":")
        if len(parts) < 2:
            return
        epic = parts[1]
        symbol = self._epic_to_symbol.get(epic)
        if symbol is None:
            # We should never subscribe to an EPIC we can't route, but
            # be defensive against subscription-config drift.
            return

        bid_close = _safe_float(fields.get("BID_CLOSE"))
        ofr_close = _safe_float(fields.get("OFR_CLOSE"))
        if bid_close <= 0 or ofr_close <= 0:
            return
        self._last_bid_ask[symbol] = (bid_close, ofr_close)
        mid = (bid_close + ofr_close) / 2.0

        utm_ms = _utm_to_utc_ms(fields.get("UTM", ""))
        if utm_ms <= 0:
            # Defensive: skip frames without a valid UTM rather than guess
            # at the bar boundary from wall clock (would land in the wrong
            # hour around the boundary itself).
            return

        ltv = _safe_float(fields.get("LTV"))
        market_open = is_market_open(symbol)
        self._aggregator.ingest_tick(symbol, utm_ms, mid, market_open=market_open, ltv=ltv)

    def _emit_aggregated_candle(self, candle: Candle) -> None:
        """Aggregator callback — write the finalised hourly Candle to
        store + DB and fire EVENT_NEW_CANDLE keyed by the canonical
        candle symbol.  Synchronous wrt the bus emit; we schedule the
        coroutine on the active event loop.
        """
        self._store.add_candle(candle)
        if self._candle_db is not None:
            try:
                self._candle_db.insert_candle(candle)
            except Exception:
                logger.exception(
                    "IGCandleLSFeed: candle_db insert failed for %s @ %d",
                    candle.symbol,
                    candle.timestamp,
                )
        # Sample the bid-ask spread once per confirmed candle, using
        # the most recent LS tick's bid/offer close.
        if self._spread_monitor is not None:
            bid_ask = self._last_bid_ask.get(candle.symbol)
            if bid_ask is not None:
                bid_close, ofr_close = bid_ask
                spread_pts = ofr_close - bid_close
                if spread_pts > 0:
                    self._spread_monitor.record(candle.symbol, spread_pts)
        if self._loop is not None:
            self._loop.create_task(self._event_bus.emit(EVENT_NEW_CANDLE, candle))
        logger.info(
            "IGCandleLSFeed: emitted %s candle @ %d close=%.4f",
            candle.symbol,
            candle.timestamp,
            candle.close,
        )


def _safe_float(v: Any) -> float:
    """Lenient float coerce — LS sends some fields as strings, empty when
    a value isn't part of the partial frame.  Returns 0.0 for anything
    that can't parse so the caller sees a falsy value uniformly."""
    if v is None or v == "":
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
