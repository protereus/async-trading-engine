"""EODHD candle feed — single-vendor replacement for TwelveData + yfinance +
IG-native candles.

Implements migration steps 3 (backfill) + 4 (live WebSocket aggregation).

Architecture
------------
1. On startup: warm the in-memory store from the local ``CandleDB`` (a restart
   re-fetches nothing already persisted), then REST-backfill any symbol still
   below ``kronos_context_bars`` from the EODHD **Intraday Historical** API
   (1h bars, deep history — replaces TD's throttled backfill *and* IG's
   allowance-burning backfill).
2. Emit ``EVENT_NEW_CANDLE`` for the freshest backfilled bar per symbol.
3. **Live**: two real-time WebSocket connections (``forex`` for FX, ``us`` for
   equities; metals are IG-native via IGCandleLSFeed) stream ticks into a shared
   ``IGCandleAggregator`` that
   rolls them into ``:00``-aligned hourly Candles, persisted + emitted on each
   hour boundary. Forex ticks use mid=(bid+ask)/2 (no volume); us trade ticks
   use last price + size. Each connection self-reconnects with backoff.

Freshness caveat (measured 2026-06-02): the intraday *historical* endpoint is
batch-updated and lags by asset class (forex ~19h, US ~25h). So backfill ends
~1 day stale at the recent edge; the step-4 WS layer fills forward from now.

Symbol set + IG-EPIC translation live in ``bot.data.eodhd_symbols``. The DB /
strategy key is the ``bot_key`` (``EUR/USD``, ``XAU/USD``, ``AAPL``); IG EPIC
translation happens only at order time.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import aiohttp
import orjson

from bot.core.event_bus import EVENT_NEW_CANDLE
from bot.core.models import Candle
from bot.core.time_constants import HOUR_MS
from bot.data.backfill import needs_backfill
from bot.data.eodhd_symbols import (
    EODHD_UNIVERSE,
    EODHDSymbol,
    WsEndpoint,
    bot_key_for_ws,
)
from bot.data.ig_candle_aggregator import IG_NATIVE_CANDLE_SYMBOLS, IGCandleAggregator
from bot.trading_hours import is_market_open, last_expected_closed_bar_ms

if TYPE_CHECKING:
    from collections.abc import Callable

    from bot.config import BotConfig
    from bot.core.event_bus import EventBus
    from bot.data.candle_db import CandleDB
    from bot.data.store import DataStore
    from bot.risk.spread_monitor import SpreadMonitor

logger = logging.getLogger(__name__)

_INTRADAY_URL = "https://eodhd.com/api/intraday/{symbol}"
# 1h history depth needed for the 400-bar Kronos context: forex fills ~400 bars
# in ~25 days, but US equities (RTH only, ~32 bars/week) need ~13 weeks. 180
# calendar days comfortably exceeds 400 bars for every asset class.
_BACKFILL_DAYS = 180
_REQUEST_TIMEOUT_S = 45.0
_INTER_REQUEST_SLEEP_S = 0.4  # gentle pacing within the EODHD rate budget

# Real-time WebSocket endpoints (verified Authorized on our token 2026-06-02).
_WS_URL = "wss://ws.eodhistoricaldata.com/ws/{endpoint}"
_WS_HEARTBEAT_S = 20.0  # aiohttp ping/pong keepalive
_WS_RECONNECT_MIN_S = 2.0
_WS_RECONNECT_MAX_S = 60.0
# Post-close gap repair: every restart
# drops the hour bar that straddles it — the aggregator discards the
# mid-forming partial bucket (correct: persisting it would corrupt OHLC) and
# EODHD REST can't serve the bar until the provider finalises it.  Each dropped
# (symbol, hour) is queued and re-fetched at ~:10 past the hour, retrying until
# served or expired.
_REPAIR_SLOT_OFFSET_S = 600  # fetch at ~:10 past the hour, once finalised
_REPAIR_MAX_AGE_MS = 48 * HOUR_MS  # give up on a bar the provider never serves
_REPAIR_SEED_LOOKBACK_HOURS = 24  # startup seed scans this window for missing hours

# A connection must stay up at least this long to count as "stable" and earn a
# backoff reset.  Resetting on mere *connect* (as the loop used to) lets a
# flapping/throttled endpoint reconnect-storm at MIN forever — the escalation to
# MAX never engages.  See the 2026-06-03 14:14 reconnect-burst incident.
_WS_STABLE_RESET_S = 30.0


def _next_reconnect_backoff(backoff: float, uptime_s: float) -> float:
    """Backoff delay for the *next* reconnect after a connection ended.

    A connection that stayed up >= ``_WS_STABLE_RESET_S`` is considered healthy
    → recover fast (reset to MIN).  A connection that dropped sooner (or never
    connected, ``uptime_s == 0``) escalates geometrically toward MAX so a
    persistent fault can't become a tight reconnect loop.
    """
    if uptime_s >= _WS_STABLE_RESET_S:
        return _WS_RECONNECT_MIN_S
    return min(backoff * 2, _WS_RECONNECT_MAX_S)


class EODHDFeed:
    """EODHD intraday-REST backfill + (step 4) WS-aggregated live 1h bars.

    Public surface matches ``TwelveDataFeed`` so ``lifecycle.py`` can select it
    behind ``candle_exchange='eodhd'``: ``__init__(store, event_bus, config,
    candle_db)``, ``run()``, ``close()``.
    """

    def __init__(
        self,
        store: DataStore,
        event_bus: EventBus,
        config: BotConfig,
        candle_db: CandleDB | None = None,
        market_open: Callable[[str], bool] = is_market_open,
        spread_monitor: SpreadMonitor | None = None,
    ) -> None:
        self._store = store
        self._event_bus = event_bus
        self._config = config
        self._candle_db = candle_db
        self._spread_monitor = spread_monitor
        # Last observed (bid, ask) per bot_key from the forex WS
        # stream; sampled into spread_monitor once per confirmed candle
        # close (see _emit_aggregated_candle) to match the documented
        # sampling cadence in spread_monitor.py, not every raw tick.
        self._last_bid_ask: dict[str, tuple[float, float]] = {}
        # Metals (IG_NATIVE_CANDLE_SYMBOLS) are sourced from the IG-native
        # Lightstreamer feed (IGCandleLSFeed), not EODHD — exclude them from both
        # REST backfill and the live WS subscription so the two feeds own
        # disjoint symbols and never write conflicting-scale candles.
        self._symbols: list[EODHDSymbol] = [
            s for s in EODHD_UNIVERSE.values() if s.bot_key not in IG_NATIVE_CANDLE_SYMBOLS
        ]
        self._by_key: dict[str, EODHDSymbol] = {s.bot_key: s for s in self._symbols}
        self._session: aiohttp.ClientSession | None = None
        self._last_confirmed: dict[str, int] = {}
        # Post-close gap repair: hour-open timestamps (ms) per bot_key whose
        # bar was lost to a partial-bucket drop and awaits a REST re-fetch.
        self._pending_repairs: dict[str, set[int]] = {}
        # Live layer: shared tick→1h aggregator (one bucket per bot_key) + the
        # market-open gate (injectable for tests). Both WS connections feed this
        # single instance — safe because they run on one event loop.
        self._market_open = market_open
        self._aggregator = IGCandleAggregator(
            emit_callback=self._emit_aggregated_candle,
            drop_callback=self._record_dropped_bucket,
        )
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------
    # REST backfill
    # ------------------------------------------------------------------

    async def _fetch_intraday(
        self, sym: EODHDSymbol, from_ts: int | None = None
    ) -> list[dict[str, Any]]:
        """Fetch 1h intraday bars for one symbol (oldest→newest). [] on failure.

        *from_ts* (epoch seconds) narrows the window — used by the gap-repair
        path to fetch just the missing recent hours instead of the full
        ``_BACKFILL_DAYS`` history.
        """
        assert self._session is not None
        if from_ts is None:
            from_ts = int(time.time()) - _BACKFILL_DAYS * 86400
        params = {
            "api_token": self._config.eodhd_api,
            "interval": "1h",
            "fmt": "json",
            "from": str(from_ts),
        }
        url = _INTRADAY_URL.format(symbol=sym.eodhd_rest)
        try:
            timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_S)
            async with self._session.get(url, params=params, timeout=timeout) as resp:
                body = await resp.read()
                if resp.status != 200:
                    logger.error(
                        "EODHD HTTP %d for %s: %.160s", resp.status, sym.eodhd_rest, body[:160]
                    )
                    return []
                data = orjson.loads(body)
        except Exception:
            logger.exception("EODHD intraday request failed for %s", sym.eodhd_rest)
            return []

        if not isinstance(data, list):
            logger.error(
                "EODHD: unexpected response for %s: %.160s", sym.eodhd_rest, str(data)[:160]
            )
            return []
        return data

    def _parse_bar(self, sym: EODHDSymbol, b: dict[str, Any]) -> Candle | None:
        """Parse one EODHD intraday bar dict → confirmed Candle.

        Returns None for null-OHLC bars (EODHD emits those for no-trade gaps)
        and for unparseable bars (logged).
        """
        try:
            ts = b.get("timestamp")
            o = b.get("open")
            h = b.get("high")
            low = b.get("low")
            c = b.get("close")
            if ts is None or o is None or h is None or low is None or c is None:
                return None
            raw_vol = b.get("volume")
            volume = 0.0
            if sym.has_volume and raw_vol is not None and raw_vol != "":
                volume = float(raw_vol)
            return Candle(
                symbol=sym.bot_key,
                timestamp=int(ts) * 1000,
                open=float(o),
                high=float(h),
                low=float(low),
                close=float(c),
                volume=volume,
                is_confirmed=True,
            )
        except Exception:
            logger.warning("EODHD: failed to parse bar for %s: %.100s", sym.bot_key, str(b)[:100])
            return None

    def _ingest(self, sym: EODHDSymbol, bars: list[dict[str, Any]]) -> int:
        """Parse EODHD bars → Candle, store + persist. Returns count stored."""
        count = 0
        for b in bars:
            candle = self._parse_bar(sym, b)
            if candle is None:
                continue
            self._store.add_candle(candle)
            if self._candle_db is not None:
                self._candle_db.insert_candle(candle)
            count += 1
        return count

    def _warm_store_from_db(self) -> None:
        """Pre-populate the in-memory store from the local CandleDB."""
        if self._candle_db is None:
            return
        limit = self._config.candle_buffer_size
        total = 0
        for sym in self._symbols:
            for c in self._candle_db.get_candles(sym.bot_key, limit=limit):
                self._store.add_candle(c)
                total += 1
        logger.info(
            "EODHD: warmed store from DB — %d candles across %d symbols",
            total,
            len(self._symbols),
        )

    async def _backfill_below_threshold(self) -> None:
        """Backfill any symbol that is short on history *or* stale.

        Threshold is ``kronos_context_bars`` (the strategy's actual requirement),
        not ``candle_buffer_size``: US equities trade RTH-only, so a 180-day 1h
        fetch yields ~800 bars — well above the 400-bar context but far below the
        3000-bar buffer. Using the buffer as the threshold would re-fetch every
        equity on every restart. FX fills the buffer over time from live bars.

        The freshness half of :func:`needs_backfill` also re-fetches a symbol
        whose buffer is deep enough but whose newest bar predates the last
        expected closed bar for its open market — the repair path for a silently
        stalled feed (2026-07-05 weekend-restart gap). The re-ingest is safe:
        ``add_candle`` drops out-of-order bars, so only the missing newer bars
        land. A not-yet-closed bar lost mid-formation (e.g. the 23:00 bucket at a
        23:53 restart) can't be REST-recovered *here* — the post-close gap
        repair (:meth:`_repair_loop`) re-fetches it once the provider has
        finalised it.
        """
        threshold = self._config.kronos_context_bars
        need = [s for s in self._symbols if needs_backfill(self._store, s.bot_key, threshold)]
        if not need:
            logger.info("EODHD: all symbols already buffered and fresh — skipping backfill")
            return
        logger.info("EODHD: backfilling %d symbols (intraday 1h, %dd)…", len(need), _BACKFILL_DAYS)
        for sym in need:
            bars = await self._fetch_intraday(sym)
            n = self._ingest(sym, bars)
            logger.info("EODHD: backfilled %s (%s) — %d bars", sym.bot_key, sym.eodhd_rest, n)
            await asyncio.sleep(_INTER_REQUEST_SLEEP_S)

    async def _emit_latest(self) -> None:
        """Emit EVENT_NEW_CANDLE for each symbol whose latest bar is fresher."""
        for sym in self._symbols:
            candle = self._store.get_latest_candle(sym.bot_key)
            if candle is None:
                continue
            if candle.timestamp > self._last_confirmed.get(sym.bot_key, 0):
                self._last_confirmed[sym.bot_key] = candle.timestamp
                await self._event_bus.emit(EVENT_NEW_CANDLE, candle)

    # ------------------------------------------------------------------
    # Post-close gap repair
    # ------------------------------------------------------------------

    def _record_dropped_bucket(self, symbol: str, hour_start_ms: int) -> None:
        """Aggregator drop callback — queue the lost hour for post-close repair.

        Fires at rollover when a partial bucket (first tick not in minute 0) is
        discarded; a mid-hour restart re-forms the straddled hour as exactly
        such a bucket, so the restart-dropped bar lands here in the new process.
        """
        if self._candle_db is None:
            return  # nowhere durable to land the repair; the store rejects old bars
        self._pending_repairs.setdefault(symbol, set()).add(hour_start_ms)
        logger.info(
            "EODHD: queued gap repair for %s @ %d (partial bucket dropped)",
            symbol,
            hour_start_ms,
        )

    def _seed_repairs_from_freshness(self) -> None:
        """Queue closed-but-missing recent bars the startup backfill couldn't serve.

        Covers what the drop callback can't: (a) the restart that straddles an
        hour boundary — the old process's partial bucket dies with it and the
        new process never re-forms it, so the callback never fires — and (b) an
        interior hole whose queued repair was lost when a restart cleared the
        in-memory list before the provider served the bar.  Scans the last
        ``_REPAIR_SEED_LOOKBACK_HOURS`` closed market-open hours for missing
        store timestamps; a queued hour the provider will never serve is pruned
        by :meth:`_repair_from_bars` once it finalises past it.
        """
        if self._candle_db is None:
            return
        for sym in self._symbols:
            latest = self._store.get_latest_candle(sym.bot_key)
            if latest is None:
                continue  # no history at all — backfill's problem, not gap repair's
            expected = last_expected_closed_bar_ms(sym.bot_key)
            if expected is None:
                continue
            start = expected - (_REPAIR_SEED_LOOKBACK_HOURS - 1) * HOUR_MS
            have = {
                c.timestamp
                for c in self._store.get_candles(sym.bot_key, limit=_REPAIR_SEED_LOOKBACK_HOURS + 2)
            }
            missing = {
                hour
                for hour in range(start, expected + 1, HOUR_MS)
                if hour not in have
                and is_market_open(sym.bot_key, datetime.fromtimestamp(hour / 1000, tz=UTC))
            }
            if not missing:
                continue
            self._pending_repairs.setdefault(sym.bot_key, set()).update(missing)
            logger.info(
                "EODHD: seeded %d gap-repair hour(s) for %s (window %d..%d)",
                len(missing),
                sym.bot_key,
                start,
                expected,
            )

    async def _repair_loop(self) -> None:
        """Wake at ~:10 past each hour and re-fetch any queued dropped bars."""
        while True:
            now = time.time()
            next_slot = (int(now) // 3600) * 3600 + _REPAIR_SLOT_OFFSET_S
            if next_slot <= now:
                next_slot += 3600
            await asyncio.sleep(next_slot - now)
            try:
                await self._repair_pending()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("EODHD: gap-repair pass failed")

    async def _repair_pending(self) -> None:
        """One repair pass: fetch, ingest, and prune every queued hour."""
        if self._candle_db is None or not self._pending_repairs:
            return
        now_ms = int(time.time() * 1000)
        for bot_key in sorted(self._pending_repairs):
            pending = self._pending_repairs.get(bot_key)
            sym = self._by_key.get(bot_key)
            if not pending or sym is None:
                self._pending_repairs.pop(bot_key, None)
                continue
            expired = {h for h in pending if now_ms - h > _REPAIR_MAX_AGE_MS}
            for h in sorted(expired):
                logger.warning(
                    "EODHD: gap repair expired for %s @ %d — bar never became available",
                    bot_key,
                    h,
                )
            pending -= expired
            if pending:
                # Fetch from one hour before the oldest pending bar: insurance
                # against `from` being boundary-exclusive at the provider, at
                # the cost of one already-persisted (idempotent) anchor bar.
                bars = await self._fetch_intraday(sym, from_ts=min(pending) // 1000 - 3600)
                repaired = self._repair_from_bars(sym, pending, bars)
                if repaired:
                    candles = self._candle_db.get_candles(
                        bot_key, limit=self._config.candle_buffer_size
                    )
                    self._store.replace_candles(bot_key, candles)
                    logger.info(
                        "EODHD: gap-repaired %d bar(s) for %s — store reloaded from DB",
                        repaired,
                        bot_key,
                    )
                await asyncio.sleep(_INTER_REQUEST_SLEEP_S)
            if not pending:
                self._pending_repairs.pop(bot_key, None)
        if self._pending_repairs:
            logger.info(
                "EODHD: gap repair — %d hour(s) still pending across %d symbol(s), retrying hourly",
                sum(len(h) for h in self._pending_repairs.values()),
                len(self._pending_repairs),
            )

    def _repair_from_bars(
        self, sym: EODHDSymbol, pending: set[int], bars: list[dict[str, Any]]
    ) -> int:
        """DB-insert the pending hours present in *bars*; prune dead hours.

        Mutates *pending* in place: a repaired hour is removed, and an hour the
        provider has finalised past without serving (a later bar exists but the
        hour itself is absent or null-OHLC — closed market / no trades) will
        never arrive and is dropped silently.  Hours not yet served stay queued
        for the next pass.  Returns the number of bars repaired.  The repaired
        bars land in the DB only — the caller reloads the store buffer, because
        ``DataStore.add_candle`` drops out-of-order bars.
        """
        assert self._candle_db is not None
        by_ts: dict[int, dict[str, Any]] = {}
        newest_ms = 0
        for b in bars:
            ts = b.get("timestamp")
            if ts is None:
                continue
            try:
                ts_ms = int(ts) * 1000
            except (TypeError, ValueError):
                logger.debug("EODHD: gap repair skipped bar with unparseable timestamp: %r", ts)
                continue
            by_ts[ts_ms] = b
            newest_ms = max(newest_ms, ts_ms)
        repaired = 0
        for hour_ms in sorted(pending):
            hit = by_ts.get(hour_ms)
            candle = self._parse_bar(sym, hit) if hit is not None else None
            if candle is not None:
                self._candle_db.insert_candle(candle)
                pending.discard(hour_ms)
                repaired += 1
            elif newest_ms > hour_ms:
                logger.debug(
                    "EODHD: gap repair pruned for %s @ %d — provider finalised past it, "
                    "no bar exists (closed market / no trades)",
                    sym.bot_key,
                    hour_ms,
                )
                pending.discard(hour_ms)
        return repaired

    # ------------------------------------------------------------------
    # Live layer — real-time WebSocket → 1h aggregation
    # ------------------------------------------------------------------

    async def _run_live(self) -> None:
        """Run one WS connection per endpoint, feeding ticks into the aggregator.

        ``forex`` carries the 12 FX pairs; ``us`` carries the 14 equities.
        Metals are excluded here (IG-native via IGCandleLSFeed). Each connection
        self-reconnects; the post-close gap-repair loop runs alongside them;
        cancellation at shutdown tears all of it down.
        """
        self._loop = asyncio.get_running_loop()
        endpoints: list[WsEndpoint] = ["forex", "us"]
        tasks = [
            asyncio.create_task(self._ws_connection_loop(ep), name=f"eodhd_ws_{ep}")
            for ep in endpoints
            if self._ws_symbols(ep)
        ]
        tasks.append(asyncio.create_task(self._repair_loop(), name="eodhd_gap_repair"))
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    def _ws_symbols(self, endpoint: WsEndpoint) -> list[str]:
        """WS subscribe codes for *endpoint*, restricted to EODHD-owned symbols.

        Derived from ``self._symbols`` (already excludes IG-native metals), so a
        metal is never subscribed here and its ticks never reach the aggregator.
        """
        return [s.ws_symbol for s in self._symbols if s.ws_endpoint == endpoint]

    async def _ws_connection_loop(self, endpoint: WsEndpoint) -> None:
        """Connect → subscribe → stream for one endpoint, reconnecting on drop."""
        assert self._session is not None
        url = _WS_URL.format(endpoint=endpoint)
        subs = ",".join(self._ws_symbols(endpoint))
        backoff = _WS_RECONNECT_MIN_S
        while True:
            connected_at: float | None = None
            try:
                async with self._session.ws_connect(
                    f"{url}?api_token={self._config.eodhd_api}",
                    heartbeat=_WS_HEARTBEAT_S,
                    timeout=aiohttp.ClientWSTimeout(ws_close=15.0),
                ) as ws:
                    await ws.send_json({"action": "subscribe", "symbols": subs})
                    logger.info(
                        "EODHD WS[%s]: connected, subscribed %d symbols",
                        endpoint,
                        len(self._ws_symbols(endpoint)),
                    )
                    connected_at = time.monotonic()
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            self._handle_ws_message(endpoint, msg.data)
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
                logger.warning("EODHD WS[%s]: disconnected — reconnecting", endpoint)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("EODHD WS[%s]: connection error", endpoint)
            # Reset to MIN only if the connection proved stable; otherwise
            # escalate so a flapping endpoint backs off instead of storming.
            uptime = (time.monotonic() - connected_at) if connected_at is not None else 0.0
            await asyncio.sleep(backoff)
            backoff = _next_reconnect_backoff(backoff, uptime)

    def _handle_ws_message(self, endpoint: WsEndpoint, raw: str) -> None:
        """Parse one WS frame and feed it to the aggregator.

        forex frame: ``{s, a(ask), b(bid), t(ms)}`` → mid price, no volume.
        us trade frame: ``{s, p(last), v(size), t(ms)}`` → last price + size.
        Auth/status frames (``status_code``/``message``) are ignored.
        """
        try:
            d = orjson.loads(raw)
        except Exception:
            logger.debug("EODHD WS[%s]: dropped unparseable frame: %.200s", endpoint, raw)
            return
        if not isinstance(d, dict) or "t" not in d or "s" not in d:
            return  # status/ack frame
        bot_key = bot_key_for_ws(endpoint, d["s"])
        if bot_key is None:
            return
        if endpoint == "forex":
            bid, ask = d.get("b"), d.get("a")
            if bid is None or ask is None:
                return
            self._last_bid_ask[bot_key] = (float(bid), float(ask))
            price = (float(bid) + float(ask)) / 2.0
            ltv = 0.0
        else:
            p = d.get("p")
            if p is None:
                return
            price = float(p)
            ltv = float(d.get("v") or 0.0)
        try:
            utm_ms = int(d["t"])
        except (TypeError, ValueError):
            logger.debug("EODHD WS[%s]: unparseable timestamp in frame: %.200s", endpoint, raw)
            return
        self._aggregator.ingest_tick(
            bot_key, utm_ms, price, market_open=self._market_open(bot_key), ltv=ltv
        )

    def _emit_aggregated_candle(self, candle: Candle) -> None:
        """Aggregator callback — persist the finalised hourly Candle + emit.

        Sync wrt the bus emit; the coroutine is scheduled on the active loop
        (same bridge pattern as ``IGCandleLSFeed._emit_aggregated_candle``).
        """
        self._store.add_candle(candle)
        if self._candle_db is not None:
            try:
                self._candle_db.insert_candle(candle)
            except Exception:
                logger.exception(
                    "EODHD: candle_db insert failed for %s @ %d", candle.symbol, candle.timestamp
                )
        self._last_confirmed[candle.symbol] = candle.timestamp
        # Sample the bid-ask spread once per confirmed candle, using
        # the most recent forex tick's bid/ask (no bid/ask for 'us' frames,
        # so this is a no-op there).
        if self._spread_monitor is not None:
            bid_ask = self._last_bid_ask.get(candle.symbol)
            if bid_ask is not None:
                bid, ask = bid_ask
                spread_pts = ask - bid
                if spread_pts > 0:
                    self._spread_monitor.record(candle.symbol, spread_pts)
        if self._loop is not None:
            self._loop.create_task(self._event_bus.emit(EVENT_NEW_CANDLE, candle))
        logger.info(
            "EODHD: emitted %s candle @ %d close=%.5f vol=%.0f",
            candle.symbol,
            candle.timestamp,
            candle.close,
            candle.volume,
        )

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Warm from DB, backfill via intraday REST, emit, then run live layer."""
        self._session = aiohttp.ClientSession()
        try:
            self._warm_store_from_db()
            await self._backfill_below_threshold()
            await self._emit_latest()
            self._seed_repairs_from_freshness()
            counts = {s.bot_key: self._store.get_candle_count(s.bot_key) for s in self._symbols}
            logger.info("EODHD: backfill complete — candle counts: %s", counts)
            await self._run_live()
        except asyncio.CancelledError:
            logger.info("EODHDFeed: cancelled")
            raise
        except Exception:
            logger.exception("EODHDFeed: unhandled error in run()")
            raise
        finally:
            await self.close()

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
            logger.debug("EODHD: aiohttp session closed")
