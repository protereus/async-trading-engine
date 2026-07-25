"""IG Group Lightstreamer market data feed.

The primary live market-data feed for the IG broker path (the legacy
ccxt.pro ``DataFeed`` was archived 2026-06-24).

Architecture
------------
The Lightstreamer Python SDK (``lightstreamer-client-lib``) uses daemon
threads internally — it is NOT asyncio-native.  Candle updates are bridged
into the event loop via an ``asyncio.Queue``::

    LightstreamerClient (SDK daemon threads)
        └─ _LSQueueListener.onItemUpdate(type_tag="chart")    [ig_ls_listeners]
               └─ loop.call_soon_threadsafe(queue.put_nowait, update_dict)
                      └─ _consume_updates() (asyncio task)
                             └─ IGFeedHandlers.handle_chart_update()  [ig_feed_handlers]
                                    └─ event_bus.emit(EVENT_NEW_CANDLE, candle)

Subscriptions
-------------
* **CHART** — ``CHART:{epic}:1MINUTE``, mode MERGE.
  ``CONS_END == "1"`` signals a confirmed closed candle.

* **TRADE** — ``TRADE:{account_id}``, mode DISTINCT.
  ``CONFIRMS`` field carries fill confirmation JSON; published as
  ``EVENT_ORDER_FILLED`` on the event bus.

* **ACCOUNT** — ``ACCOUNT:{account_id}``, mode MERGE.
  ``EQUITY`` / ``AVAILABLE_TO_DEAL`` fields; logged at DEBUG level.

Session refresh
---------------
When ``IGClient.refresh_session()`` completes, it calls
``IGLSConnection.ls_password()`` which reads the live CST/XST from the
client.  Call ``ig_feed.reconnect()`` immediately after
``refresh_session()`` to push new credentials to the Lightstreamer client
without a full reconnect.

Module layout (post §3 split

* ``ig_feed.py`` (this file) — ``IGFeed`` shell: lifecycle, REST backfill,
  consumer task that dispatches into handlers.
* ``ig_ls_listeners.py`` — SDK-thread listener classes
  (``_LSQueueListener``, ``_ConnectionListener``).
* ``ig_ls_connection.py`` — ``IGLSConnection``: LS client construction,
  reconnect, active heartbeat, full recovery.
* ``ig_feed_handlers.py`` — ``IGFeedHandlers``: CHART/TRADE/ACCOUNT update
  parsing + ``TickValidator``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

import lightstreamer.client as ls  # noqa: F401 — kept as a re-export for tests

from bot.core.models import Candle
from bot.data.ig_feed_handlers import IGFeedHandlers, TickValidator
from bot.data.ig_ls_connection import IGLSConnection

if TYPE_CHECKING:
    from bot.config import BotConfig
    from bot.core.event_bus import EventBus
    from bot.data.candle_db import CandleDB
    from bot.data.store import DataStore
    from bot.execution.ig_client import IGClient
    from bot.risk.spread_monitor import SpreadMonitor

logger = logging.getLogger(__name__)

# Heartbeat-timeout knob is owned by IGLSConnection; importing the public
# constant here would just create stale duplication.  This local refers to
# the value purely for the startup log line.
from bot.data.ig_ls_connection import _HEARTBEAT_TIMEOUT_SEC  # noqa: E402


class IGFeed:
    """Lightstreamer-based market data feed for IG spread betting.

    Exposes the standard feed interface (``run`` / ``close``) used by the
    lifecycle and the rest of the bot.

    Args:
        client:    Connected ``IGClient`` instance (tokens already loaded).
        store:     In-memory ``DataStore`` for candle buffering.
        event_bus: Async event bus for publishing candle/trade events.
        config:    Bot configuration (``ig_epics``, ``candle_timeframe``, etc.).
        candle_db: Optional SQLite persistence layer.
        spread_monitor: Optional ``SpreadMonitor`` for the pre-trade
            spread-anomaly gate; ``None`` disables it.
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

        self._epics: list[str] = config.ig_epics

        self._ls_client: ls.LightstreamerClient | None = None
        self._update_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._closing = False

        # Dedup: track last confirmed timestamp per epic
        self._last_confirmed_ts: dict[str, int] = {}

        # Per-epic outlier rejection on incoming chart ticks.
        self._tick_validator = TickValidator()

        # Per-epic spread anomaly detector.  Sampled once per confirmed
        # candle close in handle_chart_update; pre-trade gate reads
        # ctx.spread_monitor directly.  Injected from the composition root
        # (Lifecycle.init_ig) so the legacy 'ig' path shares the same
        # instance as the live EODHD/IG-candle paths; falls back to a fresh
        # instance when constructed standalone (e.g. in tests).
        if spread_monitor is None:
            from bot.risk.spread_monitor import SpreadMonitor as _SpreadMonitor

            spread_monitor = _SpreadMonitor()
        self._spread_monitor = spread_monitor

        # Reconnect state — shared with IGLSConnection.
        self._reconnect_scheduled = False
        self._reconnect_attempt = 0

        # Heartbeat state.  ``_last_tick_ts`` is monotonic and is set by
        # the SDK listener thread on every onItemUpdate.  ``_started_at`` gates
        # the initial grace window so the heartbeat doesn't trip before the
        # first snapshot arrives.  ``_adapter_set_error`` switches reconnects
        # onto the slow back-off path used for IG server mis-config (Cause 2).
        self._last_tick_ts: float = 0.0
        self._started_at: float = 0.0
        self._adapter_set_error: bool = False

        # Collaborators (back-references to this IGFeed).
        self._conn = IGLSConnection(self)
        self._handlers = IGFeedHandlers(self)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def spread_monitor(self) -> Any:
        """Per-epic spread-widening detector — read-only handle for
        the pre-trade gate in main.py."""
        return self._spread_monitor

    async def run(self) -> None:
        """Backfill all EPICs via REST, then start Lightstreamer streaming."""
        self._loop = asyncio.get_running_loop()
        self._closing = False
        self._started_at = time.monotonic()

        if not self._epics:
            raise ValueError(
                "ig_epics is empty — set IG_EPICS=... in .env (comma-separated list of EPICs)"
            )

        # Backfill from REST
        for epic in self._epics:
            await self._backfill(epic, count=self._config.candle_buffer_size)

        # Start Lightstreamer
        self._conn.connect_lightstreamer()

        # Start async consumer + active heartbeat.
        consumer = asyncio.create_task(self._consume_updates(), name="ig_feed_consumer")
        heartbeat = asyncio.create_task(self._conn.heartbeat_loop(), name="ig_feed_heartbeat")
        self._tasks.extend([consumer, heartbeat])

        logger.info(
            "IGFeed running — subscribed to %d EPIC(s), heartbeat timeout %.0fs",
            len(self._epics),
            _HEARTBEAT_TIMEOUT_SEC,
        )
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def close(self) -> None:
        """Stop streaming and disconnect from Lightstreamer."""
        self._closing = True
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        if self._ls_client is not None:
            self._ls_client.disconnect()
            self._ls_client = None
        logger.info("IGFeed closed")

    async def reconnect(self) -> None:
        """Reconnect Lightstreamer with fresh tokens after a session refresh.

        Call this immediately after ``IGClient.refresh_session()`` to push
        the new XST to the streaming connection.
        """
        if self._ls_client is not None:
            logger.info("Reconnecting Lightstreamer with refreshed tokens")
            self._ls_client.connectionDetails.setPassword(self._conn.ls_password())
            self._ls_client.disconnect()
            self._ls_client.connect()

    # ------------------------------------------------------------------
    # Listener delegate — keeps `loop.call_soon_threadsafe(feed._schedule_reconnect)`
    # working without listeners having to know about IGLSConnection.
    # ------------------------------------------------------------------

    def _schedule_reconnect(self) -> None:
        self._conn.schedule_reconnect()

    # ------------------------------------------------------------------
    # Backfill
    # ------------------------------------------------------------------

    async def _backfill(self, epic: str, count: int = 3000) -> None:
        """Load candles from DB then fill gaps from IG REST API.

        REST is limited to ~1,000 candles per call and 10k points/week.
        The backfill fetches up to ``count`` candles; for IG, 1,000 is a
        practical ceiling (≈16h of 1m data).  For deeper history, use
        5m resolution on first deploy (see ig_integration_plan.md ).
        """
        # IG REST hard limit per call — clamp to avoid quota burn
        rest_limit = min(count, 1_000)

        # Stage 1: load from DB
        db_candles: list[Candle] = []
        if self._candle_db is not None:
            db_candles = self._candle_db.get_candles(epic, limit=count)
            logger.info("Loaded %d candles from DB for %s", len(db_candles), epic)
            if len(db_candles) >= count:
                for candle in db_candles:
                    self._store.add_candle(candle)
                return

        # Stage 2: REST backfill for the gap
        rest_candles: list[Candle] = []
        earliest_in_db: int | None = None
        if self._candle_db is not None:
            earliest_in_db = self._candle_db.get_earliest_timestamp(epic)

        try:
            fetched = await self._client.fetch_ohlcv(
                epic, self._config.candle_timeframe, limit=rest_limit
            )
            # Trim candles already in DB
            if earliest_in_db is not None:
                fetched = [c for c in fetched if c.timestamp < earliest_in_db]

            for candle in fetched:
                rest_candles.append(candle)
                if self._candle_db is not None:
                    self._candle_db.insert_candle(candle)

            logger.info(
                "Backfilled %d candles from IG REST for %s (quota remaining: %s)",
                len(rest_candles),
                epic,
                self._client.datapoints_remaining,
            )
        except Exception:
            logger.exception("REST backfill failed for %s", epic)

        # Stage 3: add REST (old) then DB (new) to store in chronological order
        for candle in rest_candles:
            self._store.add_candle(candle)
        for candle in db_candles:
            self._store.add_candle(candle)

    # ------------------------------------------------------------------
    # Async consumer
    # ------------------------------------------------------------------

    async def _consume_updates(self) -> None:
        """Read updates from the queue and dispatch to handlers."""
        while True:
            try:
                update = await self._update_queue.get()
                update_type = update.get("type")
                if update_type == "chart":
                    await self._handlers.handle_chart_update(update)
                elif update_type == "trade":
                    await self._handlers.handle_trade_update(update)
                elif update_type == "account":
                    await self._handlers.handle_account_update(update)
            except asyncio.CancelledError:
                logger.info("IGFeed consumer task cancelled")
                raise
            except Exception:
                logger.exception("Error processing Lightstreamer update")
