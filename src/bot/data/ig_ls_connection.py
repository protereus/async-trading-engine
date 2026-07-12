"""Lightstreamer connection lifecycle and heartbeat recovery.

Extracted from ``ig_feed.py``.  Owns:

* Initial LS client construction + the three subscriptions (CHART, TRADE,
  ACCOUNT).
* SDK-driven reconnect (bare ``DISCONNECTED``) and IG-mis-config reconnect
  (``Adapter Set not available`` → long backoff).
* Active heartbeat loop (IG_LIVE_RISK_REFERENCE.md §3): re-establishes the session when no tick has
  been seen for ``_HEARTBEAT_TIMEOUT_SEC``.
* Full recovery sequence: tear-down → REST re-auth (CST/XST invalidated by
  drop) → fresh LS connect → re-subscribe everything.

Holds a back-reference to its parent ``IGFeed`` so it can read/mutate the
shared LS client / loop / closing / reconnect-state / IGClient / tick
validator on it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import TYPE_CHECKING

import lightstreamer.client as ls

from bot.data.ig_ls_listeners import _ConnectionListener, _LSQueueListener

if TYPE_CHECKING:
    from bot.data.ig_feed import IGFeed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reconnect / heartbeat tuning
# ---------------------------------------------------------------------------
_RETRY_DELAY_SECONDS = 5
_RECONNECT_MAX_DELAY = 60  # cap exponential backoff at 60s

# ---------------------------------------------------------------------------
# Active heartbeat (IG_LIVE_RISK_REFERENCE.md §3).
#
# The Lightstreamer SDK runs listener threads as daemons.  When the IG server
# force-drops the connection (reliably after ~2 h, sometimes daily) those
# threads can die *silently* — no DISCONNECTED event, no exception — leaving
# this process believing it is connected while no ticks flow.  Unhedged
# exposure then drifts.
#
# The fix is a main-loop heartbeat that compares the wall-clock time of the
# most recent tick across all subscriptions against ``HEARTBEAT_TIMEOUT_SEC``.
# On staleness we tear down, REST-re-auth (CST/XST invalidated by the drop),
# rebuild the LS client, and re-subscribe every channel from scratch.
# ---------------------------------------------------------------------------
_HEARTBEAT_TIMEOUT_SEC = 10.0
_HEARTBEAT_CHECK_INTERVAL_S = 2.0
# Don't trip the heartbeat until subscriptions have had a chance to deliver
# their first snapshot.  Outside trading hours the first real tick may not
# arrive for tens of seconds, so the grace window is intentionally generous.
_HEARTBEAT_INITIAL_GRACE_S = 60.0

# "Cause: 2 - Requested Adapter Set not available" is a server-side mis-config
# IG occasionally returns.  Bashing the endpoint with rapid reconnects achieves
# nothing — back off 30-120 s per §3.2.
_ADAPTER_SET_BACKOFF_MIN_S = 30.0
_ADAPTER_SET_BACKOFF_MAX_S = 120.0

# CHART fields subscribed from Lightstreamer
_CHART_FIELDS = [
    "UTM",
    "BID_OPEN",
    "BID_HIGH",
    "BID_LOW",
    "BID_CLOSE",
    "OFR_OPEN",
    "OFR_HIGH",
    "OFR_LOW",
    "OFR_CLOSE",
    "LTV",
    "CONS_END",
]

# TRADE fields subscribed from Lightstreamer
_TRADE_FIELDS = ["CONFIRMS", "OPU", "WOU"]

# ACCOUNT fields subscribed from Lightstreamer
_ACCOUNT_FIELDS = ["EQUITY", "AVAILABLE_TO_DEAL", "MARGIN", "PNL", "FUNDS"]


class IGLSConnection:
    """Owns the Lightstreamer client + reconnect + heartbeat lifecycle.

    Composed onto ``IGFeed``; mutates ``feed._ls_client`` / ``_reconnect_*`` /
    ``_last_tick_ts`` / ``_started_at`` / ``_adapter_set_error`` so the
    listener code (which reads those fields off the feed) keeps working.
    """

    def __init__(self, feed: IGFeed) -> None:
        self._feed = feed

    def ls_password(self) -> str:
        """Lightstreamer password from the client's current CST/XST tokens."""
        return self._feed._client.ls_password

    def connect_lightstreamer(self) -> None:
        """Create and connect the Lightstreamer client, subscribe to all channels."""
        feed = self._feed
        if feed._loop is None:
            raise RuntimeError("IGFeed._loop not set — call run() before connect_lightstreamer()")

        ls_client = ls.LightstreamerClient(feed._client.lightstreamer_endpoint, "DEFAULT")
        ls_client.connectionDetails.setUser(feed._client.account_id)
        ls_client.connectionDetails.setPassword(self.ls_password())

        # Connection status listener
        ls_client.addListener(_ConnectionListener(feed))

        ls_client.connect()
        feed._ls_client = ls_client

        # CHART subscriptions — one per EPIC
        chart_items = [f"CHART:{epic}:1MINUTE" for epic in feed._epics]
        chart_sub = ls.Subscription("MERGE", chart_items, _CHART_FIELDS)
        chart_sub.setDataAdapter("DEFAULT")
        chart_sub.setRequestedSnapshot("yes")
        chart_sub.addListener(_LSQueueListener(feed._update_queue, feed._loop, "chart", feed))
        ls_client.subscribe(chart_sub)

        # TRADE subscription for fill notifications
        trade_sub = ls.Subscription(
            "DISTINCT",
            [f"TRADE:{feed._client.account_id}"],
            _TRADE_FIELDS,
        )
        trade_sub.setDataAdapter("DEFAULT")
        trade_sub.addListener(_LSQueueListener(feed._update_queue, feed._loop, "trade", feed))
        ls_client.subscribe(trade_sub)

        # ACCOUNT subscription for equity updates
        account_sub = ls.Subscription(
            "MERGE",
            [f"ACCOUNT:{feed._client.account_id}"],
            _ACCOUNT_FIELDS,
        )
        account_sub.setDataAdapter("DEFAULT")
        account_sub.addListener(_LSQueueListener(feed._update_queue, feed._loop, "account", feed))
        ls_client.subscribe(account_sub)

        logger.info(
            "Lightstreamer connecting to %s  account=%s  epics=%s",
            feed._client.lightstreamer_endpoint,
            feed._client.account_id,
            feed._epics,
        )

    def schedule_reconnect(self) -> None:
        """Schedule a reconnect coroutine (called from SDK thread via call_soon_threadsafe)."""
        feed = self._feed
        if feed._reconnect_scheduled or feed._closing:
            return
        feed._reconnect_scheduled = True
        if feed._loop is None:
            return
        feed._loop.create_task(self.reconnect_with_backoff())

    async def reconnect_with_backoff(self) -> None:
        """Wait, then reconnect Lightstreamer with exponential backoff.

        Fires from bare DISCONNECTED or from an adapter-set-error.  The latter
        uses a much longer backoff per IG_LIVE_RISK_REFERENCE.md §3.2 because
        rapid reconnects do not resolve server-side mis-config.
        """
        feed = self._feed
        if feed._adapter_set_error:
            # Use a wide jitter window inside the 30-120 s envelope so multiple
            # bots reconnecting at once don't synchronise on the endpoint.
            import random

            delay = random.uniform(_ADAPTER_SET_BACKOFF_MIN_S, _ADAPTER_SET_BACKOFF_MAX_S)
            logger.warning(
                "Lightstreamer adapter-set error — backing off %.0fs before reconnect", delay
            )
        else:
            delay = min(_RETRY_DELAY_SECONDS * (2**feed._reconnect_attempt), _RECONNECT_MAX_DELAY)
            logger.info(
                "Lightstreamer reconnect attempt %d in %.0fs", feed._reconnect_attempt + 1, delay
            )
        feed._reconnect_attempt += 1
        await asyncio.sleep(delay)

        feed._reconnect_scheduled = False
        if feed._closing:
            return

        # Null out the stale reference before attempting; prevents a failed
        # connect_lightstreamer() from leaving a dead client in feed._ls_client.
        if feed._ls_client is not None:
            with contextlib.suppress(Exception):
                feed._ls_client.disconnect()
            feed._ls_client = None

        try:
            self.connect_lightstreamer()
            feed._reconnect_attempt = 0  # reset on success
            feed._adapter_set_error = False
            # Reset heartbeat baseline so the next staleness check waits for
            # the first post-reconnect tick rather than re-tripping immediately.
            feed._last_tick_ts = 0.0
            feed._started_at = time.monotonic()
            logger.info("Lightstreamer reconnected successfully")
        except Exception:
            logger.exception("Lightstreamer reconnect failed — will retry on next DISCONNECTED")
            # Next bare DISCONNECTED from the SDK will trigger schedule_reconnect again

    async def heartbeat_loop(self) -> None:
        """Active health check: re-establish the LS session if no ticks arrive.

        The full recovery sequence on staleness (per IG_LIVE_RISK_REFERENCE.md
        §3.2) is: tear down → REST re-auth (CST/XST invalidated by drop) →
        re-establish LS → re-subscribe every channel from scratch (server-side
        subs do not survive disconnect).  ``force_full_reconnect`` performs
        all four steps.
        """
        feed = self._feed
        while True:
            try:
                await asyncio.sleep(_HEARTBEAT_CHECK_INTERVAL_S)
                if feed._closing:
                    return
                if feed._reconnect_scheduled:
                    continue  # an SDK-driven reconnect is already in flight

                now = time.monotonic()
                if feed._last_tick_ts == 0.0:
                    # No tick yet — respect the initial grace window.
                    if now - feed._started_at < _HEARTBEAT_INITIAL_GRACE_S:
                        continue
                    logger.error(
                        "Lightstreamer heartbeat: no ticks within %.0fs of startup — "
                        "forcing full reconnect",
                        _HEARTBEAT_INITIAL_GRACE_S,
                    )
                    await self.force_full_reconnect("no initial ticks")
                    continue

                elapsed = now - feed._last_tick_ts
                if elapsed > _HEARTBEAT_TIMEOUT_SEC:
                    logger.error(
                        "Lightstreamer heartbeat lost: %.1fs since last tick "
                        "(threshold %.1fs) — forcing full reconnect",
                        elapsed,
                        _HEARTBEAT_TIMEOUT_SEC,
                    )
                    await self.force_full_reconnect(f"stale for {elapsed:.1f}s")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Heartbeat loop iteration failed")

    async def force_full_reconnect(self, reason: str) -> None:
        """Spec-compliant heartbeat recovery (IG_LIVE_RISK_REFERENCE.md §3.2):

        1. tear down LS, free sockets / listener threads
        2. re-auth via REST POST /session (previous CST/XST invalidated by drop)
        3. re-establish LS with fresh headers
        4. re-subscribe every MERGE + DISTINCT channel
        """
        feed = self._feed
        if feed._closing or feed._reconnect_scheduled:
            return
        feed._reconnect_scheduled = True
        try:
            # 1. Tear down
            if feed._ls_client is not None:
                with contextlib.suppress(Exception):
                    feed._ls_client.disconnect()
                feed._ls_client = None

            # 2. Re-auth via REST — IG invalidates session tokens on drop, so
            # an LS reconnect with stale CST/XST will hard-fail.
            try:
                await feed._client.refresh_session()
            except Exception:
                logger.exception(
                    "REST re-auth during heartbeat recovery failed (%s) — "
                    "will retry on next heartbeat cycle",
                    reason,
                )
                return

            # 3 + 4. Re-establish LS + re-subscribe every channel.
            try:
                self.connect_lightstreamer()
            except Exception:
                logger.exception("LS rebuild during heartbeat recovery failed (%s)", reason)
                return

            # Reset heartbeat baseline so the very next check doesn't re-trip
            # before any tick has had time to arrive.
            feed._last_tick_ts = 0.0
            feed._started_at = time.monotonic()
            feed._reconnect_attempt = 0
            # The resubscribe snapshot may break return continuity;
            # drop the rolling window so the post-recovery primer rebuilds it.
            feed._tick_validator.reset()
            logger.info("Lightstreamer recovered after heartbeat loss (%s)", reason)
        finally:
            feed._reconnect_scheduled = False
