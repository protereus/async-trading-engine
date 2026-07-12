"""Lightstreamer SDK listener classes.

Extracted from ``ig_feed.py``.

These run on the Lightstreamer SDK's daemon listener threads, NOT on the
asyncio event loop.  All interaction with the loop is via
``loop.call_soon_threadsafe`` — anything that needs to touch the
``asyncio.Queue`` or schedule a coroutine MUST stay on this side of the
boundary.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bot.data.ig_feed import IGFeed

logger = logging.getLogger(__name__)


class _LSQueueListener:  # ls.SubscriptionListener — typed as Any by the SDK
    """Bridges Lightstreamer subscription updates into the asyncio event loop.

    A single parameterised class replaces the previous _ChartListener,
    _TradeListener, and _AccountListener — they were structurally identical,
    differing only in the ``type_tag`` string dispatched into the queue.
    """

    def __init__(
        self,
        queue: asyncio.Queue[dict[str, Any]],
        loop: asyncio.AbstractEventLoop,
        type_tag: str,
        feed: IGFeed,
    ) -> None:
        self._queue = queue
        self._loop = loop
        self._type_tag = type_tag
        self._feed = feed

    def onItemUpdate(self, update: Any) -> None:  # noqa: N802
        try:
            # Update heartbeat baseline as early as possible — *any* tick on any
            # subscription proves the SDK listener thread is still alive.
            # Single-float write is atomic under CPython's GIL; no lock needed.
            self._feed._last_tick_ts = time.monotonic()

            fields = update.getFields()
            if fields:
                payload: dict[str, Any] = {
                    "type": self._type_tag,
                    "fields": dict(fields),
                }
                if self._type_tag == "chart":
                    payload["item"] = update.getItemName()
                self._loop.call_soon_threadsafe(self._queue.put_nowait, payload)
        except Exception:
            logger.exception("_LSQueueListener(%s).onItemUpdate error", self._type_tag)

    def onSubscription(self) -> None:  # noqa: N802
        logger.info("Lightstreamer %s subscription active", self._type_tag.upper())

    def onSubscriptionError(self, code: int, message: str) -> None:  # noqa: N802
        logger.error(
            "Lightstreamer %s subscription error %d: %s", self._type_tag.upper(), code, message
        )


class _ConnectionListener:  # ls.ClientListener — typed as Any by the SDK
    """Logs status changes and triggers reconnect on unexpected disconnection."""

    def __init__(self, feed: IGFeed) -> None:
        self._feed = feed

    def onStatusChange(self, status: str) -> None:  # noqa: N802
        logger.info("Lightstreamer status: %s", status)
        # Only schedule our own reconnect on bare DISCONNECTED (SDK gave up).
        # DISCONNECTED:WILL-RETRY and DISCONNECTED:TRYING-RECOVERY mean the SDK
        # is already handling retry internally — firing our reconnect on top
        # would create competing clients with duplicate subscriptions.
        if status == "DISCONNECTED" and not self._feed._closing:
            logger.warning("Lightstreamer disconnected permanently — scheduling reconnect")
            loop = self._feed._loop
            if loop is not None:
                loop.call_soon_threadsafe(self._feed._schedule_reconnect)

    def onServerError(self, code: int, message: str) -> None:  # noqa: N802
        logger.error("Lightstreamer server error %d: %s", code, message)
        # IG occasionally returns Cause: 2 — "Requested Adapter Set not available".
        # Rapid reconnects don't help (server-side mis-config); we need a long
        # backoff before re-attempting per IG_LIVE_RISK_REFERENCE.md §3.2.
        msg_text = str(message)
        if code == 2 or "Adapter Set not available" in msg_text:
            self._feed._adapter_set_error = True
            loop = self._feed._loop
            if loop is not None and not self._feed._closing:
                loop.call_soon_threadsafe(self._feed._schedule_reconnect)

    def onListenEnd(self) -> None:  # noqa: N802
        pass

    def onListenStart(self) -> None:  # noqa: N802
        pass

    def onPropertyChange(self, prop: str) -> None:  # noqa: N802
        pass
