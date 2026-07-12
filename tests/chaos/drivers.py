"""Observation utilities for the chaos suite.

The SDK's listener threads write ``feed._last_tick_ts`` (monotonic) on every
update and the bot's loggers narrate every lifecycle transition, so the
harness observes recovery through exactly two side channels:

* ``EventLog`` — a logging handler that classifies bot log records into
  coarse event kinds with monotonic timestamps.  Only the (kind, timestamp)
  pair is kept; raw messages are dropped so nothing sensitive can leak into
  results.
* tick sampling — polling ``feed._last_tick_ts`` for increases.  Resets to
  0.0 during recovery are ignored (they mark a rebuilt baseline, not data).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot.data.ig_feed import IGFeed

# Substring → event kind, checked in order.  Sourced from the log calls in
# bot.data.ig_ls_connection / ig_feed; update alongside those messages.
_CLASSIFIERS: list[tuple[str, str]] = [
    ("heartbeat lost", "heartbeat_trip"),
    ("no ticks within", "heartbeat_trip"),
    ("recovered after heartbeat loss", "heartbeat_recovery"),
    ("reconnect attempt", "sdk_reconnect_scheduled"),
    ("reconnected successfully", "sdk_reconnect_ok"),
    ("Lightstreamer connecting to", "ls_connect"),
    ("re-auth during heartbeat recovery failed", "reauth_failed"),
]


@dataclass
class LogEvent:
    t: float  # time.monotonic()
    kind: str
    level: int


class EventLog(logging.Handler):
    """Classifying log handler; thread-safe (SDK threads log off-loop)."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self._events: list[LogEvent] = []
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        kind = next((k for needle, k in _CLASSIFIERS if needle in msg), None)
        if kind is None and record.levelno >= logging.ERROR:
            kind = "error"
        if kind is not None:
            with self._lock:
                self._events.append(LogEvent(time.monotonic(), kind, record.levelno))

    def count(self, kind: str, since: float = 0.0) -> int:
        with self._lock:
            return sum(1 for e in self._events if e.kind == kind and e.t >= since)

    def first(self, kind: str, since: float = 0.0) -> float | None:
        with self._lock:
            return next((e.t for e in self._events if e.kind == kind and e.t >= since), None)

    async def wait(self, kind: str, since: float, within_s: float) -> float | None:
        """Return the monotonic time of the first matching event, or None."""
        deadline = time.monotonic() + within_s
        while time.monotonic() < deadline:  # noqa: ASYNC110 — poll across threads
            t = self.first(kind, since)
            if t is not None:
                return t
            await asyncio.sleep(0.25)
        return self.first(kind, since)


async def count_ticks(feed: IGFeed, seconds: float) -> int:
    """Count distinct update arrivals over a window by sampling the tick clock."""
    ticks = 0
    last = feed._last_tick_ts
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        await asyncio.sleep(0.2)
        now_ts = feed._last_tick_ts
        if now_ts > last:
            ticks += 1
        if now_ts != 0.0:
            last = now_ts
    return ticks


async def wait_tick(feed: IGFeed, after: float, within_s: float) -> float | None:
    """Wait for an update newer than the given monotonic instant."""
    deadline = time.monotonic() + within_s
    while time.monotonic() < deadline:  # noqa: ASYNC110 — observing SDK-thread state
        if feed._last_tick_ts > after:
            return time.monotonic()
        await asyncio.sleep(0.2)
    return None
