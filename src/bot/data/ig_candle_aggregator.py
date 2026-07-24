"""Aggregate 1-minute IG Lightstreamer prints into hourly Candles (D1 of
).

Pure aggregation logic with no Lightstreamer / IG / asyncio dependency —
the caller feeds ticks via ``ingest_tick`` and gets finalised ``Candle``
objects back through a callback on hour rollover.  Wiring into
``IGFeed._handle_chart_update`` (D1 phase 2) is a separate concern; this
module is unit-testable without any live connection.

Probe finding (2026-05-31, see ``scripts/probe_ls_chart.py``): IG LS
emits a fresh distinct UTM every minute on every subscribed EPIC
*regardless of underlying market open/closed status*, and synthesises a
non-static quote during closures.  ``CONS_END=1`` likewise fires every
minute and tells us nothing about whether real trading happened.  So the
aggregator gates on ``market_open`` (resolved by the caller via
``bot.trading_hours.is_market_open``) rather than any LS-level signal.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from bot.core.models import Candle
from bot.core.time_constants import HOUR_MS

logger = logging.getLogger(__name__)


# Canonical-symbol set (bot_keys) that the IG LS feed is the primary candle
# source for.  Under EODHD the set is the two metals.  XAU/XAG would
# otherwise be EODHD-sourced from the GLD/SLV US ETFs,
# which only tick in the US session and go dark overnight / on NYSE holidays
# (the 2026-06-19 Juneteenth idle case).  IG quotes spot gold/silver 24/5, so
# routing them through IGCandleLSFeed keeps them ticking around the clock.
# Lightstreamer is a streaming channel — it does NOT consume the IG REST
# 10k-datapoints/week historical allowance.
# Adding a symbol here requires lock-step changes:
#   1. ``eodhd_feed`` — EODHD must stop owning the symbol (it's filtered out of
#      backfill + the WS subscription via this same set).
#   2. ``ig_pip_value`` (``bot.execution.ig_quote_scale``) → 1.0 so
#      ``ig_quote_scale`` == 1.0; for EODHD symbols set it on the
#      ``EODHD_UNIVERSE`` entry (read first), not the legacy ``_IG_PIP_VALUE``.
#   3. ``tests/test_ig_quote_scale.py`` — update the parametrised cases.
IG_NATIVE_CANDLE_SYMBOLS: frozenset[str] = frozenset({"XAU/USD", "XAG/USD"})

_MINUTE_MS = 60_000

# Callback signature: takes the finalised candle, returns nothing.
# Sync because we don't want the aggregator to know about asyncio.
EmitCallback = Callable[[Candle], None]

# Invoked with (symbol, hour_start_ms) whenever a partial bucket is dropped at
# hour rollover — the hook the post-close gap repair hangs off (a restart
# mid-hour re-forms the straddled hour as exactly such a partial bucket).
DropCallback = Callable[[str, int], None]


@dataclass
class _Bucket:
    """In-progress hourly OHLCV bucket for one symbol.

    ``was_subscribed_at_open`` is the partial-hour-drop guard: the very
    first tick observed for a fresh hour must arrive within the first
    minute of that hour for the bucket to be eligible for emission.  If
    the bot starts mid-hour (or LS reconnects mid-hour), the bucket is
    accumulated but never emitted — the next full hour is the first
    real candle.
    """

    hour_start_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    was_subscribed_at_open: bool = False


class IGCandleAggregator:
    """Per-symbol 1-min-tick → 1-hour-Candle aggregator.

    Designed for the symbols in ``IG_NATIVE_CANDLE_SYMBOLS`` (the metals
    ``{"XAU/USD", "XAG/USD"}`` since the 2026-06-19 cutover).  One instance
    per bot; the symbol arg on every method distinguishes per-symbol state.

    Thread safety: NOT thread-safe.  The intended caller is
    ``IGFeed._handle_chart_update`` which runs single-threaded on the
    asyncio event loop after the LS listener marshals updates through
    ``loop.call_soon_threadsafe``.  If a future caller wants to feed
    ticks from multiple threads, add a lock — but the simpler discipline
    is "stay on the event loop".
    """

    def __init__(
        self,
        emit_callback: EmitCallback,
        drop_callback: DropCallback | None = None,
    ) -> None:
        self._emit = emit_callback
        self._drop = drop_callback
        self._buckets: dict[str, _Bucket] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_tick(
        self,
        symbol: str,
        utm_ms: int,
        mid: float,
        *,
        market_open: bool,
        ltv: float = 0.0,
    ) -> None:
        """Feed one 1-minute LS print into the aggregator.

        Args:
            symbol: Canonical candle symbol (``"XAU/USD"``, not the EPIC).
            utm_ms: LS-server UTM, epoch ms — the start of the 1-minute
                bar this tick belongs to.
            mid: Mid-price (typically ``(BID_CLOSE + OFR_CLOSE) / 2``).
            market_open: Result of ``trading_hours.is_market_open(symbol)``
                evaluated at tick-receive time.  Ticks for closed markets
                are dropped (LS synthesises continuous quotes 24/7 that
                don't represent real trading — see module docstring).
            ltv: Last traded volume for the minute.  Optional; if every
                tick passes 0 the emitted Candle has volume=0 which matches
                the zero-volume FX/index convention.

        No-op when ``market_open`` is False, when ``utm_ms`` is non-positive
        (defensive against malformed feed frames), or when ``mid`` is
        non-positive (zero/missing price — feed problem, not data).
        """
        if not market_open:
            return
        if utm_ms <= 0 or mid <= 0:
            return

        hour_start = (utm_ms // HOUR_MS) * HOUR_MS
        bucket = self._buckets.get(symbol)

        # Reject ticks pointing at a strictly earlier hour than the current
        # bucket.  This is the LS reconnect-snapshot quirk the tick validator
        # already documents — letting it through would either replace the current
        # bucket with a backwards-in-time one or replay an already-emitted
        # hour.  Drop and carry on with the active bucket.
        if bucket is not None and hour_start < bucket.hour_start_ms:
            return

        if bucket is None or bucket.hour_start_ms != hour_start:
            # Hour rollover or first ever tick for this symbol.  If the
            # outgoing bucket was eligible (subscribed at minute 0 of its
            # hour), emit it as a confirmed candle before starting the
            # new bucket.  Otherwise drop it silently — partial bars must
            # never enter the candle store.
            if bucket is not None:
                if bucket.was_subscribed_at_open:
                    self._emit(self._finalise(symbol, bucket))
                else:
                    logger.debug(
                        "IGCandleAggregator: dropping partial bucket for %s "
                        "(hour_start=%d) — first tick was not in minute 0",
                        symbol,
                        bucket.hour_start_ms,
                    )
                    if self._drop is not None:
                        self._drop(symbol, bucket.hour_start_ms)
            self._buckets[symbol] = _Bucket(
                hour_start_ms=hour_start,
                open=mid,
                high=mid,
                low=mid,
                close=mid,
                volume=ltv if ltv > 0 else 0.0,
                was_subscribed_at_open=(utm_ms - hour_start) < _MINUTE_MS,
            )
            return

        # Tick is within the current bucket's hour — update high/low/close.
        # The "open" field stays pinned to whatever the first in-hour tick set.
        if utm_ms < bucket.hour_start_ms:
            # Defensive: out-of-order tick from an earlier hour.  Drop;
            # the LS stream is well-ordered in practice but the post-
            # reconnect snapshot can occasionally backfill.
            return
        if mid > bucket.high:
            bucket.high = mid
        if mid < bucket.low:
            bucket.low = mid
        bucket.close = mid
        if ltv > 0:
            bucket.volume += ltv

    def flush(self) -> None:
        """Force-emit any bucket eligible for emission and clear state.

        Intended for graceful shutdown — if a hour is in-progress at
        shutdown time we still don't emit it (partial bars are never
        confirmed), but pre-existing emit-eligible buckets that haven't
        rolled over yet are not held back.

        In practice this is a no-op for the current design (we only
        emit on rollover), but kept as the public hook in case future
        D-phase work wants partial-flush semantics.
        """
        self._buckets.clear()

    # ------------------------------------------------------------------
    # Introspection (for tests / dashboards)
    # ------------------------------------------------------------------

    def current_bucket(self, symbol: str) -> _Bucket | None:
        """Return the in-progress bucket for *symbol*, or None if no
        ticks have been ingested for it yet.  Returns the live object —
        callers must NOT mutate it."""
        return self._buckets.get(symbol)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _finalise(symbol: str, bucket: _Bucket) -> Candle:
        return Candle(
            symbol=symbol,
            timestamp=bucket.hour_start_ms,
            open=bucket.open,
            high=bucket.high,
            low=bucket.low,
            close=bucket.close,
            volume=bucket.volume,
            is_confirmed=True,
        )
