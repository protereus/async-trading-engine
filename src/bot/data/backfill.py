"""Shared startup-backfill staleness predicate for the candle feeds.

Every feed (EODHD primary, TwelveData warm-standby, IG-native metals) warms its store from
the local DB on restart and then backfills via REST.  Historically each decided
whether to skip that backfill on buffer *depth* alone — a symbol at/above its
threshold was assumed current.  That let a full-but-stale buffer (feed silently
dropped over a weekend, restart after reopen) skip the repair fetch and wait for
the next live bucket, leaving a permanent hole (the 2026-07-05 FX gap).

:func:`needs_backfill` keeps the depth check and adds a freshness check: a
symbol whose newest buffered bar is older than the last 1h bar that should have
closed for its (open) market is stale even when the buffer is deep.  Re-running
the feed's existing fetch→ingest path is safe because ``DataStore.add_candle``
drops out-of-order bars and ``CandleDB.insert_candle`` is ``INSERT OR IGNORE`` —
so a repair fetch appends only the missing newer bars.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from bot.trading_hours import last_expected_closed_bar_ms

if TYPE_CHECKING:
    from bot.data.store import DataStore


def needs_backfill(
    store: DataStore, bot_key: str, threshold: int, now: datetime | None = None
) -> bool:
    """True if *bot_key* should be backfilled on startup.

    Triggers when either:

    * **depth** — fewer than *threshold* buffered candles (the original skip
      condition), or
    * **freshness** — the newest buffered bar is older than
      :func:`last_expected_closed_bar_ms` for the symbol's market (a silently
      stalled feed whose buffer is otherwise deep enough).

    ``now`` is injectable for tests; it defaults to the current UTC time inside
    :func:`last_expected_closed_bar_ms`.
    """
    if store.get_candle_count(bot_key) < threshold:
        return True
    last_closed = last_expected_closed_bar_ms(bot_key, now)
    if last_closed is None:
        # Market never open within the look-back horizon → no bar is expected,
        # so depth is the only signal.
        return False
    latest = store.get_latest_candle(bot_key)
    return latest is None or latest.timestamp < last_closed
