"""IG REST historical backfill helper (D2 of ).

When a symbol is moved off its Twelve Data / yfinance ETF proxy onto IG's
own price feed (D3 cutover), it needs ~``kronos_context_bars`` of hourly
history before TopK can predict on it.  Lightstreamer only delivers
realtime ticks, so the cold-start history comes from IG's REST
``/prices`` endpoint.

The 2026-05-28 REST probe (``scripts/probe_ig_history.py``) confirmed:
  - HOUR resolution returns cleanly UTC-aligned bars for the WTI / Nat Gas
    DFB EPICs.
  - Cost is 1 allowance point per returned bar — 400 points per symbol for
    a 400-bar context (USO+UNG+SLV cold start = 1 200 of the
    10 000-per-week budget).
  - MINUTE resolution also works but a 400-hour context would cost 24 000
    points per symbol (72 000 for all three) — over budget — so we backfill
    at HOUR.

This module is a thin relabelling wrapper over ``IGClient.fetch_ohlcv``:
that method returns ``Candle`` objects keyed by the IG EPIC, but the rest
of the bot keys candles by the canonical candle symbol (``"XAU/USD"``, not
``"CS.D.USCGC.TODAY.IP"``).  We fix the key so the backfilled bars land in
the same store slot the strategy reads.

It deliberately does NOT write to the store or DB — the caller (the
future IG-native feed in D1/D3) owns persistence and dedup.  Keeping this
a pure fetch+relabel makes it unit-testable without a live store.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from bot.core.models import Candle

if TYPE_CHECKING:
    from bot.execution.ig_client import IGClient

logger = logging.getLogger(__name__)

# IG REST history is hourly for backfill — see module docstring for why not
# MINUTE.  Kept as a constant so the D3 wiring and any test reference the
# same value.
IG_BACKFILL_TIMEFRAME = "1h"


async def fetch_ig_hourly_backfill(
    ig_client: IGClient,
    candle_symbol: str,
    epic: str,
    limit: int,
) -> list[Candle]:
    """Fetch ``limit`` hourly bars for *epic* and relabel them to *candle_symbol*.

    Returns the bars sorted oldest-first (``fetch_ohlcv`` already sorts).
    On any fetch error logs and returns ``[]`` — the caller treats an empty
    backfill as "warm up from live ticks instead", matching how the other
    candle feeds degrade when their source is briefly unavailable.

    The returned candles are IG-native (already in IG-level units), so the
    consumer must set ``ig_quote_scale(candle_symbol) == 1.0`` as part of
    the D3 cutover — otherwise the scale conversion would double-apply.
    """
    try:
        raw = await ig_client.fetch_ohlcv(epic, IG_BACKFILL_TIMEFRAME, limit=limit)
    except Exception:
        logger.exception(
            "IG backfill failed for %s (epic=%s) — caller should warm from live ticks",
            candle_symbol,
            epic,
        )
        return []

    # fetch_ohlcv keys Candle.symbol by the EPIC; relabel to the canonical
    # candle symbol so the bars land in the strategy's store slot.
    relabelled = [replace(c, symbol=candle_symbol) for c in raw]
    logger.info(
        "IG backfill: %s (epic=%s) → %d hourly bars",
        candle_symbol,
        epic,
        len(relabelled),
    )
    return relabelled
