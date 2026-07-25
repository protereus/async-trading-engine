"""Twelve Data REST feed — hourly OHLC, the FX warm-standby candle source.

Role
----
Secondary/failover candle feed for ``candle_exchange='twelvedata'``.  EODHD is
the live primary; this feed exists so a
one-command failover can keep the FX book trading through an EODHD provider-side
outage.  Trimmed in
2026-07 to the 12 EODHD FX pairs + XAU/USD — the old full pre-EODHD universe
(index/commodity ETFs, extra crosses, and the yfinance-fed FTSE) was removed.

Architecture
------------
1. On startup: fetch ``_BACKFILL_OUTPUTSIZE`` hourly bars per symbol in throttled
   batches (``_BATCH_SIZE`` symbols/call, within the 8-credit/min free tier).
2. Poll every hour at :05 past (outputsize=3) to catch the newly closed bar.
3. Emit EVENT_NEW_CANDLE for each symbol when a fresher bar is ingested.

Rate-limit throttling
---------------------
Free tier: 8 API credits/minute.  One batch request for N comma-separated symbols
costs N credits.  BATCH_SIZE=7 leaves 1-credit headroom.  BATCH_SLEEP_S=61s ensures
at least 60s between consecutive batch requests.

Symbol mapping
--------------
``SYMBOL_EPIC_MAP`` maps Twelve Data symbols → IG spread-bet EPICs used at order
time; ``main.py`` reads it (under ``candle_exchange='twelvedata'``) to build the
candle-symbol list and EPIC map.  XAU/USD candles come from the IG-native feed,
not Twelve Data — see ``_TD_EXCLUDE_FROM_FETCH``.

Caveat: IG EPICs are approximate.  Verify with ``scripts/search_epic.py`` before
live trading.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import aiohttp

from bot.core.event_bus import EVENT_NEW_CANDLE
from bot.core.models import Candle
from bot.data.backfill import needs_backfill
from bot.data.ig_candle_aggregator import IG_NATIVE_CANDLE_SYMBOLS
from bot.data.store import warm_store_from_db

if TYPE_CHECKING:
    from bot.config import BotConfig
    from bot.core.event_bus import EventBus
    from bot.data.candle_db import CandleDB
    from bot.data.store import DataStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Warm-standby universe: 12 FX pairs + XAU/USD (IG-fed)
# ---------------------------------------------------------------------------

SYMBOL_EPIC_MAP: dict[str, str] = {
    # --- Forex majors (7) ---
    "EUR/USD": "CS.D.EURUSD.TODAY.IP",
    "GBP/USD": "CS.D.GBPUSD.TODAY.IP",
    "USD/JPY": "CS.D.USDJPY.TODAY.IP",
    "USD/CHF": "CS.D.USDCHF.TODAY.IP",
    "USD/CAD": "CS.D.USDCAD.TODAY.IP",
    "AUD/USD": "CS.D.AUDUSD.TODAY.IP",
    "NZD/USD": "CS.D.NZDUSD.TODAY.IP",
    # --- Forex crosses (5) ---
    "EUR/GBP": "CS.D.EURGBP.TODAY.IP",
    "EUR/JPY": "CS.D.EURJPY.TODAY.IP",
    "EUR/AUD": "CS.D.EURAUD.TODAY.IP",
    "GBP/JPY": "CS.D.GBPJPY.TODAY.IP",
    "AUD/JPY": "CS.D.AUDJPY.TODAY.IP",
    # --- Metals: gold spot.  Candles come from the IG-native Lightstreamer feed,
    # not Twelve Data (excluded from the TD fetch below); the entry is kept only
    # for IG-EPIC translation + universe inclusion.
    "XAU/USD": "CS.D.USCGC.TODAY.IP",  # Spot Gold (CURRENCIES DFB) — verified on SPREADBET demo
}

# XAU/USD stays in SYMBOL_EPIC_MAP (for IG-EPIC translation + universe
# inclusion) but its candles come from the IG-native Lightstreamer feed, not
# Twelve Data.  Without this exclusion, under ``candle_exchange='twelvedata'``
# the TD gold quote (~$4043) and the IG-native gold candle (~$4150) would both
# write the XAU/USD series, interleaving two price scales in one buffer.  Track
# ``IG_NATIVE_CANDLE_SYMBOLS`` so the exclusion follows that source of truth
# (XAG/USD isn't in the TD universe, so the intersection is just XAU/USD).
_TD_EXCLUDE_FROM_FETCH: frozenset[str] = frozenset(IG_NATIVE_CANDLE_SYMBOLS)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TWELVE_DATA_URL = "https://api.twelvedata.com/time_series"
_BATCH_SIZE = 7  # Credits per request; free tier is 8/min — leave 1 headroom
_BATCH_SLEEP_S = 61.0  # Seconds between batches (> 60 s enforces the rate limit)
_BACKFILL_OUTPUTSIZE = 600  # ~600 h = 25 days calendar; covers ≥ 400 tradeable bars
_POLL_OUTPUTSIZE = 3  # Grab the 3 most-recent bars each hourly poll
_REQUEST_TIMEOUT_S = 30.0


class TwelveDataFeed:
    """Fetches hourly OHLC from Twelve Data and stores it in DataStore + CandleDB.

    Symbols are drawn from ``SYMBOL_EPIC_MAP``.  The DB/strategy key is the
    Twelve Data symbol itself (e.g. ``"EUR/USD"``), matching ``_candle_symbols``
    in ``TradingBot``.  IG EPIC translation happens at order time via
    ``_candle_epic_map``.
    """

    def __init__(
        self,
        store: DataStore,
        event_bus: EventBus,
        config: BotConfig,
        candle_db: CandleDB | None = None,
    ) -> None:
        self._store = store
        self._event_bus = event_bus
        self._config = config
        self._candle_db = candle_db
        self._symbols: list[str] = [
            sym for sym in SYMBOL_EPIC_MAP if sym not in _TD_EXCLUDE_FROM_FETCH
        ]
        self._session: aiohttp.ClientSession | None = None
        # Tracks the most-recently emitted bar timestamp per symbol (ms)
        self._last_confirmed: dict[str, int] = {}

    # ------------------------------------------------------------------
    # REST fetch
    # ------------------------------------------------------------------

    async def _fetch_batch(
        self, symbols: list[str], outputsize: int
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch one batch from Twelve Data. Returns {symbol: [values]} newest-first."""
        assert self._session is not None
        params: dict[str, Any] = {
            "symbol": ",".join(symbols),
            "interval": "1h",
            "outputsize": outputsize,
            "timezone": "UTC",
            "apikey": self._config.twelve_data_api,
        }
        try:
            timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_S)
            async with self._session.get(_TWELVE_DATA_URL, params=params, timeout=timeout) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error("Twelve Data HTTP %d for %s: %.200s", resp.status, symbols, body)
                    return {}
                raw: Any = await resp.json()
        except Exception:
            logger.exception("Twelve Data request failed for %s", symbols)
            return {}

        return self._parse_response(raw, symbols)

    def _parse_response(
        self,
        raw: Any,
        requested: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Handle single-symbol (flat) vs multi-symbol (nested) Twelve Data response."""
        if len(requested) == 1:
            # Flat: {"meta": {...}, "values": [...], "status": "ok"}
            sym = requested[0]
            if isinstance(raw, dict) and "values" in raw:
                return {sym: raw["values"]}
            logger.warning("Twelve Data: unexpected response for %s: %s", sym, str(raw)[:200])
            return {}

        # Multi-symbol: {symbol: {"meta": {...}, "values": [...], "status": "ok"}}
        # A top-level error (e.g. 429 rate-limit) returns a flat error object instead.
        if isinstance(raw, dict) and raw.get("status") == "error":
            logger.error(
                "Twelve Data: API error for batch %s — code=%s msg=%s",
                requested,
                raw.get("code"),
                raw.get("message", "")[:200],
            )
            return {}

        result: dict[str, list[dict[str, Any]]] = {}
        for sym in requested:
            entry = raw.get(sym) if isinstance(raw, dict) else None
            if isinstance(entry, dict) and "values" in entry:
                result[sym] = entry["values"]
            else:
                logger.warning("Twelve Data: missing/bad entry for %s in multi response", sym)
        return result

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def _ingest(self, symbol: str, values: list[dict[str, Any]]) -> int:
        """Parse and store values (newest-first list from Twelve Data).

        Returns the number of candles successfully stored.
        """
        count = 0
        for v in reversed(values):  # oldest → newest
            try:
                dt_str: str = v["datetime"]
                if len(dt_str) == 10:
                    # Date-only format "YYYY-MM-DD" (some instruments)
                    dt = datetime.strptime(dt_str, "%Y-%m-%d").replace(tzinfo=UTC)
                else:
                    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
                ts_ms = int(dt.timestamp() * 1000)

                raw_vol = v.get("volume")
                volume = float(str(raw_vol)) if raw_vol not in (None, "", "0", 0) else 0.0

                candle = Candle(
                    symbol=symbol,
                    timestamp=ts_ms,
                    open=float(v["open"]),
                    high=float(v["high"]),
                    low=float(v["low"]),
                    close=float(v["close"]),
                    volume=volume,
                    is_confirmed=True,
                )
                self._store.add_candle(candle)
                if self._candle_db is not None:
                    self._candle_db.insert_candle(candle)
                count += 1
            except Exception:
                logger.warning("Twelve Data: failed to parse bar for %s: %s", symbol, str(v)[:100])
        return count

    # ------------------------------------------------------------------
    # Throttled full-universe fetch
    # ------------------------------------------------------------------

    async def _throttled_fetch_all(self, outputsize: int) -> None:
        """Fetch all symbols in BATCH_SIZE chunks, sleeping between batches."""
        symbols = self._symbols
        batches = [symbols[i : i + _BATCH_SIZE] for i in range(0, len(symbols), _BATCH_SIZE)]
        for idx, batch in enumerate(batches):
            if idx > 0:
                logger.debug("Twelve Data: rate-limit sleep %.1fs", _BATCH_SLEEP_S)
                await asyncio.sleep(_BATCH_SLEEP_S)
            data = await self._fetch_batch(batch, outputsize)
            for symbol, values in data.items():
                n = self._ingest(symbol, values)
                logger.debug("Twelve Data: ingested %d bars for %s", n, symbol)

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    async def _emit_latest(self) -> None:
        """Emit EVENT_NEW_CANDLE for each symbol whose latest bar is fresher than before."""
        for symbol in self._symbols:
            candle = self._store.get_latest_candle(symbol)
            if candle is None:
                continue
            if candle.timestamp > self._last_confirmed.get(symbol, 0):
                self._last_confirmed[symbol] = candle.timestamp
                await self._event_bus.emit(EVENT_NEW_CANDLE, candle)
                logger.debug(
                    "Twelve Data: new candle %s @ %d close=%.6f",
                    symbol,
                    candle.timestamp,
                    candle.close,
                )

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def _warm_store_from_db(self) -> None:
        """Pre-populate the in-memory store from the local CandleDB.

        Called before the live backfill check so that a bot restart doesn't
        re-fetch data that was already persisted to disk.
        """
        if self._candle_db is None:
            return
        warm_store_from_db(
            self._store,
            self._candle_db,
            self._symbols,
            limit=self._config.candle_buffer_size,
            label="Twelve Data",
        )

    async def run(self) -> None:
        """Backfill all symbols, then poll hourly at :05 past the hour."""
        self._session = aiohttp.ClientSession()
        try:
            # --- warm store from local DB (avoids live API calls on restart) ---
            self._warm_store_from_db()

            # --- backfill any symbols below the buffer threshold or stale ---
            # Freshness (needs_backfill) repairs a deep-but-stale buffer after a
            # silent feed drop; without it a full buffer skips the repair fetch.
            threshold = self._config.candle_buffer_size
            need_backfill = [s for s in self._symbols if needs_backfill(self._store, s, threshold)]
            if need_backfill:
                logger.info(
                    "Twelve Data: backfilling %d symbols (outputsize=%d)…",
                    len(need_backfill),
                    _BACKFILL_OUTPUTSIZE,
                )
                # Temporarily replace symbol list for throttled fetch.  Restore
                # in finally so an exception mid-fetch can't permanently leave
                # self._symbols stuck on the backfill subset.
                _orig = self._symbols
                self._symbols = need_backfill
                try:
                    await self._throttled_fetch_all(outputsize=_BACKFILL_OUTPUTSIZE)
                finally:
                    self._symbols = _orig
            else:
                logger.info("Twelve Data: all symbols already buffered — skipping backfill")

            await self._emit_latest()
            candle_counts = {sym: self._store.get_candle_count(sym) for sym in self._symbols}
            logger.info("Twelve Data: backfill complete — candle counts: %s", candle_counts)

            # --- hourly poll ---
            await self._run_hourly_poll()

        except asyncio.CancelledError:
            logger.info("TwelveDataFeed: cancelled")
            raise
        except Exception:
            logger.exception("TwelveDataFeed: unhandled error in run()")
            raise
        finally:
            await self.close()

    async def _run_hourly_poll(self) -> None:
        """Loop forever, sleeping until :05 past each hour then fetching the latest bars."""
        while True:
            now = datetime.now(UTC)
            next_fire = now.replace(minute=5, second=0, microsecond=0)
            if next_fire <= now:
                next_fire += timedelta(hours=1)
            sleep_s = (next_fire - now).total_seconds()
            logger.info(
                "Twelve Data: next poll at %s (%.0fs)",
                next_fire.strftime("%Y-%m-%dT%H:%M:%SZ"),
                sleep_s,
            )
            await asyncio.sleep(sleep_s)

            logger.info("Twelve Data: hourly poll (outputsize=%d)…", _POLL_OUTPUTSIZE)
            await self._throttled_fetch_all(outputsize=_POLL_OUTPUTSIZE)
            await self._emit_latest()

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
            logger.debug("Twelve Data: aiohttp session closed")
