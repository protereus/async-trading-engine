"""FREDClient — daily pulls of free St. Louis Fed macro series.

The deep-research report identified six FRED series as the highest-signal
free macro overlay for 1h–5d horizons on this universe:

  - DTWEXBGS   — Nominal Broad U.S. Dollar Index (daily)
  - VIXCLS     — CBOE Volatility Index (daily)
  - DGS2       — 2-year Treasury constant-maturity yield (daily)
  - DGS10      — 10-year Treasury constant-maturity yield (daily)
  - T5YIE      — 5-year breakeven inflation rate (daily) — shortest breakeven
                 actually published on FRED (T2YIE referenced in the report
                 doesn't exist; verified live 2026-05-19)
  - DCOILWTICO — Cushing OK WTI Spot Price (daily)

Two additional monthly series are pulled for context, surfaced as the
latest-observed value (not a daily tick):

  - CPIAUCSL — CPI All Urban Consumers (monthly)
  - PAYEMS   — Total Nonfarm Payrolls (monthly)

Storage: every observation is upserted into ``macro_features`` (
schema in ``bot.data.candle_db``).  Derived features (1-day Δ, 5-day Δ,
52-week z-score) are computed on read in ``compute_macro_snapshot()`` —
nothing derived is persisted, so a parameter change does not require a
schema migration.

Cadence: FRED publishes the daily series at varying times (8:00–17:00 ET).
A once-per-day pull at startup + a 24h-period scheduler covers the cycle
with no urgency — these features move at week-to-month horizons relative
to the bot's 1h-bar Kronos signal.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import aiohttp

if TYPE_CHECKING:
    from bot.data.candle_db import CandleDB

logger = logging.getLogger(__name__)

_FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
_REQUEST_TIMEOUT_S = 30.0
_REQUEST_SLEEP_S = 0.6  # 120 req/min soft cap → ~0.5 s safe; pad with 0.6
_DEFAULT_LOOKBACK_DAYS = 400  # ≥ 52 weeks for z-score computation
_POLL_INTERVAL_S = 24 * 3600

# Canonical macro series tracked by Cadence is informational only —
# the client doesn't behave differently for monthly vs daily; it always asks
# for ``observation_start = today - 400 d`` and FRED returns whatever exists.
_TRACKED_SERIES: tuple[str, ...] = (
    "DTWEXBGS",
    "VIXCLS",
    "DGS2",
    "DGS10",
    "T5YIE",
    "DCOILWTICO",
    "CPIAUCSL",
    "PAYEMS",
)


def _date_str_to_ms(date_str: str) -> int:
    """Convert FRED's ``YYYY-MM-DD`` date strings to UTC-midnight ms epochs."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


class FREDClient:
    """Async client for FRED series-observations endpoint, persisting via CandleDB."""

    def __init__(self, api_key: str, candle_db: CandleDB) -> None:
        if not api_key:
            raise ValueError("FREDClient requires a non-empty API key")
        self._api_key = api_key
        self._candle_db = candle_db
        self._session: aiohttp.ClientSession | None = None

    # ------------------------------------------------------------------
    # HTTP fetch + parse
    # ------------------------------------------------------------------

    async def _fetch_series(
        self, series_id: str, lookback_days: int = _DEFAULT_LOOKBACK_DAYS
    ) -> list[tuple[int, float]] | None:
        """Fetch one series.  Returns ``[(observation_date_ms, value), ...]`` or None."""
        assert self._session is not None
        start = (datetime.now(UTC) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "observation_start": start,
        }
        try:
            timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_S)
            async with self._session.get(_FRED_URL, params=params, timeout=timeout) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.warning("FRED: HTTP %d for %s: %.200s", resp.status, series_id, text)
                    return None
                data: Any = await resp.json()
        except Exception:
            logger.exception("FRED: request failed for %s", series_id)
            return None

        return self._parse_observations(series_id, data)

    @staticmethod
    def _parse_observations(series_id: str, data: Any) -> list[tuple[int, float]] | None:
        """Parse the FRED JSON response into (date_ms, value) tuples.

        FRED uses ``"."`` for missing observations — those are silently dropped.
        """
        if not isinstance(data, dict):
            return None
        observations = data.get("observations")
        if not isinstance(observations, list):
            return None

        out: list[tuple[int, float]] = []
        for obs in observations:
            if not isinstance(obs, dict):
                continue
            date_str = obs.get("date")
            value_str = obs.get("value")
            if not isinstance(date_str, str) or not isinstance(value_str, str):
                continue
            if value_str == ".":
                continue
            try:
                value = float(value_str)
                obs_ms = _date_str_to_ms(date_str)
            except ValueError:
                logger.debug("FRED: skipping bad row for %s: %s", series_id, obs)
                continue
            out.append((obs_ms, value))
        return out

    # ------------------------------------------------------------------
    # Pulls + persistence
    # ------------------------------------------------------------------

    async def pull_all(self) -> dict[str, int]:
        """Fetch + persist every tracked series.  Returns ``{series_id: rows_written}``."""
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": "TradingBot/1.0 (research)"}
            )
        counts: dict[str, int] = {}
        for idx, series_id in enumerate(_TRACKED_SERIES):
            if idx > 0:
                await asyncio.sleep(_REQUEST_SLEEP_S)
            observations = await self._fetch_series(series_id)
            if not observations:
                counts[series_id] = 0
                continue
            n = self._candle_db.insert_macro_observations(series_id, observations)
            counts[series_id] = n
            logger.debug("FRED: %s — %d observations persisted", series_id, n)
        logger.info("FRED: pull_all complete — %s", counts)
        return counts

    # ------------------------------------------------------------------
    # Background scheduler
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Run forever: pull immediately on start, then once every 24 hours.

        Sleeps are split into 60 s chunks so a graceful cancel can finish
        within one minute instead of waiting up to a full day.
        """
        logger.info("FREDClient: starting daily macro overlay")
        try:
            while True:
                try:
                    await self.pull_all()
                except Exception:
                    logger.exception("FREDClient: pull_all failed (will retry tomorrow)")
                # Chunked sleep so cancellation is responsive
                remaining = _POLL_INTERVAL_S
                while remaining > 0:
                    await asyncio.sleep(min(60, remaining))
                    remaining -= 60
        except asyncio.CancelledError:
            logger.info("FREDClient: cancelled")
            raise
        finally:
            await self.close()

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
            logger.debug("FREDClient: aiohttp session closed")


# ---------------------------------------------------------------------------
# Read-side helpers — derived features computed at lookup time
# ---------------------------------------------------------------------------


def compute_macro_snapshot(
    candle_db: CandleDB, series_ids: tuple[str, ...] = _TRACKED_SERIES
) -> dict[str, dict[str, float]]:
    """Return ``{series_id: {value, delta_1d, delta_5d, zscore_52w}}``.

    Derived features are computed against the most-recent persisted
    observations.  Missing fields (insufficient history) are simply absent
    from the inner dict.
    """
    out: dict[str, dict[str, float]] = {}
    for series_id in series_ids:
        # Fetch up to ~52w of history (260 trading days + buffer)
        observations = candle_db.get_macro_series(series_id, limit=300)
        if not observations:
            continue
        snapshot: dict[str, float] = {}
        latest_ts, latest_val = observations[-1]
        snapshot["value"] = latest_val
        snapshot["observation_date_ms"] = float(latest_ts)
        if len(observations) >= 2:
            snapshot["delta_1d"] = latest_val - observations[-2][1]
        if len(observations) >= 6:
            snapshot["delta_5d"] = latest_val - observations[-6][1]
        if len(observations) >= 30:
            recent_values = [v for _, v in observations]
            mean = sum(recent_values) / len(recent_values)
            variance = sum((v - mean) ** 2 for v in recent_values) / len(recent_values)
            std = variance**0.5
            if std > 0:
                snapshot["zscore"] = (latest_val - mean) / std
        out[series_id] = snapshot
    return out
