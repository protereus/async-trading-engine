"""GdeltAgent — free global-news tone sentiment via GDELT 2.0.

The Stocktwits public stream is fully Cloudflare-blocked (HTTP 403 on every
default UA, including browser-shaped headers) as of 2026-05.  GDELT 2.0 was
the deep-research report's second-strongest free pick and works cleanly:

  - Endpoint: ``https://api.gdeltproject.org/api/v2/doc/doc``
  - Mode:     ``TimelineTone``  (15-min-bucketed average tone of matching articles)
  - Auth:     none
  - Range:    24h via ``timespan=24h``
  - Returns:  ``{"timeline": [{"series": "Average Tone", "data": [{"date", "value"}, ...]}]}``
  - Tone scale: roughly [-10, +10]; real-world values cluster in [-5, +5]

Sentiment computation is purely structural — no LLM call required:
  - Per asset, query GDELT for the chosen keyword over the last 24h
  - Average tone across the returned 15-min buckets
  - Scale tone → sentiment by dividing by 5 and clipping to [-1, +1]
  - Confidence: ramps with the number of buckets that contained articles
  - Drop assets returning fewer than ``_MIN_BUCKETS`` data points (noise floor)

Rate limit: GDELT 429s aggressively on bursts.  Requests are serialised within
the agent with ``_REQUEST_SLEEP_S`` between calls.  At a 30-min cadence and
≤ 10 covered assets the total scan stays well under one minute.

Coverage selection: we cover the assets where GDELT's English-named topical
queries have meaningful signal.  Concatenated forex queries like ``EURJPY``
empirically return no data; the high-volume USD majors do return real timelines.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote_plus

import aiohttp

from bot.sentiment.agents.base import BaseAgent
from bot.sentiment.models import RawSignal

logger = logging.getLogger(__name__)

_GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
# GDELT 429s aggressively; cap each call short so timeouts don't sit on the
# asyncio loop and starve the Kronos inference thread.  Pre-2026-05-21 default
# was 30 s; we observed 140+ failures/day at that setting.
_REQUEST_TIMEOUT_S = 8.0
_REQUEST_SLEEP_S = 3.0  # GDELT 429s on bursts; sequential calls + sleep is the safer path
_MIN_BUCKETS = 10  # Drop assets where GDELT returned too little material to be useful
_TONE_SCALE = 5.0  # tone in [-5, +5] → sentiment in [-1, +1]; clip past the rails
# GDELT TimelineTone updates ~hourly upstream, so per-asset results stay fresh
# for at least an hour.  Caching avoids the per-30-min-scan re-fetch storm.
_CACHE_TTL_S = 3600.0

# Canonical asset → GDELT query string.  English topical names beat concatenated
# tickers (EURUSD works; EURJPY returns nothing); for currency pairs we keep
# the concatenated form because GDELT's news classifier indexes them as themes.
_ASSET_QUERY_MAP: dict[str, str] = {
    # --- Metals (spot — broad query rather than ticker) ---
    "XAU/USD": "gold price",
    "XAG/USD": "silver price",
    # --- Forex majors that empirically return real data ---
    "EUR/USD": "EURUSD",
    "GBP/USD": "GBPUSD",
    "USD/JPY": "USDJPY",
}
# NB: the 14 single-name US shares are intentionally uncovered — company-name
# queries were never validated against GDELT's tone classifier, and covering
# them would blow the ≤ 10-asset scan budget documented above.


class GdeltAgent(BaseAgent):
    """Polls GDELT 2.0 TimelineTone per asset and emits average-tone sentiment.

    Per-asset results are cached for ``_CACHE_TTL_S`` (1 h, matching the
    upstream update cadence) so the 30-min sentiment scan doesn't re-fetch
    fresh data and spend 30 s timing out on a 429.
    """

    name = "gdelt"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Cache keyed by asset; value = (RawSignal, fetched_at_monotonic).
        # Only successful fetches populate it.
        self._cache: dict[str, tuple[RawSignal, float]] = {}

    async def analyze(self, assets: list[str]) -> list[RawSignal]:
        targets = {asset: query for asset, query in _ASSET_QUERY_MAP.items() if asset in assets}
        if not targets:
            logger.debug("GdeltAgent: no covered assets in target set — skipping")
            return []

        now_mono = time.monotonic()
        signals: list[RawSignal] = []
        to_fetch: list[tuple[str, str]] = []
        for asset, query in targets.items():
            cached = self._cache.get(asset)
            if cached is not None and now_mono - cached[1] < _CACHE_TTL_S:
                signals.append(cached[0])
            else:
                to_fetch.append((asset, query))

        outcomes: Counter[str] = Counter()
        for idx, (asset, query) in enumerate(to_fetch):
            if idx > 0:
                await asyncio.sleep(_REQUEST_SLEEP_S)
            scored, outcome = await self._fetch_tone(query)
            outcomes[outcome] += 1
            if scored is None:
                continue
            sig = RawSignal(
                asset=asset,
                sentiment=scored["sentiment"],
                confidence=scored["confidence"],
                reasoning=scored["reasoning"].replace("{asset}", asset),
                source="gdelt",
                fetched_at=datetime.now(UTC),
            )
            signals.append(sig)
            self._cache[asset] = (sig, time.monotonic())

        # One aggregate line per scan replaces the per-query traceback flood —
        # GDELT free-tier 429s + 8s timeouts were generating ~18 multi-line
        # tracebacks per day at the per-query exc_info debug level.
        breakdown = ", ".join(f"{k}={n}" for k, n in outcomes.most_common()) if outcomes else "none"
        logger.info(
            "GdeltAgent: %d/%d scored | cache=%d fetched=%d | %s",
            len(signals),
            len(targets),
            len(targets) - len(to_fetch),
            len(to_fetch),
            breakdown,
        )
        return signals

    async def _fetch_tone(self, query: str) -> tuple[dict[str, Any] | None, str]:
        """Fetch + score one query.

        Returns ``(data_or_None, outcome_label)`` where ``outcome_label`` is one of
        ``ok``, ``empty``, ``timeout``, ``connection``, ``http``, ``parse``,
        ``unknown``.  The label is consumed by :meth:`analyze` to produce a single
        aggregate log line per scan instead of one record per failed query.
        """
        params = {
            "query": query,
            "mode": "TimelineTone",
            "format": "json",
            "timespan": "24h",
        }
        # GDELT's URL-encoding of `&` in queries (e.g. "S&P 500") needs explicit
        # quote_plus rather than aiohttp's default form-encoding.
        encoded = "&".join(f"{k}={quote_plus(str(v))}" for k, v in params.items())
        url = f"{_GDELT_URL}?{encoded}"
        try:
            timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_S)
            async with self._session.get(url, timeout=timeout) as resp:
                if resp.status != 200:
                    return None, "http"
                # GDELT sometimes returns HTML/text with 200 on errors; force JSON parse.
                text = await resp.text()
        except TimeoutError:
            return None, "timeout"
        except aiohttp.ClientError:
            return None, "connection"
        except Exception:
            # Unexpected — keep the stack so genuine bugs stay visible.
            logger.debug("GdeltAgent: unexpected fetch error for %r", query, exc_info=True)
            return None, "unknown"

        import json

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None, "parse"

        scored = self._score_timeline(query, data)
        if scored is None:
            return None, "empty"
        return scored, "ok"

    @staticmethod
    def _score_timeline(query: str, data: Any) -> dict[str, Any] | None:
        """Reduce a TimelineTone response to a single sentiment dict.

        Returns None when fewer than ``_MIN_BUCKETS`` data points exist.
        """
        if not isinstance(data, dict):
            return None
        timeline = data.get("timeline")
        if not isinstance(timeline, list) or not timeline:
            return None
        first = timeline[0]
        if not isinstance(first, dict):
            return None
        points = first.get("data", [])
        if not isinstance(points, list):
            return None

        values: list[float] = []
        for p in points:
            if not isinstance(p, dict):
                continue
            try:
                values.append(float(p["value"]))
            except (KeyError, TypeError, ValueError):
                continue

        if len(values) < _MIN_BUCKETS:
            return None

        avg_tone = sum(values) / len(values)
        sentiment = max(-1.0, min(1.0, avg_tone / _TONE_SCALE))
        # Confidence ramps with bucket count: ~0.5 at 10 buckets, ~0.85 at 90+.
        confidence = min(0.85, 0.4 + min(len(values), 100) / 200.0)
        reasoning = (
            f"{{asset}} GDELT 24h average tone {avg_tone:+.2f} "
            f"across {len(values)} buckets (query={query!r})"
        )
        return {
            "sentiment": round(sentiment, 4),
            "confidence": round(confidence, 4),
            "reasoning": reasoning,
        }
