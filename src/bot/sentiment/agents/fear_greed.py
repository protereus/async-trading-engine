"""FearGreedAgent — maps the market fear/greed index directly to RawSignals (no Groq needed).

Data source:
- FearGreedChart Market Fear & Greed Index  → equities, metals, commodities
  https://feargreedchart.com/api/?action=all  (free, no auth, 5-min cache)

Score mapping: linear  (score − 50) / 50  → sentiment in [−1.0, +1.0]
  0   = Extreme Fear  → −1.0
  50  = Neutral       →  0.0
  100 = Extreme Greed → +1.0

Note: Alternative.me Crypto Fear & Greed (BTC/USD, ETH/USD) removed — IG only
allows crypto spread betting on professional accounts.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import aiohttp  # used for ClientTimeout

from bot.data.eodhd_symbols import EODHD_UNIVERSE
from bot.sentiment.agents.base import BaseAgent
from bot.sentiment.models import RawSignal

logger = logging.getLogger(__name__)

_MARKET_FG_URL = "https://feargreedchart.com/api/?action=all"

# Assets that track broader market fear/greed — the equities + metals slice of
# the live universe (derived, so universe changes can't drift this set).  FX
# pairs are relative-value and don't map onto a single risk-on/risk-off score.
_MARKET_ASSETS = frozenset(
    s.bot_key for s in EODHD_UNIVERSE.values() if s.asset_class in ("equity", "metal")
)

_CONFIDENCE = 0.70  # F&G is a solid indicator but a single dimension


class FearGreedAgent(BaseAgent):
    """Returns sentiment signals derived from the market fear/greed index."""

    name = "fear_greed"

    async def analyze(self, assets: list[str]) -> list[RawSignal]:
        market_score = await self._fetch_market_fg()

        signals: list[RawSignal] = []
        now = datetime.now(UTC)

        if market_score is not None:
            sentiment = self._score_to_sentiment(market_score)
            classification = self._classify(market_score)
            for asset in _MARKET_ASSETS:
                if asset in assets:
                    signals.append(
                        RawSignal(
                            asset=asset,
                            sentiment=sentiment,
                            confidence=_CONFIDENCE,
                            reasoning=f"Market Fear & Greed: {market_score}/100 ({classification})",
                            source="fear_greed",
                            fetched_at=now,
                        )
                    )

        logger.info(
            "FearGreedAgent: market=%s → %d signals",
            market_score if market_score is not None else "N/A",
            len(signals),
        )
        return signals

    # ------------------------------------------------------------------
    # Data fetcher
    # ------------------------------------------------------------------

    async def _fetch_market_fg(self) -> int | None:
        """Fetch FearGreedChart market Fear & Greed score (0–100). Returns None on failure."""
        try:
            timeout = aiohttp.ClientTimeout(total=10.0)
            async with self._session.get(_MARKET_FG_URL, timeout=timeout) as resp:
                if resp.status != 200:
                    logger.debug("FearGreedAgent: FearGreedChart HTTP %d", resp.status)
                    return None
                data: Any = await resp.json(content_type=None)
            # Response: {"score": {"score": 62, "components": [...]}, ...}
            score_field = data["score"]
            score = int(score_field["score"]) if isinstance(score_field, dict) else int(score_field)
            return score
        except Exception as exc:
            logger.debug("FearGreedAgent: market F&G fetch failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_sentiment(score: int) -> float:
        """Map 0–100 fear/greed score to −1.0…+1.0 sentiment."""
        return round((score - 50) / 50.0, 3)

    @staticmethod
    def _classify(score: int) -> str:
        if score <= 24:
            return "Extreme Fear"
        if score <= 44:
            return "Fear"
        if score <= 55:
            return "Neutral"
        if score <= 74:
            return "Greed"
        return "Extreme Greed"
