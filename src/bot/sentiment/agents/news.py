"""NewsAgent — per-symbol financial news from EODHD, interpreted by the LLM.

Rebuilt 2026-06-03. The previous source was
generic business RSS (BBC/Yahoo/MarketWatch/NYT) + Finnhub `general/forex/crypto`
categories — broad macro headlines that weren't tied to the assets we trade. It
now pulls EODHD's **ticker-tagged** news endpoint (`/api/news?s=<symbol>`), which
returns recent, financial, per-symbol articles for the whole universe (FX, US
shares, and the GLD/SLV-sourced metals).

Each EODHD article carries a pre-computed ``sentiment.polarity`` — used here only
as a cheap *neutral pre-filter* (drop near-zero articles to save LLM tokens), not
as the signal. The article **titles** are still sent to the LLM (Groq, escalating
to Gemini in the aggregator) for the trading-direction read, because EODHD's
polarity is generic news sentiment, not direction-aware (e.g. an oil supply
drawdown reads negative as "news" but is bullish for crude).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

from bot.data.eodhd_symbols import EODHD_UNIVERSE
from bot.sentiment.agents.base import BaseAgent
from bot.sentiment.models import RawSignal

logger = logging.getLogger(__name__)

_NEWS_URL = "https://eodhd.com/api/news"
_ARTICLES_PER_SYMBOL = 5  # most-recent N per symbol
_NEUTRAL_POLARITY = 0.15  # drop |polarity| below this as neutral noise (token saver)
_MAX_CONCURRENCY = 8  # cap simultaneous EODHD requests within the rate budget
_REQUEST_TIMEOUT_S = 20.0


def _eodhd_news_symbol(asset: str) -> str:
    """Map a bot_key to the EODHD news symbol (``F``→``F.US``, ``EUR/USD``→
    ``EURUSD.FOREX``, ``XAU/USD``→``GLD.US``).  Uses the universe map; falls back
    to a sensible guess for any symbol not in it (e.g. the twelvedata path)."""
    sym = EODHD_UNIVERSE.get(asset)
    if sym is not None:
        return sym.eodhd_rest
    return asset.replace("/", "") + ".FOREX" if "/" in asset else asset + ".US"


class NewsAgent(BaseAgent):
    """Per-symbol EODHD news → Groq/Gemini sentiment for each asset."""

    name = "news"

    async def analyze(self, assets: list[str]) -> list[RawSignal]:
        if not self._config.eodhd_api:
            logger.info("NewsAgent: eodhd_api empty — skipping news scan")
            return []

        per_asset = await self._fetch_news(assets)
        per_asset = {a: titles for a, titles in per_asset.items() if titles}
        if not per_asset:
            logger.info("NewsAgent: no non-neutral EODHD news for any asset")
            return []

        prompt = self._build_prompt(per_asset)
        raw = await self._call_llm(prompt, estimated_tokens=len(prompt) // 4 + 500)
        signals = [self._dict_to_raw_signal(d) for d in raw if d["asset"] in assets]
        logger.info(
            "NewsAgent: scored %d assets from EODHD news across %d with coverage",
            len(signals),
            len(per_asset),
        )
        return signals

    async def _fetch_news(self, assets: list[str]) -> dict[str, list[str]]:
        """Fetch recent EODHD news titles per asset, concurrency-capped."""
        sem = asyncio.Semaphore(_MAX_CONCURRENCY)

        async def _guarded(asset: str) -> list[str]:
            async with sem:
                return await self._fetch_symbol(asset)

        results = await asyncio.gather(*(_guarded(a) for a in assets), return_exceptions=True)
        out: dict[str, list[str]] = {}
        for asset, res in zip(assets, results, strict=False):
            if isinstance(res, list):
                out[asset] = res
        return out

    async def _fetch_symbol(self, asset: str) -> list[str]:
        """Fetch + filter the most-recent EODHD news titles for one asset."""
        params = {
            "s": _eodhd_news_symbol(asset),
            "limit": str(_ARTICLES_PER_SYMBOL),
            "api_token": self._config.eodhd_api,
            "fmt": "json",
        }
        try:
            timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_S)
            async with self._session.get(_NEWS_URL, params=params, timeout=timeout) as resp:
                if resp.status != 200:
                    logger.debug("NewsAgent: EODHD news HTTP %d for %s", resp.status, asset)
                    return []
                data: Any = await resp.json()
        except Exception:
            logger.debug("NewsAgent: EODHD news fetch failed for %s", asset, exc_info=True)
            return []
        if not isinstance(data, list):
            return []
        return self._extract_titles(data)

    @staticmethod
    def _extract_titles(articles: list[Any]) -> list[str]:
        """Pull titles, dropping near-neutral articles by EODHD polarity."""
        titles: list[str] = []
        for art in articles:
            if not isinstance(art, dict):
                continue
            title = (art.get("title") or "").strip()
            if not title:
                continue
            polarity = (art.get("sentiment") or {}).get("polarity")
            if isinstance(polarity, (int, float)) and abs(polarity) < _NEUTRAL_POLARITY:
                continue  # neutral noise — skip to save LLM tokens
            titles.append(title)
        return titles

    @staticmethod
    def _build_prompt(per_asset: dict[str, list[str]]) -> str:
        """Group titles under each asset code for the LLM to score per asset."""
        blocks: list[str] = []
        for asset, titles in per_asset.items():
            lines = "\n".join(f"  - {t}" for t in titles)
            blocks.append(f"{asset}:\n{lines}")
        body = "\n\n".join(blocks)
        return (
            "Recent financial-news headlines grouped by asset code. For each asset "
            "whose headlines carry a clear market signal, return a sentiment score; "
            "judge the likely market direction for THAT asset (e.g. an oil supply "
            "drawdown is bullish for crude, a downgrade is bearish for the stock). "
            'Use the asset code exactly as shown (e.g. "EUR/USD", "F", "XAU/USD") '
            'as the "asset" value.\n\n' + body
        )
