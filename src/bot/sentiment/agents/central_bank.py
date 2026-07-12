"""CentralBankAgent — hawkish/dovish classification of recent G10 central-bank statements.

The deep-research report identified central-bank press-release RSS as the
highest-signal addition to the sentiment overlay: statements move EUR/USD
30–60 pips within 15 minutes and are released on a stable, free, low-volume
RSS schedule.  This agent polls a curated set of G10 central-bank feeds,
prefilters by keyword + recency, and routes hits to the configured LLM
(``BaseAgent._call_llm`` — Cerebras primary, Groq fallback) for
hawkish/dovish classification.

Flow per scan:
1. Fetch each central-bank RSS in parallel
2. Parse + keyword-filter items published in the last ``_LOOKBACK_HOURS``
3. If at least one hit: batch into a single LLM call returning per-asset
   sentiment.  Currency → asset routing reuses MacroAgent's
   ``_CURRENCY_TO_ASSETS`` table.
4. Emit one RawSignal per affected asset.

If no recent hits exist across any central bank, the agent returns an empty
list and does NOT call the LLM — avoiding wasted tokens on quiet days.

Coverage: Fed/USD, ECB/EUR, BoE/GBP, BoJ/JPY by default.  Additional banks
(RBA/AUD, BoC/CAD) are easy to add — extend ``_FEEDS`` with the URL and
currency.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

import aiohttp
import defusedxml.ElementTree as ET

from bot.sentiment.agents.base import BaseAgent
from bot.sentiment.agents.macro import _CURRENCY_TO_ASSETS
from bot.sentiment.models import RawSignal

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_S = 20.0
_LOOKBACK_HOURS = 6
_MAX_ITEMS_PER_BANK = 5  # cap each feed's contribution to keep the prompt bounded
_MAX_TOTAL_ITEMS = 12  # hard cap across all banks per scan

# RSS feeds: (bank-name, currency, url).  Currency drives the routing into
# ``_CURRENCY_TO_ASSETS`` from MacroAgent.
_FEEDS: list[tuple[str, str, str]] = [
    ("Fed", "USD", "https://www.federalreserve.gov/feeds/press_monetary.xml"),
    ("ECB", "EUR", "https://www.ecb.europa.eu/rss/press.html"),
    ("BoE", "GBP", "https://www.bankofengland.co.uk/rss/news"),
    ("BoJ", "JPY", "https://www.boj.or.jp/en/rss/whatsnew.xml"),
]

# Lower-cased keywords.  An item is kept if any keyword appears in title OR description.
_KEYWORDS: tuple[str, ...] = (
    "rate",
    "decision",
    "statement",
    "hike",
    "cut",
    "cpi",
    "inflation",
    "policy",
    "outlook",
    "forecast",
    "minutes",
    "press conference",
    "balance sheet",
    "qe",
    "qt",
    "monetary",
)


class CentralBankAgent(BaseAgent):
    """Polls G10 central-bank RSS feeds and classifies recent statements hawkish/dovish."""

    name = "central_bank"

    async def analyze(self, assets: list[str]) -> list[RawSignal]:
        items_by_bank = await self._fetch_all_feeds()
        if not items_by_bank:
            logger.info("CentralBankAgent: no recent matching items across feeds")
            return []

        total_items = sum(len(items) for items in items_by_bank.values())
        prompt = self._build_prompt(items_by_bank, assets)
        logger.debug(
            "CentralBankAgent: dispatching %d items from %d banks to LLM",
            total_items,
            len(items_by_bank),
        )
        raw = await self._call_llm(prompt, estimated_tokens=len(prompt) // 4 + 400)
        # Filter to assets the bot actually tracks; the LLM may include the
        # whole _CURRENCY_TO_ASSETS expansion which can include symbols not
        # in the active universe.
        signals = [self._dict_to_raw_signal(d) for d in raw if d["asset"] in assets]
        logger.info(
            "CentralBankAgent: scored %d assets from %d items across %d banks",
            len(signals),
            total_items,
            len(items_by_bank),
        )
        return signals

    # ------------------------------------------------------------------
    # Feed fetching + parsing
    # ------------------------------------------------------------------

    async def _fetch_all_feeds(self) -> dict[str, list[dict[str, Any]]]:
        """Fetch every feed in parallel; return ``{bank_name: [items, ...]}``.

        Banks with no recent matching items (or failed fetches) are omitted.
        """
        results = await asyncio.gather(
            *(self._fetch_feed(bank, currency, url) for bank, currency, url in _FEEDS),
            return_exceptions=True,
        )

        by_bank: dict[str, list[dict[str, Any]]] = {}
        total = 0
        for (bank, _currency, _url), result in zip(_FEEDS, results, strict=False):
            if isinstance(result, BaseException):
                logger.debug("CentralBankAgent: %s feed raised: %s", bank, result)
                continue
            if not result:
                continue
            by_bank[bank] = result
            total += len(result)
            if total >= _MAX_TOTAL_ITEMS:
                # Stop accumulating once we hit the global cap so the LLM prompt
                # stays bounded — subsequent banks contribute zero items.
                break
        return by_bank

    async def _fetch_feed(self, bank: str, currency: str, url: str) -> list[dict[str, Any]]:
        """Fetch + parse one RSS feed.  Returns a list of recent matching items."""
        try:
            timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_S)
            async with self._session.get(url, timeout=timeout) as resp:
                if resp.status != 200:
                    logger.debug("CentralBankAgent: %s HTTP %d", bank, resp.status)
                    return []
                text = await resp.text()
        except TimeoutError:
            logger.debug("CentralBankAgent: %s timed out after %.0fs", bank, timeout.total)
            return []
        except aiohttp.ClientError as exc:
            logger.debug("CentralBankAgent: %s connection error: %s", bank, exc)
            return []
        except Exception:
            logger.debug("CentralBankAgent: %s fetch failed", bank, exc_info=True)
            return []

        return self._parse_rss(bank, currency, text)

    @staticmethod
    def _parse_rss(bank: str, currency: str, xml_text: str) -> list[dict[str, Any]]:
        """Parse an RSS-2.0 / Atom-ish feed body and return recent matching items."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            logger.debug("CentralBankAgent: %s feed not parseable as XML", bank)
            return []

        cutoff = datetime.now(UTC) - timedelta(hours=_LOOKBACK_HOURS)
        items: list[dict[str, Any]] = []

        # Iterate every <item> regardless of namespace — RSS 2.0, ATOM, and the
        # central-bank variants all expose their entries under one of these.
        candidates: list[Any] = []
        for tag in ("item", "{http://www.w3.org/2005/Atom}entry", "entry"):
            candidates.extend(root.iter(tag))

        for entry in candidates:
            title = (_text_of(entry, "title") or "").strip()
            description = (
                _text_of(entry, "description") or _text_of(entry, "summary") or ""
            ).strip()
            blob = f"{title} {description}".lower()
            if not any(kw in blob for kw in _KEYWORDS):
                continue
            pub_str = (
                _text_of(entry, "pubDate")
                or _text_of(entry, "{http://www.w3.org/2005/Atom}published")
                or _text_of(entry, "published")
            )
            if pub_str:
                try:
                    pub_dt = parsedate_to_datetime(pub_str)
                    if pub_dt.tzinfo is None:
                        pub_dt = pub_dt.replace(tzinfo=UTC)
                    if pub_dt < cutoff:
                        continue
                except (TypeError, ValueError):
                    # Keep the item if we can't parse the date — better to over-
                    # include for a few hours than silently drop a rate decision.
                    pass

            truncated = (description[:240] + "…") if len(description) > 240 else description
            items.append(
                {
                    "bank": bank,
                    "currency": currency,
                    "title": title,
                    "description": truncated,
                }
            )
            if len(items) >= _MAX_ITEMS_PER_BANK:
                break

        return items

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(items_by_bank: dict[str, list[dict[str, Any]]], assets: list[str]) -> str:
        # Build a flat human-readable list grouped by bank.  Including the
        # currency tag lets the LLM map a hawkish/dovish read directly onto
        # the target assets.
        lines: list[str] = []
        for bank, items in items_by_bank.items():
            for item in items:
                lines.append(f"- [{bank}/{item['currency']}] {item['title']}")
                if item["description"]:
                    lines.append(f"    {item['description']}")
        items_text = "\n".join(lines)
        assets_str = ", ".join(assets)

        # Pre-compute the affected-pair list per currency so the LLM doesn't
        # have to guess which pairs a Fed statement moves.
        cur_hints = []
        for currency, mapped in _CURRENCY_TO_ASSETS.items():
            mapped_in_universe = [a for a in mapped if a in assets]
            if mapped_in_universe:
                cur_hints.append(f"  {currency}: {', '.join(mapped_in_universe)}")
        hints_text = "\n".join(cur_hints)

        return (
            "Classify the central-bank statements below as hawkish (bullish for "
            "that currency vs others), dovish (bearish), or neutral, and emit "
            "per-asset sentiment scores.\n\n"
            f"Target assets: {assets_str}\n\n"
            f"Currency → affected assets:\n{hints_text}\n\n"
            f"Statements (last {_LOOKBACK_HOURS} hours):\n{items_text}\n\n"
            "For each asset moved by these statements, return a sentiment score "
            "where +1 = bullish for the asset and -1 = bearish.  A hawkish Fed "
            "statement implies positive sentiment for the USD across USD-quote "
            "pairs (USD/JPY, USD/CHF) and negative sentiment for USD-base pairs "
            "(EUR/USD, GBP/USD).  Apply the same logic to ECB→EUR, BoE→GBP, "
            "BoJ→JPY.  Include reasoning per asset."
        )


def _text_of(entry: Any, tag: str) -> str | None:
    """Find a child element by tag name (with or without namespace) and return text."""
    direct = entry.find(tag)
    if direct is not None and direct.text:
        return str(direct.text)
    # Fall back to a namespace-tolerant scan for the local name
    for child in entry.iter():
        ctag = child.tag.split("}", 1)[-1] if "}" in child.tag else child.tag
        if ctag == tag.split("}", 1)[-1] and child.text:
            return str(child.text)
    return None
