"""MacroAgent — fetches economic calendar events and scores their impact via Groq."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import aiohttp  # used for ClientTimeout in _fetch_forex_factory and _fetch_twelve_data
import defusedxml.ElementTree as ET

from bot.sentiment.agents.base import BaseAgent
from bot.sentiment.models import RawSignal

logger = logging.getLogger(__name__)

# Finnhub uses ISO country codes; map to currency codes for _CURRENCY_TO_ASSETS lookup
_COUNTRY_TO_CURRENCY: dict[str, str] = {
    "US": "USD",
    "GB": "GBP",
    "EU": "EUR",
    "JP": "JPY",
    "AU": "AUD",
    "CA": "CAD",
    "CH": "CHF",
    "NZ": "NZD",
    "SE": "SEK",
    "NO": "NOK",
}

# Forex Factory public XML calendar (no auth required)
_FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
# Finnhub economic calendar (replaces Twelve Data which returned 404 on free tier)
_FINNHUB_CALENDAR_URL = "https://finnhub.io/api/v1/calendar/economic"

# Map currency codes to affected live-universe assets (bot_keys).  Trimmed to
# the EODHD 28-symbol universe (12 FX + XAU/XAG); the 14 single-name US shares
# are deliberately absent — a Fed statement moves them via equity beta, which
# the LLM already sees through the target-assets list, and enumerating them
# here would drown the currency→pair hints.  SEK/NOK dropped with USD/SEK and
# USD/NOK (events for them still reach the LLM as context via
# _COUNTRY_TO_CURRENCY; they just carry no pair hint).
_CURRENCY_TO_ASSETS: dict[str, list[str]] = {
    "USD": [
        "EUR/USD",
        "GBP/USD",
        "USD/JPY",
        "USD/CHF",
        "USD/CAD",
        "AUD/USD",
        "NZD/USD",
        "XAU/USD",
        "XAG/USD",
    ],
    "EUR": ["EUR/USD", "EUR/GBP", "EUR/JPY", "EUR/AUD"],
    "GBP": ["GBP/USD", "EUR/GBP", "GBP/JPY"],
    "JPY": ["USD/JPY", "EUR/JPY", "GBP/JPY", "AUD/JPY"],
    "AUD": ["AUD/USD", "EUR/AUD", "AUD/JPY"],
    "CAD": ["USD/CAD"],
    "CHF": ["USD/CHF"],
    "NZD": ["NZD/USD"],
}


class MacroAgent(BaseAgent):
    """Fetches upcoming high-impact economic events and scores their likely market impact."""

    name = "macro"

    async def analyze(self, assets: list[str]) -> list[RawSignal]:
        events = await self._fetch_events()
        if not events:
            logger.info("MacroAgent: no economic events fetched")
            return []

        events_text = "\n".join(
            f"- [{e['currency']}] {e['title']} (impact: {e['impact']}, "
            f"forecast: {e.get('forecast', 'N/A')}, previous: {e.get('previous', 'N/A')})"
            for e in events[:25]
        )
        assets_str = ", ".join(assets)

        prompt = (
            f"Analyze these upcoming economic calendar events and assess their "
            f"likely impact on financial markets.\n\n"
            f"Target assets: {assets_str}\n\n"
            f"Economic events (next 48 hours, high/medium impact):\n{events_text}\n\n"
            f"For each target asset likely to be affected, return a sentiment score "
            f"based on what these events imply for price direction. "
            f"Consider: hawkish/dovish signals, growth indicators, risk-on/risk-off dynamics."
        )

        raw = await self._call_llm(prompt, estimated_tokens=len(prompt) // 4 + 400)
        signals = [self._dict_to_raw_signal(d) for d in raw if d["asset"] in assets]
        logger.info("MacroAgent: scored %d assets from %d events", len(signals), len(events))
        return signals

    async def _fetch_events(self) -> list[dict[str, Any]]:
        """Try Forex Factory XML first, fall back to Finnhub economic calendar."""
        events = await self._fetch_forex_factory()
        if events:
            return events
        if self._config.finnhub_api:
            return await self._fetch_finnhub_calendar()
        return []

    async def _fetch_forex_factory(self) -> list[dict[str, Any]]:
        """Parse Forex Factory weekly XML calendar."""
        try:
            timeout = aiohttp.ClientTimeout(total=15.0)
            headers = {"User-Agent": "TradingBot/1.0 (financial research)"}
            async with self._session.get(
                _FF_CALENDAR_URL, timeout=timeout, headers=headers
            ) as resp:
                if resp.status != 200:
                    logger.debug("MacroAgent: Forex Factory HTTP %d", resp.status)
                    return []
                text = await resp.text()
        except TimeoutError:
            logger.debug("MacroAgent: Forex Factory timed out after %.0fs", timeout.total)
            return []
        except aiohttp.ClientError as exc:
            logger.debug("MacroAgent: Forex Factory connection error: %s", exc)
            return []
        except Exception:
            logger.debug("MacroAgent: Forex Factory fetch failed", exc_info=True)
            return []

        return self._parse_ff_xml(text)

    def _parse_ff_xml(self, xml_text: str) -> list[dict[str, Any]]:
        """Parse Forex Factory XML into a list of event dicts."""
        events: list[dict[str, Any]] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []

        now_utc = datetime.now(UTC)
        cutoff = now_utc + timedelta(hours=48)

        for item in root.findall(".//event"):
            try:
                impact_el = item.find("impact")
                impact = impact_el.text if impact_el is not None else ""
                if impact not in ("High", "Medium"):
                    continue

                date_el = item.find("date")
                time_el = item.find("time")
                if date_el is None or not date_el.text:
                    continue

                # Parse event datetime (Forex Factory uses "MMM DD, YYYY" format)
                date_str = date_el.text.strip()
                time_str = (time_el.text or "").strip() if time_el is not None else ""
                try:
                    if time_str:
                        dt = datetime.strptime(f"{date_str} {time_str}", "%b %d, %Y %I:%M%p")
                    else:
                        dt = datetime.strptime(date_str, "%b %d, %Y")
                    dt = dt.replace(tzinfo=UTC)
                except ValueError:
                    continue

                if dt > cutoff:
                    continue

                currency_el = item.find("country")
                title_el = item.find("title")
                forecast_el = item.find("forecast")
                previous_el = item.find("previous")

                events.append(
                    {
                        "currency": (currency_el.text or "").strip()
                        if currency_el is not None
                        else "",
                        "title": (title_el.text or "").strip() if title_el is not None else "",
                        "impact": impact,
                        "forecast": (forecast_el.text or "").strip()
                        if forecast_el is not None
                        else "",
                        "previous": (previous_el.text or "").strip()
                        if previous_el is not None
                        else "",
                        "datetime": dt.isoformat(),
                    }
                )
            except Exception as exc:
                # Skip one malformed event but keep the batch — log so dropped
                # items are observable (the fetch boundary above is logged too).
                logger.debug("MacroAgent: skipping malformed Forex Factory event: %s", exc)
                continue

        return events

    async def _fetch_finnhub_calendar(self) -> list[dict[str, Any]]:
        """Fetch economic calendar from Finnhub as fallback."""
        now = datetime.now(UTC)
        params = {
            "from": now.strftime("%Y-%m-%d"),
            "to": (now + timedelta(days=2)).strftime("%Y-%m-%d"),
            "token": self._config.finnhub_api,
        }
        try:
            timeout = aiohttp.ClientTimeout(total=15.0)
            async with self._session.get(
                _FINNHUB_CALENDAR_URL, params=params, timeout=timeout
            ) as resp:
                if resp.status != 200:
                    logger.debug("MacroAgent: Finnhub calendar HTTP %d", resp.status)
                    return []
                data: Any = await resp.json()
        except TimeoutError:
            logger.debug("MacroAgent: Finnhub calendar timed out after %.0fs", timeout.total)
            return []
        except aiohttp.ClientError as exc:
            logger.debug("MacroAgent: Finnhub calendar connection error: %s", exc)
            return []
        except Exception:
            logger.debug("MacroAgent: Finnhub calendar fetch failed", exc_info=True)
            return []

        events: list[dict[str, Any]] = []
        raw_events = data.get("economicCalendar", []) if isinstance(data, dict) else []
        for item in raw_events:
            try:
                impact = str(item.get("impact", "")).lower()
                if impact not in ("high", "medium"):
                    continue
                # Map country code → currency (Finnhub uses ISO country codes)
                country = str(item.get("country", "")).upper()
                currency = _COUNTRY_TO_CURRENCY.get(country, country)
                events.append(
                    {
                        "currency": currency,
                        "title": str(item.get("event", "")),
                        "impact": impact.capitalize(),
                        "forecast": str(item.get("estimate", "")),
                        "previous": str(item.get("prev", "")),
                        "datetime": str(item.get("time", "")),
                    }
                )
            except Exception as exc:
                # Skip one malformed event but keep the batch — log so dropped
                # items are observable (the fetch boundary above is logged too).
                logger.debug("MacroAgent: skipping malformed Finnhub event: %s", exc)
                continue
        return events
