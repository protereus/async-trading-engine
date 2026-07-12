"""Tests for CentralBankAgent."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.sentiment.agents.central_bank import (
    _FEEDS,
    _LOOKBACK_HOURS,
    _MAX_ITEMS_PER_BANK,
    _MAX_TOTAL_ITEMS,
    CentralBankAgent,
)
from bot.sentiment.config import SentimentConfig
from bot.sentiment.rate_limiter import GroqRateLimiter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config() -> SentimentConfig:
    return SentimentConfig(groq_api="g")


def _make_limiter() -> GroqRateLimiter:
    lim = GroqRateLimiter()
    lim.acquire = AsyncMock()  # type: ignore[method-assign]
    lim.record_request = MagicMock()  # type: ignore[method-assign]
    return lim


def _rss_with_items(items: list[dict[str, Any]]) -> str:
    """Build a minimal RSS-2.0 body from a list of {title, description, ts_offset_hours}."""
    parts = ['<?xml version="1.0"?><rss version="2.0"><channel>']
    for it in items:
        offset = it.get("ts_offset_hours", 0)
        pub = format_datetime(datetime.now(UTC) - timedelta(hours=offset))
        title = it.get("title", "")
        desc = it.get("description", "")
        parts.append(
            f"<item><title>{title}</title>"
            f"<description>{desc}</description>"
            f"<pubDate>{pub}</pubDate></item>"
        )
    parts.append("</channel></rss>")
    return "\n".join(parts)


class _FakeResponse:
    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self._body = body

    async def text(self) -> str:
        return self._body

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# _parse_rss (pure function)
# ---------------------------------------------------------------------------


def test_parse_rss_keeps_keyword_matches_within_lookback() -> None:
    body = _rss_with_items(
        [
            {
                "title": "FOMC raises rate by 25bp",
                "description": "policy statement",
                "ts_offset_hours": 1,
            },
            {
                "title": "Annual report released",
                "description": "summary of activities",
                "ts_offset_hours": 1,
            },
            {
                "title": "Inflation outlook updated",
                "description": "CPI projections",
                "ts_offset_hours": 2,
            },
        ]
    )
    items = CentralBankAgent._parse_rss("Fed", "USD", body)
    titles = {i["title"] for i in items}
    assert "FOMC raises rate by 25bp" in titles
    assert "Inflation outlook updated" in titles
    assert "Annual report released" not in titles
    assert all(i["bank"] == "Fed" and i["currency"] == "USD" for i in items)


def test_parse_rss_drops_items_older_than_lookback() -> None:
    body = _rss_with_items(
        [
            {"title": "Rate decision", "description": "x", "ts_offset_hours": 1},
            {
                "title": "Old rate decision",
                "description": "x",
                "ts_offset_hours": _LOOKBACK_HOURS + 4,
            },
        ]
    )
    items = CentralBankAgent._parse_rss("Fed", "USD", body)
    titles = {i["title"] for i in items}
    assert "Rate decision" in titles
    assert "Old rate decision" not in titles


def test_parse_rss_respects_per_bank_cap() -> None:
    body = _rss_with_items(
        [
            {"title": "Rate change announcement", "description": "x", "ts_offset_hours": 1}
            for _ in range(_MAX_ITEMS_PER_BANK + 3)
        ]
    )
    items = CentralBankAgent._parse_rss("Fed", "USD", body)
    assert len(items) == _MAX_ITEMS_PER_BANK


def test_parse_rss_invalid_xml_returns_empty() -> None:
    assert CentralBankAgent._parse_rss("Fed", "USD", "<not xml") == []
    assert CentralBankAgent._parse_rss("Fed", "USD", "") == []


def test_parse_rss_keeps_undated_items() -> None:
    """Items without a parseable date should be kept (over-include > silently drop)."""
    body = (
        '<?xml version="1.0"?><rss version="2.0"><channel>'
        "<item><title>Rate decision pending</title><description>policy outlook</description>"
        "</item></channel></rss>"
    )
    items = CentralBankAgent._parse_rss("Fed", "USD", body)
    assert len(items) == 1


def test_parse_rss_truncates_long_descriptions() -> None:
    long_desc = "policy " + ("statement " * 80)
    body = _rss_with_items(
        [{"title": "Rate decision", "description": long_desc, "ts_offset_hours": 1}]
    )
    items = CentralBankAgent._parse_rss("Fed", "USD", body)
    assert len(items) == 1
    assert items[0]["description"].endswith("…")
    assert len(items[0]["description"]) <= 241


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def test_build_prompt_includes_bank_currency_and_routing_hints() -> None:
    items_by_bank = {
        "Fed": [{"bank": "Fed", "currency": "USD", "title": "Hawkish hike", "description": "x"}],
        "ECB": [{"bank": "ECB", "currency": "EUR", "title": "Dovish hold", "description": "x"}],
    }
    prompt = CentralBankAgent._build_prompt(items_by_bank, ["EUR/USD", "USD/JPY"])
    assert "[Fed/USD]" in prompt
    assert "[ECB/EUR]" in prompt
    assert "Hawkish hike" in prompt
    assert "Dovish hold" in prompt
    # Currency routing hints visible
    assert "USD:" in prompt and "EUR:" in prompt
    # Targeted assets surface
    assert "EUR/USD" in prompt
    assert "USD/JPY" in prompt


def test_build_prompt_omits_unavailable_assets_from_routing_hints() -> None:
    """Currency-routing hints lines should only list assets in the target universe.

    (Other parts of the prompt contain worked examples like "USD/JPY" — that's
    fine; only the dynamic hints block must reflect the active universe.)
    """
    prompt = CentralBankAgent._build_prompt(
        {"Fed": [{"bank": "Fed", "currency": "USD", "title": "Hike", "description": ""}]},
        ["EUR/USD"],  # only EUR/USD in universe
    )
    # Isolate the "Currency → affected assets:" block
    section_start = prompt.index("Currency → affected assets:")
    section_end = prompt.index("Statements ", section_start)
    section = prompt[section_start:section_end]
    assert "EUR/USD" in section
    assert "USD/CAD" not in section
    assert "USD/JPY" not in section


# ---------------------------------------------------------------------------
# analyze() — end-to-end orchestration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_no_recent_items_returns_empty_without_llm() -> None:
    agent = CentralBankAgent(_make_config(), MagicMock(), _make_limiter())
    agent._fetch_all_feeds = AsyncMock(return_value={})  # type: ignore[method-assign]
    agent._call_llm = AsyncMock()  # type: ignore[method-assign]
    result = await agent.analyze(["EUR/USD"])
    assert result == []
    agent._call_llm.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_dispatches_to_llm_and_filters_to_target_assets() -> None:
    agent = CentralBankAgent(_make_config(), MagicMock(), _make_limiter())
    agent._fetch_all_feeds = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "Fed": [{"bank": "Fed", "currency": "USD", "title": "Hike", "description": ""}]
        }
    )
    agent._call_llm = AsyncMock(  # type: ignore[method-assign]
        return_value=[
            {"asset": "EUR/USD", "sentiment": -0.6, "confidence": 0.75, "reasoning": "USD up"},
            {"asset": "USD/JPY", "sentiment": 0.5, "confidence": 0.7, "reasoning": "USD up"},
            {"asset": "BTC/USD", "sentiment": 0.1, "confidence": 0.5, "reasoning": "noise"},
        ]
    )
    result = await agent.analyze(["EUR/USD", "USD/JPY"])
    assets = {s.asset for s in result}
    # BTC/USD must be filtered (not in active universe)
    assert assets == {"EUR/USD", "USD/JPY"}
    assert all(s.source == "central_bank" for s in result)


# ---------------------------------------------------------------------------
# Feed fetching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_feed_handles_non_200() -> None:
    session = MagicMock()
    session.get = MagicMock(return_value=_FakeResponse(503, "down"))
    agent = CentralBankAgent(_make_config(), session, _make_limiter())
    items = await agent._fetch_feed("Fed", "USD", "https://example.com/feed.xml")
    assert items == []


@pytest.mark.asyncio
async def test_fetch_feed_happy_path() -> None:
    body = _rss_with_items(
        [{"title": "Rate cut announced", "description": "policy decision", "ts_offset_hours": 1}]
    )
    session = MagicMock()
    session.get = MagicMock(return_value=_FakeResponse(200, body))
    agent = CentralBankAgent(_make_config(), session, _make_limiter())
    items = await agent._fetch_feed("Fed", "USD", "https://example.com/feed.xml")
    assert len(items) == 1
    assert items[0]["bank"] == "Fed"


@pytest.mark.asyncio
async def test_fetch_all_feeds_aggregates_results() -> None:
    """All feeds run in parallel; bank-name → list of items dict is returned."""
    agent = CentralBankAgent(_make_config(), MagicMock(), _make_limiter())

    async def fake_fetch(bank: str, currency: str, url: str) -> list[dict[str, Any]]:
        return [{"bank": bank, "currency": currency, "title": f"{bank} hike", "description": ""}]

    agent._fetch_feed = AsyncMock(side_effect=fake_fetch)  # type: ignore[method-assign]
    out = await agent._fetch_all_feeds()
    # _FEEDS has 4 entries; all should appear
    assert set(out.keys()) == {bank for bank, _, _ in _FEEDS}


@pytest.mark.asyncio
async def test_fetch_all_feeds_respects_global_cap() -> None:
    """When per-bank totals would exceed _MAX_TOTAL_ITEMS, subsequent banks are skipped."""
    agent = CentralBankAgent(_make_config(), MagicMock(), _make_limiter())

    async def fake_fetch(bank: str, currency: str, url: str) -> list[dict[str, Any]]:
        # Each bank returns the per-bank max → first banks fill the budget
        return [
            {"bank": bank, "currency": currency, "title": "x", "description": ""}
            for _ in range(_MAX_ITEMS_PER_BANK)
        ]

    agent._fetch_feed = AsyncMock(side_effect=fake_fetch)  # type: ignore[method-assign]
    out = await agent._fetch_all_feeds()
    total = sum(len(v) for v in out.values())
    assert total <= _MAX_TOTAL_ITEMS + _MAX_ITEMS_PER_BANK  # last bank can push slightly over
    assert len(out) <= len(_FEEDS)
