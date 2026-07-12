"""Tests for sentiment agents (NewsAgent, MacroAgent, FearGreedAgent).

SocialAgent was retired 2026-06-01 — Finnhub's social-sentiment endpoint
requires a paid tier the account doesn't have, so every call 403'd.  The
agent + its tests were removed wholesale rather than carrying a permanently-
disabled module.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.sentiment.agents.fear_greed import FearGreedAgent
from bot.sentiment.agents.macro import MacroAgent
from bot.sentiment.agents.news import NewsAgent
from bot.sentiment.config import SentimentConfig
from bot.sentiment.rate_limiter import GroqRateLimiter


def _make_config(**kwargs: object) -> SentimentConfig:
    defaults = dict(
        groq_api="test-key", twelve_data_api="td-key", finnhub_api="fh-key", eodhd_api="eodhd-key"
    )
    defaults.update(kwargs)
    return SentimentConfig(**defaults)  # type: ignore[arg-type]


def _make_limiter() -> GroqRateLimiter:
    lim = GroqRateLimiter()
    lim.acquire = AsyncMock()  # type: ignore[method-assign]
    lim.record_request = MagicMock()  # type: ignore[method-assign]
    return lim


def _mock_groq_response(signals: list[dict]) -> dict:
    import json

    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"signals": signals}),
                }
            }
        ]
    }


# ---------------------------------------------------------------------------
# BaseAgent._parse_signals
# ---------------------------------------------------------------------------


class TestParseSignals:
    def test_valid_array_wrapped_in_object(self) -> None:
        import json

        cfg = _make_config()
        lim = _make_limiter()
        session = AsyncMock()
        agent = NewsAgent(cfg, session, lim)

        content = json.dumps(
            {
                "signals": [
                    {"asset": "BTC/USD", "sentiment": 0.7, "confidence": 0.8, "reasoning": "bull"}
                ]
            }
        )
        result = agent._parse_signals(content)
        assert len(result) == 1
        assert result[0]["asset"] == "BTC/USD"
        assert result[0]["sentiment"] == pytest.approx(0.7)

    def test_bare_array(self) -> None:
        import json

        agent = NewsAgent(_make_config(), AsyncMock(), _make_limiter())
        content = json.dumps(
            [{"asset": "EUR/USD", "sentiment": -0.3, "confidence": 0.6, "reasoning": "weak"}]
        )
        result = agent._parse_signals(content)
        assert len(result) == 1
        assert result[0]["asset"] == "EUR/USD"

    def test_empty_array(self) -> None:
        agent = NewsAgent(_make_config(), AsyncMock(), _make_limiter())
        result = agent._parse_signals("[]")
        assert result == []

    def test_malformed_json_returns_empty(self) -> None:
        agent = NewsAgent(_make_config(), AsyncMock(), _make_limiter())
        result = agent._parse_signals("not json at all")
        assert result == []

    def test_sentiment_clamped_in_dict_to_raw_signal(self) -> None:
        agent = NewsAgent(_make_config(), AsyncMock(), _make_limiter())
        sig = agent._dict_to_raw_signal(
            {"asset": "BTC/USD", "sentiment": 5.0, "confidence": -0.2, "reasoning": "extreme"}
        )
        assert sig.sentiment == pytest.approx(1.0)
        assert sig.confidence == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# NewsAgent
# ---------------------------------------------------------------------------


class TestNewsAgent:
    @pytest.mark.asyncio
    async def test_analyze_returns_signals_for_known_assets(self) -> None:
        agent = NewsAgent(_make_config(), AsyncMock(), _make_limiter())
        agent._fetch_news = AsyncMock(  # type: ignore[method-assign]
            return_value={"VZ": ["AT&T downgraded"], "EUR/USD": ["Euro slips vs USD"]}
        )
        agent._call_llm = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                {"asset": "VZ", "sentiment": -0.5, "confidence": 0.8, "reasoning": "downgrade"},
                {"asset": "EUR/USD", "sentiment": -0.4, "confidence": 0.7, "reasoning": "hawkish"},
            ]
        )
        result = await agent.analyze(["VZ", "EUR/USD", "XAU/USD"])
        assert {s.asset for s in result} == {"VZ", "EUR/USD"}
        assert all(s.source == "news" for s in result)

    @pytest.mark.asyncio
    async def test_analyze_filters_unknown_assets(self) -> None:
        agent = NewsAgent(_make_config(), AsyncMock(), _make_limiter())
        agent._fetch_news = AsyncMock(return_value={"VZ": ["headline"]})  # type: ignore[method-assign]
        agent._call_llm = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                {"asset": "NOT/IN/UNIVERSE", "sentiment": 0.5, "confidence": 0.5, "reasoning": "x"}
            ]
        )
        assert await agent.analyze(["VZ"]) == []

    @pytest.mark.asyncio
    async def test_analyze_empty_news_returns_empty(self) -> None:
        agent = NewsAgent(_make_config(), AsyncMock(), _make_limiter())
        agent._fetch_news = AsyncMock(return_value={})  # type: ignore[method-assign]
        agent._call_llm = AsyncMock()  # type: ignore[method-assign]
        assert await agent.analyze(["VZ"]) == []
        agent._call_llm.assert_not_awaited()  # no LLM call when no news

    @pytest.mark.asyncio
    async def test_analyze_skips_without_eodhd_key(self) -> None:
        agent = NewsAgent(_make_config(eodhd_api=""), AsyncMock(), _make_limiter())
        agent._fetch_news = AsyncMock()  # type: ignore[method-assign]
        assert await agent.analyze(["VZ"]) == []
        agent._fetch_news.assert_not_awaited()  # no fetch without a key

    def test_extract_titles_drops_neutral_by_polarity(self) -> None:
        articles = [
            {"title": "Strong signal", "sentiment": {"polarity": 0.9}},
            {"title": "Neutral filler", "sentiment": {"polarity": 0.05}},  # dropped
            {"title": "Bearish", "sentiment": {"polarity": -0.7}},
            {"title": "", "sentiment": {"polarity": 0.9}},  # empty title dropped
            {"title": "No-sentiment kept", "sentiment": None},  # missing polarity → kept
        ]
        titles = NewsAgent._extract_titles(articles)
        assert titles == ["Strong signal", "Bearish", "No-sentiment kept"]

    def test_build_prompt_groups_by_asset_code(self) -> None:
        prompt = NewsAgent._build_prompt({"F": ["Ford recall"], "XAU/USD": ["Gold rallies"]})
        assert "F:" in prompt and "Ford recall" in prompt
        assert "XAU/USD:" in prompt and "Gold rallies" in prompt

    def test_eodhd_news_symbol_mapping(self) -> None:
        from bot.sentiment.agents.news import _eodhd_news_symbol

        assert _eodhd_news_symbol("F") == "F.US"  # US share
        assert _eodhd_news_symbol("EUR/USD") == "EURUSD.FOREX"  # FX
        assert _eodhd_news_symbol("XAU/USD") == "GLD.US"  # metal sourced from ETF
        assert _eodhd_news_symbol("WTF/XYZ") == "WTFXYZ.FOREX"  # fallback for unknown


# ---------------------------------------------------------------------------
# MacroAgent
# ---------------------------------------------------------------------------


class TestMacroAgent:
    def test_parse_ff_xml_filters_high_medium_impact(self) -> None:
        agent = MacroAgent(_make_config(), AsyncMock(), _make_limiter())
        xml = """<?xml version="1.0"?>
        <weeklyevents>
          <event>
            <title>Non-Farm Payrolls</title>
            <country>USD</country>
            <date>Apr 14, 2026</date>
            <time>8:30am</time>
            <impact>High</impact>
            <forecast>200K</forecast>
            <previous>180K</previous>
          </event>
          <event>
            <title>Low Impact Event</title>
            <country>EUR</country>
            <date>Apr 14, 2026</date>
            <time>9:00am</time>
            <impact>Low</impact>
            <forecast></forecast>
            <previous></previous>
          </event>
        </weeklyevents>"""
        events = agent._parse_ff_xml(xml)
        assert all(e["impact"] in ("High", "Medium") for e in events)

    @pytest.mark.asyncio
    async def test_analyze_returns_signals(self) -> None:
        cfg = _make_config()
        agent = MacroAgent(cfg, AsyncMock(), _make_limiter())
        agent._fetch_events = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                {
                    "currency": "USD",
                    "title": "NFP",
                    "impact": "High",
                    "forecast": "200K",
                    "previous": "180K",
                }
            ]
        )
        agent._call_llm = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                {"asset": "EUR/USD", "sentiment": -0.5, "confidence": 0.8, "reasoning": "USD"},
            ]
        )
        result = await agent.analyze(["EUR/USD", "BTC/USD"])
        assert len(result) == 1
        assert result[0].asset == "EUR/USD"
        assert result[0].source == "macro"

    @pytest.mark.asyncio
    async def test_analyze_empty_events_returns_empty(self) -> None:
        agent = MacroAgent(_make_config(), AsyncMock(), _make_limiter())
        agent._fetch_events = AsyncMock(return_value=[])  # type: ignore[method-assign]
        result = await agent.analyze(["EUR/USD"])
        assert result == []


# ---------------------------------------------------------------------------
# FearGreedAgent
# ---------------------------------------------------------------------------


class TestFearGreedAgent:
    @pytest.mark.asyncio
    async def test_analyze_returns_market_signals(self) -> None:
        agent = FearGreedAgent(_make_config(), AsyncMock(), _make_limiter())
        agent._fetch_market_fg = AsyncMock(return_value=70)  # type: ignore[method-assign]

        assets = ["F", "XOM", "XAU/USD", "EUR/USD"]
        result = await agent.analyze(assets)

        assert len(result) == 3  # F, XOM, XAU/USD — EUR/USD (FX) not in _MARKET_ASSETS
        assert all(s.source == "fear_greed" for s in result)
        assert all(s.sentiment > 0 for s in result)  # score=70 → Greed → positive

    @pytest.mark.asyncio
    async def test_analyze_handles_fetch_failure_gracefully(self) -> None:
        agent = FearGreedAgent(_make_config(), AsyncMock(), _make_limiter())
        agent._fetch_market_fg = AsyncMock(return_value=None)  # type: ignore[method-assign]

        result = await agent.analyze(["F"])
        assert result == []

    def test_score_to_sentiment_mapping(self) -> None:
        assert FearGreedAgent._score_to_sentiment(0) == pytest.approx(-1.0)
        assert FearGreedAgent._score_to_sentiment(50) == pytest.approx(0.0)
        assert FearGreedAgent._score_to_sentiment(100) == pytest.approx(1.0)
        assert FearGreedAgent._score_to_sentiment(25) == pytest.approx(-0.5)
        assert FearGreedAgent._score_to_sentiment(75) == pytest.approx(0.5)

    def test_classify(self) -> None:
        assert FearGreedAgent._classify(10) == "Extreme Fear"
        assert FearGreedAgent._classify(35) == "Fear"
        assert FearGreedAgent._classify(50) == "Neutral"
        assert FearGreedAgent._classify(65) == "Greed"
        assert FearGreedAgent._classify(90) == "Extreme Greed"
