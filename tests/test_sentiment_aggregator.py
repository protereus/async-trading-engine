"""Tests for SentimentAggregator."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from bot.sentiment.aggregator import SentimentAggregator
from bot.sentiment.config import SentimentConfig
from bot.sentiment.models import RawSignal


def _make_config(**kwargs: object) -> SentimentConfig:
    defaults = dict(
        groq_api="test-key",
        gemini_api_key="",
        escalation_disagreement_threshold=0.5,
        escalation_confidence_threshold=0.4,
        escalation_max_per_hour=20,
    )
    defaults.update(kwargs)
    return SentimentConfig(**defaults)  # type: ignore[arg-type]


def _sig(asset: str, sentiment: float, confidence: float, source: str) -> RawSignal:
    return RawSignal(
        asset=asset,
        sentiment=sentiment,
        confidence=confidence,
        reasoning=f"{source} says {sentiment:+.1f}",
        source=source,
        fetched_at=datetime.now(UTC),
    )


class TestSentimentAggregator:
    @pytest.mark.asyncio
    async def test_basic_consensus_no_escalation(self) -> None:
        cfg = _make_config()
        agg = SentimentAggregator(cfg, session=AsyncMock())

        signals = [
            _sig("BTC/USD", 0.6, 0.8, "news"),
            _sig("BTC/USD", 0.5, 0.7, "social"),
        ]
        result = await agg.aggregate(signals, ["BTC/USD"])

        assert "BTC/USD" in result
        cs = result["BTC/USD"]
        assert not cs.escalated
        assert 0.5 <= cs.sentiment <= 0.6
        assert cs.confidence > 0

    @pytest.mark.asyncio
    async def test_agreement_perfect(self) -> None:
        cfg = _make_config()
        agg = SentimentAggregator(cfg, session=AsyncMock())

        signals = [
            _sig("EUR/USD", 0.4, 0.9, "news"),
            _sig("EUR/USD", 0.4, 0.9, "macro"),
        ]
        result = await agg.aggregate(signals, ["EUR/USD"])
        assert result["EUR/USD"].agreement == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_agreement_full_disagreement(self) -> None:
        cfg = _make_config()
        agg = SentimentAggregator(cfg, session=AsyncMock())

        signals = [
            _sig("EUR/USD", -1.0, 0.9, "news"),
            _sig("EUR/USD", 1.0, 0.9, "social"),
        ]
        result = await agg.aggregate(signals, ["EUR/USD"])
        assert result["EUR/USD"].agreement == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_assets_with_no_signals_excluded(self) -> None:
        cfg = _make_config()
        agg = SentimentAggregator(cfg, session=AsyncMock())

        signals = [_sig("BTC/USD", 0.5, 0.8, "news")]
        result = await agg.aggregate(signals, ["BTC/USD", "ETH/USD"])

        assert "BTC/USD" in result
        assert "ETH/USD" not in result

    @pytest.mark.asyncio
    async def test_escalation_triggered_on_disagreement(self) -> None:
        cfg = _make_config(gemini_api_key="sk-ant-test")
        agg = SentimentAggregator(cfg, session=AsyncMock())

        # Patch Gemini call to return a known result
        agg._call_gemini = AsyncMock(  # type: ignore[method-assign]
            return_value={"sentiment": 0.2, "confidence": 0.75, "reasoning": "balanced"}
        )

        signals = [
            _sig("XAU/USD", -0.8, 0.9, "news"),  # strongly bearish
            _sig("XAU/USD", 0.8, 0.9, "social"),  # strongly bullish — disagreement = 1.6 > 0.5
        ]
        result = await agg.aggregate(signals, ["XAU/USD"])

        assert result["XAU/USD"].escalated
        assert result["XAU/USD"].sentiment == pytest.approx(0.2)
        agg._call_gemini.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_escalation_not_triggered_without_api_key(self) -> None:
        cfg = _make_config(gemini_api_key="")
        agg = SentimentAggregator(cfg, session=AsyncMock())

        signals = [
            _sig("XAU/USD", -1.0, 0.9, "news"),
            _sig("XAU/USD", 1.0, 0.9, "social"),
        ]
        result = await agg.aggregate(signals, ["XAU/USD"])
        assert not result["XAU/USD"].escalated

    @pytest.mark.asyncio
    async def test_escalation_cap_respected(self) -> None:
        cfg = _make_config(gemini_api_key="sk-ant-test", escalation_max_per_hour=2)
        agg = SentimentAggregator(cfg, session=AsyncMock())
        agg._call_gemini = AsyncMock(  # type: ignore[method-assign]
            return_value={"sentiment": 0.0, "confidence": 0.5, "reasoning": "neutral"}
        )

        # First two escalations should fire
        for _ in range(2):
            signals = [
                _sig("BTC/USD", -1.0, 0.9, "news"),
                _sig("BTC/USD", 1.0, 0.9, "social"),
            ]
            await agg.aggregate(signals, ["BTC/USD"])

        # Third should hit the cap
        signals = [
            _sig("ETH/USD", -1.0, 0.9, "news"),
            _sig("ETH/USD", 1.0, 0.9, "social"),
        ]
        result = await agg.aggregate(signals, ["ETH/USD"])
        assert not result["ETH/USD"].escalated
        assert agg._call_gemini.await_count == 2

    @pytest.mark.asyncio
    async def test_confidence_weighted_sentiment(self) -> None:
        cfg = _make_config()
        agg = SentimentAggregator(cfg, session=AsyncMock())

        # High-confidence bearish + low-confidence bullish → net bearish
        signals = [
            _sig("USD/JPY", -0.8, 0.9, "news"),
            _sig("USD/JPY", 0.5, 0.1, "social"),
        ]
        result = await agg.aggregate(signals, ["USD/JPY"])
        assert result["USD/JPY"].sentiment < 0  # bearish wins

    @pytest.mark.asyncio
    async def test_single_agent_signal(self) -> None:
        cfg = _make_config()
        agg = SentimentAggregator(cfg, session=AsyncMock())

        signals = [_sig("SPY", 0.7, 0.85, "macro")]
        result = await agg.aggregate(signals, ["SPY"])

        assert "SPY" in result
        assert result["SPY"].sentiment == pytest.approx(0.7)
        assert result["SPY"].agreement == pytest.approx(1.0)  # only one signal = perfect agreement
