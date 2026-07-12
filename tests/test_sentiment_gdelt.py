"""Tests for GdeltAgent."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.sentiment.agents.gdelt import GdeltAgent
from bot.sentiment.config import SentimentConfig
from bot.sentiment.rate_limiter import GroqRateLimiter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config() -> SentimentConfig:
    return SentimentConfig(groq_api="test")


def _make_limiter() -> GroqRateLimiter:
    lim = GroqRateLimiter()
    lim.acquire = AsyncMock()  # type: ignore[method-assign]
    lim.record_request = MagicMock()  # type: ignore[method-assign]
    return lim


def _timeline(values: list[float]) -> dict[str, Any]:
    return {
        "timeline": [
            {
                "series": "Average Tone",
                "data": [{"date": f"20260518T{i:04d}Z", "value": v} for i, v in enumerate(values)],
            }
        ]
    }


# ---------------------------------------------------------------------------
# _score_timeline (pure function)
# ---------------------------------------------------------------------------


def test_score_timeline_positive_tone() -> None:
    data = _timeline([2.0] * 30)
    result = GdeltAgent._score_timeline("EURUSD", data)
    assert result is not None
    # avg_tone = 2.0, scale=5.0 → sentiment = 0.4
    assert result["sentiment"] == pytest.approx(0.4, abs=1e-4)
    assert 0.4 < result["confidence"] <= 0.85
    assert "EURUSD" in result["reasoning"]


def test_score_timeline_negative_tone() -> None:
    data = _timeline([-3.0] * 50)
    result = GdeltAgent._score_timeline("crude oil", data)
    assert result is not None
    assert result["sentiment"] == pytest.approx(-0.6, abs=1e-4)


def test_score_timeline_clipped_extreme() -> None:
    data = _timeline([12.0] * 30)
    result = GdeltAgent._score_timeline("x", data)
    assert result is not None
    assert result["sentiment"] == 1.0


def test_score_timeline_clipped_negative_extreme() -> None:
    data = _timeline([-12.0] * 30)
    result = GdeltAgent._score_timeline("x", data)
    assert result is not None
    assert result["sentiment"] == -1.0


def test_score_timeline_below_threshold_returns_none() -> None:
    data = _timeline([1.0] * 5)  # fewer than _MIN_BUCKETS (10)
    assert GdeltAgent._score_timeline("x", data) is None


def test_score_timeline_empty_returns_none() -> None:
    assert GdeltAgent._score_timeline("x", {"timeline": []}) is None
    assert GdeltAgent._score_timeline("x", {}) is None
    assert GdeltAgent._score_timeline("x", "not a dict") is None
    assert GdeltAgent._score_timeline("x", {"timeline": [{}]}) is None


def test_score_timeline_skips_malformed_points() -> None:
    """A few junk points alongside real ones should not crash."""
    raw = {
        "timeline": [
            {
                "series": "Average Tone",
                "data": [
                    {"value": 1.0},
                    {"value": "not a number"},
                    {"no_value_key": 0.5},
                    *[{"value": 1.0} for _ in range(15)],
                ],
            }
        ]
    }
    result = GdeltAgent._score_timeline("x", raw)
    assert result is not None
    assert result["sentiment"] > 0


def test_score_timeline_confidence_grows_with_bucket_count() -> None:
    low = GdeltAgent._score_timeline("x", _timeline([1.0] * 12))
    high = GdeltAgent._score_timeline("x", _timeline([1.0] * 90))
    assert low is not None and high is not None
    assert high["confidence"] > low["confidence"]
    assert high["confidence"] <= 0.85


# ---------------------------------------------------------------------------
# analyze() — orchestration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_filters_to_covered_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    # Skip the per-request sleep to keep the test fast
    monkeypatch.setattr("bot.sentiment.agents.gdelt._REQUEST_SLEEP_S", 0.0)
    agent = GdeltAgent(_make_config(), MagicMock(), _make_limiter())
    agent._fetch_tone = AsyncMock(  # type: ignore[method-assign]
        return_value=({"sentiment": 0.2, "confidence": 0.6, "reasoning": "x"}, "ok")
    )
    # GBP/JPY and FOO are not in _ASSET_QUERY_MAP — should be filtered
    result = await agent.analyze(["XAU/USD", "EUR/USD", "GBP/JPY", "FOO"])
    assets = {s.asset for s in result}
    assert assets == {"XAU/USD", "EUR/USD"}
    fetched = {call.args[0] for call in agent._fetch_tone.await_args_list}
    assert fetched == {"gold price", "EURUSD"}


@pytest.mark.asyncio
async def test_analyze_no_overlap_returns_empty() -> None:
    agent = GdeltAgent(_make_config(), MagicMock(), _make_limiter())
    agent._fetch_tone = AsyncMock()  # type: ignore[method-assign]
    result = await agent.analyze(["AUD/USD", "EUR/JPY"])
    assert result == []
    agent._fetch_tone.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_drops_none_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bot.sentiment.agents.gdelt._REQUEST_SLEEP_S", 0.0)
    agent = GdeltAgent(_make_config(), MagicMock(), _make_limiter())

    async def selective(query: str) -> tuple[dict[str, Any] | None, str]:
        if query == "gold price":
            return {"sentiment": 0.3, "confidence": 0.7, "reasoning": "{asset} good"}, "ok"
        return None, "empty"  # everything else below-threshold or 429

    agent._fetch_tone = AsyncMock(side_effect=selective)  # type: ignore[method-assign]
    result = await agent.analyze(["XAU/USD", "XAG/USD", "USD/JPY"])
    assert len(result) == 1
    assert result[0].asset == "XAU/USD"
    assert "XAU/USD good" in result[0].reasoning


@pytest.mark.asyncio
async def test_analyze_uses_per_request_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """The agent must sleep between requests to dodge GDELT 429s."""
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    monkeypatch.setattr("bot.sentiment.agents.gdelt._REQUEST_SLEEP_S", 7.5)
    monkeypatch.setattr("bot.sentiment.agents.gdelt.asyncio.sleep", fake_sleep)
    agent = GdeltAgent(_make_config(), MagicMock(), _make_limiter())
    agent._fetch_tone = AsyncMock(  # type: ignore[method-assign]
        return_value=({"sentiment": 0.0, "confidence": 0.5, "reasoning": "x"}, "ok")
    )
    await agent.analyze(["XAU/USD", "XAG/USD", "EUR/USD"])
    # Three covered assets → two sleeps between them (no sleep before first)
    assert sleeps == [7.5, 7.5]


# ---------------------------------------------------------------------------
# _fetch_tone HTTP handling
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status: int, text: str) -> None:
        self.status = status
        self._text = text

    async def text(self) -> str:
        return self._text

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        pass


@pytest.mark.asyncio
async def test_fetch_tone_handles_429() -> None:
    session = MagicMock()
    session.get = MagicMock(return_value=_FakeResponse(429, "Too many"))
    agent = GdeltAgent(_make_config(), session, _make_limiter())
    assert await agent._fetch_tone("EURUSD") == (None, "http")


@pytest.mark.asyncio
async def test_fetch_tone_handles_html_response() -> None:
    """GDELT sometimes returns HTML 200s for malformed queries."""
    session = MagicMock()
    session.get = MagicMock(return_value=_FakeResponse(200, "<html><body>oops</body></html>"))
    agent = GdeltAgent(_make_config(), session, _make_limiter())
    assert await agent._fetch_tone("garbage") == (None, "parse")


@pytest.mark.asyncio
async def test_fetch_tone_happy_path() -> None:
    import json

    body = json.dumps(_timeline([1.0, 2.0, -1.0] * 5))
    session = MagicMock()
    session.get = MagicMock(return_value=_FakeResponse(200, body))
    agent = GdeltAgent(_make_config(), session, _make_limiter())
    scored, outcome = await agent._fetch_tone("EURUSD")
    assert outcome == "ok"
    assert scored is not None
    assert -1.0 <= scored["sentiment"] <= 1.0


# ---------------------------------------------------------------------------
# Cache behavior (2026-05-21 — GDELT TimelineTone hourly cache)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_cache_hit_skips_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Within TTL, a second analyze() call must not re-hit the network."""
    monkeypatch.setattr("bot.sentiment.agents.gdelt._REQUEST_SLEEP_S", 0.0)
    agent = GdeltAgent(_make_config(), MagicMock(), _make_limiter())
    agent._fetch_tone = AsyncMock(  # type: ignore[method-assign]
        return_value=({"sentiment": 0.3, "confidence": 0.7, "reasoning": "{asset} hit"}, "ok")
    )

    first = await agent.analyze(["XAU/USD", "EUR/USD"])
    second = await agent.analyze(["XAU/USD", "EUR/USD"])

    assert {s.asset for s in first} == {"XAU/USD", "EUR/USD"}
    assert {s.asset for s in second} == {"XAU/USD", "EUR/USD"}
    # Two assets fetched on the first call; zero on the second (all cache hits)
    assert agent._fetch_tone.await_count == 2


@pytest.mark.asyncio
async def test_analyze_cache_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    """After TTL, the cache entry is stale and a fresh fetch fires."""
    monkeypatch.setattr("bot.sentiment.agents.gdelt._REQUEST_SLEEP_S", 0.0)
    monkeypatch.setattr("bot.sentiment.agents.gdelt._CACHE_TTL_S", 100.0)
    fake_now = [1000.0]
    monkeypatch.setattr("bot.sentiment.agents.gdelt.time.monotonic", lambda: fake_now[0])

    agent = GdeltAgent(_make_config(), MagicMock(), _make_limiter())
    agent._fetch_tone = AsyncMock(  # type: ignore[method-assign]
        return_value=({"sentiment": 0.1, "confidence": 0.6, "reasoning": "x"}, "ok")
    )

    await agent.analyze(["XAU/USD"])
    fake_now[0] += 99  # within TTL
    await agent.analyze(["XAU/USD"])
    fake_now[0] += 5  # cumulative 104 > TTL=100 → stale
    await agent.analyze(["XAU/USD"])

    # Two real fetches: t=1000 (cold) and t=1104 (stale); t=1099 hit cache.
    assert agent._fetch_tone.await_count == 2


@pytest.mark.asyncio
async def test_analyze_failed_fetch_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """A None response (429, timeout, bad JSON) must not populate the cache,
    so the next scan retries instead of returning a stale-and-wrong None."""
    monkeypatch.setattr("bot.sentiment.agents.gdelt._REQUEST_SLEEP_S", 0.0)
    agent = GdeltAgent(_make_config(), MagicMock(), _make_limiter())
    agent._fetch_tone = AsyncMock(return_value=(None, "timeout"))  # type: ignore[method-assign]

    await agent.analyze(["XAU/USD", "EUR/USD"])
    await agent.analyze(["XAU/USD", "EUR/USD"])

    # Both scans fetched both assets — 4 total fetches, no cache contamination.
    assert agent._fetch_tone.await_count == 4
    assert agent._cache == {}


@pytest.mark.asyncio
async def test_analyze_mixed_cache_and_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """One asset hits cache, another asset misses → only the miss fetches."""
    monkeypatch.setattr("bot.sentiment.agents.gdelt._REQUEST_SLEEP_S", 0.0)
    agent = GdeltAgent(_make_config(), MagicMock(), _make_limiter())
    agent._fetch_tone = AsyncMock(  # type: ignore[method-assign]
        return_value=({"sentiment": 0.2, "confidence": 0.6, "reasoning": "x"}, "ok")
    )

    await agent.analyze(["XAU/USD"])  # caches XAU/USD
    agent._fetch_tone.reset_mock()
    await agent.analyze(["XAU/USD", "EUR/USD"])  # XAU/USD hits cache, EUR/USD fetches

    fetched = {call.args[0] for call in agent._fetch_tone.await_args_list}
    assert fetched == {"EURUSD"}  # only EUR/USD query went over the wire


def test_request_timeout_is_tight() -> None:
    """Regression guard: timeout must stay ≤ 15s to avoid stalling the asyncio loop."""
    from bot.sentiment.agents.gdelt import _REQUEST_TIMEOUT_S

    assert _REQUEST_TIMEOUT_S <= 15.0


@pytest.mark.asyncio
async def test_analyze_emits_aggregate_outcome_breakdown(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """One INFO summary per scan replaces per-query failure logs.

    The aggregate must include each outcome label that was hit so the operator
    can spot timeout/parse/connection patterns without grepping for tracebacks.
    """
    monkeypatch.setattr("bot.sentiment.agents.gdelt._REQUEST_SLEEP_S", 0.0)
    agent = GdeltAgent(_make_config(), MagicMock(), _make_limiter())

    async def mixed(query: str) -> tuple[dict[str, Any] | None, str]:
        if query == "gold price":
            return {"sentiment": 0.3, "confidence": 0.7, "reasoning": "x"}, "ok"
        if query == "silver price":
            return None, "timeout"
        return None, "empty"

    agent._fetch_tone = AsyncMock(side_effect=mixed)  # type: ignore[method-assign]

    with caplog.at_level("INFO", logger="bot.sentiment.agents.gdelt"):
        await agent.analyze(["XAU/USD", "XAG/USD", "EUR/USD"])

    summary_records = [r for r in caplog.records if "scored" in r.message]
    assert len(summary_records) == 1
    msg = summary_records[0].getMessage()
    assert "1/3 scored" in msg
    assert "fetched=3" in msg
    assert "ok=1" in msg
    assert "timeout=1" in msg
    assert "empty=1" in msg
