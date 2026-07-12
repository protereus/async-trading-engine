"""Tests for ``BaseAgent._call_llm`` — Groq is the sole LLM provider.

History:
  - introduced Groq as the only provider.
  - added a Cerebras-first provider chain.
  - 2026-06-01 a head-to-head bench (gpt-oss-120b / zai-glm-4.7 on Cerebras vs
    llama-3.3-70b on Groq) showed Cerebras produced no usable output for our
    sentiment prompt.  Cerebras was removed entirely.

These tests pin the post-removal behaviour so a future regression that
re-introduces a Cerebras URL or stale config field fails loud rather than
silently 404'ing in production.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.sentiment.agents.news import NewsAgent
from bot.sentiment.config import SentimentConfig
from bot.sentiment.rate_limiter import GroqRateLimiter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kwargs: object) -> SentimentConfig:
    defaults: dict[str, Any] = dict(
        groq_api="groq-key",
        twelve_data_api="td-key",
        finnhub_api="fh-key",
    )
    defaults.update(kwargs)
    return SentimentConfig(**defaults)


def _make_limiter() -> GroqRateLimiter:
    lim = GroqRateLimiter()
    lim.acquire = AsyncMock()  # type: ignore[method-assign]
    lim.record_request = MagicMock()  # type: ignore[method-assign]
    return lim


def _good_response_body(signals: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"signals": signals}),
                }
            }
        ]
    }


class _FakeResponse:
    """Async-context-manager stand-in for aiohttp.ClientResponse."""

    def __init__(self, status: int, body: dict[str, Any] | str) -> None:
        self.status = status
        self._body = body

    async def json(self) -> dict[str, Any]:
        assert isinstance(self._body, dict)
        return self._body

    async def text(self) -> str:
        return self._body if isinstance(self._body, str) else json.dumps(self._body)

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        pass


def _session_with_responses(*responses: _FakeResponse) -> MagicMock:
    """A session whose `.post(url, ...)` returns the given responses in order.

    Also exposes a ``calls`` attribute listing the URLs hit, in order.
    """
    session = MagicMock()
    session.calls = []  # type: ignore[attr-defined]
    iter_resp = iter(responses)

    def _post(url: str, **kwargs: Any) -> _FakeResponse:
        session.calls.append(url)  # type: ignore[attr-defined]
        return next(iter_resp)

    session.post = MagicMock(side_effect=_post)
    return session


def _agent(config: SentimentConfig, session: MagicMock) -> NewsAgent:
    return NewsAgent(config, session, _make_limiter())


_SIGNALS = [{"asset": "EUR/USD", "sentiment": 0.4, "confidence": 0.8, "reasoning": "bull"}]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_groq_success_returns_signals() -> None:
    """Groq 200 + valid JSON → signals list passed through to caller."""
    config = _make_config()
    session = _session_with_responses(_FakeResponse(200, _good_response_body(_SIGNALS)))
    agent = _agent(config, session)

    result = await agent._call_llm("prompt")
    assert result == _SIGNALS
    assert len(session.calls) == 1
    assert "groq.com" in session.calls[0]


# ---------------------------------------------------------------------------
# Cerebras-removal regression guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_never_calls_cerebras_url() -> None:
    """No agent call should ever land on cerebras.ai after the 2026-06-01
    removal — the head-to-head bench proved Cerebras produced zero usable
    signals on our sentiment prompt.  This test would catch a regression
    that re-introduces a Cerebras URL (e.g. someone bringing back the
    provider chain) by checking every HTTP call is Groq-only across the
    happy path and an HTTP-failure path."""
    config = _make_config()
    # Happy: 200 from Groq
    session_ok = _session_with_responses(_FakeResponse(200, _good_response_body(_SIGNALS)))
    await _agent(config, session_ok)._call_llm("prompt")
    assert all("groq.com" in u for u in session_ok.calls)
    assert all("cerebras" not in u for u in session_ok.calls)

    # Failure: 429 from Groq — agent must NOT try cerebras as a fallback
    session_fail = _session_with_responses(_FakeResponse(429, "rate limited"))
    result = await _agent(config, session_fail)._call_llm("prompt")
    assert result == []
    assert all("groq.com" in u for u in session_fail.calls)
    assert all("cerebras" not in u for u in session_fail.calls)


def test_config_has_no_cerebras_fields() -> None:
    """SentimentConfig must NOT carry cerebras_* fields after the removal —
    the previous defaults silently kept a dead model id alive across the
    → 2026-06-01 window.  Field set is pinned so a future code
    change has to revisit this test to re-add cerebras."""
    cfg = _make_config()
    # Field-level assertions: presence checks are dataclass-attribute checks.
    assert not hasattr(cfg, "cerebras_api")
    assert not hasattr(cfg, "cerebras_model")
    assert not hasattr(cfg, "llm_provider_order")


# ---------------------------------------------------------------------------
# Failure paths — Groq returns the empty list (never raises) so the caller
# treats the agent as "no signals this scan".
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_groq_429_returns_empty() -> None:
    """Rate-limit response → empty list.  Caller treats it as no-signals
    rather than crashing the sentiment scan."""
    config = _make_config()
    session = _session_with_responses(_FakeResponse(429, "rate limited"))
    result = await _agent(config, session)._call_llm("prompt")
    assert result == []


@pytest.mark.asyncio
async def test_groq_500_returns_empty() -> None:
    """Server error → empty list (same contract as 429)."""
    config = _make_config()
    session = _session_with_responses(_FakeResponse(500, "internal server error"))
    result = await _agent(config, session)._call_llm("prompt")
    assert result == []


@pytest.mark.asyncio
async def test_groq_network_error_returns_empty() -> None:
    """Connection-level error → empty list, no exception propagates."""
    config = _make_config()
    session = MagicMock()
    session.post = MagicMock(side_effect=ConnectionError("dns fail"))
    result = await _agent(config, session)._call_llm("prompt")
    assert result == []


@pytest.mark.asyncio
async def test_empty_signals_list_returned_unchanged() -> None:
    """Groq legitimately returns {\"signals\": []} when no relevant news was
    found.  Agent must return that empty list (not [] = "failure" — same
    in shape but different in semantics)."""
    config = _make_config()
    session = _session_with_responses(_FakeResponse(200, _good_response_body([])))
    result = await _agent(config, session)._call_llm("prompt")
    assert result == []
    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_unparseable_content_returns_empty() -> None:
    """Groq returns 200 but content isn't valid JSON → empty list, no crash."""
    config = _make_config()
    session = _session_with_responses(
        _FakeResponse(200, {"choices": [{"message": {"content": "not json at all"}}]})
    )
    result = await _agent(config, session)._call_llm("prompt")
    assert result == []


@pytest.mark.asyncio
async def test_missing_message_field_returns_empty() -> None:
    """Groq returns a shape with no ``choices[0].message.content`` →
    empty list, no IndexError / KeyError surfaces."""
    config = _make_config()
    session = _session_with_responses(_FakeResponse(200, {"choices": []}))
    result = await _agent(config, session)._call_llm("prompt")
    assert result == []


@pytest.mark.asyncio
async def test_skips_when_no_groq_key() -> None:
    """Empty groq_api → no HTTP call attempted, returns []."""
    config = _make_config(groq_api="")
    session = MagicMock()
    session.post = MagicMock()
    result = await _agent(config, session)._call_llm("prompt")
    assert result == []
    session.post.assert_not_called()
