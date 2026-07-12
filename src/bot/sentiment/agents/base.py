"""BaseAgent — Groq LLM inference for sentiment agents.

History:
  - introduced Groq as the sole LLM provider.
  - (2026-04) added Cerebras as the *primary* with Groq as the
    HTTP-429/5xx fallback — Cerebras advertised a 1 M tok/day free tier
    versus Groq's ~100 k.
  - 2026-06-01 head-to-head bench against the actual account models
    available on Cerebras (``gpt-oss-120b`` Production, ``zai-glm-4.7``
    Preview) showed neither model produced usable output for our
    sentiment prompt — gpt-oss is a reasoning model that emits
    ``message.reasoning`` rather than ``message.content``; zai-glm
    consumed the full 1024-token budget without populating the signals
    array.  Cerebras was removed entirely; Groq is the sole provider
    again.

Groq's daily-request limiter is applied around every call to keep the
free-tier quota intact.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import aiohttp

from bot.sentiment.models import RawSignal

if TYPE_CHECKING:
    from bot.sentiment.config import SentimentConfig
    from bot.sentiment.rate_limiter import GroqRateLimiter

logger = logging.getLogger(__name__)

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

_SYSTEM_PROMPT = (
    "You are a financial market sentiment analyst. "
    'Analyze the text provided and return a JSON object with a single key "signals" '
    "whose value is an array. "
    "Each array element must have exactly these keys: "
    '"asset" (string), "sentiment" (float -1.0 to 1.0, bearish to bullish), '
    '"confidence" (float 0.0 to 1.0), "reasoning" (one sentence). '
    "Only include assets where the text contains clear signals. "
    'Return {"signals": []} if no relevant signals are found.'
)


class BaseAgent(ABC):
    """Abstract base for sentiment agents.

    Subclasses implement ``analyze(assets)`` which fetches domain-specific
    data and passes it to ``_call_llm`` for LLM inference.
    """

    name: str = "base"

    def __init__(
        self,
        config: SentimentConfig,
        session: aiohttp.ClientSession,
        limiter: GroqRateLimiter,
    ) -> None:
        self._config = config
        self._session = session
        self._limiter = limiter

    # ------------------------------------------------------------------
    # Multi-provider LLM inference
    # ------------------------------------------------------------------

    async def _call_llm(
        self, user_prompt: str, estimated_tokens: int = 800
    ) -> list[dict[str, Any]]:
        """Send ``user_prompt`` to Groq and return the parsed signals list.

        Returns an empty list on HTTP failure, missing API key, or unparseable
        response (never raises).  Rate-limited via ``GroqRateLimiter``.
        """
        if not self._config.groq_api:
            logger.warning(
                "%s: groq_api is empty — check sentiment config",
                self.name,
            )
            return []

        await self._limiter.acquire(estimated_tokens)

        headers = {
            "Authorization": f"Bearer {self._config.groq_api}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self._config.groq_model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
        }
        try:
            timeout = aiohttp.ClientTimeout(total=30.0)
            async with self._session.post(
                _GROQ_URL, json=body, headers=headers, timeout=timeout
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.warning("%s: groq HTTP %d: %.200s", self.name, resp.status, text)
                    return []
                data: dict[str, Any] = await resp.json()
        except Exception:
            logger.exception("%s: groq request failed", self.name)
            return []

        self._limiter.record_request()

        try:
            content: str = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            logger.warning(
                "%s: unexpected groq response structure: %s",
                self.name,
                str(data)[:200],
            )
            return []

        return self._parse_signals(content)

    def _parse_signals(self, content: str) -> list[dict[str, Any]]:
        """Extract a JSON array of signal dicts from the LLM content string."""
        # The model returns a JSON object wrapping the array (response_format=json_object)
        # Common patterns: {"signals": [...]} or {"results": [...]} or just [...]
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract a JSON array with regex as fallback
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning("%s: failed to parse Groq JSON: %.200s", self.name, content)
                    return []
            else:
                logger.warning("%s: no JSON array in Groq response: %.200s", self.name, content)
                return []

        # Unwrap if the model returned {"signals": [...]} or similar
        if isinstance(parsed, dict):
            for key in ("signals", "results", "assets", "data", "sentiment"):
                if isinstance(parsed.get(key), list):
                    parsed = parsed[key]
                    break
            else:
                # Last resort: grab the first list value
                for v in parsed.values():
                    if isinstance(v, list):
                        parsed = v
                        break
                else:
                    logger.warning("%s: Groq response is a dict with no list value", self.name)
                    return []

        if not isinstance(parsed, list):
            return []

        valid = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                valid.append(
                    {
                        "asset": str(item["asset"]),
                        "sentiment": float(item["sentiment"]),
                        "confidence": float(item["confidence"]),
                        "reasoning": str(item.get("reasoning", "")),
                    }
                )
            except (KeyError, ValueError, TypeError):
                logger.debug("%s: skipping malformed signal item: %s", self.name, item)
        return valid

    def _dict_to_raw_signal(self, d: dict[str, Any]) -> RawSignal:
        return RawSignal(
            asset=d["asset"],
            sentiment=max(-1.0, min(1.0, d["sentiment"])),
            confidence=max(0.0, min(1.0, d["confidence"])),
            reasoning=d["reasoning"],
            source=self.name,
            fetched_at=datetime.now(UTC),
        )

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def analyze(self, assets: list[str]) -> list[RawSignal]:
        """Fetch source data and return sentiment signals for ``assets``."""
        ...
