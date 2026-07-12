"""SentimentAggregator — combines raw signals from all agents into ConsensusSignals.

Escalation to Gemini fires when:
  - max_sentiment − min_sentiment > escalation_disagreement_threshold (agent disagreement), OR
  - mean confidence across agents < escalation_confidence_threshold

Both conditions subject to rolling 1-hour hard cap (escalation_max_per_hour).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import aiohttp

from bot.sentiment.models import ConsensusSignal, RawSignal

if TYPE_CHECKING:
    from bot.sentiment.config import SentimentConfig

logger = logging.getLogger(__name__)

_GEMINI_GENERATE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


class SentimentAggregator:
    """Aggregates RawSignals into ConsensusSignals with optional Gemini escalation."""

    def __init__(self, config: SentimentConfig, session: aiohttp.ClientSession) -> None:
        self._config = config
        self._session = session
        # Rolling window of escalation timestamps (monotonic seconds)
        self._escalation_times: deque[float] = deque()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def aggregate(
        self,
        raw_signals: list[RawSignal],
        assets: list[str],
    ) -> dict[str, ConsensusSignal]:
        """Aggregate raw signals into one ConsensusSignal per asset.

        Assets with no raw signals are excluded from the result.
        """
        # Group by asset
        by_asset: dict[str, list[RawSignal]] = {a: [] for a in assets}
        for sig in raw_signals:
            if sig.asset in by_asset:
                by_asset[sig.asset].append(sig)

        results: dict[str, ConsensusSignal] = {}
        escalation_tasks = []

        for asset, signals in by_asset.items():
            if not signals:
                continue

            consensus = self._compute_consensus(asset, signals, escalated=False)

            if self._should_escalate(signals, consensus):
                escalation_tasks.append((asset, signals, consensus))
            else:
                results[asset] = consensus

        # Run escalations concurrently
        if escalation_tasks:
            escalated = await asyncio.gather(
                *[
                    self._escalate(asset, signals, base_consensus)
                    for asset, signals, base_consensus in escalation_tasks
                ]
            )
            for consensus in escalated:
                results[consensus.asset] = consensus

        return results

    # ------------------------------------------------------------------
    # Consensus computation
    # ------------------------------------------------------------------

    def _compute_consensus(
        self,
        asset: str,
        signals: list[RawSignal],
        escalated: bool,
        override_sentiment: float | None = None,
        override_reasoning: str | None = None,
    ) -> ConsensusSignal:
        sentiments = [s.sentiment for s in signals]
        confidences = [s.confidence for s in signals]

        # Confidence-weighted average sentiment
        total_conf = sum(confidences)
        if total_conf > 0:
            pairs = zip(sentiments, confidences, strict=False)
            weighted_sentiment = sum(s * c for s, c in pairs) / total_conf
        else:
            weighted_sentiment = sum(sentiments) / len(sentiments)

        mean_confidence = sum(confidences) / len(confidences)

        # Agreement: 1 − (range / 2), clamped to [0, 1]
        sentiment_range = max(sentiments) - min(sentiments) if len(sentiments) > 1 else 0.0
        agreement = max(0.0, 1.0 - sentiment_range / 2.0)

        sources = list({s.source for s in signals})
        # Per-agent breakdown for the sentiment-edge measurement harness.
        # If an agent emitted multiple signals for the same asset (shouldn't
        # happen in current code, but be defensive), keep the last.
        per_agent: dict[str, float] = {s.source: s.sentiment for s in signals}
        reasoning = (
            override_reasoning
            if override_reasoning is not None
            else "; ".join(s.reasoning for s in signals if s.reasoning)
        )

        return ConsensusSignal(
            asset=asset,
            sentiment=override_sentiment if override_sentiment is not None else weighted_sentiment,
            confidence=mean_confidence,
            agreement=agreement,
            sources=sources,
            escalated=escalated,
            reasoning=reasoning[:500],
            scored_at=datetime.now(UTC),
            per_agent=per_agent,
        )

    # ------------------------------------------------------------------
    # Escalation
    # ------------------------------------------------------------------

    def _should_escalate(self, signals: list[RawSignal], consensus: ConsensusSignal) -> bool:
        """Return True if Gemini escalation should be attempted."""
        if not self._config.gemini_api_key:
            return False
        if len(signals) < 2:
            return False

        sentiments = [s.sentiment for s in signals]
        disagreement = max(sentiments) - min(sentiments)
        avg_confidence = consensus.confidence

        return (
            disagreement > self._config.escalation_disagreement_threshold
            or avg_confidence < self._config.escalation_confidence_threshold
        ) and self._try_reserve_escalation_slot()

    def _try_reserve_escalation_slot(self) -> bool:
        """Return True and record the escalation if within the hourly cap."""
        now = time.monotonic()
        # Remove entries older than 1 hour
        while self._escalation_times and now - self._escalation_times[0] > 3600:
            self._escalation_times.popleft()

        if len(self._escalation_times) >= self._config.escalation_max_per_hour:
            logger.info(
                "Sentiment: escalation cap reached (%d/hr)",
                self._config.escalation_max_per_hour,
            )
            return False

        self._escalation_times.append(now)
        return True

    async def _escalate(
        self,
        asset: str,
        signals: list[RawSignal],
        base_consensus: ConsensusSignal,
    ) -> ConsensusSignal:
        """Call Gemini to resolve conflicting agent signals."""
        sentiments = [s.sentiment for s in signals]
        raw_disagreement = max(sentiments) - min(sentiments) if sentiments else 0.0
        logger.info(
            "Sentiment: escalating %s to Gemini (raw_disagreement=%.2f, confidence=%.2f)",
            asset,
            raw_disagreement,
            base_consensus.confidence,
        )

        agent_summaries = "\n".join(
            f"  {s.source}: sentiment={s.sentiment:+.2f}, "
            f"confidence={s.confidence:.2f}, reasoning={s.reasoning}"
            for s in signals
        )
        user_prompt = (
            f"Multiple financial sentiment agents disagree about {asset}:\n\n"
            f"{agent_summaries}\n\n"
            f"As a senior market analyst, weigh the evidence and produce a single "
            f"consensus assessment. sentiment is a float in [-1.0, 1.0] (negative = "
            f"bearish, positive = bullish). confidence is a float in [0.0, 1.0]."
        )

        try:
            result = await self._call_gemini(user_prompt)
            if result:
                return self._compute_consensus(
                    asset,
                    signals,
                    escalated=True,
                    override_sentiment=float(result["sentiment"]),
                    override_reasoning=str(result.get("reasoning", "")),
                )
        except Exception:
            logger.exception("Sentiment: Gemini escalation failed for %s", asset)

        return base_consensus

    async def _call_gemini(self, user_prompt: str) -> dict[str, Any] | None:
        url = _GEMINI_GENERATE_URL.format(model=self._config.escalation_model)
        headers = {
            "x-goog-api-key": self._config.gemini_api_key,
            "content-type": "application/json",
        }
        body = {
            "system_instruction": {
                "parts": [
                    {
                        "text": (
                            "You are a financial sentiment analyst. Return a single "
                            "JSON object with keys sentiment (float in [-1, 1]), "
                            "confidence (float in [0, 1]), and reasoning (string)."
                        )
                    }
                ]
            },
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "object",
                    "properties": {
                        "sentiment": {"type": "number"},
                        "confidence": {"type": "number"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["sentiment", "confidence", "reasoning"],
                },
                "maxOutputTokens": 512,
                "temperature": 0.2,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        try:
            timeout = aiohttp.ClientTimeout(total=30.0)
            async with self._session.post(url, json=body, headers=headers, timeout=timeout) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.warning("Sentiment: Gemini HTTP %d: %.200s", resp.status, text)
                    return None
                data: Any = await resp.json()
        except Exception:
            logger.exception("Sentiment: Gemini request failed")
            return None

        try:
            content_text: str = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed: dict[str, Any] = json.loads(content_text)
            return parsed
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            logger.warning(
                "Sentiment: failed to parse Gemini response: %s — raw: %.200s",
                exc,
                json.dumps(data)[:200],
            )
            return None
