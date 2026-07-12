"""Sentiment data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class RawSignal:
    """Sentiment output from a single agent for a single asset."""

    asset: str  # Canonical bot symbol (bot_key), e.g. "EUR/USD" or "XAU/USD"
    sentiment: float  # -1.0 (very bearish) … +1.0 (very bullish)
    confidence: float  # 0.0 … 1.0
    reasoning: str  # one-sentence explanation
    source: str  # agent name: "news" | "social" | "macro"
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ConsensusSignal:
    """Aggregated sentiment for one asset from all agents."""

    asset: str
    sentiment: float  # weighted average of raw signals
    confidence: float  # mean confidence across sources
    agreement: float  # 1 − normalised range; 1.0 = perfect agreement
    sources: list[str]  # which agents contributed
    escalated: bool  # True if Gemini was consulted
    reasoning: str  # summary reasoning
    scored_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    # Per-agent breakdown — used by the sentiment-edge measurement harness
    # to bucket signals by slow-decaying (central_bank, macro) vs
    # fast-decaying (news, social, fear_greed, gdelt) source.  Always a
    # dict; missing agents simply omit their keys, which is the honest
    # "coverage<6" signal the harness logs into signal_history.
    per_agent: dict[str, float] = field(default_factory=dict)
