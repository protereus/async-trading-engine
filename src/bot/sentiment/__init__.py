"""Sentiment analysis overlay — multi-agent, Groq bulk inference, Gemini escalation."""

from bot.sentiment.engine import SentimentEngine
from bot.sentiment.models import ConsensusSignal, RawSignal

__all__ = ["SentimentEngine", "ConsensusSignal", "RawSignal"]
