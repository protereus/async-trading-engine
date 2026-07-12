"""SentimentConfig — extracted from BotConfig at engine startup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot.config import BotConfig


@dataclass
class SentimentConfig:
    groq_api: str
    gemini_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    # Groq free-tier limits.  A head-to-head bench (zai-glm-4.7 /
    # gpt-oss-120b on Cerebras vs llama-3.3-70b on Groq) showed Cerebras
    # produced no usable output for the sentiment prompt; Cerebras was
    # removed and Groq is the sole provider.  ~6 agent calls × 48 scans/day ≈ 288 calls/day
    # at ~700 tokens each → well within Groq's free-tier 1k RPD / 100k TPD
    # envelope, so the conservative 6k/6k caps below are pure operational
    # ceilings, not Groq-imposed limits.
    groq_tokens_per_minute: int = 6_000
    groq_requests_per_day: int = 6_000
    # Scan cadence
    scan_interval_minutes: int = 30
    # Gemini escalation hard cap (rolling 1-hour window)
    escalation_max_per_hour: int = 20
    escalation_model: str = "gemini-2.5-flash"
    # Escalation triggers
    escalation_disagreement_threshold: float = 0.5  # max - min > this → escalate
    escalation_confidence_threshold: float = 0.4  # avg confidence < this → escalate
    # Twelve Data key re-used for macro calendar endpoint
    twelve_data_api: str = ""
    # Finnhub API key (macro calendar; legacy news/social source)
    finnhub_api: str = ""
    # EODHD API key — NewsAgent's per-symbol financial-news source (2026-06-03)
    eodhd_api: str = ""
    # FinBERT local headline pre-filter — when True, NewsAgent runs
    # each headline through the ONNX-quantised classifier and drops neutrals
    # before LLM dispatch.  Requires `uv sync --extra finbert` for the deps.
    finbert_enabled: bool = False
    # Headlines with |finbert_score| below this are treated as ambiguous and
    # passed to the LLM unchanged; confident ones are kept verbatim.
    finbert_threshold: float = 0.4

    @classmethod
    def from_bot_config(cls, cfg: BotConfig) -> SentimentConfig:
        """Build from the app-wide ``BotConfig``.

        Owns the ``BotConfig`` -> ``SentimentConfig`` field mapping (including
        renames like ``finhub_io_api`` -> ``finnhub_api``) so the wiring site
        doesn't carry it.
        """
        return cls(
            groq_api=cfg.groq_api,
            gemini_api_key=cfg.gemini_api_key,
            scan_interval_minutes=cfg.sentiment_scan_interval_minutes,
            escalation_max_per_hour=cfg.sentiment_escalation_max_per_hour,
            twelve_data_api=cfg.twelve_data_api,
            finnhub_api=cfg.finhub_io_api,
            eodhd_api=cfg.eodhd_api,
            finbert_enabled=cfg.finbert_enabled,
        )
