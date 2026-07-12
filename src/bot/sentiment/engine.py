"""SentimentEngine — orchestrates periodic multi-agent sentiment scans.

Lifecycle
---------
1. ``run()`` starts an async loop that fires every ``scan_interval_minutes``.
2. Each scan: NewsAgent + MacroAgent + FearGreedAgent + GdeltAgent +
   CentralBankAgent run concurrently
   via ``asyncio.gather``, returning RawSignals.
3. SentimentAggregator combines them into ConsensusSignals, escalating to
   Gemini when agents disagree significantly.
4. Results are cached in ``_current_scores`` (in-memory) and persisted to DB.
5. ``get_sentiment_scores()`` returns the latest in-memory snapshot instantly.

Thread safety: all state lives in the async event loop; no thread locks needed.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path

import aiohttp

from bot.sentiment.agents.central_bank import CentralBankAgent
from bot.sentiment.agents.fear_greed import FearGreedAgent
from bot.sentiment.agents.gdelt import GdeltAgent
from bot.sentiment.agents.macro import MacroAgent
from bot.sentiment.agents.news import NewsAgent
from bot.sentiment.aggregator import SentimentAggregator
from bot.sentiment.config import SentimentConfig
from bot.sentiment.db import SentimentDB
from bot.sentiment.models import ConsensusSignal, RawSignal
from bot.sentiment.rate_limiter import GroqRateLimiter

logger = logging.getLogger(__name__)


class SentimentEngine:
    """Orchestrates multi-agent sentiment scans and exposes ``get_sentiment_scores()``."""

    def __init__(self, config: SentimentConfig, db_path: Path | None = None) -> None:
        self._sentiment_config = config
        self._assets: list[str] = []  # populated in run()
        self._current_scores: dict[str, ConsensusSignal] = {}

        self._db = SentimentDB(db_path) if db_path else SentimentDB()
        self._limiter = GroqRateLimiter(
            tokens_per_minute=self._sentiment_config.groq_tokens_per_minute,
            requests_per_day=self._sentiment_config.groq_requests_per_day,
        )
        self._session: aiohttp.ClientSession | None = None
        # If set, scans are skipped while the callback returns True. Sentiment
        # HTTP I/O on the asyncio loop contends with the Kronos inference
        # thread, so we hold scans during reranks to avoid Pass 1 OHLC slip.
        self._rerank_busy: Callable[[], bool] | None = None

    def set_rerank_busy_callback(self, cb: Callable[[], bool]) -> None:
        """Inject a predicate that returns True while a TopK rerank is running.

        When set, ``_scan_once`` skips ticks where the callback returns True.
        """
        self._rerank_busy = cb

    # ------------------------------------------------------------------
    # Public API (called from main / TopK strategy)
    # ------------------------------------------------------------------

    def get_sentiment_scores(self) -> dict[str, ConsensusSignal]:
        """Return latest in-memory ConsensusSignals keyed by Twelve Data symbol.

        Returns an empty dict if no scan has completed yet.
        """
        return dict(self._current_scores)

    def get_score(self, asset: str) -> ConsensusSignal | None:
        """Return the latest ConsensusSignal for a single asset, or None."""
        return self._current_scores.get(asset)

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    async def run(self, assets: list[str]) -> None:
        """Start periodic sentiment scans for ``assets``.

        Call this from ``asyncio.create_task()`` in main.py.
        """
        self._assets = list(assets)
        self._db.init_db()
        self._session = aiohttp.ClientSession(
            headers={"User-Agent": "TradingBot/1.0 (financial research)"}
        )

        agents: list[NewsAgent | MacroAgent | FearGreedAgent | GdeltAgent | CentralBankAgent] = [
            NewsAgent(self._sentiment_config, self._session, self._limiter),
            MacroAgent(self._sentiment_config, self._session, self._limiter),
            FearGreedAgent(self._sentiment_config, self._session, self._limiter),
            GdeltAgent(self._sentiment_config, self._session, self._limiter),
            CentralBankAgent(self._sentiment_config, self._session, self._limiter),
        ]
        aggregator = SentimentAggregator(self._sentiment_config, self._session)

        interval_s = self._sentiment_config.scan_interval_minutes * 60
        logger.info(
            "SentimentEngine: started — %d assets, scan every %dm, Groq model=%s",
            len(self._assets),
            self._sentiment_config.scan_interval_minutes,
            self._sentiment_config.groq_model,
        )

        try:
            while True:
                await self._scan_once(agents, aggregator)
                await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            logger.info("SentimentEngine: cancelled")
            raise
        except Exception:
            logger.exception("SentimentEngine: unhandled error in run()")
            raise
        finally:
            if self._session is not None and not self._session.closed:
                await self._session.close()
            self._db.close()

    async def _scan_once(
        self,
        agents: list[NewsAgent | MacroAgent | FearGreedAgent | GdeltAgent | CentralBankAgent],
        aggregator: SentimentAggregator,
    ) -> None:
        """Run one full scan: fetch → aggregate → store."""
        if self._rerank_busy is not None and self._rerank_busy():
            logger.info("SentimentEngine: skipping scan — TopK rerank in progress")
            return
        logger.info("SentimentEngine: scanning %d assets…", len(self._assets))
        try:
            # All agents run concurrently
            results = await asyncio.gather(
                *[agent.analyze(self._assets) for agent in agents],
                return_exceptions=True,
            )

            raw_signals: list[RawSignal] = []
            for agent, result in zip(agents, results, strict=False):
                if isinstance(result, BaseException):
                    logger.warning("SentimentEngine: %s agent raised: %s", agent.name, result)
                else:
                    raw_signals.extend(result)

            if not raw_signals:
                logger.info("SentimentEngine: no raw signals from any agent")
                return

            consensus = await aggregator.aggregate(raw_signals, self._assets)
            self._current_scores.update(consensus)

            for signal in consensus.values():
                self._db.insert_signal(signal)

            logger.info(
                "SentimentEngine: scan complete — %d assets scored (%d escalated to Gemini)",
                len(consensus),
                sum(1 for s in consensus.values() if s.escalated),
            )

            # Log top signals at DEBUG level
            for asset, sig in sorted(
                consensus.items(), key=lambda kv: abs(kv[1].sentiment), reverse=True
            )[:5]:
                logger.debug(
                    "  %s: sentiment=%+.2f conf=%.2f src=%s%s",
                    asset,
                    sig.sentiment,
                    sig.confidence,
                    "+".join(sig.sources),
                    " [escalated]" if sig.escalated else "",
                )

        except Exception:
            logger.exception("SentimentEngine: _scan_once failed")
