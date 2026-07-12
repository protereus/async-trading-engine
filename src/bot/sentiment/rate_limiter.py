"""Groq free-tier rate limiter.

Enforces two separate limits:
  - Token bucket: 6 000 tokens / minute (replenished continuously)
  - Daily request counter: 6 000 requests / day (resets at UTC midnight)
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


class GroqRateLimiter:
    """Async token-bucket rate limiter + daily request counter for Groq free tier."""

    def __init__(
        self,
        tokens_per_minute: int = 6_000,
        requests_per_day: int = 6_000,
    ) -> None:
        self._tokens_per_minute = tokens_per_minute
        self._requests_per_day = requests_per_day

        # Token bucket state
        self._available_tokens: float = float(tokens_per_minute)
        self._last_refill: float = time.monotonic()
        self._lock = asyncio.Lock()

        # Daily request counter
        self._day_requests: int = 0
        self._day_date: str = datetime.now(UTC).strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refill(self) -> None:
        """Replenish token bucket based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        tokens_to_add = elapsed * (self._tokens_per_minute / 60.0)
        self._available_tokens = min(
            float(self._tokens_per_minute),
            self._available_tokens + tokens_to_add,
        )
        self._last_refill = now

    def _reset_day_counter_if_needed(self) -> None:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if today != self._day_date:
            logger.debug("Groq: daily counter reset (new day: %s)", today)
            self._day_requests = 0
            self._day_date = today

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def daily_requests_remaining(self) -> int:
        self._reset_day_counter_if_needed()
        return max(0, self._requests_per_day - self._day_requests)

    async def acquire(self, estimated_tokens: int = 500) -> None:
        """Wait until the token bucket has capacity for ``estimated_tokens``."""
        async with self._lock:
            self._reset_day_counter_if_needed()
            if self._day_requests >= self._requests_per_day:
                logger.warning("Groq: daily request limit (%d) reached", self._requests_per_day)
                # Sleep until midnight UTC
                now_utc = datetime.now(UTC)
                seconds_until_midnight = (
                    86400 - now_utc.hour * 3600 - now_utc.minute * 60 - now_utc.second
                )
                await asyncio.sleep(seconds_until_midnight + 1)
                self._day_requests = 0
                self._day_date = datetime.now(UTC).strftime("%Y-%m-%d")

            # Wait for token bucket
            while True:
                self._refill()
                if self._available_tokens >= estimated_tokens:
                    self._available_tokens -= estimated_tokens
                    break
                wait_s = (estimated_tokens - self._available_tokens) / (
                    self._tokens_per_minute / 60.0
                )
                logger.debug("Groq: token bucket depleted — sleeping %.1fs", wait_s)
                await asyncio.sleep(wait_s)

    def record_request(self) -> None:
        """Increment the daily request counter (call after each successful API call)."""
        self._reset_day_counter_if_needed()
        self._day_requests += 1
