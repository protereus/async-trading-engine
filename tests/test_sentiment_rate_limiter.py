"""Tests for GroqRateLimiter."""

from __future__ import annotations

import time

import pytest

from bot.sentiment.rate_limiter import GroqRateLimiter


class TestGroqRateLimiter:
    def test_initial_state(self) -> None:
        limiter = GroqRateLimiter(tokens_per_minute=6000, requests_per_day=6000)
        assert limiter.daily_requests_remaining == 6000

    def test_record_request_decrements_counter(self) -> None:
        limiter = GroqRateLimiter(tokens_per_minute=6000, requests_per_day=6000)
        limiter.record_request()
        limiter.record_request()
        assert limiter.daily_requests_remaining == 5998

    @pytest.mark.asyncio
    async def test_acquire_within_budget(self) -> None:
        limiter = GroqRateLimiter(tokens_per_minute=6000, requests_per_day=6000)
        # Should not block for a small token request
        start = time.monotonic()
        await limiter.acquire(100)
        elapsed = time.monotonic() - start
        assert elapsed < 0.5  # no meaningful sleep

    @pytest.mark.asyncio
    async def test_acquire_depletes_bucket(self) -> None:
        limiter = GroqRateLimiter(tokens_per_minute=6000, requests_per_day=6000)
        # Consume almost all tokens
        await limiter.acquire(5900)
        # Bucket should have ~100 tokens left; requesting 50 more should be fast
        start = time.monotonic()
        await limiter.acquire(50)
        assert time.monotonic() - start < 0.5

    @pytest.mark.asyncio
    async def test_daily_limit_tracks_requests(self) -> None:
        limiter = GroqRateLimiter(tokens_per_minute=6000, requests_per_day=10)
        for _ in range(5):
            limiter.record_request()
        assert limiter.daily_requests_remaining == 5

    def test_record_request_does_not_go_below_zero(self) -> None:
        limiter = GroqRateLimiter(tokens_per_minute=6000, requests_per_day=2)
        limiter.record_request()
        limiter.record_request()
        limiter.record_request()
        # remaining is clamped at 0
        assert limiter.daily_requests_remaining == 0
