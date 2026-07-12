"""Shared pytest fixtures.

Currently this only contains an autouse patch that zeros the IG REST retry
backoff so transient-error paths don't add real seconds to test runtime.
The production constants are validated separately by the dedicated retry
tests in ``tests/test_ig_client.py`` via explicit patches.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _fast_ig_retry_backoff() -> object:
    """Patch IG REST retry delays to zero for the duration of every test.

    Without this, any test that mocks a 5xx / 403 / 429 response would burn
    ~60 s of real time exhausting the 6-step exponential backoff in
    ``IGHttp.request``.  Tests that need to *prove* the backoff fires can
    still inspect call counts and ``asyncio.sleep`` patches.

    The retry constants live in ``ig_http`` (moved there from ``ig_client``
    when the transport layer was extracted).
    """
    with (
        patch("bot.execution.ig_http._RETRY_INITIAL_DELAY_S", 0.0),
        patch("bot.execution.ig_http._RETRY_MAX_DELAY_S", 0.0),
        patch("bot.execution.ig_http._RETRY_JITTER_S", 0.0),
    ):
        yield
