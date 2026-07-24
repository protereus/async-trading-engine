"""Tests for Lifecycle._run_scale_drift_check — the D4 drift-check loop body.

The pure drift math is covered by tests/test_scale_guard.py.  These tests
cover the orchestration the loop adds on top of it: severity → action
(CRITICAL fires an operator alert; WARN/OK stay silent) and the skip paths
(missing epic / candle, market-details failure, non-positive quote).

The method is exercised by calling it unbound against a stub ``self`` that
exposes only the four attributes it touches, so no real Lifecycle (and no
network / GPU) is needed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.core.lifecycle import Lifecycle
from bot.core.models import Candle


def _candle(symbol: str, close: float) -> Candle:
    return Candle(
        timestamp=1_779_000_000_000,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=0.0,
        symbol=symbol,
        is_confirmed=True,
    )


def _bot(
    *,
    epic_map: dict[str, str],
    candle: Candle | None,
    market_details: dict[str, object],
) -> MagicMock:
    """Build a stub ``self`` exposing only what _run_scale_drift_check uses.

    ``_candle_epic_map`` is a real dict so symbols absent from it fall through
    the ``epic is None`` guard exactly as in production; only the one symbol
    under test reaches the IG call.
    """
    bot = MagicMock()
    bot._ctx.candle_epic_map = epic_map
    bot._ctx.store.get_latest_candle = MagicMock(return_value=candle)
    bot._ctx.ig_client.fetch_market_details = AsyncMock(return_value=market_details)
    bot._ctx.alerter.send_error = AsyncMock(return_value=True)
    return bot


class TestRunScaleDriftCheck:
    @pytest.mark.asyncio
    async def test_critical_drift_fires_one_alert(self) -> None:
        """A configured-vs-real scale mismatch of +30 % is CRITICAL → one alert.

        XAU/USD is the one explicitly-scaled symbol the drift loop iterates
        (``IG_SCALED_SYMBOLS``); a real scale of 1.30 (candle 100 vs IG mid 130)
        against its configured 1.0 is the classic ETF-proxy mis-scale bug
        class."""
        bot = _bot(
            epic_map={"XAU/USD": "CS.D.USCGC.TODAY.IP"},
            candle=_candle("XAU/USD", 100.0),
            # real_scale = 130 / 100 = 1.30 vs configured 1.0 → +30 %
            market_details={"snapshot": {"bid": 129.99, "offer": 130.01}},
        )
        await Lifecycle._run_scale_drift_check(bot)
        bot._ctx.alerter.send_error.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_warn_drift_does_not_alert(self) -> None:
        """+10 % drift → WARN → logged but no operator alert.  XAU/USD with a
        real scale of 1.10 (candle 100 vs IG mid 110) against configured 1.0."""
        bot = _bot(
            epic_map={"XAU/USD": "CS.D.USCGC.TODAY.IP"},
            candle=_candle("XAU/USD", 100.0),
            # real_scale = 110 / 100 = 1.10 vs configured 1.0 → +10 %
            market_details={"snapshot": {"bid": 109.5, "offer": 110.5}},
        )
        await Lifecycle._run_scale_drift_check(bot)
        bot._ctx.alerter.send_error.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_clean_symbol_does_not_alert(self) -> None:
        """XAU/USD is IG-native (2026-06-19): the candle store is already in
        IG-level units (scale 1.0), so candle ≈ IG mid → real_scale ≈ 1 ==
        configured → no drift, no alert."""
        bot = _bot(
            epic_map={"XAU/USD": "CS.D.USCGC.TODAY.IP"},
            candle=_candle("XAU/USD", 4466.75),
            # mid 4466.75 = 4466.75 × 1.0 → real_scale == expected → no drift.
            market_details={"snapshot": {"bid": 4466.25, "offer": 4467.25}},
        )
        await Lifecycle._run_scale_drift_check(bot)
        bot._ctx.alerter.send_error.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_market_details_failure_is_swallowed(self) -> None:
        """A market-details fetch error must not propagate or alert."""
        bot = _bot(
            epic_map={"XAU/USD": "CS.D.USCGC.TODAY.IP"},
            candle=_candle("XAU/USD", 130.65),
            market_details={},
        )
        bot._ctx.ig_client.fetch_market_details = AsyncMock(side_effect=RuntimeError("IG 503"))
        await Lifecycle._run_scale_drift_check(bot)  # must not raise
        bot._ctx.alerter.send_error.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_candle_skips_before_ig_call(self) -> None:
        """No candle in the store → symbol skipped, no IG call, no alert."""
        bot = _bot(
            epic_map={"XAU/USD": "CS.D.USCGC.TODAY.IP"},
            candle=None,
            market_details={"snapshot": {"bid": 8855.0, "offer": 8855.4}},
        )
        await Lifecycle._run_scale_drift_check(bot)
        bot._ctx.ig_client.fetch_market_details.assert_not_awaited()
        bot._ctx.alerter.send_error.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_positive_quote_skips(self) -> None:
        """A zero bid/offer is a feed problem, not drift → skip, no alert."""
        bot = _bot(
            epic_map={"XAU/USD": "CS.D.USCGC.TODAY.IP"},
            candle=_candle("XAU/USD", 130.65),
            market_details={"snapshot": {"bid": 0.0, "offer": 0.0}},
        )
        await Lifecycle._run_scale_drift_check(bot)
        bot._ctx.alerter.send_error.assert_not_awaited()
