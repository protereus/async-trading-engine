"""Tests for bot.data.ig_history — the D2 IG hourly backfill helper."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from bot.core.models import Candle
from bot.data.ig_history import IG_BACKFILL_TIMEFRAME, fetch_ig_hourly_backfill


def _candle(symbol: str, ts: int, close: float) -> Candle:
    return Candle(
        symbol=symbol,
        timestamp=ts,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=0.0,
        is_confirmed=True,
    )


class TestFetchIgHourlyBackfill:
    @pytest.mark.asyncio
    async def test_relabels_epic_to_candle_symbol(self) -> None:
        """fetch_ohlcv returns candles keyed by EPIC; the helper must relabel
        them to the canonical candle symbol so they land in the right store
        slot."""
        epic = "CC.D.CL.USS.IP"
        raw = [
            _candle(epic, 1_779_000_000_000, 8800.0),
            _candle(epic, 1_779_003_600_000, 8810.0),
        ]
        client = AsyncMock()
        client.fetch_ohlcv = AsyncMock(return_value=raw)

        out = await fetch_ig_hourly_backfill(client, "USO", epic, limit=400)

        assert len(out) == 2
        assert all(c.symbol == "USO" for c in out)
        # Prices / timestamps preserved exactly — only the key changed
        assert out[0].close == 8800.0
        assert out[1].timestamp == 1_779_003_600_000
        # Fetched at HOUR resolution per the REST-probe finding
        client.fetch_ohlcv.assert_awaited_once_with(epic, IG_BACKFILL_TIMEFRAME, limit=400)

    @pytest.mark.asyncio
    async def test_empty_result_returns_empty_list(self) -> None:
        client = AsyncMock()
        client.fetch_ohlcv = AsyncMock(return_value=[])
        out = await fetch_ig_hourly_backfill(client, "UNG", "CC.D.NG.USS.IP", limit=400)
        assert out == []

    @pytest.mark.asyncio
    async def test_fetch_error_degrades_to_empty(self) -> None:
        """A fetch failure must not raise — the caller warms from live ticks
        instead.  Matches how the TD / yfinance feeds degrade."""
        client = AsyncMock()
        client.fetch_ohlcv = AsyncMock(side_effect=RuntimeError("IG 503"))
        out = await fetch_ig_hourly_backfill(client, "USO", "CC.D.CL.USS.IP", limit=400)
        assert out == []

    @pytest.mark.asyncio
    async def test_backfill_timeframe_is_hourly(self) -> None:
        """Guard the resolution choice: MINUTE would blow the weekly allowance
        (24k points per symbol for a 400h context vs 400 at HOUR)."""
        assert IG_BACKFILL_TIMEFRAME == "1h"
