"""In-memory data store for OHLCV candles."""

from __future__ import annotations

import logging
from collections import deque

from bot.core.models import Candle

logger = logging.getLogger(__name__)


class DataStore:
    """Thread-safe (single asyncio loop) in-memory buffer of Candle objects.

    Maintains one deque per symbol, bounded at *buffer_size* candles.
    Duplicate candles (same symbol + timestamp) are silently dropped.
    Candles are always stored in ascending timestamp order.
    """

    def __init__(self, buffer_size: int = 500) -> None:
        self._buffer_size = buffer_size
        self._candles: dict[str, deque[Candle]] = {}

    def add_candle(self, candle: Candle) -> None:
        """Insert *candle* into the buffer, rejecting duplicates."""
        symbol = candle.symbol
        if symbol not in self._candles:
            self._candles[symbol] = deque(maxlen=self._buffer_size)

        buf = self._candles[symbol]

        # Reject exact duplicates (same timestamp)
        if buf and buf[-1].timestamp == candle.timestamp:
            # Update in-place if the incoming candle supersedes a still-forming one
            if not buf[-1].is_confirmed and candle.is_confirmed:
                buf[-1] = candle
            return

        # Guard: if the new candle is older than what we already have, skip it
        if buf and candle.timestamp < buf[-1].timestamp:
            logger.debug(
                "Skipping out-of-order candle for %s: ts=%d < last=%d",
                symbol,
                candle.timestamp,
                buf[-1].timestamp,
            )
            return

        buf.append(candle)

    def replace_candles(self, symbol: str, candles: list[Candle]) -> None:
        """Replace *symbol*'s entire buffer with *candles* (ascending order).

        ``add_candle`` deliberately drops out-of-order bars, so a repaired
        historical candle (post-close gap repair) can only become visible by
        reloading the buffer from the DB — not by appending.
        """
        self._candles[symbol] = deque(candles, maxlen=self._buffer_size)

    def get_candles(self, symbol: str, limit: int | None = None) -> list[Candle]:
        """Return candles for *symbol* in ascending timestamp order.

        If *limit* is given, returns at most the *limit* most recent candles.
        Matches ``CandleDB.get_candles`` so the two are interchangeable.
        """
        buf = self._candles.get(symbol)
        if not buf:
            return []
        if limit is None:
            return list(buf)
        return list(buf)[-limit:]

    def get_latest_candle(self, symbol: str) -> Candle | None:
        """Return the most recent candle for *symbol*, or None."""
        buf = self._candles.get(symbol)
        if not buf:
            return None
        return buf[-1]

    def get_candle_count(self, symbol: str) -> int:
        """Return the number of candles buffered for *symbol*."""
        buf = self._candles.get(symbol)
        return len(buf) if buf else 0
