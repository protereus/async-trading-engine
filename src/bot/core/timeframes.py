"""Shared timeframe → milliseconds lookup used by all data feeds."""

from __future__ import annotations

# Maps CCXT / Twelve Data timeframe strings to milliseconds.
TIMEFRAME_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


def timeframe_to_ms(timeframe: str, default: int = 60_000) -> int:
    """Return the number of milliseconds in *timeframe*, or *default* if unknown."""
    return TIMEFRAME_MS.get(timeframe, default)
