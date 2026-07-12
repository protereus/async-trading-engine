"""Tests for the in-memory data store."""

from __future__ import annotations

from bot.core.models import Candle
from bot.data.store import DataStore


def make_candle(ts: int, symbol: str = "AVAX/USDT", confirmed: bool = True) -> Candle:
    return Candle(
        timestamp=ts,
        open=10.0,
        high=11.0,
        low=9.0,
        close=10.5,
        volume=100.0,
        symbol=symbol,
        is_confirmed=confirmed,
    )


class TestDataStore:
    def test_add_and_get(self) -> None:
        store = DataStore()
        store.add_candle(make_candle(1000))
        store.add_candle(make_candle(2000))
        candles = store.get_candles("AVAX/USDT")
        assert len(candles) == 2
        assert candles[0].timestamp == 1000
        assert candles[1].timestamp == 2000

    def test_buffer_overflow_drops_oldest(self) -> None:
        store = DataStore(buffer_size=3)
        for ts in [1000, 2000, 3000, 4000]:
            store.add_candle(make_candle(ts))
        candles = store.get_candles("AVAX/USDT")
        assert len(candles) == 3
        assert candles[0].timestamp == 2000  # 1000 was dropped

    def test_duplicate_rejected(self) -> None:
        store = DataStore()
        store.add_candle(make_candle(1000))
        store.add_candle(make_candle(1000))  # duplicate
        assert store.get_candle_count("AVAX/USDT") == 1

    def test_unconfirmed_updated_by_confirmed(self) -> None:
        store = DataStore()
        store.add_candle(make_candle(1000, confirmed=False))
        store.add_candle(make_candle(1000, confirmed=True))
        candles = store.get_candles("AVAX/USDT")
        assert len(candles) == 1
        assert candles[0].is_confirmed is True

    def test_out_of_order_skipped(self) -> None:
        store = DataStore()
        store.add_candle(make_candle(2000))
        store.add_candle(make_candle(1000))  # older — should be skipped
        assert store.get_candle_count("AVAX/USDT") == 1
        assert store.get_latest_candle("AVAX/USDT").timestamp == 2000  # type: ignore[union-attr]

    def test_get_candles_with_limit(self) -> None:
        store = DataStore()
        for ts in range(10):
            store.add_candle(make_candle(ts * 1000))
        candles = store.get_candles("AVAX/USDT", limit=3)
        assert len(candles) == 3
        assert candles[-1].timestamp == 9000

    def test_get_latest_none_on_empty(self) -> None:
        store = DataStore()
        assert store.get_latest_candle("AVAX/USDT") is None

    def test_candle_count_empty(self) -> None:
        store = DataStore()
        assert store.get_candle_count("UNKNOWN/USDT") == 0

    def test_multiple_symbols_isolated(self) -> None:
        store = DataStore()
        store.add_candle(make_candle(1000, symbol="AVAX/USDT"))
        store.add_candle(make_candle(1000, symbol="BTC/USDT"))
        assert store.get_candle_count("AVAX/USDT") == 1
        assert store.get_candle_count("BTC/USDT") == 1
        assert store.get_latest_candle("AVAX/USDT").symbol == "AVAX/USDT"  # type: ignore[union-attr]
