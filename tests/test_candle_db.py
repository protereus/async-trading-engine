"""Tests for CandleDB — SQLite-backed candle persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from bot.core.models import Candle
from bot.data.candle_db import CandleDB
from bot.data.store import DataStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candle(
    symbol: str = "AVAX/USDT",
    timestamp: int = 1_000_000,
    close: float = 100.0,
    is_confirmed: bool = True,
) -> Candle:
    return Candle(
        symbol=symbol,
        timestamp=timestamp,
        open=close - 1.0,
        high=close + 2.0,
        low=close - 2.0,
        close=close,
        volume=500.0,
        is_confirmed=is_confirmed,
    )


def _make_candles(
    symbol: str = "AVAX/USDT",
    count: int = 10,
    start_ts: int = 1_000_000,
    step: int = 60_000,
) -> list[Candle]:
    return [
        _make_candle(symbol=symbol, timestamp=start_ts + i * step, close=100.0 + i)
        for i in range(count)
    ]


@pytest.fixture()
def db(tmp_path: Path) -> CandleDB:
    cdb = CandleDB(db_path=str(tmp_path / "test_candles.db"))
    cdb.init_db()
    yield cdb
    cdb.close()


# ---------------------------------------------------------------------------
# Basic operations
# ---------------------------------------------------------------------------


class TestBasicOperations:
    def test_insert_and_retrieve_candle(self, db: CandleDB) -> None:
        candle = _make_candle(timestamp=1_000_000, close=42.5)
        db.insert_candle(candle)

        results = db.get_candles("AVAX/USDT")
        assert len(results) == 1
        c = results[0]
        assert c.symbol == "AVAX/USDT"
        assert c.timestamp == 1_000_000
        assert c.open == candle.open
        assert c.high == candle.high
        assert c.low == candle.low
        assert c.close == 42.5
        assert c.volume == 500.0

    def test_duplicate_insert_no_error_no_duplicate_row(self, db: CandleDB) -> None:
        candle = _make_candle(timestamp=1_000_000)
        db.insert_candle(candle)
        db.insert_candle(candle)  # must be silently ignored

        assert len(db.get_candles("AVAX/USDT")) == 1

    def test_get_candles_ascending_order(self, db: CandleDB) -> None:
        candles = _make_candles(count=5, start_ts=1_000_000, step=60_000)
        for c in reversed(candles):  # insert in reverse order
            db.insert_candle(c)

        results = db.get_candles("AVAX/USDT")
        timestamps = [c.timestamp for c in results]
        assert timestamps == sorted(timestamps)

    def test_get_candles_with_limit_returns_most_recent_n(self, db: CandleDB) -> None:
        candles = _make_candles(count=10, start_ts=1_000_000, step=60_000)
        db.insert_candles(candles)

        results = db.get_candles("AVAX/USDT", limit=3)
        assert len(results) == 3
        expected_ts = sorted(c.timestamp for c in candles)[-3:]
        assert [c.timestamp for c in results] == expected_ts

    def test_get_candles_with_since_filters_correctly(self, db: CandleDB) -> None:
        candles = _make_candles(count=10, start_ts=1_000_000, step=60_000)
        db.insert_candles(candles)

        cutoff = candles[5].timestamp
        results = db.get_candles("AVAX/USDT", since=cutoff)
        assert all(c.timestamp >= cutoff for c in results)
        assert len(results) == 5  # candles[5..9]

    def test_get_latest_timestamp_correct(self, db: CandleDB) -> None:
        candles = _make_candles(count=5, start_ts=1_000_000, step=60_000)
        db.insert_candles(candles)

        assert db.get_latest_timestamp("AVAX/USDT") == max(c.timestamp for c in candles)

    def test_get_latest_timestamp_unknown_symbol_returns_none(self, db: CandleDB) -> None:
        assert db.get_latest_timestamp("BTC/USDT") is None

    def test_get_earliest_timestamp_correct(self, db: CandleDB) -> None:
        candles = _make_candles(count=5, start_ts=1_000_000, step=60_000)
        db.insert_candles(candles)

        assert db.get_earliest_timestamp("AVAX/USDT") == min(c.timestamp for c in candles)

    def test_get_earliest_timestamp_unknown_symbol_returns_none(self, db: CandleDB) -> None:
        assert db.get_earliest_timestamp("BTC/USDT") is None

    def test_get_candle_count_accurate(self, db: CandleDB) -> None:
        candles = _make_candles(count=7)
        db.insert_candles(candles)

        assert db.get_candle_count("AVAX/USDT") == 7

    def test_empty_db_returns_empty_list_none_zero(self, db: CandleDB) -> None:
        assert db.get_candles("AVAX/USDT") == []
        assert db.get_latest_timestamp("AVAX/USDT") is None
        assert db.get_candle_count("AVAX/USDT") == 0


# ---------------------------------------------------------------------------
# Bulk operations
# ---------------------------------------------------------------------------


class TestBulkOperations:
    def test_insert_1000_candles(self, db: CandleDB) -> None:
        candles = _make_candles(count=1000, start_ts=1_000_000, step=60_000)
        db.insert_candles(candles)

        assert db.get_candle_count("AVAX/USDT") == 1000

    def test_bulk_insert_ignores_duplicates(self, db: CandleDB) -> None:
        candles = _make_candles(count=5)
        db.insert_candles(candles)
        db.insert_candles(candles)  # full duplicate batch

        assert db.get_candle_count("AVAX/USDT") == 5

    def test_bulk_insert_partial_duplicates(self, db: CandleDB) -> None:
        first_batch = _make_candles(count=5, start_ts=1_000_000, step=60_000)
        db.insert_candles(first_batch)

        # second batch overlaps 3 existing + 2 new
        overlap_start = first_batch[2].timestamp
        second_batch = _make_candles(count=5, start_ts=overlap_start, step=60_000)
        db.insert_candles(second_batch)

        assert db.get_candle_count("AVAX/USDT") == 7  # 5 original + 2 new


class TestDeleteForSymbol:
    """Coverage for ``delete_candles_for_symbol`` — used by the D3 IG-native
    cutover to wipe pre-cutover TD-scale rows before backfilling fresh
    IG-level data.  Without it the ``INSERT OR IGNORE`` semantics would
    silently keep the old (wrong-scale) rows at colliding timestamps."""

    def test_deletes_all_rows_for_one_symbol(self, db: CandleDB) -> None:
        db.insert_candles(_make_candles(symbol="USO", count=10))
        assert db.get_candle_count("USO") == 10

        removed = db.delete_candles_for_symbol("USO")
        assert removed == 10
        assert db.get_candle_count("USO") == 0

    def test_leaves_other_symbols_untouched(self, db: CandleDB) -> None:
        db.insert_candles(_make_candles(symbol="USO", count=5))
        db.insert_candles(_make_candles(symbol="EUR/USD", count=7))

        removed = db.delete_candles_for_symbol("USO")
        assert removed == 5
        assert db.get_candle_count("USO") == 0
        assert db.get_candle_count("EUR/USD") == 7  # untouched

    def test_no_rows_returns_zero(self, db: CandleDB) -> None:
        """Idempotent: a second call on an already-empty symbol returns 0,
        doesn't raise.  Matters because the IGCandleLSFeed startup wipes
        on every run."""
        removed = db.delete_candles_for_symbol("USO")
        assert removed == 0


# ---------------------------------------------------------------------------
# WAL mode
# ---------------------------------------------------------------------------


class TestWalMode:
    def test_journal_mode_is_wal(self, db: CandleDB) -> None:
        assert db._conn is not None
        row = db._conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal"


# ---------------------------------------------------------------------------
# Integration with DataStore
# ---------------------------------------------------------------------------


class TestDataStoreIntegration:
    def test_load_candles_from_db_into_store(self, db: CandleDB) -> None:
        candles = _make_candles(count=5, start_ts=1_000_000, step=60_000)
        db.insert_candles(candles)

        store = DataStore(buffer_size=100)
        for c in db.get_candles("AVAX/USDT"):
            store.add_candle(c)

        stored = store.get_candles("AVAX/USDT")
        assert len(stored) == 5
        assert [c.timestamp for c in stored] == sorted(c.timestamp for c in candles)

    def test_candles_from_db_are_confirmed(self, db: CandleDB) -> None:
        db.insert_candle(_make_candle(is_confirmed=True))

        results = db.get_candles("AVAX/USDT")
        assert all(c.is_confirmed for c in results)


# ---------------------------------------------------------------------------
# Unconfirmed candle contract (enforced at call site in feed.py)
# ---------------------------------------------------------------------------


class TestUnconfirmedCandleContract:
    def test_only_confirmed_candles_written_in_watch_flow(self, db: CandleDB) -> None:
        confirmed = _make_candle(timestamp=1_000_000, is_confirmed=True)
        unconfirmed = _make_candle(timestamp=1_060_000, is_confirmed=False)

        # Simulate the feed.py guard: only write confirmed candles
        if confirmed.is_confirmed:
            db.insert_candle(confirmed)
        if unconfirmed.is_confirmed:
            db.insert_candle(unconfirmed)

        assert db.get_candle_count("AVAX/USDT") == 1
        assert db.get_candles("AVAX/USDT")[0].timestamp == 1_000_000


# ---------------------------------------------------------------------------
# Multiple symbols
# ---------------------------------------------------------------------------


class TestMultipleSymbols:
    def test_symbols_are_isolated(self, db: CandleDB) -> None:
        avax = _make_candles(symbol="AVAX/USDT", count=3)
        btc = _make_candles(symbol="BTC/USDT", count=5, start_ts=2_000_000)
        db.insert_candles(avax)
        db.insert_candles(btc)

        assert db.get_candle_count("AVAX/USDT") == 3
        assert db.get_candle_count("BTC/USDT") == 5
        assert all(c.symbol == "AVAX/USDT" for c in db.get_candles("AVAX/USDT"))
        assert all(c.symbol == "BTC/USDT" for c in db.get_candles("BTC/USDT"))
