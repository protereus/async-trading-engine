"""Tests for signal_history table in CandleDB.

Covers:
  - Schema migration is idempotent (init_db called twice)
  - write_signal_history_batch inserts rows correctly
  - resolve_signal_history fills realized_* only when horizon elapsed
  - resolve_signal_history skips rows with no candle data
  - get_signal_history filtering (by symbol, since_ms, resolved_only, gap)
  - Diagnostic queries return expected aggregates on a small fixture
  - gap_spanned set when horizon window contains a weekend gap
  - ALTER TABLE migration is idempotent on pre-10a DBs
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from bot.core.models import Candle
from bot.data.candle_db import CandleDB
from bot.data.signal_history_store import _detect_gap_spanned

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path: Path) -> CandleDB:
    """Fresh in-memory-style CandleDB for each test."""
    cdb = CandleDB(str(tmp_path / "test.db"))
    cdb.init_db()
    return cdb


def _make_candle(symbol: str, ts: int, price: float, high: float | None = None) -> Candle:
    return Candle(
        symbol=symbol,
        timestamp=ts,
        open=price,
        high=high if high is not None else price * 1.01,
        low=price * 0.99,
        close=price,
        volume=1000.0,
        is_confirmed=True,
    )


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


class TestSchemaIdempotent:
    def test_init_db_twice_no_error(self, db: CandleDB, tmp_path: Path) -> None:
        # Calling init_db on the same path a second time must not raise
        db2 = CandleDB(str(tmp_path / "test.db"))
        db2.init_db()  # should be a no-op (CREATE TABLE IF NOT EXISTS)
        db2.close()

    def test_signal_history_table_exists(self, db: CandleDB) -> None:
        # Verify table was created
        conn = sqlite3.connect(db._db_path)
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        conn.close()
        assert "signal_history" in tables

    def test_signal_history_indexes_exist(self, db: CandleDB) -> None:
        conn = sqlite3.connect(db._db_path)
        indexes = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        }
        conn.close()
        assert "idx_signal_history_scored_at" in indexes
        assert "idx_signal_history_symbol" in indexes


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


class TestWriteSignalHistory:
    def test_single_row_roundtrip(self, db: CandleDB) -> None:
        now_ms = int(time.time() * 1000)
        db.write_signal_history(
            scored_at=now_ms,
            symbol="EUR/USD",
            horizon_bars=120,
            mean_return=0.005,
            direction_confidence=0.75,
            uncertainty=0.8,
            entry_price=1.0850,
        )
        rows = db.get_signal_history(symbol="EUR/USD")
        assert len(rows) == 1
        row = rows[0]
        assert row["symbol"] == "EUR/USD"
        assert row["horizon_bars"] == 120
        assert abs(row["mean_return"] - 0.005) < 1e-9  # type: ignore[operator]
        assert row["realized_return_at_horizon"] is None

    def test_batch_inserts_all_rows(self, db: CandleDB) -> None:
        now_ms = int(time.time() * 1000)
        batch = [
            {
                "scored_at": now_ms,
                "symbol": sym,
                "horizon_bars": 24,
                "mean_return": 0.01,
                "direction_confidence": 0.8,
                "uncertainty": 0.5,
                "entry_price": 100.0,
            }
            for sym in ["EUR/USD", "GBP/USD", "XAU/USD"]
        ]
        db.write_signal_history_batch(batch)
        rows = db.get_signal_history()
        assert len(rows) == 3
        symbols = {r["symbol"] for r in rows}
        assert symbols == {"EUR/USD", "GBP/USD", "XAU/USD"}

    def test_null_optional_fields_stored_as_null(self, db: CandleDB) -> None:
        now_ms = int(time.time() * 1000)
        db.write_signal_history(
            scored_at=now_ms,
            symbol="USD/JPY",
            horizon_bars=120,
            mean_return=None,
            direction_confidence=None,
            uncertainty=None,
            entry_price=150.0,
        )
        rows = db.get_signal_history(symbol="USD/JPY")
        assert rows[0]["mean_return"] is None
        assert rows[0]["predicted_mfe_pct"] is None


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class TestResolveSignalHistory:
    def test_does_not_resolve_before_horizon(self, db: CandleDB) -> None:
        now_ms = int(time.time() * 1000)
        # horizon = 24 bars → resolved at now + 24h; we resolve at now (too early)
        db.write_signal_history(
            scored_at=now_ms,
            symbol="EUR/USD",
            horizon_bars=24,
            mean_return=0.01,
            direction_confidence=0.8,
            uncertainty=0.5,
            entry_price=1.08,
        )
        resolved = db.resolve_signal_history(now_ms)  # same time — not past horizon
        assert resolved == 0
        rows = db.get_signal_history(symbol="EUR/USD")
        assert rows[0]["realized_return_at_horizon"] is None

    def test_resolves_after_horizon_with_candles(self, db: CandleDB) -> None:
        # scored_at = 1h ago; horizon_bars=1 → horizon already passed
        one_hour_ms = 3_600_000
        scored_at = int(time.time() * 1000) - 2 * one_hour_ms

        db.write_signal_history(
            scored_at=scored_at,
            symbol="GBP/USD",
            horizon_bars=1,
            mean_return=0.005,
            direction_confidence=0.8,
            uncertainty=0.5,
            entry_price=1.2500,
        )
        # Insert a candle inside the horizon window
        db.insert_candle(
            _make_candle("GBP/USD", scored_at + 30 * 60_000, price=1.2600, high=1.2650)
        )

        now_ms = int(time.time() * 1000)
        resolved = db.resolve_signal_history(now_ms)
        assert resolved == 1

        rows = db.get_signal_history(symbol="GBP/USD", resolved_only=True)
        assert len(rows) == 1
        row = rows[0]
        # realized_return = (1.26 - 1.25) / 1.25 = 0.008
        assert abs(row["realized_return_at_horizon"] - 0.008) < 1e-6  # type: ignore[operator]
        # realized_max_high_pct = (1.265 - 1.25) / 1.25 = 0.012
        assert abs(row["realized_max_high_pct"] - 0.012) < 1e-6  # type: ignore[operator]
        # realized_min_low_pct = (1.25 - 1.2474) / 1.25 ≈ 0.00208
        assert row["realized_min_low_pct"] is not None
        assert row["realized_min_low_pct"] >= 0  # type: ignore[operator]

    def test_skips_rows_with_no_candles(self, db: CandleDB) -> None:
        one_hour_ms = 3_600_000
        scored_at = int(time.time() * 1000) - 2 * one_hour_ms
        db.write_signal_history(
            scored_at=scored_at,
            symbol="XAU/USD",
            horizon_bars=1,
            mean_return=0.01,
            direction_confidence=0.8,
            uncertainty=0.5,
            entry_price=2000.0,
        )
        # No candles inserted for XAU/USD
        resolved = db.resolve_signal_history(int(time.time() * 1000))
        assert resolved == 0

    def test_already_resolved_rows_not_updated_again(self, db: CandleDB) -> None:
        one_hour_ms = 3_600_000
        scored_at = int(time.time() * 1000) - 3 * one_hour_ms
        db.write_signal_history(
            scored_at=scored_at,
            symbol="USD/CAD",
            horizon_bars=1,
            mean_return=0.002,
            direction_confidence=0.75,
            uncertainty=0.6,
            entry_price=1.3600,
        )
        db.insert_candle(_make_candle("USD/CAD", scored_at + 30 * 60_000, price=1.3620))
        now_ms = int(time.time() * 1000)

        # First resolve
        resolved1 = db.resolve_signal_history(now_ms)
        assert resolved1 == 1

        # Second resolve — same row already has realized data; should skip
        resolved2 = db.resolve_signal_history(now_ms)
        assert resolved2 == 0


# ---------------------------------------------------------------------------
# get_signal_history filters
# ---------------------------------------------------------------------------


class TestGetSignalHistory:
    def _insert_rows(self, db: CandleDB, n: int = 5) -> int:
        now_ms = int(time.time() * 1000)
        for i in range(n):
            db.write_signal_history(
                scored_at=now_ms - i * 3_600_000,
                symbol="EUR/USD" if i % 2 == 0 else "GBP/USD",
                horizon_bars=24,
                mean_return=0.001 * (i + 1),
                direction_confidence=0.7,
                uncertainty=0.5,
                entry_price=1.0 + i * 0.01,
            )
        return now_ms

    def test_filter_by_symbol(self, db: CandleDB) -> None:
        self._insert_rows(db, 6)
        rows = db.get_signal_history(symbol="EUR/USD")
        assert all(r["symbol"] == "EUR/USD" for r in rows)

    def test_filter_by_since_ms(self, db: CandleDB) -> None:
        now_ms = self._insert_rows(db, 5)
        # Only rows scored in the last 2h
        since = now_ms - 2 * 3_600_000
        rows = db.get_signal_history(since_ms=since)
        # At most 3 rows (at offsets 0, 1h, 2h)
        assert len(rows) <= 3

    def test_resolved_only_filter(self, db: CandleDB) -> None:
        self._insert_rows(db, 3)
        rows = db.get_signal_history(resolved_only=True)
        assert rows == []  # none resolved yet

    def test_limit(self, db: CandleDB) -> None:
        self._insert_rows(db, 10)
        rows = db.get_signal_history(limit=3)
        assert len(rows) == 3


# ---------------------------------------------------------------------------
# Diagnostics query helpers (exercised via get_signal_history)
# ---------------------------------------------------------------------------


class TestDiagnosticAggregates:
    def test_rankic_data_available(self, db: CandleDB) -> None:
        """Verify that resolved rows contain the fields needed for RankIC computation."""
        one_hour_ms = 3_600_000
        scored_at = int(time.time() * 1000) - 3 * one_hour_ms
        symbols = ["EUR/USD", "GBP/USD", "XAU/USD"]
        for i, sym in enumerate(symbols):
            db.write_signal_history(
                scored_at=scored_at,
                symbol=sym,
                horizon_bars=1,
                mean_return=0.001 * (i + 1),
                direction_confidence=0.7 + i * 0.05,
                uncertainty=0.5,
                entry_price=1.0 + i * 0.1,
            )
            # Insert a candle to make it resolvable
            db.insert_candle(_make_candle(sym, scored_at + 30 * 60_000, price=1.01 + i * 0.1))

        db.resolve_signal_history(int(time.time() * 1000))

        rows = db.get_signal_history(resolved_only=True)
        assert len(rows) == 3
        for row in rows:
            assert row["mean_return"] is not None
            assert row["direction_confidence"] is not None
            assert row["realized_return_at_horizon"] is not None


# ---------------------------------------------------------------------------
# Path metrics wiring (regression guard for Commit 2 of fixes)
# ---------------------------------------------------------------------------


class TestPathMetricsInSignalHistory:
    """Verify that write_signal_history_batch stores non-None path metrics.

    This test class was added to catch the regression where main.py wrote None
    for predicted_mfe_pct / predicted_mae_pct / predicted_volatility / monotonicity
    instead of reading them from _topk_strategy._path_signals.
    """

    def test_path_fields_stored_when_provided(self, db: CandleDB) -> None:
        now_ms = int(time.time() * 1000)
        db.write_signal_history_batch(
            [
                {
                    "scored_at": now_ms,
                    "symbol": "EUR/USD",
                    "horizon_bars": 120,
                    "mean_return": 0.005,
                    "direction_confidence": 0.75,
                    "uncertainty": 0.8,
                    "entry_price": 1.0850,
                    "predicted_mfe_pct": 0.038,
                    "predicted_mae_pct": 0.011,
                    "predicted_volatility": 0.002,
                    "monotonicity": 0.72,
                },
                {
                    "scored_at": now_ms,
                    "symbol": "GBP/USD",
                    "horizon_bars": 120,
                    "mean_return": 0.003,
                    "direction_confidence": 0.70,
                    "uncertainty": 1.0,
                    "entry_price": 1.2700,
                    # No path signal available for this asset
                    "predicted_mfe_pct": None,
                    "predicted_mae_pct": None,
                    "predicted_volatility": None,
                    "monotonicity": None,
                },
            ]
        )

        eur_rows = db.get_signal_history(symbol="EUR/USD")
        assert len(eur_rows) == 1
        assert abs(eur_rows[0]["predicted_mfe_pct"] - 0.038) < 1e-9  # type: ignore[operator]
        assert abs(eur_rows[0]["predicted_mae_pct"] - 0.011) < 1e-9  # type: ignore[operator]
        assert abs(eur_rows[0]["predicted_volatility"] - 0.002) < 1e-9  # type: ignore[operator]
        assert abs(eur_rows[0]["monotonicity"] - 0.72) < 1e-9  # type: ignore[operator]

        gbp_rows = db.get_signal_history(symbol="GBP/USD")
        assert len(gbp_rows) == 1
        assert gbp_rows[0]["predicted_mfe_pct"] is None
        assert gbp_rows[0]["monotonicity"] is None

    def test_path_fields_null_when_omitted(self, db: CandleDB) -> None:
        """Rows written without path keys default to NULL (not crash)."""
        now_ms = int(time.time() * 1000)
        db.write_signal_history(
            scored_at=now_ms,
            symbol="USD/JPY",
            horizon_bars=120,
            mean_return=0.002,
            direction_confidence=0.72,
            uncertainty=0.9,
            entry_price=155.0,
        )
        rows = db.get_signal_history(symbol="USD/JPY")
        assert rows[0]["predicted_mfe_pct"] is None
        assert rows[0]["monotonicity"] is None


# ---------------------------------------------------------------------------
# gap_spanned detection
# ---------------------------------------------------------------------------


HOUR_MS = 3_600_000


class TestGapDetectionUnit:
    """Direct tests of the _detect_gap_spanned helper."""

    def test_contiguous_hourly_no_gap(self) -> None:
        start = 1_000_000_000_000
        ts = [start + i * HOUR_MS for i in range(120)]
        horizon_end = start + 120 * HOUR_MS
        assert _detect_gap_spanned(ts, horizon_end) == 0

    def test_weekend_gap_flagged(self) -> None:
        # 24 hourly candles, then a 48h hole, then 24 more hourly candles.
        start = 1_000_000_000_000
        first = [start + i * HOUR_MS for i in range(24)]
        gap_start = first[-1] + 48 * HOUR_MS
        second = [gap_start + i * HOUR_MS for i in range(24)]
        ts = first + second
        horizon_end = second[-1] + HOUR_MS
        assert _detect_gap_spanned(ts, horizon_end) == 1

    def test_trailing_gap_flagged(self) -> None:
        # All in-window candles are contiguous, but the horizon ends 6h after
        # the last candle.  Truncated measurement → flag.
        start = 1_000_000_000_000
        ts = [start + i * HOUR_MS for i in range(24)]
        horizon_end = ts[-1] + 6 * HOUR_MS
        assert _detect_gap_spanned(ts, horizon_end) == 1

    def test_single_candle_flagged(self) -> None:
        assert _detect_gap_spanned([1_000_000_000_000], 1_000_000_000_000 + HOUR_MS) == 1

    def test_empty_flagged(self) -> None:
        assert _detect_gap_spanned([], 1_000_000_000_000) == 1


class TestGapSpannedResolver:
    def test_contiguous_window_sets_zero(self, db: CandleDB) -> None:
        scored_at = int(time.time() * 1000) - 5 * HOUR_MS
        db.write_signal_history(
            scored_at=scored_at,
            symbol="EUR/USD",
            horizon_bars=3,
            mean_return=0.001,
            direction_confidence=0.8,
            uncertainty=0.5,
            entry_price=1.10,
        )
        # 3 hourly candles inside the window
        for i in range(1, 4):
            db.insert_candle(
                _make_candle("EUR/USD", scored_at + i * HOUR_MS, price=1.10 + 0.001 * i)
            )
        db.resolve_signal_history(int(time.time() * 1000))
        row = db.get_signal_history(symbol="EUR/USD", resolved_only=True)[0]
        assert row["gap_spanned"] == 0

    def test_weekend_gap_sets_one(self, db: CandleDB) -> None:
        # Friday-ish + Monday-ish: insert candles at hours 1, 2, 3, then jump
        # to hour 60 (≈ weekend gap), inside a 70h horizon window.
        scored_at = int(time.time() * 1000) - 80 * HOUR_MS
        db.write_signal_history(
            scored_at=scored_at,
            symbol="GBP/USD",
            horizon_bars=70,
            mean_return=0.002,
            direction_confidence=0.7,
            uncertainty=0.6,
            entry_price=1.25,
        )
        for hour_offset in (1, 2, 3, 60, 61, 62):
            db.insert_candle(_make_candle("GBP/USD", scored_at + hour_offset * HOUR_MS, price=1.25))
        db.resolve_signal_history(int(time.time() * 1000))
        row = db.get_signal_history(symbol="GBP/USD", resolved_only=True)[0]
        assert row["gap_spanned"] == 1

    def test_exclude_gap_filter(self, db: CandleDB) -> None:
        scored_at = int(time.time() * 1000) - 80 * HOUR_MS
        # Clean row
        db.write_signal_history(
            scored_at=scored_at,
            symbol="EUR/USD",
            horizon_bars=3,
            mean_return=0.001,
            direction_confidence=0.8,
            uncertainty=0.5,
            entry_price=1.10,
        )
        for i in range(1, 4):
            db.insert_candle(_make_candle("EUR/USD", scored_at + i * HOUR_MS, price=1.10))
        # Gappy row
        db.write_signal_history(
            scored_at=scored_at,
            symbol="GBP/USD",
            horizon_bars=70,
            mean_return=0.002,
            direction_confidence=0.7,
            uncertainty=0.6,
            entry_price=1.25,
        )
        for hour_offset in (1, 60):
            db.insert_candle(_make_candle("GBP/USD", scored_at + hour_offset * HOUR_MS, price=1.25))
        db.resolve_signal_history(int(time.time() * 1000))

        all_rows = db.get_signal_history(resolved_only=True)
        clean_only = db.get_signal_history(resolved_only=True, exclude_gap_spanned=True)
        assert len(all_rows) == 2
        assert len(clean_only) == 1
        assert clean_only[0]["symbol"] == "EUR/USD"


class TestComputeGapSpannedAtHorizon:
    """— re-evaluate gap_spanned at an arbitrary H.

    The persisted ``gap_spanned`` is fixed at write-time horizon_bars (120 h
    in production), which flags nearly every weekday forex row because the
    5-day window crosses a weekend.  The horizon sweep needs to ask the same
    question at shorter H — that's what this method does.
    """

    def test_short_h_clean_when_long_h_gap(self, db: CandleDB) -> None:
        # 24 contiguous hourly candles, then a 48 h hole, then 24 more.
        # At H=20 the window is the contiguous first run → clean.
        # At H=70 the window crosses the gap → gap_spanned.
        scored_at = int(time.time() * 1000) - 100 * HOUR_MS
        first_block = list(range(1, 25))
        gap_start = first_block[-1] + 48
        second_block = list(range(gap_start, gap_start + 24))
        for off in first_block + second_block:
            db.insert_candle(_make_candle("EUR/USD", scored_at + off * HOUR_MS, price=1.10))
        assert db.compute_gap_spanned_at_horizon("EUR/USD", scored_at, 20) == 0
        assert db.compute_gap_spanned_at_horizon("EUR/USD", scored_at, 70) == 1

    def test_no_candles_flagged(self, db: CandleDB) -> None:
        scored_at = int(time.time() * 1000) - 10 * HOUR_MS
        assert db.compute_gap_spanned_at_horizon("USD/JPY", scored_at, 24) == 1

    def test_trailing_gap_at_h_flagged(self, db: CandleDB) -> None:
        # 3 contiguous candles inside a 24 h horizon → trailing 21 h of empty
        # space (>2 h tolerance) → flagged.
        scored_at = int(time.time() * 1000) - 30 * HOUR_MS
        for off in (1, 2, 3):
            db.insert_candle(_make_candle("GBP/USD", scored_at + off * HOUR_MS, price=1.25))
        assert db.compute_gap_spanned_at_horizon("GBP/USD", scored_at, 24) == 1
        # At H=3 the window is exactly covered → clean.
        assert db.compute_gap_spanned_at_horizon("GBP/USD", scored_at, 3) == 0


class TestAlterMigrationIdempotent:
    def test_pre_10a_schema_migrates_to_current(self, tmp_path: Path) -> None:
        """A DB created without gap_spanned/predicted_close_path migrates cleanly."""
        db_path = tmp_path / "legacy.db"
        # Build a pre-10a signal_history table manually.
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE signal_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scored_at INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                horizon_bars INTEGER NOT NULL,
                mean_return REAL,
                direction_confidence REAL,
                uncertainty REAL,
                predicted_mfe_pct REAL,
                predicted_mae_pct REAL,
                predicted_volatility REAL,
                monotonicity REAL,
                entry_price REAL NOT NULL,
                realized_return_at_horizon REAL,
                realized_max_high_pct REAL,
                realized_min_low_pct REAL
            )
            """
        )
        # Seed a legacy row.
        conn.execute(
            """
            INSERT INTO signal_history
                (scored_at, symbol, horizon_bars, mean_return, direction_confidence,
                 uncertainty, entry_price)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (1, "EUR/USD", 120, 0.005, 0.8, 0.5, 1.10),
        )
        conn.commit()
        conn.close()

        # Open via CandleDB — migrations should add the missing columns.
        cdb = CandleDB(str(db_path))
        cdb.init_db()
        rows = cdb.get_signal_history()
        assert len(rows) == 1
        assert rows[0]["gap_spanned"] == 0  # column added with default
        assert rows[0]["predicted_close_path"] is None  # column added nullable
        assert rows[0]["var_closes_at_horizons"] is None  # 10c-prep column, nullable

        # Second init_db must be a no-op (migrations re-checked, none applied).
        cdb2 = CandleDB(str(db_path))
        cdb2.init_db()
        cdb2.close()
        cdb.close()


# ---------------------------------------------------------------------------
# — predicted_close_path persistence
# ---------------------------------------------------------------------------


class TestPredictedClosePathPersistence:
    def test_path_roundtrip(self, db: CandleDB) -> None:
        """predicted_close_path BLOB survives write → read with byte fidelity."""
        import numpy as np

        path = np.array([1.0850 + 0.0001 * i for i in range(120)], dtype="<f4")
        blob = path.tobytes()
        now_ms = int(time.time() * 1000)
        db.write_signal_history_batch(
            [
                {
                    "scored_at": now_ms,
                    "symbol": "EUR/USD",
                    "horizon_bars": 120,
                    "mean_return": 0.005,
                    "direction_confidence": 0.85,
                    "uncertainty": 0.4,
                    "entry_price": 1.0850,
                    "predicted_mfe_pct": 0.012,
                    "predicted_mae_pct": 0.003,
                    "predicted_volatility": 0.002,
                    "monotonicity": 0.88,
                    "predicted_close_path": blob,
                }
            ]
        )
        rows = db.get_signal_history(symbol="EUR/USD")
        assert len(rows) == 1
        stored = rows[0]["predicted_close_path"]
        assert stored is not None
        decoded = np.frombuffer(stored, dtype="<f4")
        assert decoded.size == 120
        # Path round-trips exactly (float32 precision).
        assert np.allclose(decoded, path, rtol=0, atol=1e-7)

    def test_path_null_when_omitted(self, db: CandleDB) -> None:
        now_ms = int(time.time() * 1000)
        db.write_signal_history_batch(
            [
                {
                    "scored_at": now_ms,
                    "symbol": "GBP/USD",
                    "horizon_bars": 120,
                    "mean_return": 0.003,
                    "direction_confidence": 0.70,
                    "uncertainty": 1.0,
                    "entry_price": 1.2700,
                    # No predicted_close_path key
                }
            ]
        )
        row = db.get_signal_history(symbol="GBP/USD")[0]
        assert row["predicted_close_path"] is None


# ---------------------------------------------------------------------------
# — var_closes_at_horizons persistence
# ---------------------------------------------------------------------------


class TestVarClosesAtHorizonsPersistence:
    def test_blob_roundtrip_to_20x6(self, db: CandleDB) -> None:
        """A (20, 6) float32 array survives write → read byte-for-byte."""
        import numpy as np

        from bot.strategy.topk_strategy import CANDIDATE_HORIZONS

        draws = np.random.default_rng(7).normal(1.085, 0.002, size=(20, len(CANDIDATE_HORIZONS)))
        arr = draws.astype("<f4")
        now_ms = int(time.time() * 1000)
        db.write_signal_history_batch(
            [
                {
                    "scored_at": now_ms,
                    "symbol": "EUR/USD",
                    "horizon_bars": 120,
                    "mean_return": 0.005,
                    "direction_confidence": 0.85,
                    "uncertainty": 0.4,
                    "entry_price": 1.0850,
                    "var_closes_at_horizons": arr.tobytes(),
                }
            ]
        )
        stored = db.get_signal_history(symbol="EUR/USD")[0]["var_closes_at_horizons"]
        assert isinstance(stored, bytes)
        assert len(stored) == 20 * len(CANDIDATE_HORIZONS) * 4  # 480 B at 20 draws
        decoded = np.frombuffer(stored, dtype="<f4").reshape(-1, len(CANDIDATE_HORIZONS))
        assert decoded.shape == (20, len(CANDIDATE_HORIZONS))
        assert np.array_equal(decoded, arr)

    def test_null_when_omitted(self, db: CandleDB) -> None:
        now_ms = int(time.time() * 1000)
        db.write_signal_history_batch(
            [
                {
                    "scored_at": now_ms,
                    "symbol": "GBP/USD",
                    "horizon_bars": 120,
                    "mean_return": 0.003,
                    "direction_confidence": 0.70,
                    "uncertainty": 1.0,
                    "entry_price": 1.2700,
                    # No var_closes_at_horizons key — pre-10c-prep writer shape
                }
            ]
        )
        row = db.get_signal_history(symbol="GBP/USD")[0]
        assert row["var_closes_at_horizons"] is None


# ---------------------------------------------------------------------------
# — resolver keys realised returns off the row's OWN horizon
# ---------------------------------------------------------------------------


class TestResolverHonoursRowHorizon:
    def test_realised_and_gap_computed_at_each_rows_horizon(self, db: CandleDB) -> None:
        """Two rows scored at the same instant with different horizon_bars
        must resolve against different windows: realised-at-H matches
        predicted-at-H, and gap_spanned answers the question at that H."""
        base = 1_700_000_000_000
        hour = 3_600_000
        # Candles only at +1h..+3h; the feed then goes dark.
        for i, px in ((1, 101.0), (2, 102.0), (3, 103.0)):
            db.insert_candle(_make_candle("EUR/USD", base + i * hour, px))

        db.write_signal_history_batch(
            [
                {
                    "scored_at": base,
                    "symbol": "EUR/USD",
                    "horizon_bars": 2,  # window fully covered by candles
                    "mean_return": 0.01,
                    "direction_confidence": 0.9,
                    "uncertainty": 0.3,
                    "entry_price": 100.0,
                },
                {
                    "scored_at": base,
                    "symbol": "EUR/USD",
                    "horizon_bars": 6,  # window truncated → trailing gap
                    "mean_return": 0.01,
                    "direction_confidence": 0.9,
                    "uncertainty": 0.3,
                    "entry_price": 100.0,
                },
            ]
        )
        resolved = db.resolve_signal_history(now_ms=base + 100 * hour)
        assert resolved == 2

        rows = sorted(
            db.get_signal_history(symbol="EUR/USD", resolved_only=True),
            key=lambda r: r["horizon_bars"],  # type: ignore[arg-type, return-value]
        )
        h2, h6 = rows
        # H=2: realised close is the +2h candle (102), window clean.
        assert h2["realized_return_at_horizon"] == pytest.approx(0.02)
        assert h2["gap_spanned"] == 0
        # H=6: realised close falls back to the last in-window candle (+3h,
        # 103) and the 3h trailing gap flags the row.
        assert h6["realized_return_at_horizon"] == pytest.approx(0.03)
        assert h6["gap_spanned"] == 1
