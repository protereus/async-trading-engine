"""Tests for the sentiment-edge measurement harness.

Two surfaces under test:

* ``candle_db.signal_history`` schema + write path: new columns survive a
  round-trip, the ALTER-TABLE migration is idempotent on a pre-migration
  DB, and ``sentiment_agent_coverage`` defaults to 0 when the row omits
  it (the honest ABSENT path for partial-overlay days).
* ``bot.analysis.sentiment_edge``: the pure partition / magnitude /
  asset-class logic, plus ``compute_harness`` end-to-end against
  synthetic signals covering each cell.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from bot.analysis.sentiment_edge import (
    MIN_CELL_N,
    SentimentSignal,
    asset_class_for,
    compute_harness,
    partition_signal,
)
from bot.data.candle_db import CandleDB

# ---------------------------------------------------------------------------
# Schema + write path
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path: Path) -> CandleDB:
    cdb = CandleDB(db_path=str(tmp_path / "test.db"))
    cdb.init_db()
    yield cdb
    cdb.close()


def _base_row(**overrides: object) -> dict[str, object]:
    """A minimal valid signal_history row.  Sentiment fields default to
    None / 0 so each test can override exactly the dimension under test."""
    row: dict[str, object] = {
        "scored_at": 1_780_000_000_000,
        "symbol": "EUR/USD",
        "horizon_bars": 120,
        "mean_return": 0.005,
        "direction_confidence": 0.9,
        "uncertainty": 1.5,
        "entry_price": 1.10,
    }
    row.update(overrides)
    return row


class TestSchemaRoundTrip:
    def test_sentiment_columns_persist_and_read_back(self, db: CandleDB) -> None:
        db.write_signal_history_batch(
            [
                _base_row(
                    sentiment_score=0.42,
                    sentiment_confidence=0.71,
                    sentiment_agent_coverage=5,
                    sentiment_slow_decay=0.3,
                    sentiment_fast_decay=0.5,
                )
            ]
        )
        assert db._conn is not None
        row = db._conn.execute(
            "SELECT sentiment_score, sentiment_confidence, sentiment_agent_coverage, "
            "       sentiment_slow_decay, sentiment_fast_decay "
            "FROM signal_history"
        ).fetchone()
        assert row == (0.42, 0.71, 5, 0.3, 0.5)

    def test_absent_sentiment_writes_nulls_and_zero_coverage(self, db: CandleDB) -> None:
        """Honest ABSENT: row omits the sentiment keys entirely → coverage
        defaults to 0, scores stay NULL.  This is the path the analysis
        script reads as the no-sentiment partition."""
        db.write_signal_history_batch([_base_row()])
        assert db._conn is not None
        row = db._conn.execute(
            "SELECT sentiment_score, sentiment_confidence, sentiment_agent_coverage, "
            "       sentiment_slow_decay, sentiment_fast_decay "
            "FROM signal_history"
        ).fetchone()
        assert row == (None, None, 0, None, None)

    def test_partial_coverage_writes_actual_count(self, db: CandleDB) -> None:
        """A 2-agent day must record coverage=2, not 0.  The overlay is
        degraded right now (FRED 400 + Cerebras 404) and the harness
        depends on this honesty to flag low-coverage cells."""
        db.write_signal_history_batch(
            [
                _base_row(
                    sentiment_score=0.1,
                    sentiment_confidence=0.4,
                    sentiment_agent_coverage=2,
                    sentiment_slow_decay=0.1,
                    sentiment_fast_decay=None,
                )
            ]
        )
        assert db._conn is not None
        cov = db._conn.execute("SELECT sentiment_agent_coverage FROM signal_history").fetchone()[0]
        assert cov == 2


class TestMigrationIdempotent:
    def test_pre_migration_db_gets_columns_added(self, tmp_path: Path) -> None:
        """Open a 'legacy' DB that has signal_history without the
        sentiment columns (simulating an older candles.db), then point
        CandleDB at it and call init_db.  The migration should add every
        new column without touching existing rows."""
        db_path = tmp_path / "legacy.db"
        legacy = sqlite3.connect(db_path)
        legacy.execute(
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
        legacy.execute(
            "INSERT INTO signal_history (scored_at, symbol, horizon_bars, "
            "mean_return, entry_price) VALUES (?, ?, ?, ?, ?)",
            (1_780_000_000_000, "USD/JPY", 120, 0.002, 156.5),
        )
        legacy.commit()
        legacy.close()

        cdb = CandleDB(db_path=str(db_path))
        cdb.init_db()
        try:
            assert cdb._conn is not None
            cols = {r[1] for r in cdb._conn.execute("PRAGMA table_info(signal_history)").fetchall()}
            for new_col in (
                "sentiment_score",
                "sentiment_confidence",
                "sentiment_agent_coverage",
                "sentiment_slow_decay",
                "sentiment_fast_decay",
            ):
                assert new_col in cols
            # Existing row still readable; sentiment cols are NULL/0
            row = cdb._conn.execute(
                "SELECT symbol, sentiment_score, sentiment_agent_coverage FROM signal_history"
            ).fetchone()
            assert row == ("USD/JPY", None, 0)

            # Running init_db again must not error (ALTER would fail if
            # we re-attempted it; the table_info check should skip every
            # already-present column).
            cdb.close()
            cdb2 = CandleDB(db_path=str(db_path))
            cdb2.init_db()
            cdb2.close()
        finally:
            cdb.close()


# ---------------------------------------------------------------------------
# Partitioning + asset-class logic
# ---------------------------------------------------------------------------


class TestAssetClassMapping:
    @pytest.mark.parametrize(
        "symbol,expected",
        [
            ("EUR/USD", "fx_major"),
            ("USD/JPY", "fx_major"),
            ("EUR/GBP", "fx_major"),
            ("EUR/AUD", "fx_major"),
            ("AUD/JPY", "fx_major"),
            ("XAU/USD", "metals"),
            ("XAG/USD", "metals"),
            # US single-name shares carry per-name equity sectors, which the
            # harness collapses to "other" (only "metals"/"equity_index" pass through).
            ("F", "other"),
            ("XOM", "other"),
            ("UNKNOWN/SYMBOL", "other"),
        ],
    )
    def test_collapses_to_three_classes_plus_other(self, symbol: str, expected: str) -> None:
        assert asset_class_for(symbol) == expected


class TestPartition:
    """The partition routing decides what the gate would have done with a
    given signal under each *reading*.  Bug-class to avoid: an off-by-one
    on the sign convention would flip the entire analysis."""

    def _sig(
        self,
        *,
        mean_return: float,
        sentiment_score: float | None,
        coverage: int = 4,
        symbol: str = "EUR/USD",
    ) -> SentimentSignal:
        return SentimentSignal(
            scored_at_ms=0,
            symbol=symbol,
            mean_return=mean_return,
            realized_return=0.0,
            sentiment_score=sentiment_score,
            sentiment_agent_coverage=coverage,
        )

    def test_momentum_long_positive_sentiment_is_agree(self) -> None:
        s = self._sig(mean_return=0.01, sentiment_score=0.4)
        assert partition_signal(s, "momentum") == "agree"

    def test_momentum_long_negative_sentiment_is_disagree(self) -> None:
        s = self._sig(mean_return=0.01, sentiment_score=-0.4)
        assert partition_signal(s, "momentum") == "disagree"

    def test_momentum_short_negative_sentiment_is_agree(self) -> None:
        s = self._sig(mean_return=-0.01, sentiment_score=-0.4)
        assert partition_signal(s, "momentum") == "agree"

    def test_contrarian_inverts_momentum_decisions(self) -> None:
        # Same input as the momentum_long_positive case → contrarian flips.
        s = self._sig(mean_return=0.01, sentiment_score=0.4)
        assert partition_signal(s, "contrarian") == "disagree"

    def test_zero_coverage_is_absent_regardless_of_reading(self) -> None:
        s = self._sig(mean_return=0.01, sentiment_score=0.4, coverage=0)
        assert partition_signal(s, "momentum") == "absent"
        assert partition_signal(s, "contrarian") == "absent"

    def test_null_sentiment_is_absent(self) -> None:
        s = self._sig(mean_return=0.01, sentiment_score=None, coverage=3)
        assert partition_signal(s, "momentum") == "absent"

    def test_kronos_flat_is_absent(self) -> None:
        # mean_return == 0 means Kronos has no direction — the gate can't
        # admit/reject a non-trade, so it routes to ABSENT.
        s = self._sig(mean_return=0.0, sentiment_score=0.5)
        assert partition_signal(s, "momentum") == "absent"

    def test_unknown_reading_raises(self) -> None:
        s = self._sig(mean_return=0.01, sentiment_score=0.4)
        with pytest.raises(ValueError):
            partition_signal(s, "neither")


class TestMagnitudeBuckets:
    """Magnitude bucketing decides whether the harness can see edge that
    appears only at the extremes.  Boundaries must be inclusive on the
    low edge and exclusive on the high (with the top bucket catching
    |s| == 1.0)."""

    def _sig(self, sentiment_score: float | None, coverage: int = 4) -> SentimentSignal:
        return SentimentSignal(
            scored_at_ms=0,
            symbol="EUR/USD",
            mean_return=0.01,
            realized_return=0.0,
            sentiment_score=sentiment_score,
            sentiment_agent_coverage=coverage,
        )

    @pytest.mark.parametrize(
        "score,bucket",
        [
            (0.0, "weak"),
            (0.10, "weak"),
            (-0.10, "weak"),
            (0.15, "moderate"),
            (0.25, "moderate"),
            (0.30, "strong"),
            (0.45, "strong"),
            (-0.45, "strong"),
            (0.60, "extreme"),
            (0.95, "extreme"),
            (-1.0, "extreme"),
        ],
    )
    def test_bucket_boundaries(self, score: float, bucket: str) -> None:
        assert self._sig(score).magnitude_bucket() == bucket

    def test_absent_returns_none(self) -> None:
        assert self._sig(None).magnitude_bucket() is None
        assert self._sig(0.5, coverage=0).magnitude_bucket() is None


# ---------------------------------------------------------------------------
# End-to-end compute_harness
# ---------------------------------------------------------------------------


class TestComputeHarness:
    def test_counts_with_sentiment_correctly(self) -> None:
        sigs = [
            SentimentSignal(0, "EUR/USD", 0.01, 0.005, 0.5, 6),
            SentimentSignal(0, "USD/JPY", 0.01, 0.002, None, 0),
            SentimentSignal(0, "XAU/USD", -0.01, 0.001, 0.3, 4),
        ]
        result = compute_harness(sigs)
        assert result.n_signals_total == 3
        assert result.n_signals_with_sentiment == 2  # the None-score row excluded

    def test_pnl_sign_follows_kronos_direction(self) -> None:
        """For LONG signals (mean_return > 0) realised_return enters as +
        into the PnL list; for SHORT signals it enters negated.  Easy to
        get wrong if AGREE / DISAGREE routing leaks into the sign too."""
        sigs = [
            # LONG Kronos, AGREE momentum → +1% PnL
            SentimentSignal(0, "EUR/USD", 0.01, 0.01, 0.5, 4),
            # SHORT Kronos, AGREE momentum → +0.5% PnL (we shorted, market went down)
            SentimentSignal(0, "EUR/USD", -0.01, -0.005, -0.5, 4),
        ]
        result = compute_harness(sigs)
        agree_cells = [
            c
            for c in result.cells
            if c.asset_class == "fx_major" and c.reading == "momentum" and c.partition == "agree"
        ]
        # Both sigs go to AGREE; only one bucket should have n=2
        total_n = sum(c.stats.n for c in agree_cells)
        assert total_n == 2
        # Mean across both should be (0.01 + 0.005) / 2 = 0.0075
        mean_pnls = [c.stats.mean_pnl * c.stats.n for c in agree_cells if c.stats.n > 0]
        assert sum(mean_pnls) / total_n == pytest.approx(0.0075)

    def test_absent_bucket_includes_zero_coverage_rows(self) -> None:
        sigs = [SentimentSignal(0, "EUR/USD", 0.01, 0.005, None, 0)]
        result = compute_harness(sigs)
        absent = next(
            c
            for c in result.cells
            if c.asset_class == "fx_major" and c.reading == "momentum" and c.partition == "absent"
        )
        assert absent.stats.n == 1

    def test_unknown_asset_class_goes_to_other(self) -> None:
        sigs = [SentimentSignal(0, "USO", 0.01, 0.002, 0.4, 4)]
        result = compute_harness(sigs)
        other_cells = [c for c in result.cells if c.asset_class == "other"]
        assert sum(c.stats.n for c in other_cells if c.partition == "agree") == 1

    def test_small_sample_flag_applies_to_low_n(self) -> None:
        """Boundary check: MIN_CELL_N - 1 must be flagged, MIN_CELL_N must
        not.  Split rows across two magnitude buckets so they land in
        distinct cells — |0.05| is 'weak', |0.4| is 'strong'."""
        from bot.analysis.sentiment_edge import report_to_json

        below = [SentimentSignal(0, "EUR/USD", 0.01, 0.005, 0.05, 4) for _ in range(MIN_CELL_N - 1)]
        at = [SentimentSignal(0, "EUR/USD", 0.01, 0.003, 0.4, 4) for _ in range(MIN_CELL_N)]
        result = compute_harness(below + at)
        payload = report_to_json(result)
        weak_cell = next(
            c
            for c in payload["cells"]
            if c["asset_class"] == "fx_major"
            and c["reading"] == "momentum"
            and c["partition"] == "agree"
            and c["magnitude_bucket"] == "weak"
        )
        assert weak_cell["n"] == MIN_CELL_N - 1
        assert weak_cell["small_sample"] is True
        strong_cell = next(
            c
            for c in payload["cells"]
            if c["asset_class"] == "fx_major"
            and c["reading"] == "momentum"
            and c["partition"] == "agree"
            and c["magnitude_bucket"] == "strong"
        )
        assert strong_cell["n"] == MIN_CELL_N
        assert strong_cell["small_sample"] is False
