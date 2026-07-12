"""Tests for scripts/calibrate_thresholds.py.

The script is not a package module — load it via importlib so the tests
exercise exactly what ``uv run python scripts/calibrate_thresholds.py`` runs.

Covers the decision logic the calibration report is built from:
  - ``_spearman`` rank correlation (perfect / inverted / degenerate inputs)
  - ``_grid_search`` cell admission, hit-rate and rate-per-day arithmetic
  - ``_load_cohort`` SQL filter semantics (resolved-only, gap exclusion,
    uncertainty cap, symbol LIKE + exclusion list)
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "calibrate_thresholds.py"
_spec = importlib.util.spec_from_file_location("calibrate_thresholds", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
calib = importlib.util.module_from_spec(_spec)
sys.modules["calibrate_thresholds"] = calib
_spec.loader.exec_module(calib)

_DAY_MS = 86_400_000


class TestSpearman:
    def test_perfect_monotonic_is_one(self) -> None:
        assert calib._spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(1.0)

    def test_perfect_inverse_is_minus_one(self) -> None:
        assert calib._spearman([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]) == pytest.approx(-1.0)

    def test_rank_based_not_linear(self) -> None:
        # Monotonic but wildly non-linear — Spearman stays 1.0 where Pearson wouldn't.
        assert calib._spearman([1.0, 2.0, 3.0], [1.0, 100.0, 10_000.0]) == pytest.approx(1.0)

    def test_fewer_than_three_points_is_nan(self) -> None:
        assert calib._spearman([1.0, 2.0], [1.0, 2.0]) != calib._spearman([1.0, 2.0], [1.0, 2.0])

    def test_constant_series_is_nan(self) -> None:
        result = calib._spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
        assert result != result  # NaN


def _row(
    scored_at: int,
    conf: float,
    mean_return: float,
    realized: float,
    symbol: str = "EUR/USD",
) -> dict:
    """_grid_search only does r[key] lookups, so a plain dict stands in for
    sqlite3.Row."""
    return {
        "scored_at": scored_at,
        "symbol": symbol,
        "mean_return": mean_return,
        "direction_confidence": conf,
        "uncertainty": 1.0,
        "realized_return_at_horizon": realized,
    }


class TestGridSearch:
    def test_empty_cohort_returns_no_cells(self) -> None:
        assert calib._grid_search([]) == []

    def test_grid_covers_every_confidence_return_pair(self) -> None:
        rows = [_row(0, 0.99, 0.02, 0.01)]
        cells = calib._grid_search(rows)
        expected = len(calib._CONFIDENCE_GRID) * len(calib._RETURN_GRID)
        assert len(cells) == expected
        assert {(c["conf"], c["ret"]) for c in cells} == {
            (conf, ret) for conf in calib._CONFIDENCE_GRID for ret in calib._RETURN_GRID
        }

    def test_admission_thresholds_are_strict_minima(self) -> None:
        rows = [
            _row(0 * _DAY_MS, conf=0.72, mean_return=0.004, realized=0.010),  # admitted @ 0.70
            _row(1 * _DAY_MS, conf=0.68, mean_return=0.004, realized=0.020),  # conf too low
            _row(2 * _DAY_MS, conf=0.90, mean_return=0.002, realized=0.030),  # ret too low
        ]
        cells = calib._grid_search(rows)
        cell = next(c for c in cells if c["conf"] == 0.70 and c["ret"] == 0.003)
        assert cell["n"] == 1
        assert cell["mean_ret"] == pytest.approx(0.010)

    def test_hit_rate_counts_positive_realized_only(self) -> None:
        rows = [
            _row(0 * _DAY_MS, 0.95, 0.02, +0.01),
            _row(1 * _DAY_MS, 0.95, 0.02, -0.01),
            _row(2 * _DAY_MS, 0.95, 0.02, +0.02),
            _row(3 * _DAY_MS, 0.95, 0.02, 0.0),  # flat is NOT a hit
        ]
        cells = calib._grid_search(rows)
        cell = next(c for c in cells if c["conf"] == 0.70 and c["ret"] == 0.001)
        assert cell["n"] == 4
        assert cell["hit"] == pytest.approx(2 / 4)

    def test_rate_is_entries_per_day_over_data_span(self) -> None:
        rows = [
            _row(0 * _DAY_MS, 0.95, 0.02, 0.01),
            _row(4 * _DAY_MS, 0.95, 0.02, 0.01),  # 4-day span, 2 entries
        ]
        cells = calib._grid_search(rows)
        cell = next(c for c in cells if c["conf"] == 0.70 and c["ret"] == 0.001)
        assert cell["rate"] == pytest.approx(2 / 4)

    def test_empty_cell_reports_zero_n_and_nan_stats(self) -> None:
        rows = [_row(0, conf=0.72, mean_return=0.002, realized=0.01)]
        cells = calib._grid_search(rows)
        cell = next(c for c in cells if c["conf"] == 0.95 and c["ret"] == 0.012)
        assert cell["n"] == 0
        assert cell["rate"] == 0.0
        assert cell["hit"] != cell["hit"]  # NaN


class TestLoadCohort:
    def _make_db(self, tmp_path: Path) -> str:
        db = tmp_path / "candles.db"
        con = sqlite3.connect(db)
        con.execute(
            """
            CREATE TABLE signal_history (
                scored_at INTEGER, symbol TEXT, mean_return REAL,
                direction_confidence REAL, uncertainty REAL,
                realized_return_at_horizon REAL, gap_spanned INTEGER
            )
            """
        )
        rows = [
            # (scored_at, symbol, mean_ret, conf, unc, realized, gap)
            (1, "EUR/USD", 0.004, 0.8, 1.0, 0.01, 0),  # clean forex — kept
            (2, "EUR/USD", 0.004, 0.8, 1.0, None, 0),  # unresolved — dropped
            (3, "GBP/USD", 0.004, 0.8, 1.0, 0.01, 1),  # gap-spanned — dropped
            (4, "USD/JPY", 0.004, 0.8, 99.0, 0.01, 0),  # uncertainty > cap — dropped
            (5, "AAPL", 0.004, 0.8, 1.0, 0.01, 0),  # no "/" — misses forex LIKE
            (6, "XAU/USD", 0.004, 0.8, 1.0, 0.01, 0),  # excluded explicitly
        ]
        con.executemany("INSERT INTO signal_history VALUES (?,?,?,?,?,?,?)", rows)
        con.commit()
        con.close()
        return str(db)

    def test_filters_compose(self, tmp_path: Path) -> None:
        db = self._make_db(tmp_path)
        rows = calib._load_cohort(db, "%/%", ("XAU/USD",))
        assert [(r["scored_at"], r["symbol"]) for r in rows] == [(1, "EUR/USD")]

    def test_symbols_like_override(self, tmp_path: Path) -> None:
        db = self._make_db(tmp_path)
        rows = calib._load_cohort(db, "AAPL", ())
        assert [r["symbol"] for r in rows] == ["AAPL"]
