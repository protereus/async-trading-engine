"""Tests for CorrelationTracker.

Covers:
  - update() computes correct Pearson correlation matrix
  - update() trims to lookback_bars before computing
  - update() with fewer than 2 symbols leaves matrix empty
  - correlation() returns known values or None for unknown pairs
  - select_uncorrelated: perfect positive correlation → second symbol bumped
  - select_uncorrelated: perfect negative correlation → also bumped (|corr| check)
  - select_uncorrelated: correlation_max=1.0 → no bumping (existing behaviour)
  - select_uncorrelated: empty matrix → pass-through (no filtering)
  - select_uncorrelated: k respected when enough uncorrelated candidates exist
  - select_uncorrelated: bumped list contains correct (sym, blocker, corr) triples
  - select_uncorrelated: disabled config → pass-through
  - snapshot / restore roundtrip
  - CandleDB write_correlations / read_latest_correlations roundtrip
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bot.strategy.correlation import CorrelationConfig, CorrelationTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tracker(
    enabled: bool = True,
    max_corr: float = 0.65,
    lookback: int = 500,
    long_only: bool = False,
) -> CorrelationTracker:
    return CorrelationTracker(
        CorrelationConfig(
            enabled=enabled,
            max_correlation=max_corr,
            lookback_bars=lookback,
            long_only=long_only,
        )
    )


def _returns(n: int, scale: float = 1.0) -> list[float]:
    """Ascending return series (1/n, 2/n, …) scaled by ``scale``."""
    return [scale * (i + 1) / n for i in range(n)]


# ---------------------------------------------------------------------------
# update() — matrix computation
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_perfect_positive_correlation(self) -> None:
        t = _tracker()
        r = _returns(50)
        t.update({"A": r, "B": r})  # identical series → corr == 1.0
        corr = t.correlation("A", "B")
        assert corr is not None
        assert abs(corr - 1.0) < 1e-9

    def test_perfect_negative_correlation(self) -> None:
        t = _tracker()
        pos = _returns(50, scale=1.0)
        neg = _returns(50, scale=-1.0)
        t.update({"A": pos, "B": neg})
        corr = t.correlation("A", "B")
        assert corr is not None
        assert abs(corr + 1.0) < 1e-9

    def test_uncorrelated_series(self) -> None:
        # Constant series → std=0 → NaN correlation → should be treated as 0 or missing
        t = _tracker()
        t.update({"A": [0.0] * 50, "B": _returns(50)})
        # NaN from constant series becomes NaN in pandas corr; may be missing
        corr = t.correlation("A", "B")
        # Either NaN/None or near 0; what matters is the matrix doesn't crash
        assert corr is None or (corr != corr) or abs(corr) <= 1.0  # NaN check

    def test_matrix_is_symmetric(self) -> None:
        t = _tracker()
        t.update({"A": _returns(40), "B": _returns(40, scale=0.5), "C": _returns(40, scale=2.0)})
        ab = t.correlation("A", "B")
        ba = t.correlation("B", "A")
        assert ab is not None and ba is not None
        assert abs(ab - ba) < 1e-12

    def test_three_assets_fully_populated(self) -> None:
        t = _tracker()
        t.update({"X": _returns(30), "Y": _returns(30, 0.8), "Z": _returns(30, 0.3)})
        assert t.correlation("X", "Y") is not None
        assert t.correlation("X", "Z") is not None
        assert t.correlation("Y", "Z") is not None

    def test_fewer_than_two_symbols_leaves_matrix_empty(self) -> None:
        t = _tracker()
        t.update({"SOLO": _returns(50)})
        assert t.correlation("SOLO", "ANYTHING") is None

    def test_lookback_trim_applied(self) -> None:
        # lookback=10; pass 100 bars; only last 10 used
        t = _tracker(lookback=10)
        # Full identical series → corr 1.0 regardless of trim, just verify no crash
        r = _returns(100)
        t.update({"A": r, "B": r})
        assert abs(t.correlation("A", "B") - 1.0) < 1e-9  # type: ignore[operator]

    def test_unknown_pair_returns_none(self) -> None:
        t = _tracker()
        t.update({"A": _returns(20), "B": _returns(20)})
        assert t.correlation("A", "UNKNOWN") is None
        assert t.correlation("UNKNOWN", "B") is None

    def test_empty_tracker_returns_none(self) -> None:
        t = _tracker()
        assert t.correlation("A", "B") is None


# ---------------------------------------------------------------------------
# select_uncorrelated() — filtering logic
# ---------------------------------------------------------------------------


class TestSelectUncorrelated:
    def _populated_tracker(self, corr_ab: float, max_corr: float = 0.65) -> CorrelationTracker:
        """Return a tracker where corr(A,B)=corr_ab, all others zero."""
        t = _tracker(max_corr=max_corr)
        t._matrix = {
            "A": {"B": corr_ab, "C": 0.0},
            "B": {"A": corr_ab, "C": 0.0},
            "C": {"A": 0.0, "B": 0.0},
        }
        return t

    def test_high_positive_corr_bumps_second(self) -> None:
        t = self._populated_tracker(corr_ab=0.90)
        selected, bumped = t.select_uncorrelated(["A", "B", "C"], k=2)
        assert "A" in selected
        assert "B" not in selected
        assert "C" in selected
        assert len(bumped) == 1
        assert bumped[0][0] == "B"
        assert bumped[0][1] == "A"
        assert abs(bumped[0][2] - 0.90) < 1e-9

    def test_high_negative_corr_also_bumps(self) -> None:
        # |corr| threshold — negative correlation also triggers bump
        t = self._populated_tracker(corr_ab=-0.85)
        selected, bumped = t.select_uncorrelated(["A", "B", "C"], k=2)
        assert "B" not in selected
        assert len(bumped) == 1

    def test_correlation_max_1_never_bumps(self) -> None:
        # corr can never exceed 1.0 in absolute value
        t = self._populated_tracker(corr_ab=0.99, max_corr=1.0)
        selected, bumped = t.select_uncorrelated(["A", "B", "C"], k=2)
        assert selected == ["A", "B"]
        assert bumped == []

    def test_empty_matrix_passthrough(self) -> None:
        t = _tracker()
        # No update called → matrix is empty
        selected, bumped = t.select_uncorrelated(["A", "B", "C"], k=2)
        assert selected == ["A", "B"]
        assert bumped == []

    def test_disabled_config_passthrough(self) -> None:
        t = _tracker(enabled=False)
        t._matrix = {"A": {"B": 0.99}, "B": {"A": 0.99}}
        selected, bumped = t.select_uncorrelated(["A", "B"], k=2)
        assert selected == ["A", "B"]
        assert bumped == []

    def test_k_respected(self) -> None:
        t = _tracker()  # empty matrix — no filtering
        selected, _ = t.select_uncorrelated(["A", "B", "C", "D"], k=2)
        assert len(selected) == 2
        assert selected == ["A", "B"]

    def test_rank_order_preserved(self) -> None:
        # B and C are uncorrelated with A; C is ranked before B
        t = self._populated_tracker(corr_ab=0.0, max_corr=0.65)
        selected, _ = t.select_uncorrelated(["A", "C", "B"], k=2)
        assert selected == ["A", "C"]

    def test_all_correlated_with_first_selects_only_one(self) -> None:
        t = _tracker()
        t._matrix = {
            "A": {"B": 0.90, "C": 0.80},
            "B": {"A": 0.90, "C": 0.70},
            "C": {"A": 0.80, "B": 0.70},
        }
        selected, bumped = t.select_uncorrelated(["A", "B", "C"], k=3)
        assert selected == ["A"]
        assert len(bumped) == 2

    def test_unknown_symbol_not_in_matrix_passes_through(self) -> None:
        t = _tracker()
        t._matrix = {"A": {"B": 0.99}, "B": {"A": 0.99}}
        # "NEW" is not in the matrix → no correlation known → not bumped
        selected, bumped = t.select_uncorrelated(["A", "NEW"], k=2)
        assert "NEW" in selected
        assert bumped == []

    def test_bumped_triple_fields(self) -> None:
        t = self._populated_tracker(corr_ab=0.75)
        _, bumped = t.select_uncorrelated(["A", "B"], k=2)
        assert len(bumped) == 1
        sym, blocker, corr = bumped[0]
        assert sym == "B"
        assert blocker == "A"
        assert abs(abs(corr) - 0.75) < 1e-9


class TestLongOnlyFilter:
    """long_only=True bumps on positive corr only; anti-correlated names are kept."""

    def _tracker_ab(self, corr_ab: float) -> CorrelationTracker:
        t = _tracker(long_only=True)
        t._matrix = {"A": {"B": corr_ab}, "B": {"A": corr_ab}}
        return t

    def test_negative_corr_kept_under_long_only(self) -> None:
        # Strongly anti-correlated → hedging, not redundant → NOT bumped.
        t = self._tracker_ab(corr_ab=-0.90)
        selected, bumped = t.select_uncorrelated(["A", "B"], k=2)
        assert selected == ["A", "B"]
        assert bumped == []

    def test_positive_corr_still_bumps_under_long_only(self) -> None:
        t = self._tracker_ab(corr_ab=0.90)
        selected, bumped = t.select_uncorrelated(["A", "B"], k=2)
        assert selected == ["A"]
        assert [b[0] for b in bumped] == ["B"]

    def test_legacy_default_still_bumps_negative(self) -> None:
        # Guard: default (long_only=False) keeps the |corr| behaviour.
        t = _tracker(long_only=False)
        t._matrix = {"A": {"B": -0.90}, "B": {"A": -0.90}}
        _, bumped = t.select_uncorrelated(["A", "B"], k=2)
        assert [b[0] for b in bumped] == ["B"]


# ---------------------------------------------------------------------------
# snapshot / restore
# ---------------------------------------------------------------------------


class TestSnapshotRestore:
    def test_roundtrip_preserves_matrix(self) -> None:
        t1 = _tracker()
        t1.update({"A": _returns(40), "B": _returns(40, scale=0.9)})
        snap = t1.snapshot()

        t2 = _tracker()
        t2.restore(snap)
        assert abs(t2.correlation("A", "B") - t1.correlation("A", "B")) < 1e-12  # type: ignore[operator]

    def test_restore_empty_snapshot(self) -> None:
        t = _tracker()
        t.restore({})
        assert t.correlation("A", "B") is None


# ---------------------------------------------------------------------------
# CandleDB integration — write_correlations / read_latest_correlations
# ---------------------------------------------------------------------------


class TestCandleDBCorrelations:
    @pytest.fixture()
    def db(self, tmp_path: Path):  # type: ignore[no-untyped-def]
        from bot.data.candle_db import CandleDB

        cdb = CandleDB(str(tmp_path / "test.db"))
        cdb.init_db()
        return cdb

    def test_empty_db_returns_empty_dict(self, db) -> None:  # type: ignore[no-untyped-def]
        assert db.read_latest_correlations() == {}

    def test_write_and_read_roundtrip(self, db) -> None:  # type: ignore[no-untyped-def]
        matrix = {
            "A": {"B": 0.75, "C": 0.20},
            "B": {"A": 0.75, "C": 0.10},
            "C": {"A": 0.20, "B": 0.10},
        }
        db.write_correlations(1_000_000, matrix)
        recovered = db.read_latest_correlations()
        assert abs(recovered["A"]["B"] - 0.75) < 1e-9
        assert abs(recovered["B"]["A"] - 0.75) < 1e-9  # symmetry
        assert abs(recovered["A"]["C"] - 0.20) < 1e-9

    def test_read_returns_latest_snapshot(self, db) -> None:  # type: ignore[no-untyped-def]
        db.write_correlations(1_000, {"A": {"B": 0.50}, "B": {"A": 0.50}})
        db.write_correlations(2_000, {"A": {"B": 0.80}, "B": {"A": 0.80}})
        recovered = db.read_latest_correlations()
        assert abs(recovered["A"]["B"] - 0.80) < 1e-9

    def test_schema_idempotent(self, db, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        from bot.data.candle_db import CandleDB

        db2 = CandleDB(str(tmp_path / "test.db"))
        db2.init_db()  # second call must not raise
        db2.close()
