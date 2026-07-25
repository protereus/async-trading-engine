"""Tests for scripts/kronos_split.py (train/val/test split manifest).

The script is not a package module — load it via importlib so the tests
exercise exactly what ``uv run python scripts/kronos_split.py`` runs.

Note: this is distinct from ``test_kronos_split_windows.py``, which tests the
shared windowing helpers in ``scripts/kronos_src/finetune_csv/split_windows.py``
directly — this file tests the manifest-building script that consumes them.

Covers:
  - ``_symbol_from_filename`` — CSV filename to bot_key convention
  - ``_compute_split`` — the no-leakage train/val/test boundary computation
    (the property the docstring calls out as safety-critical)
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "kronos_split.py"
_spec = importlib.util.spec_from_file_location("kronos_split", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
ksplit = importlib.util.module_from_spec(_spec)
sys.modules["kronos_split"] = ksplit
_spec.loader.exec_module(ksplit)


def _write_csv(path: Path, n_rows: int) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamps", "open", "high", "low", "close", "volume", "amount"])
        for i in range(n_rows):
            ts = f"2026-01-01 {i:02d}:00:00"
            writer.writerow([ts, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])


class TestSymbolFromFilename:
    def test_underscore_pair_becomes_slash_symbol(self) -> None:
        assert ksplit._symbol_from_filename("EUR_USD.csv") == "EUR/USD"

    def test_single_part_stem_is_returned_unchanged(self) -> None:
        assert ksplit._symbol_from_filename("KO.csv") == "KO"

    def test_non_csv_extension_is_kept_as_part_of_the_stem(self) -> None:
        # Only a trailing ".csv" is stripped; anything else stays.
        assert ksplit._symbol_from_filename("EUR_USD.txt") != "EUR/USD"


class TestComputeSplit:
    def test_boundaries_match_split_bounds_helper(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "EUR_USD.csv"
        _write_csv(csv_path, n_rows=200)

        result = ksplit._compute_split(
            "EUR/USD",
            csv_path,
            lookback=10,
            predict=5,
            train_frac=0.7,
            val_frac=0.15,
            included=True,
        )

        expected_val_start, expected_test_start = ksplit.split_bounds(200, 0.7, 0.15)
        assert result.val_start_idx == expected_val_start
        assert result.test_start_idx == expected_test_start
        assert result.n_bars == 200

    def test_sample_counts_match_valid_start_indices_directly(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "GBP_USD.csv"
        n_rows = 300
        lookback, predict = 20, 10
        train_frac, val_frac = 0.7, 0.15
        _write_csv(csv_path, n_rows=n_rows)

        result = ksplit._compute_split(
            "GBP/USD",
            csv_path,
            lookback=lookback,
            predict=predict,
            train_frac=train_frac,
            val_frac=val_frac,
            included=True,
        )

        val_start, test_start = ksplit.split_bounds(n_rows, train_frac, val_frac)
        assert result.n_train_samples == len(
            ksplit.valid_start_indices(n_rows, lookback, predict, 0, val_start)
        )
        assert result.n_val_samples == len(
            ksplit.valid_start_indices(n_rows, lookback, predict, val_start, test_start)
        )
        assert result.n_test_samples == len(
            ksplit.valid_start_indices(n_rows, lookback, predict, test_start, n_rows)
        )

    def test_excluded_symbol_gets_excluded_flag_only(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "XAU_USD.csv"
        _write_csv(csv_path, n_rows=200)

        result = ksplit._compute_split(
            "XAU/USD",
            csv_path,
            lookback=10,
            predict=5,
            train_frac=0.7,
            val_frac=0.15,
            included=False,
        )

        assert result.flags == ["EXCLUDED_FROM_V1"]

    def test_thin_history_flags_low_train_samples(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "THIN_SYM.csv"
        # Too few bars to reach _MIN_TRAIN_SAMPLES (50) after windowing.
        _write_csv(csv_path, n_rows=40)

        result = ksplit._compute_split(
            "THIN/SYM",
            csv_path,
            lookback=10,
            predict=5,
            train_frac=0.7,
            val_frac=0.15,
            included=True,
        )

        assert any("LOW_TRAIN_SAMPLES" in f for f in result.flags)

    def test_split_regions_do_not_leak_across_boundaries(self, tmp_path: Path) -> None:
        """The no-leakage invariant the module docstring calls safety-critical:
        0 <= val_start <= test_start <= n_bars, and every train/val/test window's
        *target* region is strictly confined inside its own split's range -- no
        window's target bars ever straddle a boundary or land in another split.
        """
        csv_path = tmp_path / "EUR_USD.csv"
        n_rows = 521
        lookback, predict = 40, 10
        train_frac, val_frac = 0.7, 0.15
        _write_csv(csv_path, n_rows=n_rows)

        result = ksplit._compute_split(
            "EUR/USD",
            csv_path,
            lookback=lookback,
            predict=predict,
            train_frac=train_frac,
            val_frac=val_frac,
            included=True,
        )
        val_start, test_start = result.val_start_idx, result.test_start_idx

        assert 0 <= val_start <= test_start <= result.n_bars == n_rows

        train_idx = ksplit.valid_start_indices(n_rows, lookback, predict, 0, val_start)
        val_idx = ksplit.valid_start_indices(n_rows, lookback, predict, val_start, test_start)
        test_idx = ksplit.valid_start_indices(n_rows, lookback, predict, test_start, n_rows)

        # Start indices must be disjoint across splits.
        assert not (set(train_idx) & set(val_idx))
        assert not (set(val_idx) & set(test_idx))
        assert not (set(train_idx) & set(test_idx))

        # The actual no-leakage property, checked independently per sample: every
        # window's *target* region is fully confined to its own split's range.
        for s in train_idx:
            assert s + lookback + predict < val_start
        for s in val_idx:
            assert val_start <= s + lookback
            assert s + lookback + predict < test_start
        for s in test_idx:
            assert test_start <= s + lookback
            assert s + lookback + predict < n_rows

        assert result.n_train_samples == len(train_idx)
        assert result.n_val_samples == len(val_idx)
        assert result.n_test_samples == len(test_idx)
