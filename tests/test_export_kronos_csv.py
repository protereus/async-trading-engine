"""Tests for scripts/export_kronos_csv.py (Phase A Kronos fine-tune data export).

The script is not a package module — load it via importlib so the tests
exercise exactly what ``uv run python scripts/export_kronos_csv.py`` runs.

Covers the pure data-quality-check logic that flags bad candle history
before it feeds the fine-tune pipeline:
  - ``_median`` — plain median (even/odd length, empty)
  - ``_volume_step_ratio`` — volume-units-discontinuity detector
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "export_kronos_csv.py"
_spec = importlib.util.spec_from_file_location("export_kronos_csv", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
export_csv = importlib.util.module_from_spec(_spec)
sys.modules["export_kronos_csv"] = export_csv
_spec.loader.exec_module(export_csv)


class TestMedian:
    def test_empty_list_is_zero(self) -> None:
        assert export_csv._median([]) == 0.0

    def test_odd_length_returns_middle_element(self) -> None:
        assert export_csv._median([3.0, 1.0, 2.0]) == 2.0

    def test_even_length_averages_two_middle_elements(self) -> None:
        assert export_csv._median([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)

    def test_single_element(self) -> None:
        assert export_csv._median([7.0]) == 7.0

    def test_unsorted_input_is_sorted_first(self) -> None:
        assert export_csv._median([5.0, 1.0, 3.0, 2.0, 4.0]) == 3.0


class TestVolumeStepRatio:
    def test_fewer_than_eight_values_returns_zero(self) -> None:
        assert export_csv._volume_step_ratio([1.0] * 7) == 0.0

    def test_stable_volume_series_ratio_near_one(self) -> None:
        vols = [100.0] * 20

        assert export_csv._volume_step_ratio(vols) == pytest.approx(1.0)

    def test_large_jump_flags_high_ratio(self) -> None:
        # First-quartile (early, low volume) vs last-quartile (late, high volume) —
        # mirrors the IG-native-cutover discontinuity this check exists to catch.
        vols = [10.0] * 10 + [80.0] * 10

        ratio = export_csv._volume_step_ratio(vols)

        assert ratio == pytest.approx(8.0)

    def test_zero_first_quartile_median_returns_zero_not_divide_by_zero(self) -> None:
        vols = [0.0] * 10 + [50.0] * 10

        assert export_csv._volume_step_ratio(vols) == 0.0
