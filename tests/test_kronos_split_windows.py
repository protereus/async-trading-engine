"""Pin the Kronos windowing logic (manifest builder == evaluation loader).

``scripts/split_windows.py`` is the single source of truth shared by
``scripts/kronos_split.py`` and the evaluation harness. These tests lock the
sample counts against the pinned manifest and assert the no-leakage invariant.
Pure logic — no torch — so it runs in the normal suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from split_windows import (  # type: ignore[import-not-found]  # noqa: E402
    split_bounds,
    valid_start_indices,
)

_LOOKBACK = 400
_PREDICT = 120
_TRAIN, _VAL = 0.70, 0.15


def _counts(n: int) -> tuple[int, int, int]:
    val_start, test_start = split_bounds(n, _TRAIN, _VAL)
    train = len(valid_start_indices(n, _LOOKBACK, _PREDICT, 0, val_start))
    val = len(valid_start_indices(n, _LOOKBACK, _PREDICT, val_start, test_start))
    test = len(valid_start_indices(n, _LOOKBACK, _PREDICT, test_start, n))
    return train, val, test


def test_split_bounds_fx() -> None:
    assert split_bounds(2063, _TRAIN, _VAL) == (1444, 1753)


def test_split_bounds_equity() -> None:
    assert split_bounds(1046, _TRAIN, _VAL) == (732, 889)


def test_fx_sample_counts_match_manifest() -> None:
    # EUR/USD and the other 2063-bar FX series — see split_manifest.json.
    assert _counts(2063) == (924, 189, 190)


def test_equity_sample_counts_match_manifest() -> None:
    # BAC (1046 bars) — see split_manifest.json.
    assert _counts(1046) == (212, 37, 37)


def test_no_leakage_train_targets_below_val_start() -> None:
    n = 2063
    val_start, _ = split_bounds(n, _TRAIN, _VAL)
    train_starts = valid_start_indices(n, _LOOKBACK, _PREDICT, 0, val_start)
    # Every train window's last (target) bar must sit below val_start.
    assert max(s + _LOOKBACK + _PREDICT for s in train_starts) < val_start


def test_splits_are_disjoint() -> None:
    n = 2063
    val_start, test_start = split_bounds(n, _TRAIN, _VAL)
    train = set(valid_start_indices(n, _LOOKBACK, _PREDICT, 0, val_start))
    val = set(valid_start_indices(n, _LOOKBACK, _PREDICT, val_start, test_start))
    test = set(valid_start_indices(n, _LOOKBACK, _PREDICT, test_start, n))
    assert not (train & val)
    assert not (val & test)
    assert not (train & test)


def test_too_short_series_yields_no_samples() -> None:
    # Below one full window (lookback+predict+1 = 521) → no samples anywhere.
    assert _counts(400) == (0, 0, 0)
