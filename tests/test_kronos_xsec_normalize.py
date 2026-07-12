"""Tests for the cross-sectional normalization prototype (Prototype A)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "kronos_eval"))

from xsec_normalize import _add_scores, _xsec_rankic  # type: ignore[import-not-found]  # noqa: E402


def _df() -> pd.DataFrame:
    # 3 symbols with strong, opposite per-symbol BIASES (range 0.20) that dwarf
    # the per-rerank skill (range 0.02). realized follows the skill, not the bias.
    # → raw ranking (bias-dominated) is wrong; per-symbol demeaning recovers skill.
    bias = {"A": 0.10, "B": 0.00, "C": -0.10}
    skill = {  # each symbol sees each skill value once → per-symbol mean equal
        1: {"A": 0.01, "B": 0.02, "C": 0.03},
        2: {"A": 0.03, "B": 0.01, "C": 0.02},
        3: {"A": 0.02, "B": 0.03, "C": 0.01},
    }
    rows = []
    for ts, sk in skill.items():
        for sym, s in sk.items():
            rows.append(
                {
                    "symbol": sym,
                    "asset_class": "forex",
                    "origin_ts": ts,
                    "mean_return": bias[sym] + s,
                    "std_return": 0.01,
                    "realized_return": s,
                }
            )
    return pd.DataFrame(rows)


def test_per_symbol_demeaning_flips_ranking_positive() -> None:
    df = _df()
    _add_scores(df)
    raw_ic, _, _ = _xsec_rankic(df, "raw")
    dm_ic, n_ts, _ = _xsec_rankic(df, "psym_demean_la")
    # Per-symbol demeaning recovers the skill order at every rerank → RankIC = 1.
    assert dm_ic == pytest.approx(1.0)
    # Raw ranking is bias-dominated and strictly worse.
    assert dm_ic > raw_ic
    assert n_ts == 3


def test_demean_removes_per_symbol_bias() -> None:
    df = _df()
    _add_scores(df)
    means = df.groupby("symbol")["psym_demean_la"].mean()
    assert means.abs().max() == pytest.approx(0.0, abs=1e-12)
