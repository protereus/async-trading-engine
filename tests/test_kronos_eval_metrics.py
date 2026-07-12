"""Tests for the Kronos eval metrics core (pure CPU — no torch)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "kronos_eval"))

from costs import load_costs, round_trip_cost  # type: ignore[import-not-found]  # noqa: E402
from metrics import (  # type: ignore[import-not-found]  # noqa: E402
    cross_sectional_rank_ic,
    hit_rate,
    ic,
    per_symbol_ic,
    rank_ic,
    signal_backtest,
    summarize,
)


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_ic_perfect_positive() -> None:
    assert ic([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.0)


def test_ic_perfect_negative() -> None:
    assert ic([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_ic_degenerate_is_nan() -> None:
    assert math.isnan(ic([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
    assert math.isnan(ic([1.0], [1.0]))


def test_rank_ic_is_monotonic_invariant() -> None:
    # Rank IC = 1 for any monotonic-increasing relationship, even if nonlinear.
    assert rank_ic([1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 9.0, 16.0]) == pytest.approx(1.0)


def test_hit_rate() -> None:
    # 3 of 4 signs agree; the zero-realized row is excluded.
    assert hit_rate([0.01, -0.02, 0.03, -0.04], [0.01, 0.02, 0.03, 0.0]) == pytest.approx(2 / 3)


def test_hit_rate_all_excluded_is_nan() -> None:
    assert math.isnan(hit_rate([0.0, 0.0], [0.0, 0.0]))


def test_signal_backtest_applies_round_trip_cost() -> None:
    costs = {"forex": 0.0002}
    df = _df(
        [
            {"asset_class": "forex", "tradeable": True, "realized_return": 0.0100},
            {"asset_class": "forex", "tradeable": True, "realized_return": -0.0050},
            {"asset_class": "forex", "tradeable": False, "realized_return": 0.5000},  # excluded
        ]
    )
    out = signal_backtest(df, costs)
    pooled = out["pooled"]
    assert pooled["n_trades"] == 2
    # nets: 0.0098, -0.0052
    assert pooled["total"] == pytest.approx(0.0098 - 0.0052)
    assert pooled["mean"] == pytest.approx((0.0098 - 0.0052) / 2)
    assert pooled["hit_rate"] == pytest.approx(0.5)


def test_signal_backtest_per_class_split() -> None:
    costs = {"forex": 0.0002, "equity": 0.0010}
    df = _df(
        [
            {"asset_class": "forex", "tradeable": True, "realized_return": 0.0100},
            {"asset_class": "equity", "tradeable": True, "realized_return": 0.0100},
        ]
    )
    out = signal_backtest(df, costs)
    assert out["by_class"]["forex"]["mean"] == pytest.approx(0.0098)
    assert out["by_class"]["equity"]["mean"] == pytest.approx(0.0090)


def test_signal_backtest_no_trades() -> None:
    df = _df([{"asset_class": "forex", "tradeable": False, "realized_return": 0.01}])
    assert signal_backtest(df, {"forex": 0.0002})["pooled"]["n_trades"] == 0


def test_cross_sectional_rank_ic_perfect() -> None:
    # Two timestamps, 3 symbols each, predicted order == realised order → RankIC 1.
    rows = []
    for ts in ("t1", "t2"):
        for i, sym in enumerate(("A", "B", "C")):
            rows.append(
                {
                    "origin_ts": ts,
                    "symbol": sym,
                    "mean_return": float(i),
                    "realized_return": float(i),
                }
            )
    assert cross_sectional_rank_ic(_df(rows)) == pytest.approx(1.0)


def test_cross_sectional_rank_ic_skips_thin_timestamps() -> None:
    rows = [{"origin_ts": "t1", "symbol": "A", "mean_return": 1.0, "realized_return": 1.0}]
    assert math.isnan(cross_sectional_rank_ic(_df(rows), min_symbols=3))


def test_summarize_structure_and_perfect_model() -> None:
    rows = [
        {
            "symbol": "EUR/USD",
            "asset_class": "forex",
            "origin_ts": "t1",
            "mean_return": 0.01,
            "direction_confidence": 0.9,
            "uncertainty": 1.0,
            "tradeable": True,
            "realized_return": 0.01,
        },
        {
            "symbol": "GBP/USD",
            "asset_class": "forex",
            "origin_ts": "t1",
            "mean_return": 0.02,
            "direction_confidence": 0.9,
            "uncertainty": 1.0,
            "tradeable": True,
            "realized_return": 0.02,
        },
        {
            "symbol": "KO",
            "asset_class": "equity",
            "origin_ts": "t1",
            "mean_return": 0.03,
            "direction_confidence": 0.9,
            "uncertainty": 1.0,
            "tradeable": True,
            "realized_return": 0.03,
        },
    ]
    out = summarize(_df(rows), load_costs(None))
    assert out["n_origins"] == 3
    assert out["n_symbols"] == 3
    assert out["n_tradeable"] == 3
    # headline (panel-correct) metrics
    psi = out["per_symbol_ic"]
    assert psi["n_symbols"] == 3
    assert set(psi["by_class"]) == {"forex", "equity"}
    # deprecated pooled IC retained for continuity
    assert out["ic"]["pooled"] == pytest.approx(1.0)
    assert set(out["ic"]["by_class"]) == {"forex", "equity"}
    assert "pooled" in out["signal_backtest"]


def test_per_symbol_ic_not_fooled_by_between_symbol_effect() -> None:
    # Within each symbol pred==real (per-symbol IC = 1), but the two symbols sit at
    # opposite mean levels so a POOLED IC would be dragged negative. The per-symbol
    # aggregate must report mean_ic≈1 and both symbols positive (Simpson guard).
    rows = []
    for sym, base in (("EUR/USD", 0.0), ("GBP/USD", 0.0)):
        for i in range(4):
            v = base + (i - 1.5) * 0.01
            rows.append(
                {"symbol": sym, "asset_class": "forex", "mean_return": v, "realized_return": v}
            )
    psi = per_symbol_ic(_df(rows))
    assert psi["n_symbols"] == 2
    assert psi["mean_ic"] == pytest.approx(1.0)
    assert psi["n_pos_ic"] == 2
    assert psi["by_class"]["forex"]["n_pos_ic"] == 2


def test_load_costs_overrides_defaults(tmp_path: Path) -> None:
    p = tmp_path / "costs.json"
    p.write_text('{"forex": 0.001}')
    costs = load_costs(str(p))
    assert round_trip_cost("forex", costs) == 0.001
    assert round_trip_cost("equity", costs) == 0.0010  # default preserved
