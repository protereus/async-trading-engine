"""Tests for kronos_signals.py — path metric extraction.

Covers:
  - extract_path_signal returns None on missing columns / empty DataFrame
  - Monotonic-up path: monotonicity ≈ 1, peak_bar at last bar, zero drawdown
  - Monotonic-down path: monotonicity ≈ -1, mean_return < 0
  - Flat path: monotonicity == 0 (flat series, std == 0)
  - V-shape path: peak_bar at bar 0, high drawdown
  - Inverse-V path: peak_bar in the middle, moderate drawdown
  - predicted_mfe_pct and predicted_mae_pct vs entry_price
  - ranking_score formula: mean_return × |monotonicity| × mfe_confirmation
  - select_top_k_path filters and ranking
"""

from __future__ import annotations

import pandas as pd

from bot.strategy.kronos_signals import KronosPathSignal, extract_path_signal, select_top_k_path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(closes: list[float], spread: float = 0.01) -> pd.DataFrame:
    """Build a minimal OHLC DataFrame from a close series.

    high = close * (1 + spread), low = close * (1 - spread), open = close.
    """
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * (1.0 + spread) for c in closes],
            "low": [c * (1.0 - spread) for c in closes],
            "close": closes,
        }
    )


def _sig(
    symbol: str = "EUR/USD",
    mean_return: float = 0.005,
    std_return: float = 0.002,
    direction_confidence: float = 0.80,
    monotonicity: float = 0.90,
    predicted_mfe_pct: float = 0.010,
    predicted_mae_pct: float = 0.003,
    predicted_peak_bar: int = 5,
    predicted_volatility: float = 0.02,
    ranking_score: float = 0.004,
) -> KronosPathSignal:
    """Construct a minimal KronosPathSignal for select_top_k_path tests."""
    entry = 1.0850
    return KronosPathSignal(
        symbol=symbol,
        mean_return=mean_return,
        std_return=std_return,
        direction_confidence=direction_confidence,
        uncertainty=std_return / (abs(mean_return) + 1e-8),
        stop_pct=max(std_return * 2.0, 0.005),
        predicted_max_high=entry * (1.0 + predicted_mfe_pct),
        predicted_min_low=entry * (1.0 - predicted_mae_pct),
        predicted_mfe_pct=predicted_mfe_pct,
        predicted_mae_pct=predicted_mae_pct,
        predicted_peak_bar=predicted_peak_bar,
        predicted_volatility=predicted_volatility,
        predicted_path_drawdown=0.01,
        monotonicity=monotonicity,
        ranking_score=ranking_score,
    )


# ---------------------------------------------------------------------------
# Null / guard cases
# ---------------------------------------------------------------------------


class TestExtractPathSignalGuards:
    def test_missing_columns_returns_none(self) -> None:
        df = pd.DataFrame({"close": [1.0, 1.1]})  # no high/low/open
        result = extract_path_signal("SYM", df, 1.0, 0.001, 0.8)
        assert result is None

    def test_empty_dataframe_returns_none(self) -> None:
        df = pd.DataFrame(columns=["open", "high", "low", "close"])
        result = extract_path_signal("SYM", df, 1.0, 0.001, 0.8)
        assert result is None

    def test_single_row_does_not_crash(self) -> None:
        df = _make_df([1.08])
        result = extract_path_signal("SYM", df, 1.08, 0.001, 0.8)
        assert result is not None
        assert result.monotonicity == 0.0  # single point — no correlation


# ---------------------------------------------------------------------------
# Monotonic-up path
# ---------------------------------------------------------------------------


class TestMonotonicUp:
    def test_monotonicity_close_to_1(self) -> None:
        closes = [1.0 + i * 0.01 for i in range(10)]  # strictly ascending
        df = _make_df(closes)
        sig = extract_path_signal("UP", df, closes[0], 0.002, 0.85)
        assert sig is not None
        assert sig.monotonicity > 0.99

    def test_peak_bar_at_last_bar(self) -> None:
        closes = [1.0 + i * 0.01 for i in range(10)]
        df = _make_df(closes)
        sig = extract_path_signal("UP", df, closes[0], 0.002, 0.85)
        assert sig is not None
        assert sig.predicted_peak_bar == len(closes) - 1

    def test_zero_drawdown(self) -> None:
        closes = [1.0 + i * 0.01 for i in range(10)]
        df = _make_df(closes)
        sig = extract_path_signal("UP", df, closes[0], 0.002, 0.85)
        assert sig is not None
        assert sig.predicted_path_drawdown < 1e-9

    def test_positive_mean_return(self) -> None:
        closes = [1.0 + i * 0.01 for i in range(10)]
        df = _make_df(closes)
        sig = extract_path_signal("UP", df, closes[0], 0.002, 0.85)
        assert sig is not None
        assert sig.mean_return > 0


# ---------------------------------------------------------------------------
# Monotonic-down path
# ---------------------------------------------------------------------------


class TestMonotonicDown:
    def test_monotonicity_close_to_minus_1(self) -> None:
        closes = [1.0 - i * 0.01 for i in range(10)]  # strictly descending
        df = _make_df(closes, spread=0.005)
        sig = extract_path_signal("DOWN", df, closes[0], 0.002, 0.30)
        assert sig is not None
        assert sig.monotonicity < -0.99

    def test_mean_return_negative(self) -> None:
        closes = [1.0 - i * 0.01 for i in range(10)]
        df = _make_df(closes, spread=0.005)
        sig = extract_path_signal("DOWN", df, closes[0], 0.002, 0.30)
        assert sig is not None
        assert sig.mean_return < 0

    def test_peak_bar_at_first_bar(self) -> None:
        closes = [1.0 - i * 0.01 for i in range(10)]
        df = _make_df(closes, spread=0.005)
        sig = extract_path_signal("DOWN", df, closes[0], 0.002, 0.30)
        assert sig is not None
        assert sig.predicted_peak_bar == 0


# ---------------------------------------------------------------------------
# Flat path
# ---------------------------------------------------------------------------


class TestFlatPath:
    def test_monotonicity_is_zero_for_flat(self) -> None:
        closes = [1.08] * 10
        df = _make_df(closes)
        sig = extract_path_signal("FLAT", df, closes[0], 0.001, 0.75)
        assert sig is not None
        assert sig.monotonicity == 0.0

    def test_mean_return_near_zero(self) -> None:
        closes = [1.08] * 10
        df = _make_df(closes)
        sig = extract_path_signal("FLAT", df, closes[0], 0.001, 0.75)
        assert sig is not None
        assert abs(sig.mean_return) < 1e-9


# ---------------------------------------------------------------------------
# V-shape path (down then up — initial dip)
# ---------------------------------------------------------------------------


class TestVShapePath:
    def test_peak_bar_near_end(self) -> None:
        # Dips to bar 5, then climbs back above start
        closes = [1.0, 0.99, 0.98, 0.97, 0.96, 0.97, 0.99, 1.01, 1.03, 1.05]
        df = _make_df(closes, spread=0.005)
        sig = extract_path_signal("V", df, closes[0], 0.003, 0.70)
        assert sig is not None
        assert sig.predicted_peak_bar == len(closes) - 1

    def test_high_drawdown(self) -> None:
        closes = [1.0, 0.99, 0.98, 0.97, 0.96, 0.97, 0.99, 1.01, 1.03, 1.05]
        df = _make_df(closes, spread=0.005)
        sig = extract_path_signal("V", df, closes[0], 0.003, 0.70)
        assert sig is not None
        # running_max starts at 1.0 and close drops to 0.96 → drawdown ≈ 4%
        assert sig.predicted_path_drawdown > 0.03

    def test_predicted_mfe_pct_positive(self) -> None:
        closes = [1.0, 0.99, 0.98, 0.97, 0.96, 0.97, 0.99, 1.01, 1.03, 1.05]
        df = _make_df(closes, spread=0.005)
        sig = extract_path_signal("V", df, closes[0], 0.003, 0.70)
        assert sig is not None
        # max high = 1.05 * 1.005; entry = 1.0 → mfe > 0
        assert sig.predicted_mfe_pct > 0

    def test_predicted_mae_pct_positive(self) -> None:
        closes = [1.0, 0.99, 0.98, 0.97, 0.96, 0.97, 0.99, 1.01, 1.03, 1.05]
        df = _make_df(closes, spread=0.005)
        sig = extract_path_signal("V", df, closes[0], 0.003, 0.70)
        assert sig is not None
        # min low = 0.96 * 0.995; entry = 1.0 → mae > 0
        assert sig.predicted_mae_pct > 0


# ---------------------------------------------------------------------------
# Inverse-V path (up then down — peak in the middle)
# ---------------------------------------------------------------------------


class TestInverseVPath:
    def test_peak_bar_in_middle(self) -> None:
        closes = [1.0, 1.02, 1.04, 1.06, 1.08, 1.06, 1.04, 1.02, 1.00, 0.98]
        df = _make_df(closes, spread=0.005)
        sig = extract_path_signal("INV_V", df, closes[0], 0.003, 0.65)
        assert sig is not None
        assert sig.predicted_peak_bar == 4  # argmax of closes

    def test_moderate_drawdown(self) -> None:
        closes = [1.0, 1.02, 1.04, 1.06, 1.08, 1.06, 1.04, 1.02, 1.00, 0.98]
        df = _make_df(closes, spread=0.005)
        sig = extract_path_signal("INV_V", df, closes[0], 0.003, 0.65)
        assert sig is not None
        # peak = 1.08, final = 0.98 → drawdown ≈ 9.3%
        assert sig.predicted_path_drawdown > 0.08

    def test_negative_mean_return(self) -> None:
        closes = [1.0, 1.02, 1.04, 1.06, 1.08, 1.06, 1.04, 1.02, 1.00, 0.98]
        df = _make_df(closes, spread=0.005)
        sig = extract_path_signal("INV_V", df, closes[0], 0.003, 0.65)
        assert sig is not None
        # final close (0.98) < entry (1.0) → mean_return < 0
        assert sig.mean_return < 0


# ---------------------------------------------------------------------------
# MFE / MAE reference values
# ---------------------------------------------------------------------------


class TestMFEandMAE:
    def test_mfe_pct_matches_formula(self) -> None:
        entry = 1.0
        closes = [1.0, 1.02, 1.05, 1.03, 1.01]
        df = pd.DataFrame(
            {
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
            }
        )
        sig = extract_path_signal("MFE", df, entry, 0.001, 0.8)
        assert sig is not None
        expected_mfe = (max(closes) - entry) / entry
        assert abs(sig.predicted_mfe_pct - expected_mfe) < 1e-9

    def test_mae_pct_matches_formula(self) -> None:
        entry = 1.05
        closes = [1.05, 1.04, 1.02, 1.03, 1.05]
        df = pd.DataFrame(
            {
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
            }
        )
        sig = extract_path_signal("MAE", df, entry, 0.001, 0.8)
        assert sig is not None
        expected_mae = (entry - min(closes)) / entry
        assert abs(sig.predicted_mae_pct - expected_mae) < 1e-9


# ---------------------------------------------------------------------------
# Ranking score formula
# ---------------------------------------------------------------------------


class TestRankingScore:
    def test_ranking_score_formula(self) -> None:
        # Fully monotonic-up, mfe_pct >> mean_return → mfe_confirmation capped at 1
        closes = [1.0 + i * 0.005 for i in range(20)]
        df = _make_df(closes, spread=0.001)
        sig = extract_path_signal("RS", df, closes[0], 0.001, 0.85)
        assert sig is not None
        # ranking_score = mean_return × max(0, monotonicity) × min(1, mfe/mean)
        abs_mean = abs(sig.mean_return)
        mfe_conf = min(1.0, sig.predicted_mfe_pct / abs_mean) if abs_mean > 1e-9 else 0.0
        expected = sig.mean_return * max(0.0, sig.monotonicity) * mfe_conf
        assert abs(sig.ranking_score - expected) < 1e-9

    def test_zero_mean_return_gives_zero_ranking_score(self) -> None:
        closes = [1.0] * 10  # flat — mean_return == 0
        df = _make_df(closes)
        sig = extract_path_signal("ZERO", df, closes[0], 0.001, 0.75)
        assert sig is not None
        assert sig.ranking_score == 0.0

    def test_negative_monotonicity_gives_zero_ranking_score(self) -> None:
        # Descending path: mean_return < 0, monotonicity ≈ -1
        # max(0, -1) = 0 → ranking_score = 0 (not selected anyway by LONG filter)
        closes = [1.0 - i * 0.005 for i in range(10)]
        df = _make_df(closes, spread=0.001)
        sig = extract_path_signal("NEG", df, closes[0], 0.001, 0.30)
        assert sig is not None
        assert sig.monotonicity < -0.99
        assert sig.ranking_score == 0.0

    def test_positive_return_negative_monotonicity_gives_zero_score(self) -> None:
        # Sharp spike at bar 1 then a slow grind down, ending just above entry.
        # mean_return > 0 (closes[-1] > entry) but the early peak dominates the
        # linear fit → negative Pearson correlation (monotonicity < 0).
        # abs() would give a positive ranking weight; max(0, .) correctly zeroes it.
        closes = [1.0, 1.3, 1.2, 1.1, 1.05, 1.03, 1.02, 1.01, 1.005, 1.002]
        df = _make_df(closes, spread=0.0)  # zero spread so highs == lows == closes
        sig = extract_path_signal("SPIKE", df, closes[0], 0.003, 0.70)
        assert sig is not None
        assert sig.mean_return > 0  # ends above entry
        assert sig.monotonicity < 0  # early spike pulls Pearson corr negative
        assert sig.ranking_score == 0.0  # max(0, negative) = 0

    def test_clean_rise_ranked_above_noisy_rise(self) -> None:
        # Same mean_return; clean monotonic path should outscore a noisy one.
        # Clean: strictly ascending (monotonicity ≈ 1)
        clean = [1.0 + i * 0.005 for i in range(20)]
        # Noisy: zigzag ending at the same level (lower monotonicity)
        noisy = [1.0 + (i % 3 - 1) * 0.002 + i * 0.005 for i in range(20)]
        df_clean = _make_df(clean, spread=0.001)
        df_noisy = _make_df(noisy, spread=0.001)
        entry = 1.0
        sig_clean = extract_path_signal("CLEAN", df_clean, entry, 0.001, 0.85)
        sig_noisy = extract_path_signal("NOISY", df_noisy, entry, 0.001, 0.85)
        assert sig_clean is not None and sig_noisy is not None
        assert sig_clean.monotonicity > sig_noisy.monotonicity
        assert sig_clean.ranking_score > sig_noisy.ranking_score


# ---------------------------------------------------------------------------
# Raw arrays stored on the signal
# ---------------------------------------------------------------------------


class TestRawArrays:
    def test_closes_highs_lows_stored(self) -> None:
        closes = [1.0 + i * 0.01 for i in range(5)]
        df = _make_df(closes, spread=0.005)
        sig = extract_path_signal("ARR", df, closes[0], 0.001, 0.8)
        assert sig is not None
        assert len(sig.predicted_closes) == 5
        assert len(sig.predicted_highs) == 5
        assert len(sig.predicted_lows) == 5
        for c, h, lo in zip(
            sig.predicted_closes, sig.predicted_highs, sig.predicted_lows, strict=True
        ):
            assert h >= c >= lo


# ---------------------------------------------------------------------------
# select_top_k_path
# ---------------------------------------------------------------------------


class TestSelectTopKPath:
    def test_returns_top_k_symbols(self) -> None:
        s1 = _sig("A", ranking_score=0.010)
        s2 = _sig("B", ranking_score=0.005)
        s3 = _sig("C", ranking_score=0.003)
        result = select_top_k_path([s1, s2, s3], k=2)
        assert result == ["A", "B"]

    def test_filters_negative_mean_return(self) -> None:
        s1 = _sig("A", mean_return=-0.005)
        s2 = _sig("B", mean_return=0.005)
        result = select_top_k_path([s1, s2], k=2)
        assert result == ["B"]

    def test_filters_below_min_predicted_return(self) -> None:
        s1 = _sig("A", mean_return=0.0005)  # below default 0.001
        s2 = _sig("B", mean_return=0.005)
        result = select_top_k_path([s1, s2], k=2)
        assert result == ["B"]

    def test_filters_low_confidence(self) -> None:
        s1 = _sig("A", direction_confidence=0.50)  # below 0.70
        s2 = _sig("B", direction_confidence=0.80)
        result = select_top_k_path([s1, s2], k=2)
        assert result == ["B"]

    def test_filters_high_volatility(self) -> None:
        s1 = _sig("A", predicted_volatility=0.10)  # above default 0.05
        s2 = _sig("B", predicted_volatility=0.02)
        result = select_top_k_path([s1, s2], k=2)
        assert result == ["B"]

    def test_respects_k_limit(self) -> None:
        signals = [_sig(f"S{i}", ranking_score=float(i)) for i in range(10)]
        result = select_top_k_path(signals, k=3)
        assert len(result) == 3

    def test_empty_signals_returns_empty(self) -> None:
        assert select_top_k_path([], k=3) == []

    def test_no_tradeable_signals_returns_empty(self) -> None:
        s1 = _sig("A", direction_confidence=0.30)
        result = select_top_k_path([s1], k=3)
        assert result == []


# ---------------------------------------------------------------------------
# — horizon-matched ranking slice
# ---------------------------------------------------------------------------


class TestRankingHorizonSlice:
    """Verify the ranking_horizon_bars slice in extract_path_signal."""

    def test_zero_uses_terminal_bar_legacy(self) -> None:
        """ranking_horizon_bars=0 must be byte-identical to pre-10b behaviour."""
        # Monotonic rising path: 1.00, 1.01, 1.02, ..., 1.09
        closes = [1.00 + 0.01 * i for i in range(10)]
        df = _make_df(closes)
        sig = extract_path_signal(
            symbol="EUR/USD",
            pred_df=df,
            entry_price=1.00,
            std_return=0.002,
            direction_confidence=0.85,
            ranking_horizon_bars=0,
        )
        assert sig is not None
        # Terminal bar (close[-1] = 1.09) → return = 0.09
        assert abs(sig.mean_return - 0.09) < 1e-9

    def test_nonzero_h_slices_at_that_bar(self) -> None:
        """ranking_horizon_bars=3 uses close at bar index 2 (1-indexed)."""
        closes = [1.00 + 0.01 * i for i in range(10)]
        df = _make_df(closes)
        sig = extract_path_signal(
            symbol="EUR/USD",
            pred_df=df,
            entry_price=1.00,
            std_return=0.002,
            direction_confidence=0.85,
            ranking_horizon_bars=3,
        )
        assert sig is not None
        # close at bar 3 (1-indexed) = closes[2] = 1.02 → return = 0.02
        assert abs(sig.mean_return - 0.02) < 1e-9

    def test_h_clamped_to_rollout_length(self) -> None:
        """ranking_horizon_bars > rollout length clamps to the final bar."""
        closes = [1.00, 1.01, 1.02]
        df = _make_df(closes)
        sig = extract_path_signal(
            symbol="EUR/USD",
            pred_df=df,
            entry_price=1.00,
            std_return=0.002,
            direction_confidence=0.85,
            ranking_horizon_bars=120,  # rollout has only 3 bars
        )
        assert sig is not None
        assert abs(sig.mean_return - 0.02) < 1e-9

    def test_path_metrics_remain_full_path(self) -> None:
        """MFE/MAE/peak_bar/monotonicity stay computed over the full path
        even when ranking is sliced at a shorter horizon."""
        # Path peaks at bar 5 (close=1.05), then drops back to 1.01 by bar 9
        closes = [1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.04, 1.03, 1.02, 1.01]
        df = _make_df(closes, spread=0.005)
        sig_short = extract_path_signal(
            symbol="EUR/USD",
            pred_df=df,
            entry_price=1.00,
            std_return=0.002,
            direction_confidence=0.85,
            ranking_horizon_bars=3,
        )
        sig_full = extract_path_signal(
            symbol="EUR/USD",
            pred_df=df,
            entry_price=1.00,
            std_return=0.002,
            direction_confidence=0.85,
            ranking_horizon_bars=0,
        )
        assert sig_short is not None and sig_full is not None
        # mean_return differs (short uses bar 3, full uses bar 9)
        assert sig_short.mean_return != sig_full.mean_return
        # Path metrics are identical (computed across full path)
        assert sig_short.predicted_max_high == sig_full.predicted_max_high
        assert sig_short.predicted_min_low == sig_full.predicted_min_low
        assert sig_short.predicted_peak_bar == sig_full.predicted_peak_bar
        assert sig_short.predicted_mfe_pct == sig_full.predicted_mfe_pct
        assert sig_short.predicted_mae_pct == sig_full.predicted_mae_pct
        # peak_bar (close peak) is at bar 5
        assert sig_short.predicted_peak_bar == 5
