"""Tests for bot.risk.scale_guard — the D4 quote-scale drift detector.

Pins the drift math against the real ``ig_quote_scale`` table so the
ETF-proxy mis-scaling failure mode can't recur silently.
"""

from __future__ import annotations

import pytest

from bot.execution.ig_quote_scale import ig_quote_scale
from bot.risk.scale_guard import (
    DriftSeverity,
    ScaleDriftResult,
    classify_drift,
    compute_drift,
)


class TestClassifyDrift:
    @pytest.mark.parametrize(
        ("drift", "expected"),
        [
            (0.0, DriftSeverity.OK),
            (0.05, DriftSeverity.OK),
            (-0.09, DriftSeverity.OK),
            (0.10, DriftSeverity.WARN),
            (-0.10, DriftSeverity.WARN),
            (0.24, DriftSeverity.WARN),
            (0.25, DriftSeverity.CRITICAL),
            (-0.322, DriftSeverity.CRITICAL),  # a real observed ETF-proxy drift
            (0.381, DriftSeverity.CRITICAL),  # the real UNG drift
        ],
    )
    def test_thresholds_by_magnitude(self, drift: float, expected: DriftSeverity) -> None:
        assert classify_drift(drift) == expected

    def test_custom_thresholds(self) -> None:
        assert classify_drift(0.06, warn=0.05, critical=0.20) == DriftSeverity.WARN
        assert classify_drift(0.21, warn=0.05, critical=0.20) == DriftSeverity.CRITICAL


class TestComputeDrift:
    def test_clean_symbol_zero_drift(self) -> None:
        """XAU/USD (EODHD: GLD candle → IG spot level, scale ~10.84): when the
        IG mid sits exactly at candle × expected_scale, drift is zero."""
        scale = ig_quote_scale("XAU/USD")
        result = compute_drift("XAU/USD", candle_price=412.0, ig_mid=412.0 * scale)
        assert result is not None
        assert result.expected_scale == pytest.approx(scale)
        assert result.real_scale == pytest.approx(scale)
        assert result.drift == pytest.approx(0.0, abs=1e-9)
        assert result.severity is DriftSeverity.OK

    def test_ig_native_metal_is_drift_invariant(self) -> None:
        """IG-native metals (XAG/USD): candle_price IS the IG level, scale=1.0,
        so drift is zero by construction — the ETF-proxy scale-drift failure
        mode (an ETF proxy quoted on a different scale) can't occur for them."""
        # Same IG-level reading on both sides of the ratio → zero drift.
        result = compute_drift("XAG/USD", candle_price=7313.0, ig_mid=7313.0)
        assert result is not None
        assert result.expected_scale == 1.0
        assert result.drift == pytest.approx(0.0)
        assert result.severity is DriftSeverity.OK

    def test_critical_drift_still_flagged(self) -> None:
        """A configured-vs-real scale mismatch is still caught.  XAU is
        IG-native (configured scale 1.0); a real scale of 1.30 (candle 100 vs
        IG mid 130) → +30 % drift → CRITICAL."""
        result = compute_drift("XAU/USD", candle_price=100.0, ig_mid=130.0)
        assert result is not None
        assert result.expected_scale == 1.0
        assert result.drift == pytest.approx(0.30)
        assert result.severity is DriftSeverity.CRITICAL
        assert result.implied_pnl_error == pytest.approx(0.30)

    def test_warn_drift_classifies_warn(self) -> None:
        """XAU configured scale 1.0; a real scale of 1.10 → +10 % drift → WARN
        (not CRITICAL)."""
        result = compute_drift("XAU/USD", candle_price=100.0, ig_mid=110.0)
        assert result is not None
        assert result.drift == pytest.approx(0.10)
        assert result.severity is DriftSeverity.WARN

    def test_real_scale_matches_inverse_pip(self) -> None:
        """When IG mid exactly equals candle × configured-scale, drift is 0
        for every symbol regardless of its pip table entry."""
        for symbol in ("EUR/USD", "USD/JPY", "XAU/USD", "XAG/USD", "F", "XOM"):
            scale = ig_quote_scale(symbol)
            result = compute_drift(symbol, candle_price=100.0, ig_mid=100.0 * scale)
            assert result is not None
            assert result.drift == pytest.approx(0.0)
            assert result.severity is DriftSeverity.OK

    def test_non_positive_inputs_return_none(self) -> None:
        """A missing / zero quote is a feed problem, not a drift event — skip."""
        assert compute_drift("USO", candle_price=0.0, ig_mid=8855.0) is None
        assert compute_drift("USO", candle_price=130.0, ig_mid=0.0) is None
        assert compute_drift("USO", candle_price=-1.0, ig_mid=8855.0) is None

    def test_result_is_frozen(self) -> None:
        import dataclasses

        result = compute_drift("USO", candle_price=130.65, ig_mid=8855.20)
        assert isinstance(result, ScaleDriftResult)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.drift = 0.0  # type: ignore[misc]
