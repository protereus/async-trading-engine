"""Tests for IG retail margin classification and pre-trade margin estimate.

References:
- IG_LIVE_RISK_REFERENCE.md §4.1 (retail margin rates)

"""

from __future__ import annotations

import pytest

from bot.risk.ig_margin import (
    GUARANTEED_STOP_PREMIUM_PCT,
    MARGIN_RATES,
    SLIPPAGE_PCT,
    AssetClass,
    classify_symbol,
    estimate_guaranteed_stop_premium_gbp,
    estimate_margin_gbp,
    estimate_slippage_pts,
    guaranteed_stop_premium_pct_for,
    margin_rate_for,
    slippage_pct_for,
)


class TestClassifySymbol:
    @pytest.mark.parametrize(
        "symbol",
        [
            "EUR/USD",
            "GBP/USD",
            "USD/JPY",
            "USD/CHF",
            "USD/CAD",
            "AUD/USD",
            "NZD/USD",
            "EUR/GBP",
            "EUR/JPY",
            "GBP/JPY",
        ],
    )
    def test_ten_majors_classified_as_major(self, symbol: str) -> None:
        assert classify_symbol(symbol) == AssetClass.FOREX_MAJOR

    @pytest.mark.parametrize("symbol", ["EUR/AUD", "AUD/JPY"])
    def test_crosses_classified_as_minor(self, symbol: str) -> None:
        assert classify_symbol(symbol) == AssetClass.FOREX_MINOR

    def test_gold_classified_as_spot_gold(self) -> None:
        assert classify_symbol("XAU/USD") == AssetClass.SPOT_GOLD

    def test_silver_is_commodity(self) -> None:
        """XAG/USD (IG Spot Silver DFB) attracts the 10 % commodity rate, not
        the 5 % spot-gold rate."""
        assert classify_symbol("XAG/USD") == AssetClass.COMMODITY

    @pytest.mark.parametrize("symbol", ["F", "XOM", "PFE"])
    def test_us_shares_are_equity(self, symbol: str) -> None:
        """The 14 US single-name shares trade as IG US-share DFBs → 20 % equity."""
        assert classify_symbol(symbol) == AssetClass.EQUITY_ETF

    def test_unknown_forex_pair_falls_back_to_minor(self) -> None:
        """Conservative default — over-estimating margin is the safe side."""
        assert classify_symbol("ZAR/MXN") == AssetClass.FOREX_MINOR

    def test_unknown_non_forex_falls_back_to_commodity(self) -> None:
        """Higher (10 %) default for unknown non-forex symbols."""
        assert classify_symbol("LITHIUM_2026") == AssetClass.COMMODITY


class TestMarginRateFor:
    def test_known_symbol_uses_table_rate(self) -> None:
        assert margin_rate_for("EUR/USD") == pytest.approx(0.0333)
        assert margin_rate_for("EUR/AUD") == pytest.approx(0.05)
        assert margin_rate_for("XAU/USD") == pytest.approx(0.05)
        assert margin_rate_for("XAG/USD") == pytest.approx(0.10)
        assert margin_rate_for("F") == pytest.approx(0.20)

    def test_all_asset_classes_have_a_rate(self) -> None:
        for cls in AssetClass:
            assert cls in MARGIN_RATES, f"missing rate for {cls}"
            assert 0 < MARGIN_RATES[cls] <= 0.5

    def test_etf_rate_present_even_though_we_dont_use_it(self) -> None:
        """If we ever start trading listed equities/ETFs directly, the lookup
        must still return the documented 20 % rate."""
        assert MARGIN_RATES[AssetClass.EQUITY_ETF] == pytest.approx(0.20)


class TestEstimateMarginGbp:
    def test_eur_usd_one_pound_per_pt_at_eleven_thousand(self) -> None:
        """EUR/USD £1/pt at IG level 11000 → notional ≈ £11,000 → margin ≈ £366.
        (Major-forex rate 3.33 %.)"""
        margin = estimate_margin_gbp(symbol="EUR/USD", size_per_pt=1.0, ig_level=11000.0)
        assert margin == pytest.approx(11000.0 * 0.0333, rel=1e-6)

    def test_gold_one_pound_per_pt_at_4700(self) -> None:
        """XAU/USD £1/pt at level 4700 → notional ≈ £4,700 → margin ≈ £235.
        (Spot-gold rate 5 %.)"""
        margin = estimate_margin_gbp(symbol="XAU/USD", size_per_pt=1.0, ig_level=4700.0)
        assert margin == pytest.approx(4700.0 * 0.05, rel=1e-6)

    def test_silver_uses_ten_percent_rate(self) -> None:
        """XAG/USD → IG Spot Silver commodity DFB → 10 %."""
        margin = estimate_margin_gbp(symbol="XAG/USD", size_per_pt=2.0, ig_level=7400.0)
        assert margin == pytest.approx(2.0 * 7400.0 * 0.10, rel=1e-6)

    def test_us_share_uses_equity_rate(self) -> None:
        """A US single-name share → IG US-share DFB → 20 % equity rate."""
        margin = estimate_margin_gbp(symbol="F", size_per_pt=1.0, ig_level=1200.0)
        assert margin == pytest.approx(1200.0 * 0.20, rel=1e-6)

    def test_scales_linearly_with_size(self) -> None:
        m1 = estimate_margin_gbp(symbol="EUR/USD", size_per_pt=1.0, ig_level=11000.0)
        m2 = estimate_margin_gbp(symbol="EUR/USD", size_per_pt=3.5, ig_level=11000.0)
        assert m2 == pytest.approx(3.5 * m1, rel=1e-6)

    def test_zero_size_returns_zero(self) -> None:
        assert estimate_margin_gbp(symbol="EUR/USD", size_per_pt=0.0, ig_level=11000.0) == 0.0

    def test_zero_or_negative_level_returns_zero(self) -> None:
        assert estimate_margin_gbp(symbol="EUR/USD", size_per_pt=1.0, ig_level=0.0) == 0.0
        assert estimate_margin_gbp(symbol="EUR/USD", size_per_pt=1.0, ig_level=-100.0) == 0.0


# ===========================================================================
# Slippage + guaranteed-stop premium per asset class
# IG_LIVE_RISK_REFERENCE.md §1.1
# ===========================================================================


class TestSlippagePct:
    def test_known_symbols(self) -> None:
        assert slippage_pct_for("EUR/USD") == pytest.approx(SLIPPAGE_PCT[AssetClass.FOREX_MAJOR])
        assert slippage_pct_for("EUR/AUD") == pytest.approx(SLIPPAGE_PCT[AssetClass.FOREX_MINOR])
        assert slippage_pct_for("XAU/USD") == pytest.approx(SLIPPAGE_PCT[AssetClass.SPOT_GOLD])
        assert slippage_pct_for("XAG/USD") == pytest.approx(SLIPPAGE_PCT[AssetClass.COMMODITY])

    def test_table_has_every_asset_class(self) -> None:
        for cls in AssetClass:
            assert cls in SLIPPAGE_PCT
            assert 0 < SLIPPAGE_PCT[cls] <= 0.01  # sanity: under 1 %

    def test_majors_cheaper_than_minors_than_commodities(self) -> None:
        assert SLIPPAGE_PCT[AssetClass.FOREX_MAJOR] < SLIPPAGE_PCT[AssetClass.FOREX_MINOR]
        assert SLIPPAGE_PCT[AssetClass.FOREX_MINOR] < SLIPPAGE_PCT[AssetClass.COMMODITY]


class TestEstimateSlippagePts:
    """``entry_price`` is in Twelve Data / native units (same as the input to
    ``RiskManager.compute_ig_size``), not the IG-level scale form."""

    def test_eur_usd_one_bp_at_native_price(self) -> None:
        """1 bp slippage on EUR/USD at native 1.10 with pip 0.0001 →
        1.10 × 0.0001 / 0.0001 = 1.1 IG points (≈ 1 pip)."""
        pts = estimate_slippage_pts("EUR/USD", entry_price=1.10, pip_value=0.0001)
        assert pts == pytest.approx(1.10 * SLIPPAGE_PCT[AssetClass.FOREX_MAJOR] / 0.0001)
        assert pts == pytest.approx(1.10)

    def test_gold_5bp_at_4500(self) -> None:
        """5 bp on gold at $4500/oz with pip 1.0 → 2.25 points."""
        pts = estimate_slippage_pts("XAU/USD", entry_price=4500.0, pip_value=1.0)
        assert pts == pytest.approx(4500.0 * 0.0005 / 1.0)
        assert pts == pytest.approx(2.25)

    def test_zero_or_negative_inputs_return_zero(self) -> None:
        assert estimate_slippage_pts("EUR/USD", entry_price=0.0, pip_value=0.0001) == 0.0
        assert estimate_slippage_pts("EUR/USD", entry_price=1.10, pip_value=0.0) == 0.0
        assert estimate_slippage_pts("EUR/USD", entry_price=-1.0, pip_value=0.0001) == 0.0


class TestGuaranteedStopPremium:
    def test_premium_pct_lookup(self) -> None:
        assert guaranteed_stop_premium_pct_for("EUR/USD") == pytest.approx(0.0030)
        assert guaranteed_stop_premium_pct_for("XAG/USD") == pytest.approx(0.0070)

    def test_table_covers_every_asset_class(self) -> None:
        for cls in AssetClass:
            assert cls in GUARANTEED_STOP_PREMIUM_PCT
            assert 0 < GUARANTEED_STOP_PREMIUM_PCT[cls] <= 0.05

    def test_estimate_gbp(self) -> None:
        """EUR/USD £1/pt at level 11000 → notional ≈ £11,000 → premium ≈ £33."""
        gbp = estimate_guaranteed_stop_premium_gbp(
            symbol="EUR/USD", size_per_pt=1.0, ig_level=11000.0
        )
        assert gbp == pytest.approx(11000.0 * 0.0030)

    def test_estimate_zero_for_invalid_inputs(self) -> None:
        assert (
            estimate_guaranteed_stop_premium_gbp(
                symbol="EUR/USD", size_per_pt=0.0, ig_level=11000.0
            )
            == 0.0
        )
        assert (
            estimate_guaranteed_stop_premium_gbp(symbol="EUR/USD", size_per_pt=1.0, ig_level=0.0)
            == 0.0
        )
