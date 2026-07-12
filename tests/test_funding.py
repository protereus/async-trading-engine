"""Tests for the overnight funding cost estimator (IG_LIVE_RISK_REFERENCE.md §5).

Reference: IG_LIVE_RISK_REFERENCE.md §5 (overnight funding model with Wed ×3
for FX, Fri ×3 for equities / indices / commodities).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from bot.risk.funding import (
    BENCHMARK_RATE_USD,
    COMMODITY_ADMIN_PCT,
    COMMODITY_DIVISOR,
    EQUITY_ADMIN_PCT,
    EQUITY_DIVISOR,
    FOREX_ADMIN_PCT,
    FOREX_DIVISOR,
    daily_funding_pct,
    estimate_funding_over_horizon_pct,
    estimate_overnight_cost_gbp,
    is_equity_triple_day,
    is_fx_triple_day,
)
from bot.risk.ig_margin import AssetClass

# A specific reference date for each weekday — 2026-05-18 is a Monday
_MONDAY = datetime(2026, 5, 18, 22, 0, tzinfo=UTC)
_TUESDAY = datetime(2026, 5, 19, 22, 0, tzinfo=UTC)
_WEDNESDAY = datetime(2026, 5, 20, 22, 0, tzinfo=UTC)
_THURSDAY = datetime(2026, 5, 21, 22, 0, tzinfo=UTC)
_FRIDAY = datetime(2026, 5, 22, 22, 0, tzinfo=UTC)


@pytest.mark.preflight
class TestWeekdayTriggers:
    def test_fx_triple_is_wednesday_only(self) -> None:
        assert is_fx_triple_day(_WEDNESDAY) is True
        for d in (_MONDAY, _TUESDAY, _THURSDAY, _FRIDAY):
            assert is_fx_triple_day(d) is False

    def test_equity_triple_is_friday_only(self) -> None:
        assert is_equity_triple_day(_FRIDAY) is True
        for d in (_MONDAY, _TUESDAY, _WEDNESDAY, _THURSDAY):
            assert is_equity_triple_day(d) is False


class TestDailyFundingPct:
    def test_fx_long_normal_day(self) -> None:
        """FX long on a Monday → 1.5 %/360 per day with no multiplier."""
        pct = daily_funding_pct(asset_class=AssetClass.FOREX_MAJOR, side="BUY", now_utc=_MONDAY)
        assert pct == pytest.approx(FOREX_ADMIN_PCT / FOREX_DIVISOR)

    def test_fx_short_uses_same_admin(self) -> None:
        """Both sides pay the admin spread in our simplified model."""
        long_pct = daily_funding_pct(
            asset_class=AssetClass.FOREX_MAJOR, side="BUY", now_utc=_MONDAY
        )
        short_pct = daily_funding_pct(
            asset_class=AssetClass.FOREX_MAJOR, side="SELL", now_utc=_MONDAY
        )
        assert long_pct == pytest.approx(short_pct)

    def test_fx_wednesday_triples(self) -> None:
        """Wed-night roll covers Wed→Mon (T+2 settlement), so 3× the daily cost."""
        normal = daily_funding_pct(asset_class=AssetClass.FOREX_MAJOR, side="BUY", now_utc=_MONDAY)
        triple = daily_funding_pct(
            asset_class=AssetClass.FOREX_MAJOR, side="BUY", now_utc=_WEDNESDAY
        )
        assert triple == pytest.approx(3.0 * normal)

    def test_commodity_friday_triples(self) -> None:
        """Fri-night roll covers Fri→Mon (3 carry days) for non-FX assets."""
        normal = daily_funding_pct(asset_class=AssetClass.COMMODITY, side="BUY", now_utc=_MONDAY)
        triple = daily_funding_pct(asset_class=AssetClass.COMMODITY, side="BUY", now_utc=_FRIDAY)
        assert triple == pytest.approx(3.0 * normal)
        # And the absolute number matches IG's 3.4 %/360 admin formula
        assert normal == pytest.approx(COMMODITY_ADMIN_PCT / COMMODITY_DIVISOR)

    def test_gold_uses_commodity_rate(self) -> None:
        """Spot gold is in the commodity bucket — 3.4 %/360 admin."""
        gold = daily_funding_pct(asset_class=AssetClass.SPOT_GOLD, side="BUY", now_utc=_MONDAY)
        commodity = daily_funding_pct(asset_class=AssetClass.COMMODITY, side="BUY", now_utc=_MONDAY)
        assert gold == pytest.approx(commodity)

    def test_index_long_pays_benchmark_plus_admin(self) -> None:
        """§5.2 — long pays (bench + admin) annualised, divided by 365."""
        pct = daily_funding_pct(asset_class=AssetClass.INDEX_MAJOR, side="BUY", now_utc=_MONDAY)
        expected = (BENCHMARK_RATE_USD + EQUITY_ADMIN_PCT) / EQUITY_DIVISOR
        assert pct == pytest.approx(expected)

    def test_index_short_can_receive_carry(self) -> None:
        """§5.2 — short receives (bench - admin)/365.  In our simplified model
        with bench=4.5 % and admin=3 % the short *receives* net.  Sign
        convention: negative = credit to holder."""
        long_pct = daily_funding_pct(
            asset_class=AssetClass.INDEX_MAJOR, side="BUY", now_utc=_MONDAY
        )
        short_pct = daily_funding_pct(
            asset_class=AssetClass.INDEX_MAJOR, side="SELL", now_utc=_MONDAY
        )
        # Short cost = (admin - bench)/365 with admin < bench → negative
        assert short_pct < 0
        assert long_pct > 0
        # Magnitudes diverge by 2× admin (since both sides see admin, but bench
        # flips sign between long/short).
        diff = long_pct - short_pct  # = 2 × bench / 365
        assert diff == pytest.approx(2 * BENCHMARK_RATE_USD / EQUITY_DIVISOR)

    def test_index_friday_triples_both_sides(self) -> None:
        long_normal = daily_funding_pct(
            asset_class=AssetClass.INDEX_MAJOR, side="BUY", now_utc=_MONDAY
        )
        long_fri = daily_funding_pct(
            asset_class=AssetClass.INDEX_MAJOR, side="BUY", now_utc=_FRIDAY
        )
        assert long_fri == pytest.approx(3.0 * long_normal)


class TestEstimateOvernightCostGbp:
    def test_eur_usd_long_one_pound_per_pt(self) -> None:
        """£1/pt EUR/USD at IG level 11000 on a Monday →
        notional ≈ £11,000 × 1.5 %/360 ≈ £0.46 per night."""
        cost = estimate_overnight_cost_gbp(
            symbol="EUR/USD",
            size_per_pt=1.0,
            ig_level=11000.0,
            side="BUY",
            now_utc=_MONDAY,
        )
        assert cost == pytest.approx(11000.0 * FOREX_ADMIN_PCT / FOREX_DIVISOR)

    def test_eur_usd_wednesday_triples(self) -> None:
        mon = estimate_overnight_cost_gbp(
            symbol="EUR/USD",
            size_per_pt=2.0,
            ig_level=11000.0,
            side="BUY",
            now_utc=_MONDAY,
        )
        wed = estimate_overnight_cost_gbp(
            symbol="EUR/USD",
            size_per_pt=2.0,
            ig_level=11000.0,
            side="BUY",
            now_utc=_WEDNESDAY,
        )
        assert wed == pytest.approx(3.0 * mon)

    def test_uso_friday_triples(self) -> None:
        """USO routes to a commodity DFB; Fri ×3 applies."""
        mon = estimate_overnight_cost_gbp(
            symbol="USO",
            size_per_pt=1.0,
            ig_level=9000.0,
            side="BUY",
            now_utc=_MONDAY,
        )
        fri = estimate_overnight_cost_gbp(
            symbol="USO",
            size_per_pt=1.0,
            ig_level=9000.0,
            side="BUY",
            now_utc=_FRIDAY,
        )
        assert fri == pytest.approx(3.0 * mon)

    def test_zero_or_negative_inputs_return_zero(self) -> None:
        assert (
            estimate_overnight_cost_gbp(
                symbol="EUR/USD",
                size_per_pt=0.0,
                ig_level=11000.0,
                side="BUY",
                now_utc=_MONDAY,
            )
            == 0.0
        )
        assert (
            estimate_overnight_cost_gbp(
                symbol="EUR/USD",
                size_per_pt=1.0,
                ig_level=0.0,
                side="BUY",
                now_utc=_MONDAY,
            )
            == 0.0
        )

    def test_default_now_uses_current_utc(self) -> None:
        """Omitting now_utc must not throw — fills from datetime.now(UTC)."""
        cost = estimate_overnight_cost_gbp(
            symbol="EUR/USD",
            size_per_pt=1.0,
            ig_level=11000.0,
            side="BUY",
        )
        # Value depends on today's weekday but must be non-negative for FX longs.
        assert cost >= 0


class TestEstimateFundingOverHorizon:
    def test_horizon_zero_returns_zero(self) -> None:
        assert (
            estimate_funding_over_horizon_pct(
                symbol="EUR/USD", side="BUY", horizon_days=0, now_utc=_MONDAY
            )
            == 0.0
        )

    def test_five_day_horizon_starting_monday_no_wed_in_window_for_fx(self) -> None:
        """Mon→Fri = 5 days; the Wed in the middle triggers FX ×3 once,
        the other 4 days are normal → total = 4 + 3 = 7 day-equivalents."""
        pct = estimate_funding_over_horizon_pct(
            symbol="EUR/USD", side="BUY", horizon_days=5, now_utc=_MONDAY
        )
        daily = FOREX_ADMIN_PCT / FOREX_DIVISOR
        # Day 0 (Mon) + Day 1 (Tue) + Day 2 (Wed, ×3) + Day 3 (Thu) + Day 4 (Fri)
        expected = daily * (1 + 1 + 3 + 1 + 1)
        assert pct == pytest.approx(expected)

    def test_five_day_horizon_starting_monday_picks_up_friday_for_equity(self) -> None:
        """Mon→Fri equity horizon → Fri ×3 lands on the last day."""
        pct = estimate_funding_over_horizon_pct(
            symbol="USO", side="BUY", horizon_days=5, now_utc=_MONDAY
        )
        daily = COMMODITY_ADMIN_PCT / COMMODITY_DIVISOR
        expected = daily * (1 + 1 + 1 + 1 + 3)
        assert pct == pytest.approx(expected)
