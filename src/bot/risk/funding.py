"""IG overnight funding cost estimator (IG_LIVE_RISK_REFERENCE.md §5).

Reference: ``IG_LIVE_RISK_REFERENCE.md §5`` — overnight funding model with
Wednesday ×3 for FX (T+2 settlement covers Wed→Mon) and Friday ×3 for
equities / indices / commodities (weekend roll).

This module is intentionally conservative: it uses fixed benchmark-rate proxies
(SONIA / SOFR / ESTR) and a zero default tom-next rate rather than wiring a
live rate feed.  Refining those into live values is a follow-up — the structure
here keeps the call sites stable while we plug in real rates later.

For our 28-asset universe the per-day funding cost as a fraction of notional is:

| Asset class     | Daily cost long  | Daily cost short | Weekday ×3    |
|-----------------|------------------|------------------|---------------|
| Forex (major)   | 1.5 % / 360      | 1.5 % / 360      | Wed           |
| Forex (minor)   | 1.5 % / 360      | 1.5 % / 360      | Wed           |
| Index DFB       | (bench+3 %) /365 | (admin-bench)/365| Fri           |
| Spot gold       | 3.4 % / 360      | 3.4 % / 360      | Fri           |
| Commodity DFB   | 3.4 % / 360      | 3.4 % / 360      | Fri           |
| Equity ETF      | (bench+3 %) /365 | (admin-bench)/365| Fri           |

The IG admin markup (1.5 % FX, 3.0 % equities, 3.4 % commodities) is the
*absolute capital destruction* — it's charged regardless of which side of the
benchmark the position sits on.  Index/equity shorts can therefore go net
negative (credit) when the benchmark is high and the admin smaller; we surface
that via the sign of the returned figure.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from bot.data.eodhd_symbols import AssetClass
from bot.risk.ig_margin import classify_symbol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Admin markups (annualised) — IG_LIVE_RISK_REFERENCE.md §5
# ---------------------------------------------------------------------------
FOREX_ADMIN_PCT = 0.015  # §5.1
EQUITY_ADMIN_PCT = 0.030  # §5.2 mid of 2.5-3.4 % range
COMMODITY_ADMIN_PCT = 0.034  # §5.3 — covers spot gold and other commodities

# Year divisors per IG convention.  GBP-base equities use 365, everything else
# 360.  We're on a GBP account so the equity-side divisor is 365 — small
# difference but kept for fidelity with IG's published numbers.
FOREX_DIVISOR = 360.0
EQUITY_DIVISOR = 365.0
COMMODITY_DIVISOR = 360.0

# Benchmark-rate proxies (annualised) — SONIA / SOFR / ESTR are all in the
# 3-5 % range as of 2026-Q2.  We use a single conservative USD proxy because
# most of our index / commodity universe is USD-denominated.  Refresh
# manually until a live feed is wired.
BENCHMARK_RATE_USD = 0.045
BENCHMARK_RATE_GBP = 0.045
BENCHMARK_RATE_EUR = 0.030

# Weekday multipliers — applied when the 22:00 UTC rollover *falls on* the
# named weekday.  FX settles T+2, so a Wed-night roll covers Wed→Mon (3
# carry days).  Equities/indices/commodities use a Fri-night ×3 for the
# weekend.
FX_WEEKDAY_MULT_WED = 3.0
EQUITY_WEEKDAY_MULT_FRI = 3.0

# Annualised → daily cost helpers (positive = position holder pays).
_FOREX_DAILY = FOREX_ADMIN_PCT / FOREX_DIVISOR
_COMMODITY_DAILY = COMMODITY_ADMIN_PCT / COMMODITY_DIVISOR


def is_fx_triple_day(now_utc: datetime) -> bool:
    """True if the upcoming 22:00 UTC rollover triggers FX ×3 (Wed-night)."""
    return now_utc.weekday() == 2  # Mon=0, Wed=2


def is_equity_triple_day(now_utc: datetime) -> bool:
    """True if the upcoming 22:00 UTC rollover triggers equity ×3 (Fri-night)."""
    return now_utc.weekday() == 4  # Fri=4


def _weekday_multiplier(asset_class: AssetClass, now_utc: datetime) -> float:
    if asset_class in (AssetClass.FOREX_MAJOR, AssetClass.FOREX_MINOR):
        return FX_WEEKDAY_MULT_WED if is_fx_triple_day(now_utc) else 1.0
    if is_equity_triple_day(now_utc):
        return EQUITY_WEEKDAY_MULT_FRI
    return 1.0


def daily_funding_pct(
    *,
    asset_class: AssetClass,
    side: str,
    now_utc: datetime,
) -> float:
    """Daily funding charge as a fraction of position notional, signed.

    Positive = the position holder pays carry.  Negative = the position holder
    *receives* carry (short equity above the admin spread).  Apply the
    appropriate weekday multiplier already baked in.
    """
    is_long = side.upper() == "BUY"
    mult = _weekday_multiplier(asset_class, now_utc)

    if asset_class in (AssetClass.FOREX_MAJOR, AssetClass.FOREX_MINOR):
        # FX swap per §5.1: tom-next ± admin/360.  We approximate tom-next as 0,
        # so both sides pay the admin daily.  Conservative for a circuit
        # check — actual carry is usually within 1-2 bp of this.
        return _FOREX_DAILY * mult

    if asset_class in (AssetClass.SPOT_GOLD, AssetClass.COMMODITY):
        # §5.3 — basis adjustment is cash-neutral; IG admin is absolute.
        return _COMMODITY_DAILY * mult

    if asset_class in (AssetClass.INDEX_MAJOR, AssetClass.EQUITY_ETF):
        # §5.2 — long pays (bench + admin); short receives (bench - admin).
        bench = BENCHMARK_RATE_USD  # most of our index/ETF universe is USD
        if is_long:
            return (bench + EQUITY_ADMIN_PCT) / EQUITY_DIVISOR * mult
        # Short: receive bench, pay admin.  Sign convention: positive = cost.
        return (EQUITY_ADMIN_PCT - bench) / EQUITY_DIVISOR * mult

    return 0.0


def estimate_overnight_cost_gbp(
    *,
    symbol: str,
    size_per_pt: float,
    ig_level: float,
    side: str,
    now_utc: datetime | None = None,
) -> float:
    """Estimate the GBP cost of *one* overnight roll for a £/pt spread bet.

    Positive = the holder pays.  Uses the same conservative notional
    approximation as ``ig_margin.estimate_margin_gbp`` (size_per_pt × ig_level).
    Returns 0.0 on invalid inputs so callers can use the result unconditionally.
    """
    if size_per_pt <= 0 or ig_level <= 0:
        return 0.0
    if now_utc is None:
        now_utc = datetime.now(UTC)
    asset_class = classify_symbol(symbol)
    notional = size_per_pt * ig_level
    return notional * daily_funding_pct(asset_class=asset_class, side=side, now_utc=now_utc)


def estimate_funding_over_horizon_pct(
    *,
    symbol: str,
    side: str,
    horizon_days: int,
    now_utc: datetime | None = None,
) -> float:
    """Cumulative funding as a fraction of notional over *horizon_days*.

    Sums each daily charge across the holding period, applying the weekday
    multipliers correctly (one Wed in 7 days for FX → ×3, one Fri in 7 days
    for equities → ×3).  Useful for the strategy-level EV check: if this
    figure approaches the projected gross return, the trade isn't worth the
    funding drag.
    """
    if horizon_days <= 0:
        return 0.0
    if now_utc is None:
        now_utc = datetime.now(UTC)
    asset_class = classify_symbol(symbol)
    total = 0.0
    for d in range(horizon_days):
        roll_ts = now_utc.replace(hour=22, minute=0, second=0, microsecond=0)
        # Roll on day d after now's date — calendar-day, not trading-day.
        from datetime import timedelta

        roll_day = roll_ts + timedelta(days=d)
        total += daily_funding_pct(asset_class=asset_class, side=side, now_utc=roll_day)
    return total


def log_overnight_funding_estimate(symbol: str, size_per_pt: float, ig_level: float) -> None:
    """Log the GBP cost of the *next* overnight roll for a new LONG
    position.  Includes Wed (FX) and Fri (equities/commodities) ×3 multipliers
    so the operator sees the realistic carry, not a generic hour-of-day
    warning.  Cheap: only fires at order-placement time."""
    if size_per_pt <= 0 or ig_level <= 0:
        return
    now_utc = datetime.now(UTC)
    cost_gbp = estimate_overnight_cost_gbp(
        symbol=symbol,
        size_per_pt=size_per_pt,
        ig_level=ig_level,
        side="BUY",
        now_utc=now_utc,
    )
    triple = ""
    if is_fx_triple_day(now_utc):
        triple = " (Wed FX ×3)"
    elif is_equity_triple_day(now_utc):
        triple = " (Fri equity ×3)"
    logger.info(
        "Overnight funding estimate: %s LONG £%.2f/pt → next-roll cost £%.2f%s",
        symbol,
        size_per_pt,
        cost_gbp,
        triple,
    )
