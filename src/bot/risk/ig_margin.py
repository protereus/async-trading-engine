"""IG retail margin rates + execution-cost estimates per asset class.

References:
- ``IG_LIVE_RISK_REFERENCE.md §1.1`` (slippage + guaranteed-stop premium)
- ``IG_LIVE_RISK_REFERENCE.md §4.1`` (retail margin rates)
- ``IG_LIVE_RISK_REFERENCE.md §4.2`` (tiered margining)

Covers tier-aware margin sizing, extended with
per-asset-class slippage and guaranteed-stop premium estimators so the size
calc bakes a worst-case fill into the risk budget rather than assuming the
stop ``level`` is the realised price.

Under FCA jurisdiction, retail spread-bet accounts have fixed minimum margin
floors per asset class:

| Asset class        | Margin rate | Max leverage |
|--------------------|-------------|--------------|
| Forex major        | 3.33 %      | 1:30         |
| Forex minor        | 5.00 %      | 1:20         |
| Index major        | 5.00 %      | 1:20         |
| Spot gold          | 5.00 %      | 1:20         |
| Other commodities  | 10.00 %     | 1:10         |
| Equity / ETFs      | 20.00 %     |  1:5         |

Symbols are classified via ``EODHD_UNIVERSE`` (the live 28-asset universe)
first, falling back to a small hand-maintained table for the twelvedata
warm-standby (a subset of the EODHD universe).  The default for unknown symbols
intentionally over-estimates margin.

Tier-2 escalations (margin rates rising with notional size) are not modelled:
our retail account would need to push past £150 k+ notional on a single FX
position before they kick in, which is several multiples of typical equity.
If we ever scale past that we revisit; for now the flat Tier-1 rates suffice.
"""

from __future__ import annotations

from enum import StrEnum

from bot.data.eodhd_symbols import EODHD_UNIVERSE


class AssetClass(StrEnum):
    FOREX_MAJOR = "forex_major"
    FOREX_MINOR = "forex_minor"
    INDEX_MAJOR = "index_major"
    SPOT_GOLD = "spot_gold"
    COMMODITY = "commodity"  # silver, oil, gas, copper, sovereign bonds
    EQUITY_ETF = "equity_etf"  # only if traded as direct equity, not index DFB


# Retail margin floors per IG_LIVE_RISK_REFERENCE.md §4.1.
MARGIN_RATES: dict[AssetClass, float] = {
    AssetClass.FOREX_MAJOR: 0.0333,
    AssetClass.FOREX_MINOR: 0.05,
    AssetClass.INDEX_MAJOR: 0.05,
    AssetClass.SPOT_GOLD: 0.05,
    AssetClass.COMMODITY: 0.10,
    AssetClass.EQUITY_ETF: 0.20,
}

# Slippage estimates as a fraction of entry price.  These are the
# *expected worst-case fill drift past the stop trigger*, sized to bake a
# realistic buffer into the position size so a real stop hit doesn't blow
# past the £-risked budget.  Conservative defaults from typical IG demo
# observations + IG_LIVE_RISK_REFERENCE.md §1.1 — refine once we have live
# fill-vs-trigger telemetry.
SLIPPAGE_PCT: dict[AssetClass, float] = {
    AssetClass.FOREX_MAJOR: 0.0001,  # 1 bp  (≈ 1 pip on EUR/USD)
    AssetClass.FOREX_MINOR: 0.0002,  # 2 bp  (wider spreads in minors)
    AssetClass.INDEX_MAJOR: 0.0005,  # 5 bp
    AssetClass.SPOT_GOLD: 0.0005,  # 5 bp  (~$2 on $4500 gold)
    AssetClass.COMMODITY: 0.0010,  # 10 bp  (oil/gas gap-prone)
    AssetClass.EQUITY_ETF: 0.0010,  # 10 bp
}

# Guaranteed-stop premium as a fraction of position notional.  IG
# charges this at order placement and refunds it iff the guaranteed stop is
# never triggered.  Defaults follow the volatility-tier table in
# IG_LIVE_RISK_REFERENCE.md §1.1 (low 0.30 % / medium 0.70 % / high 1.00 %).
# Currently unused — our orders set ``guaranteedStop=False`` — but exposed so
# any future toggle deducts the premium from projected EV.
GUARANTEED_STOP_PREMIUM_PCT: dict[AssetClass, float] = {
    AssetClass.FOREX_MAJOR: 0.0030,
    AssetClass.FOREX_MINOR: 0.0030,
    AssetClass.INDEX_MAJOR: 0.0030,
    AssetClass.SPOT_GOLD: 0.0030,
    AssetClass.COMMODITY: 0.0070,
    AssetClass.EQUITY_ETF: 0.0100,
}


# Fallback classification for the twelvedata warm-standby universe (12 FX +
# XAU/USD).  EODHD-universe symbols (incl. the 14 US shares + XAG) are classified
# via ``EODHD_UNIVERSE`` first (see ``classify_symbol``); this table only catches
# the warm-standby path, whose symbols are a subset of the EODHD universe.
_SYMBOL_ASSET_CLASS: dict[str, AssetClass] = {
    # --- Forex majors (IG counts these as "major" for retail margin) ---
    "EUR/USD": AssetClass.FOREX_MAJOR,
    "GBP/USD": AssetClass.FOREX_MAJOR,
    "USD/JPY": AssetClass.FOREX_MAJOR,
    "USD/CHF": AssetClass.FOREX_MAJOR,
    "USD/CAD": AssetClass.FOREX_MAJOR,
    "AUD/USD": AssetClass.FOREX_MAJOR,
    "NZD/USD": AssetClass.FOREX_MAJOR,
    "EUR/GBP": AssetClass.FOREX_MAJOR,
    "EUR/JPY": AssetClass.FOREX_MAJOR,
    "GBP/JPY": AssetClass.FOREX_MAJOR,
    # --- Forex minors ---
    "EUR/AUD": AssetClass.FOREX_MINOR,
    "AUD/JPY": AssetClass.FOREX_MINOR,
    # --- Metals ---
    "XAU/USD": AssetClass.SPOT_GOLD,
}


def classify_symbol(symbol: str) -> AssetClass:
    """Classify a Twelve Data symbol into an IG asset class.

    Unknown symbols fall back to the *higher* margin class within their broad
    type (minor forex / commodity) — over-estimating margin is safe for a
    circuit-breaker check; under-estimating is not.
    """
    eodhd = EODHD_UNIVERSE.get(symbol)
    if eodhd is not None:
        return AssetClass(eodhd.ig_margin_class)
    if symbol in _SYMBOL_ASSET_CLASS:
        return _SYMBOL_ASSET_CLASS[symbol]
    if "/" in symbol:
        return AssetClass.FOREX_MINOR
    return AssetClass.COMMODITY


def margin_rate_for(symbol: str) -> float:
    """Return the IG retail margin rate (as a decimal fraction) for *symbol*."""
    return MARGIN_RATES[classify_symbol(symbol)]


def estimate_margin_gbp(
    *,
    symbol: str,
    size_per_pt: float,
    ig_level: float,
) -> float:
    """Estimate the GBP margin a spread-bet position will consume.

    For IG spread bets in a GBP account, the rough working formula is::

        notional_gbp ≈ bet_size_per_pt × ig_level
        margin_gbp    = notional_gbp × margin_rate

    This treats USD/EUR-denominated notionals as if they were GBP, which
    over-estimates margin by 15-30 % depending on the cross-rate.  That bias
    is *intentional*: it feeds the pre-trade circuit-breaker
    gate, where over-estimating margin is the safe side of the trade-off.

    For exact post-fill accounting we rely on IG's own margin field via the
    LS ACCOUNT stream.
    """
    if size_per_pt <= 0 or ig_level <= 0:
        return 0.0
    notional = size_per_pt * ig_level
    return notional * margin_rate_for(symbol)


# ---------------------------------------------------------------------------
# Slippage + guaranteed-stop accounting (IG_LIVE_RISK_REFERENCE.md §1.1)
# ---------------------------------------------------------------------------


def slippage_pct_for(symbol: str) -> float:
    """Expected worst-case slippage past a stop trigger, as a fraction of the
    entry price.  Used by ``estimate_slippage_pts`` to size positions against
    the realistic fill, not the trigger level."""
    return SLIPPAGE_PCT[classify_symbol(symbol)]


def guaranteed_stop_premium_pct_for(symbol: str) -> float:
    """IG's guaranteed-stop premium as a fraction of position notional.

    Charged on order placement when ``guaranteedStop=True``; refunded if the
    guaranteed stop never fires.  Only relevant for orders that actually attach
    a guaranteed stop — our default orders do not, so the helper is exposed
    for future-use EV deductions rather than wired into the live size calc.
    """
    return GUARANTEED_STOP_PREMIUM_PCT[classify_symbol(symbol)]


def estimate_slippage_pts(symbol: str, entry_price: float, pip_value: float) -> float:
    """Return the slippage buffer in IG points for *symbol* at *entry_price*.

    Mirrors the stop-distance math in ``RiskManager.compute_ig_size``::

        slip_pts = entry_price × slippage_pct / pip_value

    Returns 0.0 on invalid inputs so callers can pass the result blindly into
    the size calc without a None-check.
    """
    if entry_price <= 0 or pip_value <= 0:
        return 0.0
    return entry_price * slippage_pct_for(symbol) / pip_value


def estimate_guaranteed_stop_premium_gbp(
    *,
    symbol: str,
    size_per_pt: float,
    ig_level: float,
) -> float:
    """GBP cost IG charges for attaching a guaranteed stop to a £/pt position.

    Computed as ``notional × premium_rate`` with the same notional
    approximation as ``estimate_margin_gbp`` — conservative for GBP accounts
    trading USD/EUR-denominated instruments.
    """
    if size_per_pt <= 0 or ig_level <= 0:
        return 0.0
    notional = size_per_pt * ig_level
    return notional * guaranteed_stop_premium_pct_for(symbol)
