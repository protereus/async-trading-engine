"""Static sector classification for concentration-risk caps.

Used by ``RiskManager.evaluate_ig_order`` to enforce
``max_sector_risk_pct`` in addition to ``max_total_risk_pct``.  Without
this, the bot could park its entire risk-on budget in one sector — e.g.
three yen-cross longs whose pairwise correlations are each below the
0.65 ``correlation_threshold`` but which all move together when the
yen moves.  The pairwise correlation cap in
``bot.strategy.correlation`` catches *pairs*; this catches *clusters*.

Sectors are static and intentional: they reflect *fundamental*
co-movement (shared currency leg / asset class), not transient
statistical correlation.  Symbols outside the live universe map to
``"other"`` so the cap remains conservative if the watchlist is
extended without updating this file.

Lookup accepts either the candle/strategy symbol (e.g. ``"EUR/USD"``)
or the IG EPIC (e.g. ``"CS.D.EURUSD.TODAY.IP"``), since the risk
manager keys open positions by EPIC while strategy code uses candle
symbols.
"""

from __future__ import annotations

from bot.data.eodhd_symbols import EODHD_UNIVERSE

# Candle symbol → sector name.
# Grouping rationale:
#   fx_usd          — pairs with USD as one of the two legs (shared USD beta)
#   fx_eur_cross    — EUR-base crosses without USD (shared EUR beta)
#   fx_gbp_cross    — GBP-base crosses without USD/EUR (shared GBP beta)
#   fx_jpy_cross    — JPY-quoted crosses without USD/EUR/GBP (shared JPY beta)
#   metals          — XAU + silver
# Single-name US shares carry their own per-name sector via EODHD_UNIVERSE.
SECTOR_MAP: dict[str, str] = {
    # FX USD pairs (one leg is USD)
    "EUR/USD": "fx_usd",
    "GBP/USD": "fx_usd",
    "USD/JPY": "fx_usd",
    "USD/CHF": "fx_usd",
    "USD/CAD": "fx_usd",
    "AUD/USD": "fx_usd",
    "NZD/USD": "fx_usd",
    # FX EUR crosses (no USD)
    "EUR/GBP": "fx_eur_cross",
    "EUR/JPY": "fx_eur_cross",
    "EUR/AUD": "fx_eur_cross",
    # FX GBP crosses (no USD, no EUR-base)
    "GBP/JPY": "fx_gbp_cross",
    # FX JPY-quoted crosses (no USD/EUR/GBP base)
    "AUD/JPY": "fx_jpy_cross",
    # Precious metals
    "XAU/USD": "metals",
    "XAG/USD": "metals",
}

# Inverse map: IG EPIC → candle symbol, built once at import from the live EODHD
# universe.  The risk manager keys open positions by EPIC, so we accept either
# form when resolving sector membership.  The twelvedata warm-standby trades a
# subset of the EODHD universe (12 FX + XAU/USD) on the same IG EPICs, so no
# separate rollback map is needed.
_EPIC_TO_SYMBOL: dict[str, str] = {s.ig_epic: s.bot_key for s in EODHD_UNIVERSE.values()}

OTHER_SECTOR = "other"


def sector_for(key: str) -> str:
    """Return the sector for *key*, which may be a candle symbol or an IG EPIC.

    Unknown symbols / EPICs return ``OTHER_SECTOR``.  Unknown symbols
    still get a sector bucket so the cap remains active for them — it
    just means anything unmapped lumps together, which is the
    conservative behaviour we want when the universe is extended
    without updating this file.
    """
    if key in SECTOR_MAP:
        return SECTOR_MAP[key]
    # EODHD universe carries its own sector (incl. the per-name equity buckets).
    eodhd = EODHD_UNIVERSE.get(key)
    if eodhd is not None:
        return eodhd.sector
    candle_sym = _EPIC_TO_SYMBOL.get(key)
    if candle_sym is not None and candle_sym in SECTOR_MAP:
        return SECTOR_MAP[candle_sym]
    # Also resolve an IG EPIC for an EODHD symbol → its sector.
    for s in EODHD_UNIVERSE.values():
        if s.ig_epic == key:
            return s.sector
    return OTHER_SECTOR
