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

# Candle symbol → sector name, derived from EODHD_UNIVERSE (the live
# universe's single source of truth) so this can't drift out of sync with
# eodhd_symbols.py's own sector assignment the way a hand-duplicated table
# could. Grouping rationale (see eodhd_symbols._fx_sector):
#   fx_usd          — pairs with USD as one of the two legs (shared USD beta)
#   fx_eur_cross    — EUR-base crosses without USD (shared EUR beta)
#   fx_gbp_cross    — GBP-base crosses without USD/EUR (shared GBP beta)
#   fx_jpy_cross    — JPY-quoted crosses without USD/EUR/GBP (shared JPY beta)
#   metals          — XAU + silver
# Single-name US shares carry their own per-name sector via EODHD_UNIVERSE too
# (included here so SECTOR_MAP is a complete live-universe lookup).
SECTOR_MAP: dict[str, str] = {s.bot_key: s.sector for s in EODHD_UNIVERSE.values()}

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
    # Not a candle symbol — try resolving as an IG EPIC instead. SECTOR_MAP
    # and _EPIC_TO_SYMBOL are both built from EODHD_UNIVERSE, so any epic
    # that resolves here is guaranteed a SECTOR_MAP entry too.
    candle_sym = _EPIC_TO_SYMBOL.get(key)
    if candle_sym is not None:
        return SECTOR_MAP.get(candle_sym, OTHER_SECTOR)
    return OTHER_SECTOR
