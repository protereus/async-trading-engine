"""IG quote-scale conversion — single source of truth.

For every IG spread bet, the relationship between the candle-source price
(EODHD, or Twelve Data on the warm-standby path) and the IG-quoted level is::

    ig_level = candle_price / _IG_PIP_VALUE[symbol]

This module owns the ``_IG_PIP_VALUE`` table and the two helpers
(``ig_pip_value``, ``ig_quote_scale``) plus the inverse converter
``ig_level_to_display_price``.  Both ``bot.main`` and ``webgui.data``
import from here so the table can't drift between processes.

History
-------
Before 2026-05-29 the table was duplicated in ``webgui/data.py`` "to keep
webgui import-light".  When FTSE was added on 2026-05-24 the webgui copy
wasn't updated, so the dashboard rendered the FTSE entry as 1.04 instead
of 10,400.  The duplicate is gone; both importers reach into this module
instead.

Adding a symbol
---------------
Three places need updating in lock-step:

1. ``_IG_PIP_VALUE`` here (the table itself).
2. ``EODHD_UNIVERSE`` in ``bot/data/eodhd_symbols.py`` (the active universe;
   its derived ``SYMBOL_EPIC_MAP`` is the order route).
3. ``SECTOR_MAP`` in ``bot/risk/sectors.py`` (concentration cap).

``tests/test_ig_quote_scale.py`` includes a cross-reference test that
fails fast if any of these maps falls out of sync.
"""

from __future__ import annotations

from bot.data.eodhd_symbols import EODHD_UNIVERSE

# IG spread bet point size: the price change (in native units) that equals
# 1 IG point.
#   Standard non-JPY forex: 0.0001 (1 pip = 4th decimal place of the rate).
#   JPY-denominated pairs : 0.01   (1 pip = 2nd decimal place).
#   Gold (XAU/USD)        : 1.0    (gold futures-style pricing, 1 pt = $1).
#   Symbols not listed here AND not containing "JPY" default to 0.0001.
# EODHD-universe symbols carry their own ig_pip_value (see ``ig_pip_value``);
# this override table is only the twelvedata warm-standby's XAU/USD.
_IG_PIP_VALUE: dict[str, float] = {
    # Gold: 1 IG point = $1 move (bid≈4590 = spot $4590/oz). Verified 2026-04-28.
    "XAU/USD": 1.0,
}

# Public, read-only view of the symbols that carry an explicit scale entry.
# These are the only symbols whose candle-source↔IG-level ratio can drift
# (everything else falls through to the stable forex defaults in
# ``ig_pip_value``).  The D4 scale-drift guard iterates this set instead of
# reaching into the private ``_IG_PIP_VALUE`` table.
IG_SCALED_SYMBOLS: frozenset[str] = frozenset(_IG_PIP_VALUE)


def ig_pip_value(symbol: str) -> float:
    """Return the IG point size (in native price units) for *symbol*.

    EODHD-first: post-migration symbols carry their own ``ig_pip_value`` (FX
    0.0001/0.01, equities 0.01, IG-native metals 1.0 since the 2026-06-19
    cutover — candles already in IG-level units). The ``_IG_PIP_VALUE`` table is
    the override for the twelvedata warm-standby's XAU/USD (its candles come from
    the IG-native feed at scale 1.0).
    """
    eodhd = EODHD_UNIVERSE.get(symbol)
    if eodhd is not None:
        return eodhd.ig_pip_value
    if symbol in _IG_PIP_VALUE:
        return _IG_PIP_VALUE[symbol]
    if "JPY" in symbol:
        return 0.01
    return 0.0001


def ig_quote_scale(symbol: str) -> float:
    """Multiplier converting a candle-source price into the matching IG level.

    ``ig_level = candle_price × ig_quote_scale(symbol)``.  Examples:

      - 4dp forex (EUR/USD, EUR/AUD, …)        : pip 0.0001 → scale 10000
      - JPY-denominated forex                  : pip 0.01   → scale 100
      - Gold XAU/USD                           : pip 1.0    → scale 1

    Use this whenever a strategy needs to compare a candle-source price
    with an IG-quoted level (entry_price, stop_distance, etc.).
    """
    return 1.0 / ig_pip_value(symbol)


def ig_level_to_display_price(symbol: str, level: float) -> float:
    """Convert an IG fill level back to the candle-source face-value price.

    Inverse of ``ig_quote_scale``: for any symbol the relationship is
    ``candle_price = ig_level / ig_quote_scale(symbol)``.  ETFs, JPY
    pairs, gold and forex all flow through the same formula.
    """
    return level / ig_quote_scale(symbol)


# ---------------------------------------------------------------------------
# Display-only divisors for the natural quote unit (dashboard + Telegram).
# Metals are IG-native since 2026-06-19 (IGCandleLSFeed): the candle store holds
# the IG spot level directly and ``ig_quote_scale`` == 1.0, so the display input
# already IS the IG level. Divisor 1.0 shows it as-is (gold ~4466, silver
# ~7456). Adjust the silver divisor only if the dashboard reads in cents rather
# than $/oz.
# ---------------------------------------------------------------------------
_DISPLAY_DIVISOR: dict[str, float] = {
    "XAU/USD": 1.0,
    "XAG/USD": 1.0,
}


def ig_display_price(symbol: str, level: float) -> float:
    """Convert an IG fill level into the natural quote-unit number used by
    the dashboard and Telegram alerts.

    For most symbols this matches ``ig_level_to_display_price`` exactly.
    The IG-native metals (XAU/XAG) carry an explicit divisor so the
    dashboard reads the $/oz spot level directly.  Decoupled from
    ``ig_level_to_display_price`` so the "inverse of ``ig_quote_scale``"
    contract that the tick-validator and signal tests rely on is
    preserved.
    """
    if symbol in _DISPLAY_DIVISOR:
        return level / _DISPLAY_DIVISOR[symbol]
    return ig_level_to_display_price(symbol, level)
