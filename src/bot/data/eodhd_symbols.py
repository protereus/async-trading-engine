"""EODHD universe — single source of truth for the post-migration symbol set.

Replaces ``twelve_data_feed.SYMBOL_EPIC_MAP`` once ``candle_exchange="eodhd"``.

verification (2026-06-02). Every entry is confirmed on an EODHD WebSocket
endpoint **and** on IG spread-bet, and is a clean signal=order (equities),
spot↔spot (metals), or FX↔FX match — no proxy/basis instruments.

Each symbol carries everything the feed, strategy, and risk modules need:

* ``bot_key``        — DB/strategy key (``EUR/USD``, ``XAU/USD``, ``AAPL``).
* ``eodhd_rest``     — full EODHD symbol for the intraday-historical REST
                       backfill (``EURUSD.FOREX``, ``F.US``). ``ws_symbol`` strips
                       the suffix.
* ``ws_endpoint``    — which WS stream carries it: ``forex`` or ``us``.
* ``ig_epic``        — IG spread-bet EPIC for order placement (verified on demo).
* ``asset_class``    — ``forex`` | ``metal`` | ``equity`` (coarse data grouping).
* ``has_volume``     — True for US equities + the IG-native metals (LS LTV carries
                       size → Kronos volume group); False for FX (bid/ask only).
* ``ig_pip_value``   — IG point size in candle-source price units. ``ig_quote_scale``
                       consults this first (EODHD-first), so the IG-side stop /
                       level math is correct without touching the TD tables.
* ``ig_margin_class``— ``bot.risk.ig_margin.AssetClass`` value for margin/slippage.
* ``sector``         — ``bot.risk.sectors`` bucket for the concentration cap.

Metals nuance: since 2026-06-19 gold/silver are **IG-native** — candles stream
from IG spot via ``IGCandleLSFeed`` (the metals are in ``IG_NATIVE_CANDLE_SYMBOLS``)
and the order routes to the same IG Spot Gold/Silver EPIC, so the candle store is
already in IG-level units and ``ig_pip_value`` is **1.0** (``ig_quote_scale`` == 1.0,
no conversion). The ``eodhd_rest`` GLD/SLV symbols are retained only as historical
provenance — EODHD no longer fetches them. (Pre-2026-06-19 metals were EODHD-sourced
from the GLD/SLV ETFs on a calibrated cross-instrument scale.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

WsEndpoint = Literal["forex", "us"]
AssetClass = Literal["forex", "metal", "equity"]

# Pairs IG counts as "major" for retail margin (3.33 %); the rest are minors (5 %).
# Mirrors ``ig_margin._SYMBOL_ASSET_CLASS`` so the two never disagree.
_FX_MAJORS: frozenset[str] = frozenset(
    {
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
    }
)


@dataclass(frozen=True)
class EODHDSymbol:
    bot_key: str
    eodhd_rest: str
    ws_endpoint: WsEndpoint
    ig_epic: str
    asset_class: AssetClass
    has_volume: bool
    ig_pip_value: float
    ig_margin_class: str
    sector: str

    @property
    def ws_symbol(self) -> str:
        """Subscribe/identify code on the WS stream (REST symbol minus suffix)."""
        return self.eodhd_rest.split(".")[0]


def _fx_sector(bot_key: str) -> str:
    if "USD" in bot_key:
        return "fx_usd"
    if bot_key.startswith("EUR/"):
        return "fx_eur_cross"
    if bot_key.startswith("GBP/"):
        return "fx_gbp_cross"
    return "fx_jpy_cross"  # remaining crosses are JPY-quoted


def _fx(bot_key: str, ig_epic: str) -> EODHDSymbol:
    return EODHDSymbol(
        bot_key=bot_key,
        eodhd_rest=bot_key.replace("/", "") + ".FOREX",
        ws_endpoint="forex",
        ig_epic=ig_epic,
        asset_class="forex",
        has_volume=False,
        # JPY-quoted pairs are 2dp (pip 0.01); all others 4dp (pip 0.0001).
        ig_pip_value=0.01 if bot_key.endswith("/JPY") else 0.0001,
        ig_margin_class="forex_major" if bot_key in _FX_MAJORS else "forex_minor",
        sector=_fx_sector(bot_key),
    )


def _metal(bot_key: str, etf: str, ig_epic: str, pip: float, margin_class: str) -> EODHDSymbol:
    """Gold/silver: candles + order both on the IG spot metal (24/5).

    Since 2026-06-19 metals are IG-native (``IGCandleLSFeed`` streams the spot
    EPIC and aggregates to 1h), so the candle store is already in IG-level units
    and ``pip`` is 1.0 (no scale conversion — ``ig_quote_scale`` == 1.0). The
    ``etf`` arg is retained only as historical provenance; EODHD no longer
    fetches it (metals are in ``IG_NATIVE_CANDLE_SYMBOLS``). ``margin_class`` is
    spot_gold (5 %) / commodity (10 %).
    """
    return EODHDSymbol(
        bot_key=bot_key,
        eodhd_rest=etf + ".US",
        ws_endpoint="us",
        ig_epic=ig_epic,
        asset_class="metal",
        has_volume=True,
        ig_pip_value=pip,
        ig_margin_class=margin_class,
        sector="metals",
    )


def _eq(bot_key: str, ig_epic: str, sector: str) -> EODHDSymbol:
    """US single-name share: IG quotes in cents → pip 0.01 (scale 100); 20 % margin."""
    return EODHDSymbol(
        bot_key=bot_key,
        eodhd_rest=bot_key + ".US",
        ws_endpoint="us",
        ig_epic=ig_epic,
        asset_class="equity",
        has_volume=True,
        ig_pip_value=0.01,
        ig_margin_class="equity_etf",
        sector=sector,
    )


# --- FX core (12) — forex WS, 24/5, no-volume; cleanest Kronos data ---
_FX = [
    _fx("EUR/USD", "CS.D.EURUSD.TODAY.IP"),
    _fx("GBP/USD", "CS.D.GBPUSD.TODAY.IP"),
    _fx("USD/JPY", "CS.D.USDJPY.TODAY.IP"),
    _fx("USD/CHF", "CS.D.USDCHF.TODAY.IP"),
    _fx("USD/CAD", "CS.D.USDCAD.TODAY.IP"),
    _fx("AUD/USD", "CS.D.AUDUSD.TODAY.IP"),
    _fx("NZD/USD", "CS.D.NZDUSD.TODAY.IP"),
    _fx("EUR/GBP", "CS.D.EURGBP.TODAY.IP"),
    _fx("EUR/JPY", "CS.D.EURJPY.TODAY.IP"),
    _fx("GBP/JPY", "CS.D.GBPJPY.TODAY.IP"),
    _fx("AUD/JPY", "CS.D.AUDJPY.TODAY.IP"),
    _fx("EUR/AUD", "CS.D.EURAUD.TODAY.IP"),
]

# --- Metals (2) — IG-native candles + order (spot gold/silver, 24/5). pip 1.0
#     because IGCandleLSFeed writes candles in IG-level units directly, so
#     ig_quote_scale == 1.0 (same invariant USO/UNG/SLV got at the D3 cutover).
#     The flip native→IG-level candle + pip→1.0 is sizing/stop/display-neutral:
#     compute_ig_size uses entry_price/pip_value (= the IG level), which is
#     preserved.  pip was the GLD/SLV→IG cross scale (gold 0.092237, silver
#     0.009119) while metals were EODHD-sourced (retired 2026-06-19). ---
_METALS = [
    _metal("XAU/USD", "GLD", "CS.D.USCGC.TODAY.IP", 1.0, "spot_gold"),
    _metal("XAG/USD", "SLV", "CS.D.USCSI.TODAY.IP", 1.0, "commodity"),
]

# --- US equities (14) — us WS, volume group; lower-priced sleeve (min-risk
#     £19–£180, fits 1%/£20k). IG "(24 Hours)" share DFB; signal=order. ---
_EQUITIES = [
    _eq("F", "SC.D.F.DAILY.IP", "equity_consumer"),  # Ford
    _eq("T", "SG.D.T.DAILY.IP", "equity_comm"),  # AT&T
    _eq("PFE", "SE.D.PFE.DAILY.IP", "equity_health"),  # Pfizer
    _eq("VZ", "SH.D.VZ.DAILY.IP", "equity_comm"),  # Verizon
    _eq("NKE", "SE.D.NKE.DAILY.IP", "equity_consumer"),  # Nike
    _eq("BAC", "SA.D.BAC.DAILY.IP", "equity_financials"),  # Bank of America
    _eq("BMY", "SA.D.BMYUS.DAILY.IP", "equity_health"),  # Bristol-Myers
    _eq("MO", "SE.D.MO.DAILY.IP", "equity_staples"),  # Altria
    _eq("KO", "SD.D.KO.DAILY.IP", "equity_staples"),  # Coca-Cola
    _eq("WFC", "SH.D.WFC.DAILY.IP", "equity_financials"),  # Wells Fargo
    _eq("CVS", "SB.D.CVS.DAILY.IP", "equity_health"),  # CVS Health
    _eq("INTC", "UB.D.INTC.DAILY.IP", "equity_tech"),  # Intel
    _eq("CSCO", "UA.D.CSCO.DAILY.IP", "equity_tech"),  # Cisco
    _eq("XOM", "SH.D.XOM.DAILY.IP", "equity_energy"),  # Exxon
]

EODHD_UNIVERSE: dict[str, EODHDSymbol] = {s.bot_key: s for s in (*_FX, *_METALS, *_EQUITIES)}

# Compatibility shim: main.py reads SYMBOL_EPIC_MAP to build _candle_symbols /
# _candle_epic_map — keep the same shape as twelve_data_feed.SYMBOL_EPIC_MAP.
SYMBOL_EPIC_MAP: dict[str, str] = {k: s.ig_epic for k, s in EODHD_UNIVERSE.items()}

# Bot keys whose Kronos inference uses the volume-bearing tokeniser path.
VOLUME_SYMBOLS: frozenset[str] = frozenset(k for k, s in EODHD_UNIVERSE.items() if s.has_volume)


def ws_symbols(endpoint: WsEndpoint) -> list[str]:
    """WS subscribe codes for one endpoint (e.g. ['EURUSD', 'XAUUSD'])."""
    return [s.ws_symbol for s in EODHD_UNIVERSE.values() if s.ws_endpoint == endpoint]


def bot_key_for_ws(endpoint: WsEndpoint, ws_symbol: str) -> str | None:
    """Reverse-map an incoming WS message's symbol back to its bot key."""
    for s in EODHD_UNIVERSE.values():
        if s.ws_endpoint == endpoint and s.ws_symbol == ws_symbol:
            return s.bot_key
    return None
