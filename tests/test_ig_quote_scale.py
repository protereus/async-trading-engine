"""Tests for ``bot.execution.ig_quote_scale``.

Pins the table contents and the lookup semantics.  After the 2026-07 rollback
trim the ``_IG_PIP_VALUE`` override table holds only XAU/USD (the twelvedata
warm-standby's IG-native gold); every EODHD-universe symbol resolves its pip
value from ``EODHDSymbol.ig_pip_value`` (EODHD-first).  The cross-reference
tests at the bottom fail loud if a table entry drifts out of the universe.
"""

from __future__ import annotations

import pytest

from bot.data.eodhd_symbols import EODHD_UNIVERSE, SYMBOL_EPIC_MAP
from bot.execution.ig_quote_scale import (
    _IG_PIP_VALUE,
    ig_display_price,
    ig_level_to_display_price,
    ig_pip_value,
    ig_quote_scale,
)


class TestPipValueLookup:
    @pytest.mark.parametrize("symbol", ["USD/JPY", "EUR/JPY", "GBP/JPY", "AUD/JPY"])
    def test_jpy_pairs_default_to_2dp(self, symbol: str) -> None:
        """All JPY-quoted pairs default to 0.01 unless explicitly overridden."""
        assert ig_pip_value(symbol) == 0.01

    @pytest.mark.parametrize("symbol", ["EUR/USD", "GBP/USD", "EUR/AUD", "AUD/USD"])
    def test_non_jpy_forex_defaults_to_4dp(self, symbol: str) -> None:
        assert ig_pip_value(symbol) == 0.0001

    def test_unknown_symbol_falls_through_to_forex_default(self) -> None:
        """Defensive: an unrecognised symbol with no JPY in its name picks up
        the 4dp forex default rather than raising.  Strategy code calls this
        unconditionally for every position symbol; raising here would crash
        the heartbeat path."""
        assert ig_pip_value("UNKNOWN/SYMBOL") == 0.0001


class TestQuoteScale:
    @pytest.mark.parametrize(
        ("symbol", "expected"),
        [
            ("EUR/USD", 10000.0),
            ("USD/JPY", 100.0),
            ("XAU/USD", 1.0),  # IG-native gold, scale 1.0
        ],
    )
    def test_scale_is_inverse_of_pip(self, symbol: str, expected: float) -> None:
        assert ig_quote_scale(symbol) == expected
        # And the round-trip identity holds
        assert ig_pip_value(symbol) * ig_quote_scale(symbol) == pytest.approx(1.0)


class TestEODHDOverrides:
    """EODHD-first pip/scale for the live universe (eodhd_symbols)."""

    @pytest.mark.parametrize(
        ("symbol", "pip", "scale"),
        [
            # Metals are IG-native since 2026-06-19 (IGCandleLSFeed): candles are
            # already in IG-level units, so pip 1.0 → scale 1.0.
            ("XAU/USD", 1.0, 1.0),  # IG Spot Gold (CS.D.USCGC.TODAY.IP)
            ("XAG/USD", 1.0, 1.0),  # IG Spot Silver (CS.D.USCSI.TODAY.IP)
            ("F", 0.01, 100.0),  # US share, cents-quoted
            ("XOM", 0.01, 100.0),
            ("EUR/USD", 0.0001, 10000.0),
            ("USD/JPY", 0.01, 100.0),  # JPY pair
        ],
    )
    def test_eodhd_pip_and_scale(self, symbol: str, pip: float, scale: float) -> None:
        assert ig_pip_value(symbol) == pytest.approx(pip)
        assert ig_quote_scale(symbol) == pytest.approx(scale)
        assert ig_pip_value(symbol) * ig_quote_scale(symbol) == pytest.approx(1.0)


class TestLevelToDisplay:
    def test_jpy_pair_divides_by_100(self) -> None:
        """USD/JPY at IG level 15600 displays as 156.0."""
        assert ig_level_to_display_price("USD/JPY", 15600.0) == pytest.approx(156.0)

    def test_forex_4dp_divides_by_10000(self) -> None:
        """EUR/USD at IG level 11000 displays as 1.1."""
        assert ig_level_to_display_price("EUR/USD", 11000.0) == pytest.approx(1.1)


class TestUniverseCoverage:
    """Cross-reference tests against the live universe — catch a table entry
    that drifts out of both the active EODHD universe and the twelvedata
    warm-standby universe."""

    def test_pip_table_keys_are_known_symbols(self) -> None:
        """Every entry in ``_IG_PIP_VALUE`` must be a tradable candle symbol in
        the active EODHD universe or the twelvedata warm-standby universe.  A
        stale entry wastes lookup time and is a silent doc lie."""
        from bot.data.twelve_data_feed import SYMBOL_EPIC_MAP as TD_STANDBY_MAP

        known = set(SYMBOL_EPIC_MAP) | set(TD_STANDBY_MAP)
        unknown = set(_IG_PIP_VALUE) - known
        assert unknown == set(), (
            f"Pip table contains symbols in neither the active nor the warm-standby "
            f"universe: {unknown}"
        )

    def test_universe_pip_values_match_asset_class(self) -> None:
        """Every active-universe symbol must resolve a pip value consistent with
        its asset class — otherwise the forex default of 0.0001 silently
        mis-scales it.  Post-EODHD-migration the pip value rides on the universe
        entry itself (``EODHDSymbol.ig_pip_value``)."""
        expected_by_class = {
            "equity": {0.01},  # US shares quote in cents
            "metal": {1.0},  # IG-native since 2026-06-19 — levels are $/oz
            "forex": {0.0001, 0.01},  # 4dp pairs / JPY-quoted pairs
        }
        for symbol, entry in EODHD_UNIVERSE.items():
            allowed = expected_by_class[entry.asset_class]
            assert ig_pip_value(symbol) in allowed, (
                f"{symbol!r} ({entry.asset_class}) resolves pip "
                f"{ig_pip_value(symbol)} — expected one of {sorted(allowed)}"
            )
            if entry.asset_class == "forex":
                expected = 0.01 if "JPY" in symbol else 0.0001
                assert ig_pip_value(symbol) == expected, (
                    f"{symbol!r} resolves pip {ig_pip_value(symbol)} — expected {expected}"
                )


class TestIgDisplayPrice:
    """``ig_display_price`` is the operator-facing converter used by the webgui
    dashboard and the Telegram rerank alert.  It mirrors
    ``ig_level_to_display_price`` for every symbol except the IG-native metals
    (XAU/XAG), which carry an explicit divisor so the dashboard reads the $/oz
    spot level directly."""

    def test_xau_xag_display_ig_spot_level(self) -> None:
        """IG-native metals: the operator display reads the IG $/oz spot fill
        level directly via divisor 1.0."""
        assert ig_display_price("XAU/USD", 4466.0) == pytest.approx(4466.0)
        assert ig_display_price("XAG/USD", 7456.0) == pytest.approx(7456.0)

    def test_other_symbols_match_ig_level_to_display_price(self) -> None:
        """Everything outside the metals override inherits the legacy
        ``ig_level_to_display_price`` semantics — display divisor equals trading
        scale.  Spot-check USD/JPY (scale 100), a 4dp forex (scale 10000), and a
        US share (scale 100)."""
        for symbol, level in (
            ("USD/JPY", 15605.0),
            ("EUR/USD", 11000.0),
            ("F", 1234.0),
        ):
            assert ig_display_price(symbol, level) == pytest.approx(
                ig_level_to_display_price(symbol, level)
            )
