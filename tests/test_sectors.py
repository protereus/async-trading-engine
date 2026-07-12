"""Tests for bot.risk.sectors — static sector classification used by the
per-sector concentration cap in RiskManager.evaluate_ig_order."""

from __future__ import annotations

from bot.data.eodhd_symbols import SYMBOL_EPIC_MAP
from bot.risk.sectors import OTHER_SECTOR, sector_for


class TestSectorMap:
    def test_every_active_universe_symbol_has_a_sector(self) -> None:
        """Every symbol in the active EODHD universe must resolve to a real
        sector — never the fallback bucket — otherwise the concentration cap
        would silently lump them together.  Active-universe sectors ride on
        ``EODHDSymbol.sector``, so this asserts through ``sector_for`` (the
        lookup the risk manager actually uses) rather than the SECTOR_MAP
        dict."""
        for symbol in SYMBOL_EPIC_MAP:
            assert sector_for(symbol) != OTHER_SECTOR, (
                f"{symbol} resolves to OTHER_SECTOR — set a sector on its "
                "EODHD_UNIVERSE entry or extend SECTOR_MAP"
            )

    def test_warm_standby_universe_covered_by_sector_map(self) -> None:
        """The twelvedata warm-standby trades a subset of the EODHD universe
        (12 FX + XAU/USD); every one of those must resolve a sector so the
        concentration cap stays active on the failover path."""
        from bot.data.twelve_data_feed import SYMBOL_EPIC_MAP as TD_STANDBY_MAP

        missing = sorted(s for s in TD_STANDBY_MAP if sector_for(s) == OTHER_SECTOR)
        assert not missing, (
            f"Warm-standby symbols without a sector: {missing} — "
            "extend SECTOR_MAP in bot/risk/sectors.py"
        )

    def test_warm_standby_map_is_an_epic_identical_subset_of_the_live_universe(self) -> None:
        """Every EPIC-keyed lookup (``sectors._EPIC_TO_SYMBOL``, risk-manager
        position reconciliation, …) is built from ``EODHD_UNIVERSE`` only; the
        2026-07-10 warm-standby trim leans on the TD map trading the *same IG
        EPICs*.  A TD entry with a different EPIC would silently bucket as
        ``OTHER_SECTOR`` (and dodge the concentration cap) on the failover
        path — pin membership *and* EPIC identity."""
        from bot.data.twelve_data_feed import SYMBOL_EPIC_MAP as TD_STANDBY_MAP

        drifted = sorted(set(TD_STANDBY_MAP.items()) - set(SYMBOL_EPIC_MAP.items()))
        assert not drifted, (
            f"Warm-standby entries not (symbol, EPIC)-identical to EODHD_UNIVERSE: {drifted}"
        )

    def test_sector_for_accepts_epic(self) -> None:
        """The risk manager keys open positions by EPIC, so sector_for must
        resolve EPICs too."""
        assert sector_for("CS.D.EURUSD.TODAY.IP") == "fx_usd"
        assert sector_for("CS.D.USCSI.TODAY.IP") == "metals"  # XAG/USD silver-spot EPIC

    def test_sector_for_unknown_returns_other(self) -> None:
        assert sector_for("UNKNOWN/SYMBOL") == OTHER_SECTOR
        assert sector_for("CS.D.NEVER.HEARD.OF.IP") == OTHER_SECTOR

    def test_expected_sector_groupings(self) -> None:
        """Spot-check the intended buckets: USD pairs, EUR/GBP/JPY crosses, and
        metals each map to distinct sectors; US shares get per-name equity
        sectors from EODHD_UNIVERSE."""
        assert sector_for("EUR/USD") == "fx_usd"
        assert sector_for("GBP/USD") == "fx_usd"
        assert sector_for("USD/JPY") == "fx_usd"

        assert sector_for("EUR/GBP") == "fx_eur_cross"
        assert sector_for("EUR/JPY") == "fx_eur_cross"
        assert sector_for("EUR/AUD") == "fx_eur_cross"

        assert sector_for("GBP/JPY") == "fx_gbp_cross"
        assert sector_for("AUD/JPY") == "fx_jpy_cross"

        assert sector_for("XAU/USD") == "metals"
        assert sector_for("XAG/USD") == "metals"

        # US single-name shares carry per-name equity sectors (EODHD_UNIVERSE)
        assert sector_for("F") == "equity_consumer"
