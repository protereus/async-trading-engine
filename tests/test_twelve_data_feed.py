"""TwelveData feed invariants (the rollback / warm-standby candle path)."""

from bot.data.ig_candle_aggregator import IG_NATIVE_CANDLE_SYMBOLS
from bot.data.twelve_data_feed import _TD_EXCLUDE_FROM_FETCH, SYMBOL_EPIC_MAP


class TestIgNativeMetalsNotDoubleSourced:
    """Under ``candle_exchange='twelvedata'`` the IG Lightstreamer feed still
    owns the IG-native metals (XAU/XAG). ``TwelveDataFeed`` must NOT also fetch
    them: two sources writing the same symbol interleave incompatible price
    scales in one candle series (TD gold ~$4043 vs IG-native gold ~$4150) — the
    scale-discontinuity class the 2026-06-19 metals cutover was meant to remove.
    """

    def test_ig_native_metals_excluded_from_td_fetch_set(self) -> None:
        for sym in IG_NATIVE_CANDLE_SYMBOLS:
            assert sym in _TD_EXCLUDE_FROM_FETCH, (
                f"{sym} is IG-native but not in _TD_EXCLUDE_FROM_FETCH — "
                "TwelveDataFeed would double-source it against the IG feed"
            )

    def test_td_fetch_list_never_includes_ig_native_metals(self) -> None:
        # Mirrors TwelveDataFeed.__init__'s derivation of the symbols it fetches.
        fetched = {s for s in SYMBOL_EPIC_MAP if s not in _TD_EXCLUDE_FROM_FETCH}
        assert not (fetched & set(IG_NATIVE_CANDLE_SYMBOLS)), (
            "TwelveDataFeed fetch list overlaps IG_NATIVE_CANDLE_SYMBOLS"
        )
