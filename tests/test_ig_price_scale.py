"""Tests for IG price-scale normalisation in stop-loss and P&L calculations.

``ig_quote_scale`` bridges the candle-source price to the IG fill level so
stop-loss math, P&L, and Telegram alerts all operate in matching units. It is
EODHD-first (consults ``EODHD_UNIVERSE`` before the legacy override table):

  * 4dp forex (EUR/USD, EUR/AUD …)          — pip 0.0001 → scale 10000
  * JPY-denominated forex                    — pip 0.01   → scale 100
  * US single-name shares (cents → dollars)  —            → scale 100
  * IG-native metals (XAU/XAG, since D3/cutover) — candles already at IG level → scale 1.0

Live candles come from EODHD; the Twelve Data feed survives only as the
``CANDLE_EXCHANGE=twelvedata`` warm-standby (FX + XAU).
"""

from __future__ import annotations

import pytest

from bot.execution.ig_quote_scale import (
    ig_level_to_display_price as _ig_level_to_display_price,
)
from bot.execution.ig_quote_scale import ig_pip_value as _ig_pip_value
from bot.execution.ig_quote_scale import ig_quote_scale as _ig_quote_scale


class TestIgQuoteScale:
    @pytest.mark.parametrize("symbol", ["USD/JPY", "EUR/JPY", "GBP/JPY", "AUD/JPY"])
    def test_jpy_pairs_return_100(self, symbol: str) -> None:
        assert _ig_quote_scale(symbol) == 100.0

    @pytest.mark.parametrize("symbol", ["EUR/USD", "GBP/USD", "EUR/AUD", "AUD/USD"])
    def test_4dp_forex_returns_10000(self, symbol: str) -> None:
        assert _ig_quote_scale(symbol) == pytest.approx(10000.0)

    @pytest.mark.parametrize("symbol", ["XAU/USD", "XAG/USD"])
    def test_ig_native_metals_return_1(self, symbol: str) -> None:
        # Metals moved to the IG-native candle feed (IGCandleLSFeed): the candle
        # store already speaks IG-level units, so scale collapses to 1.0.
        assert _ig_quote_scale(symbol) == pytest.approx(1.0)

    @pytest.mark.parametrize("symbol", ["F", "XOM", "PFE"])
    def test_us_shares_return_100(self, symbol: str) -> None:
        # US single-name shares quote in cents on IG → scale 100.
        assert _ig_quote_scale(symbol) == pytest.approx(100.0)

    def test_scale_is_inverse_of_pip_value(self) -> None:
        # By construction: scale = 1 / pip_value for every symbol family.
        for symbol in ("EUR/USD", "EUR/AUD", "USD/JPY", "EUR/JPY", "XAU/USD", "XAG/USD", "F"):
            assert _ig_quote_scale(symbol) == pytest.approx(1.0 / _ig_pip_value(symbol)), symbol


class TestStopLossWithScale:
    """Verify the stop-loss formula gives sensible results once scaled."""

    def _loss_pct(self, entry_ig: float, candle_close: float, symbol: str) -> float:
        scale = _ig_quote_scale(symbol)
        ig_current = candle_close * scale
        return (entry_ig - ig_current) / entry_ig

    def test_usd_jpy_profitable_position_is_not_a_loss(self) -> None:
        # Entry IG level 15658.5 = 156.585 FX; current 157.052 = gain
        loss = self._loss_pct(15658.5, 157.052, "USD/JPY")
        assert loss < 0, "Price went up → should be a profit (negative loss)"

    def test_usd_jpy_small_real_loss_is_within_stop(self) -> None:
        # Entry 15658.5, price drops 0.3% to 155.884 FX = IG 15588.4
        loss = self._loss_pct(15658.5, 155.884, "USD/JPY")
        assert 0 < loss < 0.009, "~0.45% real loss should be below 0.9% stop"

    def test_usd_jpy_large_loss_exceeds_stop(self) -> None:
        loss = self._loss_pct(15658.5, 154.242, "USD/JPY")
        assert loss > 0.009, "1.5% real loss should exceed 0.9% stop"

    def test_eur_usd_at_entry_is_not_a_loss(self) -> None:
        # EUR/USD fill at FX 1.17225 → IG level 11722.5; candle close at FX 1.17225
        # should round-trip to zero loss.  Without the 4dp-forex fix this would
        # compute (11722.5 - 1.17225) / 11722.5 ≈ 99.99 % and stop-loss instantly.
        loss = self._loss_pct(11722.5, 1.17225, "EUR/USD")
        assert pytest.approx(loss, abs=1e-6) == 0.0

    def test_eur_usd_small_loss_is_within_stop(self) -> None:
        loss = self._loss_pct(11722.5, 1.16756, "EUR/USD")
        assert 0 < loss < 0.005

    def test_eur_usd_large_loss_exceeds_stop(self) -> None:
        loss = self._loss_pct(11722.5, 1.16053, "EUR/USD")
        assert loss > 0.009

    def test_xau_usd_stop_unaffected(self) -> None:
        # Gold is IG-native (2026-06-19): candle store is in IG-level units
        # (scale 1.0), so entry and candle are both IG levels — identity arithmetic.
        loss = self._loss_pct(4466.0, 4456.0, "XAU/USD")
        assert 0 < loss < 0.01

    def test_xag_usd_at_entry_is_not_a_loss(self) -> None:
        # Silver is IG-native (scale 1.0): entry IG == candle close → zero loss.
        loss = self._loss_pct(7313.0, 7313.0, "XAG/USD")
        assert pytest.approx(loss, abs=1e-9) == 0.0

    def test_us_share_at_entry_is_not_a_loss(self) -> None:
        # US share at $12.00 → IG level 1200 with scale 100.
        loss = self._loss_pct(1200.0, 12.00, "F")
        assert pytest.approx(loss, abs=1e-6) == 0.0

    def test_us_share_small_loss_within_stop(self) -> None:
        # Entry IG 1200 ($12.00); price drops 0.4% to $11.952 → IG 1195.2.
        loss = self._loss_pct(1200.0, 11.952, "F")
        assert 0 < loss < 0.01

    def test_us_share_large_loss_exceeds_stop(self) -> None:
        # Entry IG 1200; price drops 2% to $11.76 → IG 1176.
        loss = self._loss_pct(1200.0, 11.76, "F")
        assert loss > 0.015


class TestPnlWithScale:
    """Verify pnl_pct and display price calculations in _close_position."""

    def _pnl_and_display(
        self, entry_ig: float, candle_close: float, symbol: str
    ) -> tuple[float, float]:
        scale = _ig_quote_scale(symbol)
        ig_current = candle_close * scale
        pnl_pct = (ig_current - entry_ig) / entry_ig * 100
        entry_display = entry_ig / scale
        return pnl_pct, entry_display

    def test_usd_jpy_profitable_pnl_is_positive(self) -> None:
        pnl, entry_display = self._pnl_and_display(15658.5, 157.052, "USD/JPY")
        assert pnl > 0, "Price moved up → profit"
        assert pytest.approx(entry_display, abs=0.01) == 156.585

    def test_usd_jpy_entry_display_is_fx_rate(self) -> None:
        _, entry_display = self._pnl_and_display(15658.5, 157.052, "USD/JPY")
        assert 150 < entry_display < 165, "Entry display should be in FX rate range"

    def test_eur_usd_pnl_matches_underlying_return(self) -> None:
        pnl, entry_display = self._pnl_and_display(11722.5, 1.165, "EUR/USD")
        expected = (1.165 - 1.17225) / 1.17225 * 100
        assert pytest.approx(pnl, abs=1e-4) == expected
        assert pytest.approx(entry_display, abs=1e-6) == 1.17225

    def test_xau_entry_display_is_ig_level(self) -> None:
        # IG-native metal: the "display" price IS the IG level (scale 1.0).
        _, entry_display = self._pnl_and_display(4466.0, 4466.0, "XAU/USD")
        assert pytest.approx(entry_display, abs=1e-6) == 4466.0

    def test_xau_profitable_pnl(self) -> None:
        pnl, _ = self._pnl_and_display(4466.0, 4546.0, "XAU/USD")
        assert pnl > 0
        assert pytest.approx(pnl, abs=0.05) == (4546.0 - 4466.0) / 4466.0 * 100

    def test_us_share_pnl_matches_underlying_return(self) -> None:
        # Entry IG 1200 = $12.00; price rises to $12.24 (IG 1224) → +2.0%.
        pnl, _ = self._pnl_and_display(1200.0, 12.24, "F")
        assert pytest.approx(pnl, abs=0.01) == 2.0


class TestTradeAlertDisplayPrice:
    """Verify display_price normalization for Telegram fill alerts."""

    def _display_price(self, ig_fill: float, symbol: str) -> float:
        return ig_fill / _ig_quote_scale(symbol)

    def test_usd_jpy_fill_shows_fx_rate(self) -> None:
        price = self._display_price(15658.5, "USD/JPY")
        assert pytest.approx(price, abs=0.001) == 156.585

    def test_eur_jpy_fill_shows_fx_rate(self) -> None:
        price = self._display_price(18346.7, "EUR/JPY")
        assert pytest.approx(price, abs=0.001) == 183.467

    def test_eur_usd_fill_shows_fx_rate(self) -> None:
        price = self._display_price(11722.5, "EUR/USD")
        assert pytest.approx(price, abs=1e-6) == 1.17225

    def test_eur_aud_fill_shows_fx_rate(self) -> None:
        price = self._display_price(16788.9, "EUR/AUD")
        assert pytest.approx(price, abs=1e-5) == 1.67889

    def test_xau_usd_fill_shows_ig_level(self) -> None:
        # IG-native gold: display = IG level / 1.0 = IG level.
        price = self._display_price(4466.0, "XAU/USD")
        assert pytest.approx(price, abs=1e-6) == 4466.0

    def test_xag_usd_fill_shows_ig_level(self) -> None:
        price = self._display_price(7313.0, "XAG/USD")
        assert pytest.approx(price, abs=1e-6) == 7313.0

    def test_us_share_fill_shows_dollars(self) -> None:
        # US share IG fill 1200 → $12.00 (÷100).
        price = self._display_price(1200.0, "F")
        assert pytest.approx(price, abs=1e-6) == 12.00


class TestLevelToDisplayPriceMatchesScale:
    """_ig_level_to_display_price is the inverse of _ig_quote_scale."""

    @pytest.mark.parametrize(
        "symbol",
        ["EUR/USD", "EUR/AUD", "USD/JPY", "EUR/JPY", "XAU/USD", "XAG/USD", "F", "XOM"],
    )
    def test_inverse_relationship(self, symbol: str) -> None:
        # td_price → ig_level → td_price round-trips for any symbol
        td = 100.0
        ig = td * _ig_quote_scale(symbol)
        assert pytest.approx(_ig_level_to_display_price(symbol, ig), rel=1e-9) == td
