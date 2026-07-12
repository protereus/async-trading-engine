"""Tests for TelegramAlerter — message formatting, rate limiting, and failure handling."""

from __future__ import annotations

import time
from html.parser import HTMLParser
from unittest.mock import AsyncMock, patch

import pytest

from bot.core.models import OrderResult, OrderSide, OrderStatus, OrderType, RiskEvent
from bot.monitoring.telegram_alerts import TelegramAlerter


def _html_balanced(msg: str) -> bool:
    """True if every tag in *msg* is opened and closed in order — the property
    Telegram's HTML parser requires (an unbalanced entity → 400 'can't parse
    entities', the bug this whole module guards against)."""
    stack: list[str] = []
    ok = True

    class _P(HTMLParser):
        def handle_starttag(self, tag: str, attrs: object) -> None:
            stack.append(tag)

        def handle_endtag(self, tag: str) -> None:
            nonlocal ok
            if not stack or stack.pop() != tag:
                ok = False

    _P().feed(msg)
    return ok and not stack


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _alerter(enabled: bool = True) -> TelegramAlerter:
    if enabled:
        return TelegramAlerter(bot_token="faketoken", chat_id="12345")
    return TelegramAlerter(bot_token="", chat_id="")


def _order_result(side: OrderSide = OrderSide.BUY) -> OrderResult:
    return OrderResult(
        order_id="ord001",
        client_order_id="cli001",
        symbol="AVAX/USDT",
        side=side,
        order_type=OrderType.MARKET,
        status=OrderStatus.FILLED,
        requested_quantity=10.0,
        filled_quantity=10.0,
        average_price=22.45,
        fee=0.02,
        fee_currency="USDT",
        timestamp=1_700_000_000_000,
    )


# ---------------------------------------------------------------------------
# Disabled alerter
# ---------------------------------------------------------------------------


class TestDisabledAlerter:
    @pytest.mark.asyncio
    async def test_disabled_send_returns_false(self):
        alerter = _alerter(enabled=False)
        result = await alerter.send("hello")
        assert result is False

    @pytest.mark.asyncio
    async def test_disabled_trade_alert_returns_false(self):
        alerter = _alerter(enabled=False)
        result = await alerter.send_trade_alert(_order_result())
        assert result is False

    @pytest.mark.asyncio
    async def test_disabled_never_makes_http_call(self):
        alerter = _alerter(enabled=False)
        with patch("aiohttp.ClientSession") as mock_session:
            await alerter.send("test")
            mock_session.assert_not_called()


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    def test_messages_within_limit_are_allowed(self):
        alerter = _alerter()
        alerter._send_times = [time.time() - 1.0] * 29
        assert not alerter._is_rate_limited()

    def test_messages_at_limit_are_blocked(self):
        alerter = _alerter()
        alerter._send_times = [time.time() - 1.0] * 30
        assert alerter._is_rate_limited()

    def test_old_timestamps_expire_from_window(self):
        alerter = _alerter()
        alerter._send_times = [time.time() - 61.0] * 30
        assert not alerter._is_rate_limited()

    @pytest.mark.asyncio
    async def test_rate_limited_send_returns_false_without_http_call(self):
        alerter = _alerter()
        alerter._send_times = [time.time() - 1.0] * 30
        with patch("aiohttp.ClientSession") as mock_session:
            result = await alerter.send("blocked message")
            assert result is False
            mock_session.assert_not_called()


# ---------------------------------------------------------------------------
# Send failure handling
# ---------------------------------------------------------------------------


class TestSendFailures:
    @pytest.mark.asyncio
    async def test_http_error_returns_false_no_raise(self):
        alerter = _alerter()
        mock_resp = AsyncMock()
        mock_resp.status = 400
        mock_resp.text = AsyncMock(return_value="bad request")
        resp_cm = AsyncMock()
        resp_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        resp_cm.__aexit__ = AsyncMock(return_value=None)
        session = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        from unittest.mock import MagicMock

        session.post = MagicMock(return_value=resp_cm)

        with patch("aiohttp.ClientSession", return_value=session):
            result = await alerter.send("test")
            assert result is False

    @pytest.mark.asyncio
    async def test_network_exception_returns_false_no_raise(self):
        alerter = _alerter()
        with patch("aiohttp.ClientSession", side_effect=Exception("connection refused")):
            result = await alerter.send("test")
            assert result is False

    @pytest.mark.asyncio
    async def test_successful_send_returns_true(self):
        alerter = _alerter()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        resp_cm = AsyncMock()
        resp_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        resp_cm.__aexit__ = AsyncMock(return_value=None)
        session = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        from unittest.mock import MagicMock

        session.post = MagicMock(return_value=resp_cm)

        with patch("aiohttp.ClientSession", return_value=session):
            result = await alerter.send("test")
            assert result is True


# ---------------------------------------------------------------------------
# Message formatting
# ---------------------------------------------------------------------------


class TestMessageFormatting:
    @pytest.mark.asyncio
    async def test_buy_trade_alert_contains_symbol_and_qty(self):
        alerter = _alerter()
        captured: list[str] = []

        async def _fake_send(msg: str, **_: object) -> bool:
            captured.append(msg)
            return True

        alerter.send = _fake_send  # type: ignore[method-assign]
        await alerter.send_trade_alert(_order_result(side=OrderSide.BUY))
        assert len(captured) == 1
        assert "AVAX/USDT" in captured[0]
        assert "BUY" in captured[0]
        assert "10.0000" in captured[0]

    @pytest.mark.asyncio
    async def test_sell_trade_alert_contains_sell(self):
        alerter = _alerter()
        captured: list[str] = []

        async def _fake_send(msg: str, **_: object) -> bool:
            captured.append(msg)
            return True

        alerter.send = _fake_send  # type: ignore[method-assign]
        await alerter.send_trade_alert(_order_result(side=OrderSide.SELL))
        assert "SELL" in captured[0]

    @pytest.mark.asyncio
    async def test_daily_summary_contains_key_fields(self):
        alerter = _alerter()
        captured: list[str] = []

        async def _fake_send(msg: str, **_: object) -> bool:
            captured.append(msg)
            return True

        alerter.send = _fake_send  # type: ignore[method-assign]
        await alerter.send_daily_summary(
            {
                "date": "2026-03-28",
                "trades": 3,
                "wins": 2,
                "losses": 1,
                "pnl": 45.2,
                "pnl_pct": 0.45,
                "equity": 10045.2,
                "drawdown_pct": 1.2,
                "open_positions": 1,
            }
        )
        assert "2026-03-28" in captured[0]
        assert "45.20" in captured[0]

    @pytest.mark.asyncio
    async def test_risk_alert_contains_event_type(self):
        alerter = _alerter()
        captured: list[str] = []

        async def _fake_send(msg: str, **_: object) -> bool:
            captured.append(msg)
            return True

        alerter.send = _fake_send  # type: ignore[method-assign]
        event = RiskEvent(
            timestamp=0,
            event_type="drawdown_yellow",
            details={"drawdown_pct": 0.052},
        )
        await alerter.send_risk_alert(event)
        assert "drawdown_yellow" in captured[0]

    @pytest.mark.asyncio
    async def test_startup_message_contains_started(self):
        alerter = _alerter()
        captured: list[str] = []

        async def _fake_send(msg: str, **_: object) -> bool:
            captured.append(msg)
            return True

        alerter.send = _fake_send  # type: ignore[method-assign]
        await alerter.send_startup("Mode: PAPER TRADING")
        assert "started" in captured[0].lower()
        assert "PAPER TRADING" in captured[0]

    @pytest.mark.asyncio
    async def test_take_profit_reasoning_is_html_safe(self):
        """Reasoning strings containing field names like ``mean_return`` must NOT
        break the parse.  Under legacy Markdown the underscores left the ``_..._``
        italic span open and Telegram 400'd the whole send (2026-06-04).  With
        HTML the underscores are literal and the ``<i>`` wrapper stays balanced."""
        alerter = _alerter()
        captured: list[str] = []

        async def _fake_send(msg: str, **_: object) -> bool:
            captured.append(msg)
            return True

        alerter.send = _fake_send  # type: ignore[method-assign]
        await alerter.alert_take_profit(
            symbol="USD/SEK",
            reason="signal_decay_mean_flip",
            entry_price=9347.82,
            exit_price=9364.40,
            pnl_pct=0.18,
            reasoning="mean_return=-0.0003 flipped from entry=0.0026",
        )
        msg = captured[0]
        # Underscores pass through literally — no backslash escaping.
        assert "mean_return=-0.0003 flipped from entry=0.0026" in msg
        assert "\\_" not in msg
        # Wrapped in a balanced italic entity; whole message parses cleanly.
        assert "<i>mean_return=-0.0003 flipped from entry=0.0026</i>" in msg
        assert _html_balanced(msg)

    @pytest.mark.asyncio
    async def test_send_error_is_html_safe(self):
        alerter = _alerter()
        captured: list[str] = []

        async def _fake_send(msg: str, **_: object) -> bool:
            captured.append(msg)
            return True

        alerter.send = _fake_send  # type: ignore[method-assign]
        # Underscores must survive untouched; HTML specials must be escaped.
        await alerter.send_error("close failed for _internal_ <tag> & co")
        msg = captured[0]
        assert "_internal_" in msg
        assert "\\_" not in msg
        assert "&lt;tag&gt;" in msg
        assert "&amp;" in msg
        assert "<tag>" not in msg  # raw angle brackets neutralised
        assert _html_balanced(msg)

    @pytest.mark.asyncio
    async def test_ampersand_in_symbol_label_is_escaped(self):
        """A bare ``&`` in a symbol label would break HTML parsing if not
        escaped to ``&amp;``.  ``_label`` must escape it — regression guard for
        the labels that flow into every alert."""
        alerter = _alerter()
        captured: list[str] = []

        async def _fake_send(msg: str, **_: object) -> bool:
            captured.append(msg)
            return True

        alerter.send = _fake_send  # type: ignore[method-assign]
        await alerter.send_trade_alert(_order_result(side=OrderSide.BUY), display_symbol="A&B")
        msg = captured[0]
        assert "A&amp;B" in msg
        assert "A&B" not in msg.replace("A&amp;B", "")  # no bare &
        assert _html_balanced(msg)

    @pytest.mark.asyncio
    async def test_shutdown_message_contains_reason(self):
        alerter = _alerter()
        captured: list[str] = []

        async def _fake_send(msg: str, **_: object) -> bool:
            captured.append(msg)
            return True

        alerter.send = _fake_send  # type: ignore[method-assign]
        await alerter.send_shutdown("SIGINT (manual shutdown)")
        assert "SIGINT" in captured[0]


class TestTopKRerank:
    def _fake_signal(self, symbol: str, ret: float = 0.02) -> object:
        from dataclasses import make_dataclass

        Sig = make_dataclass(
            "Sig",
            ["symbol", "mean_return", "direction_confidence", "uncertainty", "tradeable"],
        )
        return Sig(
            symbol=symbol,
            mean_return=ret,
            direction_confidence=0.80,
            uncertainty=0.9,
            tradeable=True,
        )

    async def test_bumped_section_appears_when_provided(self) -> None:
        alerter = _alerter()
        captured: list[str] = []

        async def _fake_send(msg: str, **_: object) -> bool:
            captured.append(msg)
            return True

        alerter.send = _fake_send  # type: ignore[method-assign]

        sigs = [self._fake_signal("EUR/USD", 0.03), self._fake_signal("GBP/USD", 0.02)]
        bumped = [("GBP/USD", "EUR/USD", 0.78)]
        await alerter.send_topk_rerank(sigs, ["EUR/USD"], 3, bumped=bumped)

        assert len(captured) == 1
        msg = captured[0]
        assert "Bumped" in msg or "bumped" in msg.lower() or "\U0001f500" in msg
        assert "GBP" in msg
        assert "0.78" in msg

    async def test_full_rerank_message_is_html_balanced(self) -> None:
        """The richest message (status + positions + open/closed signals +
        bumps, including a friendly-name label) must parse cleanly."""
        alerter = _alerter()
        captured: list[str] = []

        async def _fake_send(msg: str, **_: object) -> bool:
            captured.append(msg)
            return True

        alerter.send = _fake_send  # type: ignore[method-assign]
        sigs = [
            self._fake_signal("XAU/USD", 0.03),  # friendly "Gold"
            self._fake_signal("EUR/USD", 0.02),
            self._fake_signal("GBP/USD", 0.01),
        ]
        positions = {
            "XAU/USD": {"entry_price": 500.0, "quantity": 2, "stop_price": 490.0, "stop_pct": 0.02}
        }
        await alerter.send_topk_rerank(
            sigs,
            ["XAU/USD"],
            3,
            positions=positions,
            current_prices={"XAU/USD": 505.0},
            equity=10000.0,
            risk_summary={"pnl_24h": -5.0, "current_drawdown_pct": 0.03, "drawdown_tier": "YELLOW"},
            bumped=[("GBP/USD", "EUR/USD", 0.78)],
            open_market={"XAU/USD": True, "EUR/USD": False, "GBP/USD": False}.__getitem__,
        )
        msg = captured[0]
        assert "Gold" in msg  # XAU/USD renders via its friendly name
        assert _html_balanced(msg)

    async def test_no_bumped_section_when_none(self) -> None:
        alerter = _alerter()
        captured: list[str] = []

        async def _fake_send(msg: str, **_: object) -> bool:
            captured.append(msg)
            return True

        alerter.send = _fake_send  # type: ignore[method-assign]

        sigs = [self._fake_signal("EUR/USD", 0.03)]
        await alerter.send_topk_rerank(sigs, ["EUR/USD"], 3)

        assert len(captured) == 1
        assert "\U0001f500" not in captured[0]

    async def test_bumped_empty_list_no_section(self) -> None:
        alerter = _alerter()
        captured: list[str] = []

        async def _fake_send(msg: str, **_: object) -> bool:
            captured.append(msg)
            return True

        alerter.send = _fake_send  # type: ignore[method-assign]

        sigs = [self._fake_signal("EUR/USD", 0.03)]
        await alerter.send_topk_rerank(sigs, ["EUR/USD"], 3, bumped=[])

        assert "\U0001f500" not in captured[0]

    async def test_open_market_splits_tradeable_into_open_and_closed(self) -> None:
        """When ``open_market`` is supplied, the green ✅ tick only fires for
        signals whose market is currently open; market-closed tradeable signals
        get a separate 🌙 line and never claim a ✅ even when "selected".
        """
        alerter = _alerter()
        captured: list[str] = []

        async def _fake_send(msg: str, **_: object) -> bool:
            captured.append(msg)
            return True

        alerter.send = _fake_send  # type: ignore[method-assign]

        sigs = [
            self._fake_signal("EUR/USD", 0.03),  # forex — pretend market closed
            self._fake_signal("XAU/USD", 0.02),  # gold — pretend market open
        ]
        open_market = {"XAU/USD": True, "EUR/USD": False}.__getitem__

        await alerter.send_topk_rerank(
            sigs,
            ["EUR/USD", "XAU/USD"],  # both "selected" by Kronos
            3,
            open_market=open_market,
        )

        assert len(captured) == 1
        msg = captured[0]
        # Open-market XAU/USD gets the green tick (rendered via friendly name "Gold").
        open_lines = [ln for ln in msg.splitlines() if ln.startswith("✅")]
        assert len(open_lines) == 1
        assert "Gold" in open_lines[0]
        # Closed-market EUR/USD never gets ✅ even though it is in `selected`.
        eur_lines = [ln for ln in msg.splitlines() if "EUR/USD" in ln]
        assert eur_lines, "EUR/USD should appear somewhere in the message"
        assert all("✅" not in ln for ln in eur_lines)
        # Closed bucket is summarised on its own line with the 🌙 marker.
        assert "\U0001f319" in msg
        assert "ready but market closed" in msg

    async def test_open_market_omitted_keeps_legacy_behaviour(self) -> None:
        """Without ``open_market``, every Kronos-tradeable signal is rendered
        as if the market were open — backwards-compat with existing callers.
        """
        alerter = _alerter()
        captured: list[str] = []

        async def _fake_send(msg: str, **_: object) -> bool:
            captured.append(msg)
            return True

        alerter.send = _fake_send  # type: ignore[method-assign]

        sigs = [self._fake_signal("EUR/USD", 0.03)]
        await alerter.send_topk_rerank(sigs, ["EUR/USD"], 3)

        assert len(captured) == 1
        msg = captured[0]
        assert "✅" in msg
        assert "\U0001f319" not in msg
        assert "ready but market closed" not in msg

    async def test_no_open_signals_message_when_all_closed(self) -> None:
        """When every tradeable signal is market-closed and nothing was
        selected, the closing line points at the waiting count rather than
        the generic "no tradeable signals" line.
        """
        alerter = _alerter()
        captured: list[str] = []

        async def _fake_send(msg: str, **_: object) -> bool:
            captured.append(msg)
            return True

        alerter.send = _fake_send  # type: ignore[method-assign]

        sigs = [self._fake_signal("EUR/USD", 0.03), self._fake_signal("USD/SEK", 0.01)]
        await alerter.send_topk_rerank(
            sigs,
            [],  # nothing selected (e.g. weekend, all candidates market-closed)
            3,
            open_market=lambda _sym: False,
        )

        msg = captured[0]
        assert "No new entries until market reopens" in msg
        assert "2 signal(s) waiting" in msg


class TestRerankPositionsStopColumn:
    """The 2026-06-01 patch adds ``stop_price`` + ``stop_pct`` keys to the
    ``positions`` dict passed to ``send_topk_rerank``.  The renderer must
    show them when present and degrade gracefully when absent so old
    callers (e.g. a stub TradingBot in tests) don't break."""

    def _fake_signal(self, symbol: str, ret: float) -> object:
        from unittest.mock import MagicMock

        sig = MagicMock()
        sig.symbol = symbol
        sig.mean_return = ret
        sig.tradeable = True
        sig.direction_confidence = 1.0
        sig.uncertainty = 1.0
        sig.stop_pct = 0.01
        sig.direction = "LONG" if ret >= 0 else "SHORT"
        return sig

    async def test_stop_rendered_when_provided(self) -> None:
        alerter = _alerter()
        captured: list[str] = []

        async def _fake_send(msg: str, **_: object) -> bool:
            captured.append(msg)
            return True

        alerter.send = _fake_send  # type: ignore[method-assign]
        positions = {
            "FTSE": {
                "entry_price": 10321.6,
                "quantity": 1.08,
                "stop_price": 10138.7,
                "stop_pct": 0.0177,
            }
        }
        await alerter.send_topk_rerank(
            [self._fake_signal("FTSE", 0.008)],
            ["FTSE"],
            3,
            positions=positions,
        )
        msg = captured[0]
        # Stop level + percent appear on the position line, both formatted.
        assert "stop=10138.7000" in msg
        assert "-1.77%" in msg

    async def test_position_without_stop_keys_renders_unchanged(self) -> None:
        """Backward-compat: callers that don't supply stop info still get
        a position line, just without the stop suffix.  Important for tests
        that fake a position via dataclass-style object access (no .get())."""
        alerter = _alerter()
        captured: list[str] = []

        async def _fake_send(msg: str, **_: object) -> bool:
            captured.append(msg)
            return True

        alerter.send = _fake_send  # type: ignore[method-assign]
        positions = {
            "FTSE": {
                "entry_price": 10321.6,
                "quantity": 1.08,
                # stop_price / stop_pct absent — should render no stop suffix
            }
        }
        await alerter.send_topk_rerank(
            [self._fake_signal("FTSE", 0.008)],
            ["FTSE"],
            3,
            positions=positions,
        )
        msg = captured[0]
        assert "FTSE" in msg
        assert "stop=" not in msg
