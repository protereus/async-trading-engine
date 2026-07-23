"""Async Telegram alert sender via aiohttp.

Sends trade notifications, risk alerts, daily P&L summaries, startup /
shutdown messages, and error notifications via the Telegram Bot API.

Rate limit: max 30 messages per minute (conservative) to avoid 429s during
alert floods.  Send failures are logged but never propagated — Telegram is
monitoring, not critical path.  If bot_token or chat_id is empty the alerter
disables itself silently.
"""

from __future__ import annotations

import html
import logging
import time
from collections.abc import Callable
from typing import Any

import aiohttp

from bot.core.models import OrderResult, OrderSide, RiskEvent

logger = logging.getLogger(__name__)

_RATE_LIMIT_PER_MINUTE = 30
_SEND_TIMEOUT_S = 10.0
_API_URL = "https://api.telegram.org"

# Human-readable names for the 28-asset universe.  Forex pairs and single-name
# US shares are self-explanatory; only the metals need a friendly label.
_FRIENDLY_NAMES: dict[str, str] = {
    "XAU/USD": "Gold",
    "XAG/USD": "Silver",
}


def _friendly(symbol: str) -> str:
    """Return a human-readable label for *symbol*, falling back to the symbol itself."""
    return _FRIENDLY_NAMES.get(symbol, symbol)


# Alerts are sent with ``parse_mode="HTML"`` (not legacy Markdown, which has no
# backslash escaping — see the 2026-06-04 "can't parse entities" incident where
# an underscore in ``mean_return=…`` left an italic span open and Telegram 400'd
# the whole send).  HTML only needs ``< > &`` escaped, and every dynamic value
# (symbol labels like "S&P 500", reason/reasoning strings, error text) MUST go
# through ``_h`` before interpolation.
def _h(text: str) -> str:
    """Escape *text* for Telegram HTML parse mode (only ``< > &``)."""
    return html.escape(text, quote=False)


def _label(symbol: str) -> str:
    """HTML-safe human-readable label for *symbol*."""
    return _h(_friendly(symbol))


class TelegramAlerter:
    """Async Telegram message sender with per-minute rate limiting."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._enabled = bool(bot_token and chat_id)
        if not self._enabled:
            logger.warning("Alerts disabled: bot_token or chat_id is empty")
        # Rolling 60-second window for rate limiting
        self._send_times: list[float] = []

    # ------------------------------------------------------------------
    # Core send
    # ------------------------------------------------------------------

    def _is_rate_limited(self) -> bool:
        """Return True if the rolling 60-s window is exhausted."""
        now = time.time()
        self._send_times = [t for t in self._send_times if now - t < 60.0]
        return len(self._send_times) >= _RATE_LIMIT_PER_MINUTE

    async def send(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send *message* to the configured chat.  Returns True on success."""
        if not self._enabled:
            return False
        if self._is_rate_limited():
            logger.warning("Rate limit reached — message dropped: %.80s", message)
            return False

        url = f"{_API_URL}/bot{self._bot_token}/sendMessage"
        payload: dict[str, Any] = {
            "chat_id": self._chat_id,
            "text": message,
            "parse_mode": parse_mode,
        }
        try:
            timeout = aiohttp.ClientTimeout(total=_SEND_TIMEOUT_S)
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(url, json=payload) as resp,
            ):
                if resp.status != 200:
                    body = await resp.text()
                    logger.error("Alert send failed: HTTP %d — %.200s", resp.status, body)
                    return False
            self._send_times.append(time.time())
            return True
        except Exception:
            logger.exception("Alert send exception")
            return False

    # ------------------------------------------------------------------
    # Specialised formatters
    # ------------------------------------------------------------------

    async def send_trade_alert(
        self,
        result: OrderResult,
        display_symbol: str | None = None,
        display_price: float | None = None,
    ) -> bool:
        """Format and send an order fill notification."""
        label = _label(display_symbol or result.symbol)
        price = display_price if display_price is not None else result.average_price
        order_id = _h(str(result.order_id))
        if result.side == OrderSide.BUY:
            msg = (
                f"\U0001f4c8 <b>BUY {label}</b>\n"
                f"Size: {result.filled_quantity:.4f} @ {price:.4f}\n"
                f"Order: <code>{order_id}</code>"
            )
        else:
            msg = (
                f"\U0001f4c9 <b>SELL {label}</b>\n"
                f"Size: {result.filled_quantity:.4f} @ {price:.4f}\n"
                f"Order: <code>{order_id}</code>"
            )
        return await self.send(msg)

    async def send_risk_alert(self, event: RiskEvent) -> bool:
        """Send a risk tier change or limit-breach notification."""
        details_str = ", ".join(f"{k}={v}" for k, v in event.details.items())
        msg = (
            f"\u26a0\ufe0f <b>RISK ALERT</b>\n"
            f"Type: <code>{_h(event.event_type)}</code>\n"
            f"Details: {_h(details_str[:300])}"
        )
        return await self.send(msg)

    async def send_daily_summary(self, summary: dict[str, Any]) -> bool:
        """Send end-of-day P&L summary."""
        date_str = summary.get("date", "")
        trades = summary.get("trades", 0)
        wins = summary.get("wins", 0)
        losses = summary.get("losses", 0)
        pnl = float(summary.get("pnl", 0.0))
        pnl_pct = float(summary.get("pnl_pct", 0.0))
        equity = float(summary.get("equity", 0.0))
        drawdown = float(summary.get("drawdown_pct", 0.0))
        open_pos = summary.get("open_positions", 0)
        msg = (
            f"\U0001f4ca <b>Daily Summary \u2014 {_h(str(date_str))}</b>\n"
            f"Trades: {trades} ({wins} wins, {losses} losses)\n"
            f"PnL: {pnl:+.2f} ({pnl_pct:+.2f}%)\n"
            f"Equity: ${equity:,.2f}\n"
            f"Drawdown: {drawdown:.1f}%\n"
            f"Open positions: {open_pos}"
        )
        return await self.send(msg)

    async def send_error(self, error: str) -> bool:
        """Send an error notification."""
        msg = f"\u274c <b>ERROR</b>\n{_h(error[:500])}"
        return await self.send(msg)

    async def send_topk_rerank(
        self,
        signals: list[Any],
        selected: list[str],
        k: int,
        *,
        positions: dict[str, Any] | None = None,
        risk_summary: dict[str, Any] | None = None,
        current_prices: dict[str, float] | None = None,
        equity: float | None = None,
        cash: float | None = None,
        open_pnl: float | None = None,
        bumped: list[tuple[str, str, float]] | None = None,  # (dropped, blocker, corr)
        open_market: Callable[[str], bool] | None = None,
    ) -> bool:
        """Send TopK rerank result with bot status, open positions, and signal rankings.

        When ``cash`` and ``open_pnl`` are provided alongside ``equity``, the status
        line shows the three-part IG breakdown (Cash | P&L | Eq) so the displayed
        figure matches what's visible on the IG.com web platform.

        When ``open_market`` is supplied (typically ``trading_hours.is_safe_for_entry``),
        the tradeable bucket is split into "market-open" (✅/▫️) and "market-closed"
        (🌙) so the green tick only fires when an entry could actually be placed
        right now.  Without the callback, the legacy behaviour treats every
        Kronos-tradeable signal as market-open.
        """
        selected_set = set(selected)
        lines = [f"\U0001f504 <b>TopK Rerank \u2014 {len(selected)}/{k} selected</b>"]

        # --- Bot status line ---
        if risk_summary is not None or equity is not None:
            status_parts = []
            if cash is not None and open_pnl is not None and equity is not None:
                pnl_sign = "+" if open_pnl >= 0 else ""
                status_parts.append(
                    f"Cash: \xa3{cash:,.0f} | "
                    f"Open P&L: {pnl_sign}\xa3{open_pnl:,.0f} | "
                    f"Eq: \xa3{equity:,.0f}"
                )
            elif equity is not None:
                status_parts.append(f"Eq: \xa3{equity:,.0f}")
            if risk_summary is not None:
                # Legacy key "daily_pnl" supported until any mocked test snapshots
                # are refreshed; the live RiskManager only emits "pnl_24h" now.
                pnl_24h = risk_summary.get("pnl_24h", risk_summary.get("daily_pnl", 0.0))
                dd = risk_summary.get("current_drawdown_pct", 0.0) * 100
                tier = risk_summary.get("drawdown_tier", "NORMAL")
                sign = "+" if pnl_24h >= 0 else ""
                status_parts.append(f"P&L: {sign}\xa3{pnl_24h:.2f}")
                if dd > 0.1:
                    _tier_icons = {
                        "YELLOW": "\U0001f7e1",
                        "ORANGE": "\U0001f7e0",
                        "RED": "\U0001f534",
                    }
                    tier_emoji = _tier_icons.get(tier, "")
                    dd_str = f"DD: {dd:.1f}%"
                    if tier_emoji:
                        dd_str += f" {tier_emoji}"
                    status_parts.append(dd_str)
                consecutive = risk_summary.get("consecutive_losses", 0)
                if consecutive > 0:
                    status_parts.append(f"Losses: {consecutive}")
            lines.append(" | ".join(status_parts))

        # --- Open positions ---
        if positions:
            lines.append("")
            lines.append("<b>Open positions:</b>")
            prices = current_prices or {}
            for sym, pos in positions.items():
                label = _label(sym)
                entry = (
                    pos.entry_price if hasattr(pos, "entry_price") else pos.get("entry_price", 0)
                )
                qty = pos.quantity if hasattr(pos, "quantity") else pos.get("quantity", 0)
                # Stop level + stop_pct are dict-only (added 2026-06-01 so
                # the rerank alert tells the operator where IG would stop
                # the position out without having to open the dashboard).
                stop_price: float | None = (
                    None if hasattr(pos, "entry_price") else pos.get("stop_price")
                )
                stop_pct: float | None = (
                    None if hasattr(pos, "entry_price") else pos.get("stop_pct")
                )
                stop_suffix = ""
                if stop_price is not None and stop_pct is not None:
                    stop_suffix = f" stop={stop_price:.4f} (-{stop_pct * 100:.2f}%)"
                cur = prices.get(sym, 0.0)
                if cur > 0 and entry > 0:
                    pnl_pct = (cur - entry) / entry * 100
                    pnl_sign = "+" if pnl_pct >= 0 else ""
                    lines.append(
                        f"  \U0001f4bc {label}: entry={entry:.4f} now={cur:.4f} "
                        f"({pnl_sign}{pnl_pct:.2f}%){stop_suffix} size={_h(str(qty))}"
                    )
                else:
                    lines.append(
                        f"  \U0001f4bc {label}: entry={entry:.4f}{stop_suffix} size={_h(str(qty))}"
                    )
        else:
            lines.append("<i>No open positions</i>")

        # --- Signal rankings ---
        lines.append("")
        lines.append("<b>Signal rankings:</b>")
        tradeable = [s for s in signals if s.tradeable]
        non_tradeable = [s for s in signals if not s.tradeable]
        # Split tradeable into market-open vs market-closed so the \u2705 tick
        # only fires when an entry could actually be placed right now. When
        # no ``open_market`` callback was supplied we fall back to the
        # legacy behaviour and treat every Kronos-tradeable signal as open.
        if open_market is None:
            tradeable_open = tradeable
            tradeable_closed: list[Any] = []
        else:
            tradeable_open = [s for s in tradeable if open_market(s.symbol)]
            tradeable_closed = [s for s in tradeable if not open_market(s.symbol)]
        for sig in sorted(tradeable_open, key=lambda s: s.mean_return, reverse=True):
            icon = "\u2705" if sig.symbol in selected_set else "\u25ab\ufe0f"
            label = _label(sig.symbol)
            lines.append(
                f"{icon} {label}: {sig.mean_return * 100:+.2f}% "
                f"conf={sig.direction_confidence:.0%} cv={sig.uncertainty:.2f}"
            )
        if tradeable_closed:
            closed_labels = ", ".join(
                _label(s.symbol)
                for s in sorted(tradeable_closed, key=lambda s: s.mean_return, reverse=True)
            )
            lines.append(
                f"<i>\U0001f319 {len(tradeable_closed)} ready but market closed: "
                f"{closed_labels}</i>"
            )
        if non_tradeable:
            lines.append(f"<i>\u274c {len(non_tradeable)} below threshold (SHORT or low conf)</i>")

        if not selected:
            if tradeable_closed and not tradeable_open:
                lines.append(
                    "<i>No new entries until market reopens \u2014 "
                    f"{len(tradeable_closed)} signal(s) waiting</i>"
                )
            else:
                lines.append("<i>No tradeable signals \u2014 no new entries until next rerank</i>")

        # --- Correlation bumps ---
        # Only selection-changing bumps reach here: each `sym` would have been a
        # top-k pick on score but was dropped because it correlated with a
        # higher-ranked selection, pulling a lower-ranked name into the book.
        if bumped:
            lines.append("")
            lines.append("\U0001f500 <b>Dropped for correlation</b> (would-be top pick):")
            for sym, blocker, corr in bumped:
                lines.append(f"  {_label(sym)} ↔ {_label(blocker)} ({abs(corr):.2f})")

        return await self.send("\n".join(lines))

    async def alert_take_profit(
        self,
        symbol: str,
        reason: str,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
        reasoning: str = "",
    ) -> bool:
        """Send a take-profit or signal/time/sentiment exit notification."""
        _EMOJIS = {
            "static_take_profit": "\U0001f3af",
            "trailing_breakeven": "\U0001f512",
            "trailing_ratchet": "\U0001f4c8",
            "signal_decay_mean_flip": "\U0001f504",
            "signal_decay_strikes_exhausted": "\U000026a1",
            "time_limit": "⏰",
            "sentiment_reversal": "\U0001f321",
        }
        emoji = _EMOJIS.get(reason, "❌")
        pnl_sign = "+" if pnl_pct >= 0 else ""
        label = _label(symbol)
        msg = (
            f"{emoji} <b>Exit: {label}</b>\n"
            f"Reason: <code>{_h(reason)}</code>\n"
            f"Entry: {entry_price:.4f}  Exit: {exit_price:.4f}\n"
            f"PnL: {pnl_sign}{pnl_pct:.2f}%"
        )
        if reasoning:
            msg += f"\n<i>{_h(reasoning[:200])}</i>"
        return await self.send(msg)

    async def send_startup(self, config_summary: str) -> bool:
        """Send bot startup notification."""
        msg = f"\U0001f7e2 <b>Bot started</b>\n{_h(config_summary)}"
        return await self.send(msg)

    async def send_shutdown(self, reason: str) -> bool:
        """Send bot shutdown notification."""
        msg = f"\U0001f534 <b>Bot stopped</b>\nReason: {_h(reason)}"
        return await self.send(msg)
