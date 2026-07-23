"""Event-bus wiring + callbacks for TradingBot.

Extracted from ``main.py``.  Owns the seven cross-cutting handlers
that translate event-bus signals into bot state changes:

* ``wire`` — subscribes the handlers to their topics on the shared bus.
* ``handle_account_update`` — forwards LS ACCOUNT pushes into
  ``RiskManager.update_margin_state``.
* ``handle_margin_breaker`` — executes the de-risking action a margin
  breaker transition requested (``halt_entries`` / ``close_worst`` /
  ``flatten`` / ``liquidation_alert``).
* ``handle_order_filled`` — bookkeeping for BUY / SELL fills: update
  ``RiskManager``, send the Telegram trade alert, mutate
  ``_state.positions`` + ``_ig_deal_ids``, and emit
  ``EVENT_POSITION_CLOSED`` with the realised P&L on SELL.
* ``handle_position_closed`` — tells ``RiskManager.on_position_closed``
  about the realised P&L so loss windows / drawdown tracking update.
* ``handle_risk_alert`` — forwards a ``RiskEvent`` to Telegram.
* ``handle_shutdown`` — sets ``_shutdown_event`` so the supervisor can
  cancel every task.

Operates on the shared ``BotContext`` like every collaborator — reads /
mutates ``ctx.event_bus``, ``ctx.risk_manager``, ``ctx.alerter``,
``ctx.state``, ``ctx.ig_deal_ids``, ``ctx.closer``, ``ctx.shutdown_event``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from bot.core.event_bus import (
    EVENT_ACCOUNT_UPDATE,
    EVENT_MARGIN_BREAKER,
    EVENT_ORDER_FILLED,
    EVENT_POSITION_CLOSED,
    EVENT_RISK_ALERT,
    EVENT_SHUTDOWN,
)
from bot.core.models import (
    AccountUpdate,
    MarginBreakerEvent,
    OrderResult,
    OrderSide,
    PositionClosed,
    RiskEvent,
)
from bot.execution.ig_quote_scale import ig_quote_scale

if TYPE_CHECKING:
    from bot.core.bot_context import BotContext

logger = logging.getLogger(__name__)


class EventWiring:
    """Event-bus subscription + handlers collaborator for TradingBot."""

    def __init__(self, ctx: BotContext) -> None:
        self._ctx = ctx

    def wire(self) -> None:
        """Register event handlers on the bus."""
        bus = self._ctx.event_bus
        bus.subscribe(EVENT_ORDER_FILLED, self.handle_order_filled)
        bus.subscribe(EVENT_POSITION_CLOSED, self.handle_position_closed)
        bus.subscribe(EVENT_RISK_ALERT, self.handle_risk_alert)
        bus.subscribe(EVENT_SHUTDOWN, self.handle_shutdown)
        # IGFeed emits AccountUpdate on every LS ACCOUNT push; we forward
        # to RiskManager to recompute margin ratio + circuit-breaker state.
        # On state transitions RiskManager emits EVENT_MARGIN_BREAKER which the
        # action handler below executes (defensive close / flatten).
        bus.subscribe(EVENT_ACCOUNT_UPDATE, self.handle_account_update)
        bus.subscribe(EVENT_MARGIN_BREAKER, self.handle_margin_breaker)

    async def handle_account_update(self, data: Any) -> None:
        """Forward LS ACCOUNT pushes into RiskManager so the margin circuit
        breakers update in real time."""
        if isinstance(data, AccountUpdate):
            self._ctx.risk_manager.update_margin_state(data)

    async def handle_margin_breaker(self, data: Any) -> None:
        """Execute the de-risking action requested by a margin breaker
        transition.  ``halt_entries`` is enforced by the pre-trade gate
        in evaluate_ig_order — nothing to do here.  ``close_worst`` picks the
        worst-performing open position and closes it.  ``flatten`` closes
        every open position.  ``liquidation_alert`` only logs + alerts —
        IG's own auto-close-out is already running by then."""
        ctx = self._ctx
        if not isinstance(data, MarginBreakerEvent):
            return

        try:
            await ctx.alerter.send_error(
                f"Margin circuit breaker → {data.state} (action={data.action}) "
                f"ratio={data.ratio:.2f}  equity=£{data.equity:.2f}  "
                f"margin_req=£{data.margin_required:.2f}"
            )
        except Exception:
            logger.exception("Failed to send margin-breaker alert")

        if data.action == "halt_entries":
            # Pre-trade gate handles this; nothing to do beyond the alert.
            return

        if data.action == "liquidation_alert":
            logger.critical("Margin LIQUIDATION alert — IG auto-close-out may be in progress")
            return

        # Both defensive-close and flatten need a current snapshot of open
        # positions keyed by candle-symbol (which close_position expects).
        symbols = list(ctx.state.positions.keys())
        if not symbols:
            logger.warning("Margin breaker %s but no tracked positions to close", data.action)
            return

        if data.action == "close_worst":
            worst_symbol = ctx.closer.pick_worst_performer(symbols)
            if worst_symbol is None:
                logger.warning("close_worst requested but no live prices for any position")
                return
            logger.warning(
                "Margin breaker DEFENSIVE_CLOSE: closing worst performer %s (ratio=%.2f)",
                worst_symbol,
                data.ratio,
            )
            await ctx.closer.request_close(
                worst_symbol,
                reason="margin_breaker_defensive",
                reasoning=f"Margin ratio {data.ratio:.2f} ≤ defensive threshold",
            )
            return

        if data.action == "flatten":
            logger.critical(
                "Margin breaker EMERGENCY_FLATTEN: closing %d position(s) (ratio=%.2f)",
                len(symbols),
                data.ratio,
            )
            for symbol in symbols:
                await ctx.closer.request_close(
                    symbol,
                    reason="margin_breaker_flatten",
                    reasoning=f"Margin ratio {data.ratio:.2f} ≤ emergency threshold",
                )
            return

    async def handle_order_filled(self, data: Any) -> None:
        ctx = self._ctx
        if not isinstance(data, OrderResult):
            return

        # For IG, data.symbol is the EPIC (e.g. CS.D.USCGC.TODAY.IP).  Translate back
        # to the candle symbol (e.g. XAU/USD) so _state.positions is keyed consistently.
        state_symbol = ctx.candle_for(data.symbol)

        pnl_for_event: float | None = None
        if data.side == OrderSide.SELL and data.filled_quantity > 0:
            pos = ctx.risk_manager.get_position(data.symbol)
            if pos is not None:
                pnl_for_event = (data.average_price - pos.entry_price) * data.filled_quantity

        new_pos = ctx.risk_manager.on_fill(data)
        await ctx.alerter.send_trade_alert(
            data,
            display_symbol=state_symbol,
            display_price=data.average_price / ig_quote_scale(state_symbol),
        )

        if data.side == OrderSide.BUY and data.filled_quantity > 0:
            if new_pos is not None:
                ctx.state.positions[state_symbol] = new_pos
            # _ig_deal_ids[state_symbol] is set before event emission in
            # process_candle_ig_topk, so nothing to record here.
        elif data.side == OrderSide.SELL and data.filled_quantity > 0:
            ctx.state.positions.pop(state_symbol, None)
            # _ig_deal_ids[state_symbol] is already removed by close_ig_position.
            if pnl_for_event is not None:
                await ctx.event_bus.emit(
                    EVENT_POSITION_CLOSED,
                    PositionClosed(symbol=data.symbol, pnl=pnl_for_event),
                )

    async def handle_position_closed(self, data: Any) -> None:
        if isinstance(data, PositionClosed):
            self._ctx.risk_manager.on_position_closed(data.symbol, data.pnl)

    async def handle_risk_alert(self, data: Any) -> None:
        if isinstance(data, RiskEvent):
            await self._ctx.alerter.send_risk_alert(data)

    async def handle_shutdown(self, data: Any) -> None:
        logger.critical("Risk manager triggered shutdown: %s", data)
        self._ctx.shutdown_event.set()
