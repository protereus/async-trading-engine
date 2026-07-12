"""Async event bus for inter-module communication.

Provides a publish/subscribe pattern for bot events.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from collections import defaultdict
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Well-known event type constants
EVENT_NEW_CANDLE = "new_candle"
EVENT_SIGNAL = "signal"
EVENT_ORDER_PLACED = "order_placed"
EVENT_ORDER_FILLED = "order_filled"
EVENT_ORDER_CANCELLED = "order_cancelled"
EVENT_POSITION_CLOSED = "position_closed"
EVENT_RISK_ALERT = "risk_alert"
EVENT_SHUTDOWN = "shutdown"
# Margin circuit-breaker — payload is a MarginBreakerEvent with the
# requested action ("close_worst" or "flatten") and current ratio.  Emitted
# only on state transitions so subscribers don't loop on a stuck high state.
EVENT_MARGIN_BREAKER = "margin_breaker"
# Raw LS ACCOUNT push translated into (equity_gbp, margin_required_gbp,
# available_to_deal_gbp, unrealised_pnl_gbp).  RiskManager subscribes to
# update its circuit-breaker state in real time.
EVENT_ACCOUNT_UPDATE = "account_update"


class EventBus:
    """Simple async pub/sub event bus.

    Handlers are called concurrently via asyncio.gather; exceptions in one
    handler do not prevent other handlers from running.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[..., Any]]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Callable[..., Any]) -> None:
        """Register *handler* to be called when *event_type* is emitted."""
        self._subscribers[event_type].append(handler)
        logger.debug("Subscribed %s to event '%s'", handler.__name__, event_type)

    def unsubscribe(self, event_type: str, handler: Callable[..., Any]) -> None:
        """Remove *handler* from the subscriber list for *event_type*."""
        try:
            self._subscribers[event_type].remove(handler)
            logger.debug("Unsubscribed %s from event '%s'", handler.__name__, event_type)
        except ValueError:
            logger.warning(
                "unsubscribe: handler %s was not subscribed to '%s'",
                handler.__name__,
                event_type,
            )

    async def emit(self, event_type: str, data: Any = None) -> None:
        """Emit an event, calling all registered handlers concurrently.

        Handler exceptions are logged with full traceback but do not
        propagate — the event bus will not crash.
        """
        handlers = list(self._subscribers.get(event_type, []))
        if not handlers:
            return

        results = await asyncio.gather(
            *(self._call_handler(h, data) for h in handlers),
            return_exceptions=True,
        )

        for handler, result in zip(handlers, results, strict=False):
            if isinstance(result, BaseException):
                logger.error(
                    "Handler %s raised an exception for event '%s':\n%s",
                    handler.__name__,
                    event_type,
                    "".join(traceback.format_exception(type(result), result, result.__traceback__)),
                )

    @staticmethod
    async def _call_handler(handler: Callable[..., Any], data: Any) -> Any:
        if asyncio.iscoroutinefunction(handler):
            return await handler(data)
        return handler(data)
