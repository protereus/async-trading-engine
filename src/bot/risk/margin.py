"""Margin-utilisation circuit-breaker (extracted from risk_manager.py).

Implements IG_LIVE_RISK_REFERENCE.md §4.3: track
``equity / total_margin_required`` from every LS ACCOUNT push and emit
``EVENT_MARGIN_BREAKER`` on transitions between the five circuit-breaker
states (NORMAL > HALT_ENTRIES > DEFENSIVE_CLOSE > EMERGENCY_FLATTEN >
LIQUIDATION).  Caller is responsible for the actual de-risking action —
this module just classifies and announces.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from bot.core.event_bus import EVENT_MARGIN_BREAKER, EVENT_SHUTDOWN
from bot.core.models import (
    AccountUpdate,
    MarginAction,
    MarginBreakerEvent,
    MarginCircuitState,
)
from bot.risk.risk_config import RiskConfig

if TYPE_CHECKING:
    from bot.core.event_bus import EventBus

logger = logging.getLogger(__name__)

# Used when no positions are open (and therefore total_margin_required = 0).
_MARGIN_RATIO_INFINITY = float("inf")

# Callback signature: (event_type, payload) -> None — called for every
# transition that should be logged via the host's risk-event ledger.
RiskEventCallback = Callable[[str, dict[str, Any]], None]


class MarginCircuitBreaker:
    """Real-time margin-utilisation classifier with edge-triggered events."""

    def __init__(
        self,
        config: RiskConfig,
        event_bus: EventBus,
        clock_fn: Callable[[], int],
        risk_event_callback: RiskEventCallback | None = None,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._clock = clock_fn
        self._risk_event_callback = risk_event_callback

        self._ratio: float = _MARGIN_RATIO_INFINITY
        self._margin_required: float = 0.0
        self._state: MarginCircuitState = MarginCircuitState.NORMAL

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ratio(self) -> float:
        return self._ratio

    @property
    def state(self) -> MarginCircuitState:
        return self._state

    @property
    def margin_required(self) -> float:
        return self._margin_required

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, ratio: float) -> MarginCircuitState:
        """Map a margin-utilisation ratio onto a circuit-breaker state.

        Thresholds come from ``RiskConfig`` and default to the values in
        IG_LIVE_RISK_REFERENCE.md §4.3 — well above the broker's 0.50
        liquidation floor so we have several steps of warning.
        """
        cfg = self._config
        if ratio <= cfg.margin_liquidation_floor:
            return MarginCircuitState.LIQUIDATION
        if ratio <= cfg.margin_emergency_ratio:
            return MarginCircuitState.EMERGENCY_FLATTEN
        if ratio <= cfg.margin_defensive_ratio:
            return MarginCircuitState.DEFENSIVE_CLOSE
        if ratio <= cfg.margin_halt_ratio:
            return MarginCircuitState.HALT_ENTRIES
        return MarginCircuitState.NORMAL

    # ------------------------------------------------------------------
    # Update entry-point
    # ------------------------------------------------------------------

    def update(self, update: AccountUpdate) -> MarginCircuitState:
        """Recompute the circuit-breaker state from a fresh account snapshot.

        Returns the new state.  Emits ``EVENT_MARGIN_BREAKER`` only on
        *transitions* so action handlers don't loop while utilisation stays
        elevated.  On transition to LIQUIDATION also emits ``EVENT_SHUTDOWN``.
        """
        equity = max(0.0, update.equity)
        margin_required = max(0.0, update.margin_required)
        ratio = _MARGIN_RATIO_INFINITY if margin_required <= 0 else equity / margin_required

        self._ratio = ratio
        self._margin_required = margin_required

        old_state = self._state
        new_state = self.classify(ratio)
        self._state = new_state

        if new_state == old_state:
            return new_state

        logger.warning(
            "Margin circuit breaker: %s → %s  equity=%.2f margin_req=%.2f ratio=%.3f",
            old_state.value,
            new_state.value,
            equity,
            margin_required,
            ratio,
        )
        if self._risk_event_callback is not None:
            self._risk_event_callback(
                "margin_circuit_transition",
                {
                    "from": old_state.value,
                    "to": new_state.value,
                    "equity": equity,
                    "margin_required": margin_required,
                    "ratio": ratio,
                },
            )

        action_map: dict[MarginCircuitState, MarginAction] = {
            MarginCircuitState.HALT_ENTRIES: "halt_entries",
            MarginCircuitState.DEFENSIVE_CLOSE: "close_worst",
            MarginCircuitState.EMERGENCY_FLATTEN: "flatten",
            MarginCircuitState.LIQUIDATION: "liquidation_alert",
        }
        action = action_map.get(new_state)
        if action is not None:
            event = MarginBreakerEvent(
                timestamp=self._clock(),
                state=new_state,
                action=action,
                ratio=ratio,
                equity=equity,
                margin_required=margin_required,
            )
            self._emit_async(EVENT_MARGIN_BREAKER, event)

        # The 0.50 floor is the broker's number — by the time we observe it,
        # IG's auto-close-out has already started.  Flag loudly and shut down.
        if new_state == MarginCircuitState.LIQUIDATION:
            logger.critical(
                "MARGIN LIQUIDATION FLOOR BREACHED  equity=%.2f margin_req=%.2f ratio=%.3f "
                "— IG auto-liquidation may already be in progress",
                equity,
                margin_required,
                ratio,
            )
            self._emit_async(EVENT_SHUTDOWN, "margin_liquidation_floor_breached")

        return new_state

    # ------------------------------------------------------------------
    # Async bridge
    # ------------------------------------------------------------------

    def _emit_async(self, event_type: str, payload: Any) -> None:
        """Schedule an emission onto the running loop without blocking.

        Falls back to a debug log if no loop is running (e.g. during unit
        tests that create a tracker outside an async context)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("No running loop for %s emission — payload=%r", event_type, payload)
            return
        loop.create_task(self._event_bus.emit(event_type, payload))
