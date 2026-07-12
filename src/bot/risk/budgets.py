"""Risk-on budget ledger (extracted from risk_manager.py).

Tracks per-EPIC £-at-risk so the IG-path total-risk gate can sum live
budgets across all open positions when deciding whether a new entry would
breach ``max_total_risk_pct``.  An injected trailing-stop lookup shrinks
the live risk as TakeProfitManager ratchets stops favourably.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from bot.core.models import Position

logger = logging.getLogger(__name__)


class RiskBudgetLedger:
    """Per-EPIC risk-on budget map + trailing-stop-aware live-risk calc.

    Lifecycle:
      - ``set_risk_budget(epic, gbp)`` — recorded at fill time by main.py
        as ``size × stop_distance`` (£/pt × points)
      - ``clear_risk_budget(epic)`` — on close
      - ``set_trailing_stop_lookup(fn)`` — inject a callable returning the
        live trailing-stop level for an EPIC (in IG-level units, same as
        ``Position.entry_price``); ``None`` for "no trail armed"
      - ``live_risk_gbp(epic, position)`` — current £-at-risk, shrunk by
        any locked-in profit from the trailing stop
    """

    def __init__(self) -> None:
        self._budgets: dict[str, float] = {}
        self._trailing_stop_lookup: Callable[[str], float | None] | None = None

    # ------------------------------------------------------------------
    # Budget ledger
    # ------------------------------------------------------------------

    def set(self, epic: str, gbp: float) -> None:
        if gbp < 0:
            gbp = 0.0
        self._budgets[epic] = gbp

    def clear(self, epic: str) -> None:
        self._budgets.pop(epic, None)

    def get(self, epic: str, default: float = 0.0) -> float:
        return self._budgets.get(epic, default)

    def snapshot(self) -> dict[str, float]:
        return dict(self._budgets)

    def load_snapshot(self, budgets: dict[str, float]) -> None:
        self._budgets = {epic: float(gbp) for epic, gbp in budgets.items()}

    # ------------------------------------------------------------------
    # Trailing-stop lookup injection
    # ------------------------------------------------------------------

    def set_trailing_stop_lookup(self, fn: Callable[[str], float | None] | None) -> None:
        """Inject a trail-stop lookup so the total-risk gate sees live stops.

        ``fn(epic)`` returns the current trailing-stop level (in the same
        units as ``Position.entry_price`` — IG levels for the IG path) or
        ``None`` if no trail is armed.  If the callable raises, the gate
        falls back to the entry-time budget (conservative).
        """
        self._trailing_stop_lookup = fn

    # ------------------------------------------------------------------
    # Live-risk computation
    # ------------------------------------------------------------------

    def live_risk_gbp(self, epic: str, position: Position) -> float:
        """Return live risk-on for a position, shrinking as the trail moves up.

        BUY-only (the bot does not open shorts).  If no trail is armed (or
        the lookup is not wired), returns the entry-time budget unchanged.
        """
        budget = self._budgets.get(epic, 0.0)
        if budget <= 0:
            return 0.0
        lookup = self._trailing_stop_lookup
        if lookup is None or position.quantity <= 0 or position.entry_price <= 0:
            return budget
        try:
            trail_level = lookup(epic)
        except Exception:
            logger.debug("Trailing-stop lookup raised for %s — using full budget", epic)
            return budget
        if trail_level is None:
            return budget
        # Original stop distance reverse-engineered from the entry-time budget:
        #   budget = size × original_stop_distance
        size = position.quantity
        original_stop_distance = budget / size
        original_stop_level = position.entry_price - original_stop_distance
        # Trail only ratchets favourably; clamp in case of a stale lookup.
        effective_stop_level = max(original_stop_level, trail_level)
        remaining_distance = max(0.0, position.entry_price - effective_stop_level)
        return size * remaining_distance
