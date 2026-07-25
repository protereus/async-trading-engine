"""Risk manager: the gatekeeper for all order placement.

Every order must pass evaluate_ig_order() before reaching the broker.
All decision logic is synchronous; event-bus emissions are scheduled
onto the running asyncio loop via create_task so they do not block.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from bot.core.event_bus import (
    EVENT_RISK_ALERT,
)
from bot.core.models import (
    AccountUpdate,
    DrawdownTier,
    IGOrderRequest,
    MarginCircuitState,
    OrderResult,
    OrderSide,
    Position,
    RiskDecision,
    RiskEvent,
    RiskLevel,
    RiskState,
)
from bot.core.time_constants import DAY_MS, HOUR_MS, MONTH_MS, WEEK_MS
from bot.risk import sizing
from bot.risk.budgets import RiskBudgetLedger
from bot.risk.drawdown import DrawdownTracker
from bot.risk.fscs import FSCSCeiling
from bot.risk.loss_windows import LossWindowTracker
from bot.risk.margin import MarginCircuitBreaker
from bot.risk.risk_config import RiskConfig
from bot.risk.sectors import sector_for

if TYPE_CHECKING:
    from bot.core.event_bus import EventBus

logger = logging.getLogger(__name__)


class RiskManager:
    """Evaluates every order request against configurable risk rules.

    Decision methods are fully synchronous so they can be called from
    anywhere in the codebase.  Event-bus integration uses
    ``asyncio.get_running_loop().create_task()`` for async emission.
    """

    def __init__(
        self,
        config: RiskConfig,
        event_bus: EventBus,
        clock_fn: Callable[[], int] | None = None,
        in_blackout_fn: Callable[[], bool] | None = None,
    ) -> None:
        self._config = config
        self._event_bus_inner = event_bus
        self._clock: Callable[[], int] = clock_fn or (lambda: int(time.time() * 1000))

        # --- equity / drawdown (extracted to DrawdownTracker) ---
        # Tier-change handler is wired late (after __init__ body so it can
        # reach self).  Assigning the bound method via a closure works either
        # way; doing it inline keeps the dependency direction clean.
        self._drawdown = DrawdownTracker(
            config, on_tier_change=lambda old, new, dd, eq: self._on_tier_change(old, new, dd, eq)
        )

        # --- loss windows (extracted to LossWindowTracker) ---
        self._loss_windows = LossWindowTracker()

        # --- order rate limiting ---
        self._orders_this_hour: deque[int] = deque()  # ts_ms of approved orders

        # --- positions ---
        self._open_positions: dict[str, Position] = {}
        # Risk-on budget ledger (extracted to RiskBudgetLedger).  Set by
        # main.py at fill time
        # via set_risk_budget(epic, gbp).  ``live_risk_gbp`` shrinks the
        # entry-time budget by any locked-in profit from a trailing stop.
        # ``_risk_budgets`` / ``_trailing_stop_lookup`` remain as
        # backward-compatible property aliases so the rest of this file (and
        # any external snapshotter) can still reach into them.
        self._budget_ledger = RiskBudgetLedger()

        # --- ATR / volatility ---
        self._atr_values: dict[str, deque[float]] = {}
        self._current_atr: dict[str, float] = {}

        # --- halt state ---
        self._trading_halted: bool = False
        self._halt_reason: str = ""
        # Drawdown breaker hardening (2026-06-05): debounce RED on N consecutive
        # reads, and freeze evaluation during the IG rollover/maintenance window
        # (injected predicate — None = no guard, used by tests).  RED now halts
        # NEW ENTRIES only and auto-clears on recovery; it no longer shuts down.
        self._consecutive_red: int = 0
        self._in_blackout_fn = in_blackout_fn

        # --- margin circuit breakers (extracted to MarginCircuitBreaker) ---
        # ``update_margin_state`` delegates to the breaker; the host receives
        # transition log events via the injected callback so the existing
        # risk-event ledger is kept intact.
        self._margin_breaker = MarginCircuitBreaker(
            config,
            event_bus,
            self._clock,
            risk_event_callback=self._log_risk_event,
        )

        # --- FSCS soft ceiling ---
        # Warns once on £100K crossing and clamps sizing equity at £120K so
        # incremental profits past the FSCS line don't grow per-trade £ risk.
        self._fscs = FSCSCeiling(config, risk_event_callback=self._log_risk_event)

        # --- event log ---
        self._risk_events: list[RiskEvent] = []

    # ------------------------------------------------------------------
    # Margin circuit-breaker monitor (delegates to MarginCircuitBreaker)
    # ------------------------------------------------------------------

    @property
    def margin_ratio(self) -> float:
        """Current ``equity / total_margin_required``.  ``inf`` if no positions."""
        return self._margin_breaker.ratio

    @property
    def equity(self) -> float:
        """Current account equity tracked by the drawdown breaker."""
        return self._drawdown.equity

    @property
    def peak_equity(self) -> float:
        """Peak account equity tracked by the drawdown breaker."""
        return self._drawdown.peak_equity

    def _classify_margin_ratio(self, ratio: float) -> MarginCircuitState:
        """Delegates to ``MarginCircuitBreaker.classify``."""
        return self._margin_breaker.classify(ratio)

    def update_margin_state(self, update: AccountUpdate) -> MarginCircuitState:
        """Delegates to ``MarginCircuitBreaker.update``."""
        return self._margin_breaker.update(update)

    # Legacy aliases for callers / tests that read the private attrs directly.
    @property
    def _margin_ratio(self) -> float:
        return self._margin_breaker.ratio

    @property
    def _margin_required(self) -> float:
        return self._margin_breaker.margin_required

    @property
    def _margin_circuit_state(self) -> MarginCircuitState:
        return self._margin_breaker.state

    # Tests swap `_event_bus` post-construction to capture emissions.
    # Keep the legacy attribute name working by routing the setter through
    # all collaborators that hold their own bus reference.
    @property
    def _event_bus(self) -> EventBus:
        return self._event_bus_inner

    @_event_bus.setter
    def _event_bus(self, bus: EventBus) -> None:
        self._event_bus_inner = bus
        # Propagate to collaborators holding their own bus reference.
        if hasattr(self, "_margin_breaker"):
            self._margin_breaker._event_bus = bus  # noqa: SLF001

    def _common_entry_gates(self, orig_size: float, equity: float) -> RiskDecision | None:
        """Account-level pre-trade gates — the broker-agnostic prelude of
        ``evaluate_ig_order``.

        Returns a rejection ``RiskDecision`` if any gate trips — trading
        halted, daily/weekly/monthly loss limits, consecutive-loss pause, or
        max open positions — or ``None`` when all pass.  These read account
        equity plus the risk manager's own loss/position state; ``orig_size``
        is the order's stake used only to echo back into the rejection.
        """
        # 1. Trading halted?
        if self._trading_halted:
            return RiskDecision(
                approved=False,
                original_quantity=orig_size,
                adjusted_quantity=0.0,
                reason=f"Trading halted: {self._halt_reason}",
                risk_level=RiskLevel.HALTED,
            )

        now_ms = self._clock()

        # 2-4. Loss limits (daily / weekly / monthly windows)
        if equity > 0:
            daily = self._window_pnl(now_ms, DAY_MS)
            if daily <= -(self._config.daily_loss_limit_pct * equity):
                return self._reject(orig_size, "Daily loss limit reached", RiskLevel.CRITICAL)

            weekly = self._window_pnl(now_ms, WEEK_MS)
            if weekly <= -(self._config.weekly_loss_limit_pct * equity):
                return self._reject(orig_size, "Weekly loss limit reached", RiskLevel.CRITICAL)

            monthly = self._window_pnl(now_ms, MONTH_MS)
            if monthly <= -(self._config.monthly_loss_limit_pct * equity):
                return self._reject(orig_size, "Monthly loss limit reached", RiskLevel.CRITICAL)

        # 5. Consecutive losses
        if self._loss_windows.consecutive_losses >= self._config.consecutive_loss_pause:
            return self._reject(
                orig_size,
                f"Paused after {self._loss_windows.consecutive_losses} consecutive losses",
                RiskLevel.ELEVATED,
            )

        # 6. Max open positions
        if len(self._open_positions) >= self._config.max_open_positions:
            return self._reject(orig_size, "Maximum open positions reached", RiskLevel.ELEVATED)

        return None

    # ------------------------------------------------------------------
    # IG spread betting interface
    # ------------------------------------------------------------------

    # ``compute_ig_size`` delegates to bot.risk.sizing.  Kept
    # as a static method so all existing call sites (RiskManager.compute_ig_size)
    # continue to work without touching them.
    compute_ig_size = staticmethod(sizing.compute_ig_size)

    def evaluate_ig_order(
        self,
        order: IGOrderRequest,
        margin_used: float,
        equity_gbp: float,
        estimated_margin_gbp: float = 0.0,
    ) -> RiskDecision:
        """Run all IG-specific risk checks for a spread bet order.

        ``adjusted_quantity`` in the returned ``RiskDecision`` is the
        (possibly reduced) stake in £/point after drawdown-tier scaling.
        The caller should create a new ``IGOrderRequest`` with that size.

        Args:
            order:       The proposed IG spread bet order.
            margin_used: Current GBP margin in use across all open positions
                         (from ``IGClient.fetch_balance()["margin"]``).
            equity_gbp:  Current account equity in GBP.
            estimated_margin_gbp: Optional pre-fill estimate of margin this
                         order will consume.  When > 0 we run the tier-aware
                         pre-trade check: would adding this position push
                         ``equity / (margin_used + estimated)`` below the
                         halt threshold?  Caller is responsible for the
                         estimate; use ``bot.risk.ig_margin.estimate_margin_gbp``.
        """
        orig_size = order.size

        # 1-6. Account-level gates shared with the generic path.
        gate = self._common_entry_gates(orig_size, equity_gbp)
        if gate is not None:
            return gate

        now_ms = self._clock()

        # 6b-6c. Total risk-on exposure and Per-sector concentration cap.
        exposure_gate = self._check_exposure_limits(order, orig_size, equity_gbp)
        if exposure_gate is not None:
            return exposure_gate

        # 7. Margin limits (utilisation hard cap, circuit breaker, tier-aware check)
        margin_gate = self._check_margin_limits(
            orig_size, margin_used, equity_gbp, estimated_margin_gbp
        )
        if margin_gate is not None:
            return margin_gate

        # 8. Order rate limit
        self._prune_order_history(now_ms)
        if len(self._orders_this_hour) >= self._config.max_orders_per_hour:
            return self._reject(orig_size, "Order rate limit reached", RiskLevel.ELEVATED)

        # 9. Volatility circuit breaker (uses epic as symbol key)
        volatility_gate = self._check_volatility_circuit_breaker(orig_size, order.epic)
        if volatility_gate is not None:
            return volatility_gate

        # 10. Drawdown tier size reduction
        adjusted_size = orig_size
        tier = self._drawdown.tier
        if tier == DrawdownTier.RED:
            return self._reject(orig_size, "RED drawdown tier: trading halted", RiskLevel.HALTED)
        if tier == DrawdownTier.YELLOW:
            adjusted_size *= 0.5
        elif tier == DrawdownTier.ORANGE:
            adjusted_size *= 0.25

        # 11. Overnight DFB funding warning
        self._warn_if_overnight(order.epic)

        # All checks passed
        self._orders_this_hour.append(now_ms)

        risk_level = RiskLevel.NORMAL if tier == DrawdownTier.NORMAL else RiskLevel.ELEVATED
        if adjusted_size >= orig_size:
            reason = "Approved"
        else:
            reason = (
                f"Stake reduced {orig_size:.2f} → {adjusted_size:.2f} £/pt "
                f"({tier.value} drawdown tier)"
            )
        return RiskDecision(
            approved=True,
            original_quantity=orig_size,
            adjusted_quantity=adjusted_size,
            reason=reason,
            risk_level=risk_level,
        )

    # ------------------------------------------------------------------
    # State update methods
    # ------------------------------------------------------------------

    def on_fill(self, result: OrderResult) -> Position | None:
        """Called when an order fill is confirmed.

        Returns the newly-opened :class:`Position` on a BUY fill (so callers
        don't have to re-read tracked state), or ``None`` on a SELL/no-op.
        """
        if result.side == OrderSide.BUY and result.filled_quantity > 0:
            pos = Position(
                symbol=result.symbol,
                side=result.side,
                entry_price=result.average_price,
                quantity=result.filled_quantity,
                current_price=result.average_price,
                unrealised_pnl=0.0,
                realised_pnl=0.0,
                opened_at=result.timestamp,
                updated_at=result.timestamp,
            )
            self._open_positions[result.symbol] = pos
            logger.info(
                "Risk: position opened for %s qty=%.4f @ %.4f",
                result.symbol,
                result.filled_quantity,
                result.average_price,
            )
            return pos
        if result.side == OrderSide.SELL and result.filled_quantity > 0:
            removed = self._open_positions.pop(result.symbol, None)
            self._budget_ledger.clear(result.symbol)
            if removed is not None:
                logger.info(
                    "Risk: position removed for %s (SELL fill @ %.4f)",
                    result.symbol,
                    result.average_price,
                )
            else:
                logger.warning(
                    "Risk: SELL fill for %s but no tracked position",
                    result.symbol,
                )
        return None

    # -- Open-position accessors -----------------------------------------
    # Collaborators (event wiring, close/reconcile) used to read and mutate
    # ``_open_positions`` directly across the fill/close seam.  These narrow
    # accessors keep that dict private to the risk manager so the open-position
    # set has a single owner.  All keys are IG EPICs (``OrderResult.symbol``).

    def get_position(self, symbol: str) -> Position | None:
        """Return the tracked open position for *symbol*, or ``None``."""
        return self._open_positions.get(symbol)

    def seed_position(self, symbol: str, position: Position) -> None:
        """Insert/replace a tracked position without an order fill.

        Used by reconciliation to re-seed positions that pre-date a restart
        (``on_fill`` never fired for them) so the open-position count, risk
        budget, and loss accounting stay correct.
        """
        self._open_positions[symbol] = position

    def drop_position(self, symbol: str) -> Position | None:
        """Remove and return a tracked position (externally-closed/reconciled)."""
        return self._open_positions.pop(symbol, None)

    def on_position_closed(self, symbol: str, pnl: float) -> None:
        """Called when a position is fully closed."""
        now_ms = self._clock()

        # The loss-window tracker owns the rolling deque, consecutive-loss
        # counter, and PnL accumulators in one place.
        self._loss_windows.record_close(now_ms, pnl)

        # Remove from open positions
        self._open_positions.pop(symbol, None)

        # Check if we just breached a limit
        if self._drawdown.equity > 0:
            if self._loss_windows.consecutive_losses >= self._config.consecutive_loss_pause:
                self._log_risk_event(
                    "consecutive_losses",
                    {
                        "count": self._loss_windows.consecutive_losses,
                        "symbol": symbol,
                        "pnl": pnl,
                    },
                )
            daily = self._window_pnl(now_ms, DAY_MS)
            if daily <= -(self._config.daily_loss_limit_pct * self._drawdown.equity):
                self._log_risk_event(
                    "daily_limit_hit",
                    {"daily_pnl": daily, "limit_pct": self._config.daily_loss_limit_pct},
                )

        logger.info(
            "Risk: position closed %s pnl=%.4f consecutive_losses=%d",
            symbol,
            pnl,
            self._loss_windows.consecutive_losses,
        )

    def update_equity(self, equity: float) -> None:
        """Update current equity, fire FSCS warn-once, then check drawdown
        tiers (drawdown is delegated to ``DrawdownTracker``; tier-change
        callback was wired at init.  The FSCS hook runs first so the
        warn risk event is timestamped before any drawdown event from the
        same update)."""
        # Maintenance guard — during the daily IG rollover/maintenance window
        # (FX funding + metals maintenance) account marks are unreliable, so a
        # bad tick can't pollute the peak or trip the breaker.  New entries are
        # already blocked across this window by is_safe_for_entry, so freezing
        # the (entry-only) breaker here costs nothing.
        if (
            self._config.drawdown_maintenance_guard
            and self._in_blackout_fn is not None
            and self._in_blackout_fn()
        ):
            return
        self._fscs.update_equity(equity)
        self._drawdown.update_equity(equity)
        self._evaluate_drawdown_halt()

    def _evaluate_drawdown_halt(self) -> None:
        """Debounced RED-drawdown entry-halt with auto-resume.

        A single transient equity reading no longer halts: we require
        ``drawdown_red_confirm_count`` consecutive RED reads.  When the drawdown
        recovers out of RED the halt auto-clears.  RED halts *new entries* only
        (open positions keep their own stops) and alerts via Telegram — it no
        longer shuts the bot down (2026-06-05 incident)."""
        tier = self._drawdown.tier
        dd = self._drawdown.drawdown_pct
        if tier == DrawdownTier.RED:
            self._consecutive_red += 1
            if (
                self._consecutive_red >= self._config.drawdown_red_confirm_count
                and not self._trading_halted
            ):
                self._trading_halted = True
                self._halt_reason = (
                    f"RED drawdown {dd:.1%} confirmed on {self._consecutive_red} consecutive reads"
                )
                logger.critical(
                    "TRADING HALTED (new entries): %s (peak=%.2f current=%.2f)",
                    self._halt_reason,
                    self._drawdown.peak_equity,
                    self._drawdown.equity,
                )
                self._schedule_emit(
                    EVENT_RISK_ALERT,
                    RiskEvent(
                        timestamp=self._clock(),
                        event_type="drawdown_red_halt",
                        details={
                            "drawdown_pct": dd,
                            "consecutive_reads": self._consecutive_red,
                            "peak_equity": self._drawdown.peak_equity,
                            "equity": self._drawdown.equity,
                            "action": "new entries halted",
                        },
                    ),
                )
        else:
            # Recovered out of RED — reset the debounce and auto-clear the halt.
            self._consecutive_red = 0
            if self._trading_halted:
                self._trading_halted = False
                self._halt_reason = ""
                logger.warning(
                    "TRADING RESUMED: drawdown recovered to %s (%.2f%%) — entry halt cleared",
                    tier.value,
                    dd * 100,
                )
                self._schedule_emit(
                    EVENT_RISK_ALERT,
                    RiskEvent(
                        timestamp=self._clock(),
                        event_type="drawdown_halt_cleared",
                        details={
                            "drawdown_pct": dd,
                            "tier": tier.value,
                            "action": "new entries resumed",
                        },
                    ),
                )

    def equity_for_sizing(self, equity: float) -> float:
        """FSCS-capped equity for position sizing (IG_LIVE_RISK_REFERENCE.md §7.2).

        Returns ``min(equity, fscs_cap_gbp)``.  Callers in the order path
        should pass this to ``compute_ig_size`` instead of raw equity so
        incremental profits past the FSCS line don't grow per-trade £
        risk.  Loss-limit and margin gates inside ``evaluate_ig_order``
        intentionally still see real equity.
        """
        return self._fscs.cap_for_sizing(equity)

    def update_atr(self, symbol: str, atr: float) -> None:
        """Append a new ATR value for *symbol*; maintains rolling lookback."""
        self._current_atr[symbol] = atr
        if symbol not in self._atr_values:
            self._atr_values[symbol] = deque(maxlen=self._config.volatility_atr_lookback)
        self._atr_values[symbol].append(atr)

    # ------------------------------------------------------------------
    # Risk-on budget tracking (IG)
    # ------------------------------------------------------------------

    # Budget helpers delegate to the extracted RiskBudgetLedger.
    def set_risk_budget(self, epic: str, gbp: float) -> None:
        """Record the £ at risk on this position if its stop hits.  Computed
        by the entry path as ``size × stop_distance`` (£/pt × pts) and used by
        ``evaluate_ig_order`` to enforce ``max_total_risk_pct``."""
        self._budget_ledger.set(epic, gbp)

    def clear_risk_budget(self, epic: str) -> None:
        self._budget_ledger.clear(epic)

    def set_trailing_stop_lookup(self, fn: Callable[[str], float | None] | None) -> None:
        """Inject a trail-stop lookup so the total-risk gate sees live stops."""
        self._budget_ledger.set_trailing_stop_lookup(fn)

    def _live_risk_gbp(self, epic: str, position: Position) -> float:
        """Return live risk-on for a position, shrinking as the trail moves up."""
        return self._budget_ledger.live_risk_gbp(epic, position)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def current_drawdown_tier(self) -> DrawdownTier:
        return self._drawdown.tier

    @property
    def is_trading_halted(self) -> bool:
        return self._trading_halted

    @property
    def current_drawdown_pct(self) -> float:
        return self._drawdown.drawdown_pct

    def get_risk_summary(self) -> dict[str, Any]:
        """Snapshot for the heartbeat / webgui / Telegram alerter.

        ``pnl_24h`` is the rolling 24-hour realised-P&L sum (not calendar-day
        since 00:00 UTC).  A trade closed exactly 24 h ago will drop off the
        trailing edge between two consecutive summaries — that's expected
        behaviour, not a bookkeeping glitch.
        """
        now_ms = self._clock()
        pnl_24h = self._window_pnl(now_ms, DAY_MS)
        total_exposure = sum(p.quantity * p.entry_price for p in self._open_positions.values())
        equity = self._drawdown.equity
        total_exposure_pct = total_exposure / equity if equity > 0 else 0.0
        return {
            "current_drawdown_pct": self.current_drawdown_pct,
            "drawdown_tier": self._drawdown.tier.value,
            "pnl_24h": pnl_24h,
            "consecutive_losses": self._loss_windows.consecutive_losses,
            "open_positions": len(self._open_positions),
            "trading_halted": self._trading_halted,
            "total_exposure_pct": total_exposure_pct,
        }

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def snapshot_state(self) -> RiskState:
        """Prune expired trade results, then snapshot internal state for
        persistence.

        Not a pure read: entries that have aged out of every loss window are
        dropped from ``_trade_results`` first, so the persisted snapshot never
        carries stale rows back into the next process.
        """
        now_ms = self._clock()
        self._prune_trade_results(now_ms)
        return RiskState(
            peak_equity=self._drawdown.peak_equity,
            daily_pnl=self._loss_windows.daily_pnl,
            weekly_pnl=self._loss_windows.weekly_pnl,
            monthly_pnl=self._loss_windows.monthly_pnl,
            consecutive_losses=self._loss_windows.consecutive_losses,
            trade_results=[[ts, pnl] for ts, pnl in self._loss_windows._trade_results],
            trading_halted=self._trading_halted,
            halt_reason=self._halt_reason,
            risk_events=[
                {
                    "timestamp": e.timestamp,
                    "event_type": e.event_type,
                    "details": e.details,
                    "resolved": e.resolved,
                    "resolved_at": e.resolved_at,
                }
                for e in self._risk_events[-100:]
            ],
            atr_values={sym: list(vals) for sym, vals in self._atr_values.items()},
            risk_budgets=self._budget_ledger.snapshot(),
        )

    def load_state(self, state: RiskState) -> None:
        """Restore internal state from a previously persisted RiskState."""
        self._drawdown.set_peak_equity(state.peak_equity)
        self._loss_windows.load_snapshot(
            daily_pnl=state.daily_pnl,
            weekly_pnl=state.weekly_pnl,
            monthly_pnl=state.monthly_pnl,
            consecutive_losses=state.consecutive_losses,
            trade_results=state.trade_results,
        )
        self._trading_halted = state.trading_halted
        self._halt_reason = state.halt_reason
        self._risk_events = [
            RiskEvent(
                timestamp=e["timestamp"],
                event_type=e["event_type"],
                details=e.get("details", {}),
                resolved=e.get("resolved", False),
                resolved_at=e.get("resolved_at"),
            )
            for e in state.risk_events
        ]
        self._atr_values = {
            sym: deque(vals, maxlen=self._config.volatility_atr_lookback)
            for sym, vals in state.atr_values.items()
        }
        self._budget_ledger.load_snapshot(state.risk_budgets)
        # Recompute tier from restored equity (tracker owns the math).
        self._drawdown.recompute_tier()

        logger.info(
            "Risk state restored: peak_equity=%.2f halted=%s consecutive_losses=%d",
            self._drawdown.peak_equity,
            self._trading_halted,
            self._loss_windows.consecutive_losses,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reject(qty: float, reason: str, level: RiskLevel) -> RiskDecision:
        return RiskDecision(
            approved=False,
            original_quantity=qty,
            adjusted_quantity=0.0,
            reason=reason,
            risk_level=level,
        )

    def _window_pnl(self, now_ms: int, window_ms: int) -> float:
        """Delegates to ``LossWindowTracker``."""
        return self._loss_windows.window_pnl(now_ms, window_ms)

    def _prune_trade_results(self, now_ms: int) -> None:
        """Delegates to ``LossWindowTracker``."""
        self._loss_windows.prune(now_ms)

    def _prune_order_history(self, now_ms: int) -> None:
        cutoff = now_ms - HOUR_MS
        while self._orders_this_hour and self._orders_this_hour[0] < cutoff:
            self._orders_this_hour.popleft()

    def _tier_for_drawdown(self, drawdown_pct: float) -> DrawdownTier:
        """Delegates to ``DrawdownTracker``."""
        return self._drawdown.tier_for_drawdown(drawdown_pct)

    def _check_exposure_limits(
        self, order: IGOrderRequest, orig_size: float, equity_gbp: float
    ) -> RiskDecision | None:
        # 6b. Total risk-on exposure: sum of (live) stop-loss budgets across
        #     all open positions, plus the proposed risk for this order.
        #     Live = entry-time budget shrunk by any locked-in profit from
        #     TakeProfitManager's trailing stop.  Drawdown-tier scaling on
        #     ``adjusted_size`` is applied below; we use the pre-scale order
        #     size here so the check is conservative.
        if equity_gbp <= 0 or order.stop_distance is None:
            return None

        existing_risk_gbp = sum(
            self._live_risk_gbp(epic, pos) for epic, pos in self._open_positions.items()
        )
        proposed_risk_gbp = order.size * order.stop_distance
        total_risk_pct = (existing_risk_gbp + proposed_risk_gbp) / equity_gbp
        if total_risk_pct > self._config.max_total_risk_pct:
            return self._reject(
                orig_size,
                f"Total risk-on {total_risk_pct:.2%} > "
                f"{self._config.max_total_risk_pct:.0%} cap "
                f"(open={existing_risk_gbp:.2f} + new={proposed_risk_gbp:.2f} £)",
                RiskLevel.ELEVATED,
            )

        # 6c. Per-sector concentration cap.  Pairwise correlation only
        #     catches pairs; this catches clusters (e.g. three yen-cross
        #     longs each below the 0.65 pairwise threshold but all moving
        #     with the yen).  Static buckets in bot.risk.sectors.
        proposed_sector = sector_for(order.epic)
        existing_sector_risk: dict[str, float] = {}
        for epic, pos in self._open_positions.items():
            sector = sector_for(epic)
            existing_sector_risk[sector] = existing_sector_risk.get(
                sector, 0.0
            ) + self._live_risk_gbp(epic, pos)
        projected_sector_risk = existing_sector_risk.get(proposed_sector, 0.0) + proposed_risk_gbp
        sector_pct = projected_sector_risk / equity_gbp
        if sector_pct > self._config.max_sector_risk_pct:
            return self._reject(
                orig_size,
                f"Sector '{proposed_sector}' risk-on {sector_pct:.2%} > "
                f"{self._config.max_sector_risk_pct:.1%} cap "
                f"(open_in_sector={existing_sector_risk.get(proposed_sector, 0.0):.2f} "
                f"+ new={proposed_risk_gbp:.2f} £)",
                RiskLevel.ELEVATED,
            )

        return None

    def _check_margin_limits(
        self, orig_size: float, margin_used: float, equity_gbp: float, estimated_margin_gbp: float
    ) -> RiskDecision | None:
        # 7. Margin utilisation hard cap (pre-trade snapshot).  This is the
        # legacy gate using fetch_balance() — kept as a belt-and-braces check
        # for the moments between LS ACCOUNT pushes.
        if equity_gbp > 0 and margin_used / equity_gbp >= self._config.max_margin_pct:
            return self._reject(
                orig_size,
                f"Margin utilisation {margin_used / equity_gbp:.1%} >= "
                f"{self._config.max_margin_pct:.0%} hard cap",
                RiskLevel.CRITICAL,
            )

        # 7b. Margin circuit breaker — anything above NORMAL refuses new
        # entries.  Defensive-close / flatten actions are dispatched via
        # EVENT_MARGIN_BREAKER from update_margin_state(); this gate just
        # ensures we don't ADD exposure while a breaker is active.
        if self._margin_circuit_state != MarginCircuitState.NORMAL:
            return self._reject(
                orig_size,
                f"Margin circuit breaker {self._margin_circuit_state.value} active "
                f"(equity/margin={self._margin_ratio:.2f}, halt≤"
                f"{self._config.margin_halt_ratio:.2f})",
                RiskLevel.CRITICAL,
            )

        # 7c. Tier-aware pre-trade margin check.  Caller passes the
        # per-asset-class margin estimate from ``ig_margin.estimate_margin_gbp``.
        # We project the post-fill ratio and refuse the order if it would
        # immediately put the account into HALT_ENTRIES territory.  This is
        # the *prospective* counterpart to the reactive ratio check.
        if estimated_margin_gbp > 0 and equity_gbp > 0:
            projected_margin = margin_used + estimated_margin_gbp
            if projected_margin > 0:
                projected_ratio = equity_gbp / projected_margin
                if projected_ratio < self._config.margin_halt_ratio:
                    return self._reject(
                        orig_size,
                        f"Pre-trade margin check: post-fill ratio "
                        f"{projected_ratio:.2f} < halt {self._config.margin_halt_ratio:.2f} "
                        f"(margin_used={margin_used:.2f}, "
                        f"estimated_new={estimated_margin_gbp:.2f}, "
                        f"equity={equity_gbp:.2f})",
                        RiskLevel.CRITICAL,
                    )

        return None

    def _check_volatility_circuit_breaker(self, orig_size: float, epic: str) -> RiskDecision | None:
        if epic in self._atr_values:
            current_atr = self._current_atr.get(epic, 0.0)
            atr_buf = self._atr_values[epic]
            if len(atr_buf) >= 2 and current_atr > 0:
                avg_atr = statistics.mean(atr_buf)
                if avg_atr > 0 and current_atr > avg_atr * self._config.volatility_multiplier:
                    return self._reject(
                        orig_size,
                        f"Volatility circuit breaker: ATR {current_atr:.4f} > "
                        f"{self._config.volatility_multiplier}x avg {avg_atr:.4f}",
                        RiskLevel.CRITICAL,
                    )
        return None

    def _warn_if_overnight(self, epic: str) -> None:
        """Log a coarse warning if opening a DFB position close to the daily
        rollover.  IG UK DFB rollover is at 22:00 UTC.  The per-asset-class
        GBP estimate lives in ``main._process_candle_ig_topk`` where
        the IG-level price is already in scope; this method stays
        decoupled from price data so it can run from any caller."""
        now_utc = datetime.now(UTC)
        if now_utc.hour >= self._config.ig_overnight_warning_hour_utc:
            logger.warning(
                "DFB overnight funding risk: opening %s at %02d:%02d UTC — "
                "position may incur overnight funding charge if held past 22:00 UTC",
                epic,
                now_utc.hour,
                now_utc.minute,
            )

    def _on_tier_change(
        self,
        old_tier: DrawdownTier,
        new_tier: DrawdownTier,
        drawdown: float,
        equity: float,
    ) -> None:
        level = "INFO"
        if new_tier in (DrawdownTier.ORANGE, DrawdownTier.RED):
            level = "WARNING"

        log = getattr(logger, level.lower())
        log(
            "Drawdown tier: %s -> %s  drawdown=%.2f%%  equity=%.2f",
            old_tier.value,
            new_tier.value,
            drawdown * 100,
            equity,
        )

        event = RiskEvent(
            timestamp=self._clock(),
            event_type=f"drawdown_{new_tier.value}",
            details={
                "old_tier": old_tier.value,
                "new_tier": new_tier.value,
                "drawdown_pct": drawdown,
                "equity": equity,
                "peak_equity": self._drawdown.peak_equity,
            },
        )
        self._log_risk_event_obj(event)

        # Every tier transition (including entering RED) raises a Telegram risk
        # alert.  The RED *entry-halt* itself is owned by the debounced
        # ``_evaluate_drawdown_halt`` — it no longer shuts the bot down here.
        self._schedule_emit(EVENT_RISK_ALERT, event)

    def _log_risk_event(self, event_type: str, details: dict[str, Any]) -> None:
        event = RiskEvent(
            timestamp=self._clock(),
            event_type=event_type,
            details=details,
        )
        self._log_risk_event_obj(event)
        self._schedule_emit(EVENT_RISK_ALERT, event)

    def _log_risk_event_obj(self, event: RiskEvent) -> None:
        self._risk_events.append(event)
        if len(self._risk_events) > 200:
            self._risk_events = self._risk_events[-100:]

    def _schedule_emit(self, event_type: str, data: Any) -> None:
        """Schedule an async event emission from synchronous context."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._event_bus.emit(event_type, data))
        except RuntimeError:
            # No running event loop (e.g., in synchronous unit tests)
            logger.debug("No running loop -- skipping emit of '%s'", event_type)
