"""TopK entry-order execution: sizing, margin gate, spread gate, order placement.

Extracted from ``RerankRunner`` as a dedicated collaborator, mirroring the
``IGCloseManager`` / ``IGSession`` / ``IGHttp`` extraction pattern already
used elsewhere in ``bot.execution``.  ``RerankRunner.check_exit_and_entry``
calls ``ctx.entry_executor.maybe_enter`` once a symbol clears the trading-
hours / entry-lock gate; everything below that is this collaborator's.
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Any

from bot.core.event_bus import EVENT_ORDER_FILLED
from bot.core.models import ExchangeError, IGOrderRequest, MarketClosedError
from bot.execution.ig_convert import IG_MIN_STOP_PCT
from bot.execution.ig_quote_scale import ig_pip_value, ig_quote_scale
from bot.risk.funding import log_overnight_funding_estimate
from bot.risk.ig_margin import estimate_margin_gbp as ig_margin_estimate
from bot.risk.ig_margin import estimate_slippage_pts as ig_slippage_pts
from bot.risk.risk_manager import RiskManager
from bot.trading_hours import is_safe_for_entry

if TYPE_CHECKING:
    from bot.core.bot_context import BotContext

logger = logging.getLogger(__name__)

_SIZE_GRID_GBP_PER_PT = 0.1


def _snap_size_up_to_grid(size: float, min_deal_size: float | None) -> float:
    """Round *size* UP to the next 0.1 £/pt, clamped to ``>= min_deal_size``.

    Used to recover from an IG ``SIZE_INCREMENT`` reject: the risk-sized 2-dp
    stake (e.g. 0.28) gets snapped to the next grid point (0.3) and retried.
    The bump is bounded to ``< 0.1`` £/pt, so the order's already-approved
    risk / margin gate stays valid.  Returns a value rounded to 1 dp; an input
    already on the grid is returned unchanged (caller then skips the retry).
    """
    # Subtract a tiny epsilon so a value already exactly on the grid isn't
    # bumped to the next point by floating-point noise in the division.
    snapped = round(math.ceil(size / _SIZE_GRID_GBP_PER_PT - 1e-9) * _SIZE_GRID_GBP_PER_PT, 1)
    if min_deal_size is not None and snapped < min_deal_size:
        snapped = round(
            math.ceil(min_deal_size / _SIZE_GRID_GBP_PER_PT - 1e-9) * _SIZE_GRID_GBP_PER_PT, 1
        )
    return snapped


class EntryExecutor:
    """TopK entry-order execution collaborator on ``TradingBot``."""

    def __init__(self, ctx: BotContext) -> None:
        self._ctx = ctx

    async def maybe_enter(self, symbol: str, current_price: float) -> None:
        """Entry path: selection/scan/hours gate, then the entry lock + re-check
        + ``_attempt_topk_entry``."""
        ctx = self._ctx
        # Entry: scan has run and this symbol is currently selected
        if not ctx.topk_scanned or symbol not in ctx.topk_selected:
            return

        # Trading hours guard — skip entry if the market is currently closed
        # or inside the daily funding/maintenance buffer (22:00 UTC funding tick
        # for forex; 22:00–23:00 UTC maintenance for metals, widened ±5 min).

        if not is_safe_for_entry(symbol):
            logger.debug("TopK entry skipped: %s market closed or in funding window", symbol)
            return

        # Serialise entries: the risk gate (max_open_positions / max_total_risk /
        # sector caps) reads shared position state that isn't registered until
        # after the order's network round-trips. Without this lock, concurrent
        # hour-boundary candle handlers each pass the gate on a stale snapshot
        # and collectively overshoot the caps. Entries are low-frequency, so full
        # serialisation costs nothing.
        async with ctx.entry_lock:
            # Re-check under the lock: a peer entry that filled while we waited
            # for the lock may have opened this symbol (double-entry guard) or
            # filled the last slot (evaluate_ig_order re-reads fresh state in the
            # helper, so the cap is enforced even then).
            if ctx.state.positions.get(symbol) is not None:
                logger.debug("TopK entry skipped: %s already open (raced a peer entry)", symbol)
                return
            if symbol not in ctx.topk_selected:
                return
            await self._attempt_topk_entry(symbol, current_price)

    async def _place_with_size_retry(
        self, symbol: str, order: IGOrderRequest, min_deal_size: float | None
    ) -> tuple[Any, float]:
        """``place_order`` + ``confirm_order``; on SIZE_INCREMENT snap the stake
        up one 0.1 £/pt grid point and retry once (bump < 0.1 £/pt so the already-
        approved risk/margin gate still holds).  Returns ``(confirmed, filled_size)``
        — the filled size is the snapped stake on retry, else the requested size.

        SIZE_INCREMENT means the stake isn't a valid multiple of the market's
        (API-hidden) deal-size increment.  A size already on the grid
        (snapped == requested) or a non-size reject re-raises to the caller — the
        slot is left unfilled and the next rerank re-selects, as before."""
        ctx = self._ctx
        assert ctx.ig_client is not None, (
            "ig_client must be set by init_ig() before the rerank loop runs"
        )
        try:
            pending = await ctx.ig_client.place_order(order)
            confirmed = await ctx.ig_client.confirm_order(pending.order_id)
            return confirmed, order.size
        except ExchangeError as exc:
            if "SIZE_INCREMENT" not in str(exc):
                raise
            snapped = _snap_size_up_to_grid(order.size, min_deal_size)
            if snapped <= order.size:
                raise
            logger.warning(
                "TopK IG entry SIZE_INCREMENT for %s at size=%.2f £/pt — "
                "retrying once at snapped size=%.2f £/pt",
                symbol,
                order.size,
                snapped,
            )
            retry = IGOrderRequest(
                epic=order.epic,
                direction=order.direction,
                size=snapped,
                stop_distance=order.stop_distance,
            )
            pending = await ctx.ig_client.place_order(retry)
            confirmed = await ctx.ig_client.confirm_order(pending.order_id)
            return confirmed, snapped

    async def _attempt_topk_entry(self, symbol: str, current_price: float) -> None:
        """Place a single TopK entry for *symbol*.

        The caller holds ``ctx.entry_lock`` and has re-checked selection +
        position state under it, so the risk gate below and the position
        registration that follows are serialised — concurrent entries can't
        overshoot the caps on a stale ``_open_positions`` snapshot.
        """
        ctx = self._ctx
        logger.info(
            "TopK entry check: %s scanned=%s selected=%s",
            symbol,
            ctx.topk_scanned,
            ctx.topk_selected,
        )

        equity_gbp, margin_used = await self._sync_balance_state(symbol)
        if equity_gbp is None or margin_used is None:
            return

        topk_signal = next((s for s in ctx.topk_signals if s.symbol == symbol), None)
        stop_pct = topk_signal.stop_pct if topk_signal is not None else ctx.config.topk_min_stop_pct
        if stop_pct <= 0:
            logger.info("TopK stop_pct=0 for %s — skipping", symbol)
            return

        # Translate candle symbol → IG EPIC for stop enforcement and order placement
        ig_epic = ctx.epic_for(symbol)

        # Sizing and risk checks
        order = self._prepare_entry_order(symbol, ig_epic, current_price, equity_gbp, stop_pct)
        if order is None:
            return

        # Pre-trade margin gate
        order, final_size = self._check_margin_gate(
            symbol, ig_epic, current_price, equity_gbp, margin_used, order
        )
        if order is None or final_size is None:
            return

        # Spread anomaly check
        if not self._check_spread_anomaly(symbol):
            return

        # Execute trade
        await self._execute_entry_order(symbol, ig_epic, order, final_size, topk_signal)

    async def _sync_balance_state(self, symbol: str) -> tuple[float | None, float | None]:
        ctx = self._ctx
        try:
            balance = await ctx.refresh_balance()
        except Exception:
            logger.warning("Could not fetch IG balance", exc_info=True)
            return None, None

        equity_gbp = balance["equity"]
        margin_used = balance["margin"]
        logger.info("TopK balance: %s equity=%.2f margin=%.2f", symbol, equity_gbp, margin_used)
        return equity_gbp, margin_used

    def _prepare_entry_order(
        self, symbol: str, ig_epic: str, current_price: float, equity_gbp: float, stop_pct: float
    ) -> IGOrderRequest | None:
        ctx = self._ctx
        effective_stop_pct = max(stop_pct, IG_MIN_STOP_PCT.get(ig_epic, 0.0))

        pip_value = ig_pip_value(symbol)
        # Bake worst-case fill slippage into the size denominator so a
        # real stop hit doesn't blow past the £-risked budget.  Per-asset-class
        # estimate from bot.risk.ig_margin (1 bp forex major → 10 bp commodity).
        slip_pts = ig_slippage_pts(symbol, current_price, pip_value)
        # Clamp the sizing-equity at the FSCS line so incremental
        # profits past £120K don't grow per-trade £ risk.  Loss-limit and
        # margin gates in evaluate_ig_order still see real equity below.
        sizing_equity = ctx.risk_manager.equity_for_sizing(equity_gbp)

        size = RiskManager.compute_ig_size(
            sizing_equity,
            ctx.risk_config.risk_per_trade_pct,
            current_price,
            effective_stop_pct,
            pip_value,
            slippage_buffer_pts=slip_pts,
        )
        if size <= 0:
            logger.info(
                "TopK size=0 for %s equity=%.2f price=%.4f stop=%.4f — skipping",
                symbol,
                equity_gbp,
                current_price,
                effective_stop_pct,
            )
            return None

        stop_distance = round(current_price * effective_stop_pct / pip_value, 2)
        return IGOrderRequest(epic=ig_epic, direction="BUY", size=size, stop_distance=stop_distance)

    def _check_margin_gate(
        self,
        symbol: str,
        ig_epic: str,
        current_price: float,
        equity_gbp: float,
        margin_used: float,
        proposed: IGOrderRequest,
    ) -> tuple[IGOrderRequest | None, float | None]:
        ctx = self._ctx
        # B6 — pre-trade tier-aware margin estimate.  ``estimate_margin_gbp``
        # uses IG's retail rate per asset class (forex major 3.33 %, minor 5 %,
        # indices/gold 5 %, other commodities 10 %).  The estimate is in
        # IG-level units (post-scale price) and is intentionally conservative;
        # the risk manager projects post-fill ``equity / margin`` against the
        # halt ratio and refuses if it would immediately trip the breaker.
        ig_level = current_price * ig_quote_scale(symbol)
        estimated_margin = ig_margin_estimate(
            symbol=symbol, size_per_pt=proposed.size, ig_level=ig_level
        )
        decision = ctx.risk_manager.evaluate_ig_order(
            proposed, margin_used, equity_gbp, estimated_margin_gbp=estimated_margin
        )
        if not decision.approved:
            logger.info("TopK IG order rejected: %s — %s", symbol, decision.reason)
            return None, None

        final_size = decision.adjusted_quantity
        order = IGOrderRequest(
            epic=ig_epic, direction="BUY", size=final_size, stop_distance=proposed.stop_distance
        )
        # Log expected overnight funding for this position so the
        # operator can see Wed FX ×3 / Fri equity ×3 multipliers concretely
        # rather than the old "you opened after 18:00 UTC" hour warning.
        log_overnight_funding_estimate(symbol, final_size, ig_level)
        return order, final_size

    def _check_spread_anomaly(self, symbol: str) -> bool:
        ctx = self._ctx
        # Refuse new entries when the live bid-ask spread is anomalously
        # wide (> mean + 2σ of the 30-day rolling window).  Quiet no-op until
        # the window has enough history primed in.  ctx.spread_monitor is
        # always set by Lifecycle.init_ig() regardless of candle_exchange.
        if ctx.spread_monitor is not None and ctx.spread_monitor.is_anomalous(symbol):
            current = ctx.spread_monitor.latest_spread(symbol)
            stats = ctx.spread_monitor.stats(symbol)
            if stats is not None:
                mean, stdev = stats
                logger.warning(
                    "TopK IG entry blocked by spread monitor: %s current=%.2fpt vs "
                    "mean=%.2f ±%.2f (%.1fσ above mean)",
                    symbol,
                    current or 0.0,
                    mean,
                    stdev,
                    ((current or 0.0) - mean) / stdev if stdev > 0 else 0.0,
                )
            return False
        return True

    async def _execute_entry_order(
        self,
        symbol: str,
        ig_epic: str,
        order: IGOrderRequest,
        final_size: float,
        topk_signal: Any | None,
    ) -> None:
        ctx = self._ctx
        assert ctx.ig_client is not None, (
            "ig_client must be set by init_ig() before the rerank loop runs"
        )
        assert ctx.topk_strategy is not None, "topk_rerank_loop only runs when topk_strategy is set"
        try:
            # B4 — pre-trade market-status gate (docs/ig_live_readiness_plan.md).
            # Refuse to send the order if IG has the EPIC in any state other than
            # TRADEABLE (CLOSED, MARKET_CLOSED_WITH_EDITS, ON_AUCTION, SUSPENDED…).
            # The demo environment hides these transitions; in live they routinely
            # appear around macro events and end-of-session.
            try:
                min_deal_size = await ctx.ig_client.require_tradeable(ig_epic)
            except MarketClosedError as exc:
                logger.warning("TopK IG entry blocked by market-status gate: %s — %s", ig_epic, exc)
                return

            # Skip a risk-sized stake below IG's minimum deal size rather than eat a
            # MINIMUM_ORDER_SIZE_ERROR reject (higher-priced US shares with a wide
            # vol-stop can size under the ~0.24 £/pt floor).  The slot is left
            # unfilled; the next rerank re-selects.
            if min_deal_size is not None and final_size < min_deal_size:
                logger.info(
                    "TopK IG entry skipped: %s size %.3f £/pt < IG min deal size %.3f "
                    "(stop too wide for the 1%% risk budget at this price)",
                    symbol,
                    final_size,
                    min_deal_size,
                )
                return

            confirmed, final_size = await self._place_with_size_retry(symbol, order, min_deal_size)
            ctx.ig_deal_ids[symbol] = confirmed.order_id

            # Record the risk-on budget (£ lost if stop hits) so the
            # total-risk gate can sum it across open positions at the next
            # entry.  IG fills the requested stake exactly, so final_size is
            # authoritative.
            assert order.stop_distance is not None  # set in _prepare_entry_order
            ctx.risk_manager.set_risk_budget(ig_epic, final_size * order.stop_distance)
            await ctx.event_bus.emit(EVENT_ORDER_FILLED, confirmed)
            fill_price = confirmed.average_price

            logger.info(
                "TopK IG BUY: %s (epic=%s) size=%.2f £/pt @ %.4f  dealId=%s",
                symbol,
                ig_epic,
                final_size,
                fill_price,
                confirmed.order_id,
            )

            if ctx.tp_manager is not None and topk_signal is not None:
                _path_sig = ctx.topk_strategy.path_signal_for(symbol)
                ctx.tp_manager.register_position(
                    symbol,
                    fill_price,
                    topk_signal,
                    int(time.time() * 1000),
                    path_signal=_path_sig,
                )
        except Exception:
            logger.exception("TopK IG entry failed for %s (epic=%s)", symbol, ig_epic)
            try:
                await ctx.alerter.send_error(f"TopK entry failed for {symbol} (epic={ig_epic})")
            except Exception:
                logger.exception("Entry-failure alert failed for %s — continuing", symbol)
