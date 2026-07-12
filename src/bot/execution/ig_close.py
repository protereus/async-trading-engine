"""IG position close + reconciliation logic.

Extracted from ``main.py``.  Owns the four methods that drive every
position exit on the IG side:

* ``close_ig_position`` — single IG REST close with three-way result
  (success, hard-fail → reconcile, deferred → IG inside funding window).
* ``close_position`` — unified TP / rerank close path that wraps
  ``close_ig_position`` with TP-manager deregistration and Telegram alert.
* ``reconcile_positions_with_ig`` — bidirectional sync of ``_state.positions``
  with the live IG account: prunes stale locals, alerts on orphans, re-seeds
  the risk-manager view and budget ledger.
* ``pick_worst_performer`` — defensive-close helper that picks the symbol
  with the worst unrealised P&L%.

Operates on the shared ``BotContext`` — the explicit collaborator seam
(same pattern as ``HealthMonitor`` / ``RerankRunner``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from bot.core.event_bus import EVENT_ORDER_FILLED
from bot.core.models import MarketClosedError
from bot.execution.ig_convert import parse_ig_pnl, safe_float
from bot.execution.ig_quote_scale import ig_quote_scale

if TYPE_CHECKING:
    from bot.core.bot_context import BotContext

logger = logging.getLogger(__name__)


class IGCloseManager:
    """Position close + reconciliation collaborator for TradingBot."""

    def __init__(self, ctx: BotContext) -> None:
        self._ctx = ctx

    def pick_worst_performer(self, symbols: list[str]) -> str | None:
        """Return the symbol with the most negative unrealised P&L%, or None
        if no current prices are available."""
        ctx = self._ctx
        worst_pct: float | None = None
        worst_symbol: str | None = None
        for symbol in symbols:
            pos = ctx.state.positions.get(symbol)
            if pos is None:
                continue
            latest = ctx.store.get_latest_candle(symbol)
            if latest is None or pos.entry_price <= 0:
                continue
            ig_current = latest.close * ig_quote_scale(symbol)
            pnl_pct = (ig_current - pos.entry_price) / pos.entry_price
            if worst_pct is None or pnl_pct < worst_pct:
                worst_pct = pnl_pct
                worst_symbol = symbol
        return worst_symbol

    async def reconcile_positions_with_ig(self, *, verbose: bool = False) -> None:
        """Sync ``_state.positions`` with the live IG account.

        Fetches ``/positions`` and rebuilds ``_ig_deal_ids`` from the response.
        Reconciliation is bidirectional:

        * **Local → IG**: any local position whose EPIC is not on IG is treated
          as closed externally (server-side stop, contract rollover, manual web
          UI close) and purged from ``_state.positions``, ``_ig_deal_ids``,
          ``_risk_manager._open_positions``, and the TP manager.
        * **IG → Local**: any IG position whose EPIC is not in
          ``_state.positions`` is an *orphan* — most commonly created when
          ``confirm_order`` 404s on ``error.confirms.deal-not-found`` (IG hasn't
          materialised the confirm yet) but the order actually fills on IG.
          The bot does NOT auto-close (could conflict with server-side stops)
          and does NOT auto-adopt (no Kronos signal / TP target to manage it).
          Just logs + Telegram-alerts so the operator can close manually via
          the IG portal or ``scripts/close_stale_positions.py``.

        Called at startup (``verbose=True``), once per rerank (``verbose=False``),
        and after every ``close_ig_position`` failure (to disambiguate flaky
        endpoint vs ghost dealId).  Hourly cadence is well within IG's
        non-trading rate limit (30 req/min/app).
        """
        ctx = self._ctx
        try:
            raw_positions = await ctx.ig_client.fetch_positions_raw()
            live_symbols: set[str] = set()
            # EPICs that still have an unclaimed local _state.positions entry.
            # We pop on first match so a second IG position for the same EPIC
            # is correctly flagged as an orphan rather than silently
            # overwriting _ig_deal_ids (the May 14 duplicate-USD/NOK bug).
            unclaimed_epics = {ctx.epic_for(s) for s in ctx.state.positions}
            for entry in raw_positions:
                epic = entry["market"]["epic"]
                deal_id = entry["position"]["dealId"]
                candle_sym = ctx.candle_for(epic)
                if candle_sym in ctx.state.positions and epic in unclaimed_epics:
                    ctx.ig_deal_ids[candle_sym] = deal_id
                    live_symbols.add(candle_sym)
                    unclaimed_epics.discard(epic)
                    # Re-seed the risk-manager view: _open_positions is not
                    # persisted across restarts and on_fill never fires for
                    # pre-restart positions. Without this, the first SELL
                    # close after a restart logs "no tracked position" and
                    # its P&L doesn't roll into consecutive_losses /
                    # daily_pnl, and max_open_positions briefly under-counts.
                    ctx.risk_manager.seed_position(epic, ctx.state.positions[candle_sym])
                    # Re-seed the risk-on budget from the live IG payload.
                    # IG's GET /positions response normalises stops to
                    # ``stopLevel`` (absolute price) and leaves
                    # ``stopDistance`` null even when the order was placed
                    # with stop_distance, so derive distance from level −
                    # stopLevel (BUY-only — the bot does not open shorts).
                    # size is £/pt.  If size or stopLevel is missing the
                    # position contributes 0 to the cap (safest default
                    # for stop-less or malformed payloads).
                    pos_payload = entry.get("position", {})
                    ig_size = safe_float(pos_payload.get("size"))
                    ig_level = safe_float(pos_payload.get("level"))
                    ig_stop_level = safe_float(pos_payload.get("stopLevel"))
                    if ig_size > 0 and ig_level > 0 and ig_stop_level > 0:
                        stop_dist_pts = ig_level - ig_stop_level
                        if stop_dist_pts > 0:
                            ctx.risk_manager.set_risk_budget(epic, ig_size * stop_dist_pts)
                    if verbose:
                        logger.info("Reconciled deal ID for %s: %s", candle_sym, deal_id)
                elif deal_id not in ctx.alerted_orphan_deals:
                    pos = entry["position"]
                    logger.warning(
                        "Orphan IG position: dealId=%s epic=%s dir=%s size=%s level=%s",
                        deal_id,
                        epic,
                        pos.get("direction"),
                        pos.get("size"),
                        pos.get("level"),
                    )
                    try:
                        await ctx.alerter.send_error(
                            f"Untracked IG position: {epic} {pos.get('direction')} "
                            f"size={pos.get('size')} level={pos.get('level')} "
                            f"dealId={deal_id} — close via the IG portal or "
                            "scripts/close_stale_positions.py if unintended"
                        )
                    except Exception:
                        logger.exception("Orphan alert failed for %s — continuing", deal_id)
                    ctx.alerted_orphan_deals.add(deal_id)

            stale = [s for s in ctx.state.positions if s not in live_symbols]
            for sym in stale:
                logger.warning("Purging stale position %s (not found on IG)", sym)
                pos = ctx.state.positions[sym]
                # Try to fetch the real IG close fill from /history/transactions
                # so the alert reports the actual exit level + P&L instead of
                # an estimate from the local Twelve Data candle.
                try:
                    txn = await ctx.ig_client.fetch_closed_transaction(
                        opened_at_ms=pos.opened_at,
                    )
                except Exception:
                    logger.exception("fetch_closed_transaction failed for %s", sym)
                    txn = None
                if not isinstance(txn, dict):
                    txn = None
                scale = ig_quote_scale(sym)
                entry_display = pos.entry_price / scale
                if txn is not None:
                    close_level = safe_float(txn.get("closeLevel"))
                    pnl_raw = parse_ig_pnl(txn.get("profitAndLoss", ""))
                    if close_level > 0:
                        ig_close_level = close_level
                        close_display = close_level / scale
                        pnl_pct = (ig_close_level - pos.entry_price) / pos.entry_price * 100
                        reasoning = (
                            f"IG closed externally @ {close_display:.4f} "
                            f"(real P&L {pnl_raw:+.2f} GBP) — "
                            "server-side stop, contract rollover, or manual close"
                        )
                    else:
                        # Transaction matched but no closeLevel — fall back to candle
                        txn = None
                if txn is None:
                    latest = ctx.store.get_latest_candle(sym)
                    if latest is None:
                        # No price reference at all — purge silently with a stub alert
                        close_display = entry_display
                        pnl_pct = 0.0
                        reasoning = (
                            "Position no longer present on IG — no fill record found "
                            "(transactions API empty or window outside lookback)"
                        )
                    else:
                        close_display = latest.close
                        ig_current = latest.close * scale
                        pnl_pct = (ig_current - pos.entry_price) / pos.entry_price * 100
                        reasoning = (
                            "Position no longer present on IG — estimated P&L "
                            "from local candle (real fill unavailable from /history)"
                        )
                try:
                    await ctx.alerter.alert_take_profit(
                        sym,
                        "reconciled_external_close",
                        entry_display,
                        close_display,
                        pnl_pct,
                        reasoning,
                    )
                except Exception:
                    logger.exception("External-close alert failed for %s — continuing purge", sym)
                del ctx.state.positions[sym]
                # _risk_manager._open_positions is keyed by IG EPIC (set in
                # IGClient via OrderResult.symbol=order.epic), not by the
                # candle symbol used in _state.positions.
                ctx.risk_manager.drop_position(ctx.epic_for(sym))
                ctx.risk_manager.clear_risk_budget(ctx.epic_for(sym))
                ctx.ig_deal_ids.pop(sym, None)
                if ctx.tp_manager is not None:
                    ctx.tp_manager.deregister_position(sym)
        except Exception:
            logger.exception("IG position reconciliation failed — stop-losses may not close")

    async def close_ig_position(
        self, symbol: str, current_price: float, current_position: Any
    ) -> float | bool | None:
        """Close an open IG spread bet position.

        Returns:
          * ``float`` — close succeeded; value is the realised IG-level fill
            price (``result.average_price``).  Callers use it as the
            authoritative exit price rather than the last candle close.
          * ``False`` — close failed (network error, ghost dealId, IG reject).
            Callers should reconcile against ``/positions`` to disambiguate
            "still live on IG" (alert + retry) from "ghost dealId" (purge).
          * ``None``  — close *deferred* because IG returned
            ``MARKET_CLOSED_WITH_EDITS`` (or similar).  The position is still
            alive on IG; do NOT reconcile (would mis-purge) and do NOT send
            an error alert.  TP state and deal IDs are left intact so the
            next candle/rerank can retry once the window ends.

        ``symbol`` is the candle/strategy symbol (e.g. AVAX/USDT or the IG EPIC
        when candle_exchange='ig').  The IG EPIC for the REST call is looked up
        from ``_candle_epic_map``.
        """
        ctx = self._ctx
        epic = ctx.epic_for(symbol)
        deal_id = ctx.ig_deal_ids.get(symbol)
        if not deal_id:
            logger.warning("No deal_id tracked for %s — cannot close position", symbol)
            return False
        try:
            result = await ctx.ig_client.close_position(
                deal_id=deal_id,
                epic=epic,
                direction="SELL",
                size=current_position.quantity,
            )
            ctx.ig_deal_ids.pop(symbol, None)
            await ctx.event_bus.emit(EVENT_ORDER_FILLED, result)
            pnl = result.average_price - current_position.entry_price
            logger.info(
                "IG position closed: %s (epic=%s) @ %.4f  pnl=%.4f £/pt",
                symbol,
                epic,
                result.average_price,
                pnl,
            )
            return float(result.average_price)
        except MarketClosedError as exc:
            logger.warning(
                "IG close deferred — market closed for edits: %s epic=%s (%s)",
                symbol,
                epic,
                exc,
            )
            return None
        except Exception:
            logger.exception(
                "IG close_position failed: %s epic=%s dealId=%s", symbol, epic, deal_id
            )
            return False

    async def close_position(self, symbol: str, reason: str, reasoning: str = "") -> None:
        """Unified close path for all TP and rerank exits.

        Looks up current position and price, calls the IG REST closer,
        deregisters from TakeProfitManager, and fires a Telegram alert.
        Stop-loss closes bypass this (they call close_ig_position directly).

        If IG rejects the close, TP state is preserved so the next candle/rerank
        can retry, and an error alert is sent in place of the take-profit alert.
        """
        ctx = self._ctx
        pos = ctx.state.positions.get(symbol)
        if pos is None:
            return
        latest = ctx.store.get_latest_candle(symbol)
        current_price = latest.close if latest else pos.entry_price
        logger.info("Closing %s: reason=%s %s", symbol, reason, reasoning)
        fill_ig = await self.close_ig_position(symbol, current_price, pos)
        if fill_ig is None:
            # Deferred — IG is inside its daily funding/maintenance window.
            # Position is still live on IG; do NOT reconcile (would mis-purge)
            # and do NOT alert.  Next candle/rerank will retry.
            return
        if fill_ig is False:
            # Probe IG immediately: if the dealId is a ghost (position already
            # gone server-side, e.g. contract rollover), the reconciler purges
            # the local entry and fires its own external-close alert. Otherwise
            # the position is still live on IG and we surface the close error.
            await self.reconcile_positions_with_ig()
            if symbol not in ctx.state.positions:
                return
            await ctx.alerter.send_error(
                f"Close failed for {symbol} ({reason}) — position still open on IG, "
                f"TP tracking preserved for retry"
            )
            return
        if ctx.tp_manager is not None:
            ctx.tp_manager.deregister_position(symbol)
        # Report the *realised* IG fill (``close_ig_position`` returns
        # ``result.average_price`` in IG-level units), not the last candle
        # close.  The candle is a feed mark and can disagree with the executed
        # fill by the full spread — the SELL fill alert already shows the real
        # price, so sourcing the exit summary + P&L from the same fill keeps the
        # two alerts consistent (previously the summary used the stale candle
        # close and could even flip the P&L sign).
        scale = ig_quote_scale(symbol)
        pnl_pct = (fill_ig - pos.entry_price) / pos.entry_price * 100
        await ctx.alerter.alert_take_profit(
            symbol, reason, pos.entry_price / scale, fill_ig / scale, pnl_pct, reasoning
        )
