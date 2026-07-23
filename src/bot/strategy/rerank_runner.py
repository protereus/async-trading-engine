"""TopK rerank + per-candle dispatch + signal-history resolver loop.

Extracted from ``main.py``.  This is the largest single extraction:
the rerank loop alone is the longest method in the codebase (~230 lines).

``RerankRunner`` owns:

* ``topk_rerank_loop`` — deadline-based hourly rerank.  Runs ``TopK.scan``
  across the watchlist, writes signal_history, applies the sentiment gate,
  fires signal-decay / sentiment-reversal exits via ``IGCloseManager``,
  and sends the Telegram rerank summary.
* ``signal_resolver_loop`` — hourly fill of ``realized_*`` columns on past
  signal_history rows.
* ``subscribe_candle_handler`` — wires the ``EVENT_NEW_CANDLE`` subscriber
  and parks on the shutdown event.
* ``process_candle`` — dispatcher invoked on every confirmed candle.
* ``process_candle_ig_topk`` — per-candle TopK logic: stop-loss check,
  take-profit evaluation, fresh-entry placement.

Operates on the shared ``BotContext`` like every collaborator — reads /
mutates ``ctx.state`` / ``ctx.topk_*`` / ``ctx.ig_*`` / ``ctx.risk_manager``
/ ``ctx.tp_manager`` / ``ctx.closer`` in place.  The rerank task is created
by ``Lifecycle.start`` as ``asyncio.create_task(ctx.runner.topk_rerank_loop(),
name="topk_rerank")``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

from bot.core.event_bus import EVENT_NEW_CANDLE, EVENT_ORDER_FILLED
from bot.core.models import Candle, ExchangeError, IGOrderRequest, MarketClosedError, Position
from bot.data.candle_db import SignalHistoryRow
from bot.execution.ig_convert import IG_MIN_STOP_PCT, apply_sentiment_gate
from bot.execution.ig_quote_scale import (
    ig_display_price,
    ig_pip_value,
    ig_quote_scale,
)
from bot.risk.funding import log_overnight_funding_estimate
from bot.risk.ig_margin import estimate_margin_gbp as ig_margin_estimate
from bot.risk.ig_margin import estimate_slippage_pts as ig_slippage_pts
from bot.risk.risk_manager import RiskManager
from bot.strategy import _kronos_progress
from bot.trading_hours import is_market_open, is_safe_for_entry

if TYPE_CHECKING:
    from bot.core.bot_context import BotContext
    from bot.sentiment.models import ConsensusSignal
    from bot.strategy.topk_strategy import AssetSignal

logger = logging.getLogger(__name__)


# Sentiment-edge measurement harness.
#
# Six agents emit RawSignals; ``ConsensusSignal.per_agent`` carries the
# per-agent sentiment.  We split into two decay buckets so the analysis
# script can ask "does the slow part add anything the fast part doesn't"
# without storing all six columns.
_SLOW_DECAY_AGENTS: frozenset[str] = frozenset({"central_bank", "macro"})
# ``social`` dropped 2026-06-01 — SocialAgent retired (Finnhub social endpoint
# requires paid tier; agent emitted 403s for every ETF asset).  Set kept as
# the lookup is forgiving — any new agent name shows up as an unbucketed
# contributor in ``sentiment_agent_coverage`` and falls into neither slow
# nor fast.
_FAST_DECAY_AGENTS: frozenset[str] = frozenset({"news", "fear_greed", "gdelt"})


class _SentimentCaptureFields(TypedDict):
    """The sentiment subset of ``SignalHistoryRow`` — what
    ``_sentiment_capture`` contributes to each signal_history row."""

    sentiment_score: float | None
    sentiment_confidence: float | None
    sentiment_agent_coverage: int
    sentiment_slow_decay: float | None
    sentiment_fast_decay: float | None


def _sentiment_capture(
    snapshot: dict[str, ConsensusSignal],
    symbol: str,
) -> _SentimentCaptureFields:
    """Pull sentiment fields for *symbol* out of the per-rerank snapshot.

    Returns a dict ready to splat into a ``signal_history`` row:

      ``sentiment_score`` (float | None)
      ``sentiment_confidence`` (float | None)
      ``sentiment_agent_coverage`` (int, 0..6)
      ``sentiment_slow_decay`` (float | None — mean of central_bank+macro)
      ``sentiment_fast_decay`` (float | None — mean of news/fear_greed/gdelt)

    A missing symbol or empty ``per_agent`` writes nulls + coverage=0,
    routing the row into the ABSENT partition of the analysis.  A symbol
    present with coverage<6 still records the partial scores so the
    harness can see whether degraded overlay correlates with weaker edge.
    """
    consensus = snapshot.get(symbol)
    if consensus is None:
        return {
            "sentiment_score": None,
            "sentiment_confidence": None,
            "sentiment_agent_coverage": 0,
            "sentiment_slow_decay": None,
            "sentiment_fast_decay": None,
        }
    per_agent = consensus.per_agent
    slow_vals = [v for k, v in per_agent.items() if k in _SLOW_DECAY_AGENTS]
    fast_vals = [v for k, v in per_agent.items() if k in _FAST_DECAY_AGENTS]
    return {
        "sentiment_score": consensus.sentiment,
        "sentiment_confidence": consensus.confidence,
        "sentiment_agent_coverage": len(per_agent),
        "sentiment_slow_decay": sum(slow_vals) / len(slow_vals) if slow_vals else None,
        "sentiment_fast_decay": sum(fast_vals) / len(fast_vals) if fast_vals else None,
    }


# IG rejects a stake that isn't a multiple of a market's deal-size increment
# with ``SIZE_INCREMENT``.  That increment is NOT exposed anywhere in the REST
# API (``dealingRules`` carries ``minDealSize`` but no size step — confirmed
# 2026-06-19, a known IG limitation), so we can't round to it ahead of time.
# Empirically the stakes IG accepts on the affected US shares are all multiples
# of 0.1 £/pt, so 0.1 is a safe superset grid to snap UP to on a reject + retry.
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


class RerankRunner:
    """TopK rerank loop + per-candle dispatch.  Collaborator on TradingBot."""

    def __init__(self, ctx: BotContext) -> None:
        self._ctx = ctx

    async def topk_rerank_loop(self) -> None:
        """Periodically re-scan watchlist, update selection, and close dropped positions.

        The cadence is *deadline-based*: the next rerank fires
        ``topk_rerank_interval_minutes`` from the previous one's *start* (not its
        completion).  Kronos inference for the 28-asset universe takes ~55–60 min,
        so a "sleep after scan" loop produced an effective ~2 h cadence instead of
        the documented 1 h.  We track ``_next_rerank_at_mono`` (monotonic clock)
        and sleep until that point; if a scan overruns, the next pass fires as
        soon as it finishes (and we never let missed deadlines pile up — the
        deadline is advanced past ``now`` in one step).
        """
        ctx = self._ctx
        # Wait for backfill to load enough candles before running the first scan.
        # Without this gate the loop fires immediately at startup (0 candles), overwrites
        # _topk_selected with [] and destroys any topk_state restored from disk.
        _watchlist_init = ctx.config.topk_watchlist or ctx.candle_symbols
        _min_bars = ctx.config.kronos_context_bars
        while not ctx.shutdown_event.is_set():
            if all(len(ctx.store.get_candles(s)) >= _min_bars for s in _watchlist_init):
                break
            await asyncio.sleep(30)

        interval = float(ctx.config.topk_rerank_interval_minutes * 60)
        next_rerank_at = time.monotonic()  # first scan fires immediately
        while not ctx.shutdown_event.is_set():
            ctx.rerank_in_progress = True
            try:
                await self._run_rerank_once(interval, next_rerank_at)
            except Exception:
                logger.exception("TopK rerank loop error")
            finally:
                ctx.rerank_in_progress = False

            # Advance the deadline by exactly one interval.  If the scan
            # overran (next_rerank_at already in the past), jump straight to
            # the next future tick so we don't accumulate missed reranks.
            now_mono = time.monotonic()
            next_rerank_at += interval
            ctx.rerank_status.update(
                in_progress=False,
                phase="idle",
                phase_started_at=time.time(),
                current_batch=None,
                next_rerank_at=ctx.mono_to_wall(max(next_rerank_at, now_mono)),
            )
            if next_rerank_at <= now_mono:
                missed = int((now_mono - next_rerank_at) // interval) + 1
                logger.warning(
                    "TopK rerank scan overran interval — skipping %d missed rerank(s)",
                    missed,
                )
                next_rerank_at = now_mono  # fire next pass immediately
            sleep_for = max(0.0, next_rerank_at - now_mono)
            try:
                await asyncio.wait_for(asyncio.shield(ctx.shutdown_event.wait()), timeout=sleep_for)
                break
            except TimeoutError:
                pass

    async def _run_rerank_once(self, interval: float, next_rerank_at: float) -> None:
        """One full rerank pass: reconcile → inference → select → signal_history
        → correlation → sentiment_gate → decay_eval → balance_alert.  Each phase
        owns its own ``_rerank_status.update(phase=...)`` so the dashboard phase
        tracking is unchanged; the deadline math + in-progress flag stay in the
        scheduler (``topk_rerank_loop``)."""
        ctx = self._ctx
        watchlist = ctx.config.topk_watchlist or ctx.candle_symbols
        _now_wall = time.time()
        _batches_total = ctx.topk_strategy.expected_batches(watchlist)
        ctx.rerank_status.update(
            in_progress=True,
            phase="reconcile",
            started_at=_now_wall,
            phase_started_at=_now_wall,
            interval_s=interval,
            batches_done=0,
            batches_total=_batches_total,
            next_rerank_at=ctx.mono_to_wall(next_rerank_at + interval),
            current_batch=None,
        )
        await self._rerank_reconcile()
        signals = await self._rerank_inference(watchlist)
        new_selected = self._rerank_select(signals)
        # One shared ``scored_at`` timestamp across the signal_history rows and
        # the asset_correlations write (returned by the signal-history helper).
        _now_ms = self._rerank_write_signal_history(signals, _batches_total)
        await self._rerank_write_correlations(_now_ms)
        new_selected = self._rerank_apply_sentiment_gate(new_selected, signals)
        ctx.topk_selected = new_selected
        ctx.topk_scanned = True
        await self._rerank_run_decay_exits(signals)
        await self._rerank_send_alert(signals, new_selected)

    async def _rerank_reconcile(self) -> None:
        """Sync state with IG before any rerank-driven exits run, so we don't try
        to close (or signal-decay) a position that was already closed externally
        (IG server-side stop, manual web UI close).  Hourly cadence is well within
        IG's rate limit.  The ``reconcile`` phase is set by the run-init update."""
        await self._ctx.closer.reconcile_positions_with_ig()

    async def _rerank_inference(self, watchlist: list[str]) -> list[AssetSignal]:
        """Pass-1/Pass-2 Kronos inference for the watchlist; returns the per-asset
        signals and stashes them on ``ctx.topk_signals``."""
        ctx = self._ctx

        async def _fetch(epic: str) -> list[Candle]:
            return ctx.store.get_candles(epic)

        _kronos_progress.reset_counter()
        ctx.rerank_status.update(phase="inference", phase_started_at=time.time())
        signals: list[AssetSignal] = await ctx.topk_strategy.scan(watchlist, _fetch)
        ctx.topk_signals = signals
        return signals

    def _rerank_select(self, signals: list[AssetSignal]) -> list[str]:
        """Exclude monitor-only symbols then pick the open-market top-K."""
        ctx = self._ctx
        # Exclude monitor-only symbols (e.g. metals) from selection while
        # still letting their signals reach signal_history (written from
        # ``signals`` above, before this filter).  An open position on
        # an excluded symbol falls out of the new top-K on the next
        # rerank and is rotated out — intended behaviour.
        _exclude = set(ctx.config.topk_exclude_from_selection)
        selectable = [s for s in signals if s.symbol not in _exclude]
        # Only rank assets whose market is open right now, so the k
        # entry slots aren't consumed by names that can't be filled
        # (US shares overnight / on NYSE holidays while FX stays open).
        # The top-K-miss exit counter below is made market-aware to
        # compensate, so held positions aren't evicted during their own
        # market's closure.
        selected: list[str] = ctx.topk_strategy.select_top_k(selectable, is_open=is_market_open)
        return selected

    def _rerank_write_signal_history(self, signals: list[Any], batches_total: int) -> int:
        """Persist every asset's signal row (predicted-close path blob, per-draw
        var blob, sentiment capture) for post-hoc RankIC / calibration analysis.
        Returns the shared ``scored_at`` ms timestamp so the correlation write
        tags the same instant."""
        ctx = self._ctx
        ctx.rerank_status.update(
            phase="signal_history",
            phase_started_at=time.time(),
            batches_done=batches_total,
            current_batch=None,
        )

        # Sentiment-edge measurement harness (2026-06-01).  Snapshot
        # the engine's cached scores ONCE per rerank so every
        # signal_history row writes the same view; partial overlay
        # (FRED/Cerebras degraded) is captured honestly via the
        # agent-coverage count rather than silently nulled.  The
        # capture runs unconditionally of ``sentiment_gate_enabled``
        # so we can decide whether to enable the gate from data we
        # already have.
        _sentiment_snapshot = (
            ctx.sentiment_engine.get_sentiment_scores() if ctx.sentiment_engine is not None else {}
        )

        # Record every asset's signal for post-hoc RankIC / calibration analysis.
        # persist the full predicted close path (little-endian
        # float32 bytes) so an offline horizon back-test can read any
        # bar from the path without re-running inference.
        _now_ms = int(time.time() * 1000)
        _sig_rows: list[SignalHistoryRow] = []
        for _sig in signals:
            _latest = ctx.store.get_latest_candle(_sig.symbol)
            _entry_px = _latest.close if _latest is not None else _sig.predicted_close
            _ps = ctx.topk_strategy.path_signal_for(_sig.symbol)
            _path_blob: bytes | None = None
            if _ps is not None and _ps.predicted_closes:
                _path_blob = np.asarray(_ps.predicted_closes, dtype="<f4").tobytes()
            # per-draw Pass-2 closes at every candidate
            # ranking horizon.  Little-endian float32, shape
            # (sample_count, len(CANDIDATE_HORIZONS)) row-major — see
            # CandleDB.write_signal_history_batch for the layout doc.
            _varh_blob: bytes | None = None
            if _sig.var_closes_at_horizons:
                _varh_blob = np.asarray(_sig.var_closes_at_horizons, dtype="<f4").tobytes()
            _sentiment_fields = _sentiment_capture(_sentiment_snapshot, _sig.symbol)
            _sig_rows.append(
                {
                    "scored_at": _now_ms,
                    "symbol": _sig.symbol,
                    # The symbol's *effective* ranking horizon (class
                    # override → global → pred_len), so the resolver
                    # realises returns — and computes gap_spanned —
                    # over the same window the prediction was sliced
                    # at.  Constant pred_len until the per-class map
                    # is enabled.
                    "horizon_bars": ctx.topk_strategy.signal_horizon_bars(_sig.symbol),
                    "mean_return": _sig.mean_return,
                    "direction_confidence": _sig.direction_confidence,
                    "uncertainty": _sig.uncertainty,
                    "entry_price": _entry_px,
                    "predicted_mfe_pct": _ps.predicted_mfe_pct if _ps else None,
                    "predicted_mae_pct": _ps.predicted_mae_pct if _ps else None,
                    "predicted_volatility": _ps.predicted_volatility if _ps else None,
                    "monotonicity": _ps.monotonicity if _ps else None,
                    "predicted_close_path": _path_blob,
                    "var_closes_at_horizons": _varh_blob,
                    **_sentiment_fields,
                }
            )
        if _sig_rows:
            try:
                ctx.candle_db.write_signal_history_batch(_sig_rows)
            except Exception:
                logger.exception("signal_history write failed — continuing")
        return _now_ms

    async def _rerank_write_correlations(self, now_ms: int) -> None:
        """Persist the rolling asset-correlation snapshot, tagged with the shared
        ``scored_at`` timestamp from the signal-history write."""
        ctx = self._ctx
        ctx.rerank_status.update(phase="correlation", phase_started_at=time.time())
        try:
            _corr_snap = ctx.topk_strategy.snapshot_correlation()
            if _corr_snap:
                await asyncio.to_thread(ctx.candle_db.write_correlations, now_ms, _corr_snap)
        except Exception:
            logger.exception("asset_correlations write failed — continuing")

    def _rerank_apply_sentiment_gate(
        self, new_selected: list[str], signals: list[Any]
    ) -> list[str]:
        """Optional sentiment gate — direction-aware: LONGs need bullish sentiment,
        SHORTs need bearish sentiment. Assets with no sentiment pass through."""
        ctx = self._ctx
        if ctx.config.sentiment_gate_enabled and ctx.sentiment_engine is not None:
            ctx.rerank_status.update(phase="sentiment_gate", phase_started_at=time.time())
            scores = ctx.sentiment_engine.get_sentiment_scores()
            new_selected, blocked = apply_sentiment_gate(
                new_selected,
                {s.symbol: s.mean_return for s in signals},
                {sym: cs.sentiment for sym, cs in scores.items()},
                ctx.config.sentiment_gate_long_threshold,
                ctx.config.sentiment_gate_short_threshold,
            )
            if blocked:
                logger.info("Sentiment gate blocked %d assets: %s", len(blocked), blocked)
        return new_selected

    async def _rerank_run_decay_exits(self, signals: list[Any]) -> None:
        """Signal-decay exits (priorities 4 & 7) and sentiment reversal (priority 8)."""
        ctx = self._ctx
        if ctx.tp_manager is not None and ctx.state.positions:
            ctx.rerank_status.update(phase="decay_eval", phase_started_at=time.time())
            signal_by_symbol = {s.symbol: s for s in signals}
            sentiment_scores: dict[str, Any] = {}
            if (
                ctx.config.sentiment_enabled
                and ctx.config.tp_sentiment_reversal_enabled
                and ctx.sentiment_engine is not None
            ):
                sentiment_scores = ctx.sentiment_engine.get_sentiment_scores()

            for sym in list(ctx.state.positions.keys()):
                latest_signal = signal_by_symbol.get(sym)
                # A held position whose own market is shut couldn't be
                # ranked among the open-market candidates (select_top_k
                # filters by is_market_open), so a top-K "miss" here is
                # an artefact of the closure, not signal decay.  Treat it
                # as still-selected to avoid evicting it overnight / on
                # NYSE holidays.  Signal-quality strikes (flip /
                # confidence / uncertainty) still apply — Kronos keeps
                # scoring the symbol regardless of market state.
                in_topk = sym in ctx.topk_selected or not is_market_open(sym)
                decision = ctx.tp_manager.evaluate_signal(sym, latest_signal, in_topk)
                if decision.should_exit:
                    await ctx.closer.request_close(sym, decision.reason.value, decision.reasoning)
                    continue
                # Sentiment reversal check — uses latest scan's cached scores
                if ctx.config.tp_sentiment_reversal_enabled:
                    sentiment = sentiment_scores.get(sym)
                    if sentiment is not None:
                        sent_decision = ctx.tp_manager.evaluate_sentiment(sym, sentiment)
                        if sent_decision.should_exit:
                            await ctx.closer.request_close(
                                sym,
                                sent_decision.reason.value,
                                sent_decision.reasoning,
                            )

    async def _rerank_send_alert(self, signals: list[Any], new_selected: list[str]) -> None:
        """Gather live prices + balance, build the display-positions dict, and send
        the TopK rerank Telegram alert."""
        ctx = self._ctx
        # Gather current prices for open positions' P&L display.
        # Convert the candle-source close to display units so post-D3
        # USO/UNG/SLV render as $/bbl / $/MMBtu / $/oz rather than the
        # raw IG-level numbers (8726 → 87.26 etc.).  Forex / SPY / DIA
        # / QQQ flow through the existing ``ig_quote_scale`` divisor
        # unchanged.
        ctx.rerank_status.update(phase="balance_alert", phase_started_at=time.time())
        _cur_prices: dict[str, float] = {}
        # Prefer IG's LIVE per-position price (market bid) so the alert's
        # per-position P&L matches the live aggregate Open P&L — both from
        # IG — rather than the last *closed* hourly candle, which can be up
        # to an hour stale and disagree in sign on a near-flat position.
        _live_by_epic: dict[str, Any] = {}
        try:
            for _lp in await ctx.ig_client.fetch_positions():
                _live_by_epic[_lp.symbol] = _lp
        except Exception:
            logger.debug(
                "TopK rerank: live /positions fetch failed — per-position "
                "prices fall back to the last candle",
                exc_info=True,
            )
        for _sym in ctx.state.positions:
            _lp = _live_by_epic.get(ctx.epic_for(_sym))
            if _lp is not None and _lp.current_price > 0:
                _cur_prices[_sym] = ig_display_price(_sym, _lp.current_price)
                continue
            _c = ctx.store.get_latest_candle(_sym)
            if _c is not None:
                _cur_prices[_sym] = ig_display_price(_sym, _c.close * ig_quote_scale(_sym))
        _equity_now: float | None = None
        _cash_now: float | None = None
        _open_pnl_now: float | None = None
        try:
            _bal = await ctx.ig_client.fetch_balance()
            _equity_now = _bal["equity"]
            _cash_now = _bal["balance"]
            _open_pnl_now = _bal["open_pnl"]
            ctx.risk_manager.update_equity(_equity_now)
            ctx.state.cash = _cash_now
            ctx.state.open_pnl = _open_pnl_now
        except Exception:
            logger.debug(
                "TopK rerank: IG balance fetch failed — alert will omit equity",
                exc_info=True,
            )
        # Build the alert-side positions dict: entry, size, and the
        # current active stop level (initial stop until breakeven /
        # trail arms, then the live trailing stop).  All prices use
        # ``ig_display_price`` so post-D3 USO/UNG/SLV read as natural
        # quote units in the Telegram alert.
        _tp_snapshot = ctx.tp_manager.snapshot() if ctx.tp_manager is not None else {}
        _display_positions: dict[str, dict[str, Any]] = {}
        for _sym, _pos in ctx.state.positions.items():
            _stop_pct: float | None = None
            _stop_display: float | None = None
            _tp = _tp_snapshot.get(_sym)
            if isinstance(_tp, dict) and _pos.entry_price > 0:
                _tp_entry = float(_tp.get("entry_price") or _pos.entry_price)
                _trail = _tp.get("current_trailing_stop")
                if _trail is not None:
                    _stop_level_ig = float(_trail)
                    _stop_pct = (_tp_entry - _stop_level_ig) / _tp_entry
                else:
                    _raw_pct = _tp.get("entry_stop_pct")
                    if isinstance(_raw_pct, (int, float)):
                        _stop_pct = float(_raw_pct)
                        _stop_level_ig = _tp_entry * (1.0 - _stop_pct)
                    else:
                        _stop_level_ig = None
                if _stop_level_ig is not None:
                    _stop_display = ig_display_price(_sym, _stop_level_ig)
            _display_positions[_sym] = {
                "entry_price": ig_display_price(_sym, _pos.entry_price),
                "quantity": _pos.quantity,
                "stop_price": _stop_display,
                "stop_pct": _stop_pct,
            }
        # Dedup-on-change: only surface the correlation section when the
        # set of dropped picks differs from the previous rerank, so a
        # steady-state bump isn't repeated on every alert.
        _bumped = ctx.topk_strategy.material_bumps_if_changed()
        await ctx.alerter.send_topk_rerank(
            signals,
            new_selected,
            ctx.config.topk_k,
            positions=_display_positions,
            risk_summary=ctx.risk_manager.get_risk_summary(),
            current_prices=_cur_prices,
            equity=_equity_now,
            cash=_cash_now,
            open_pnl=_open_pnl_now,
            bumped=_bumped,
            open_market=is_safe_for_entry,
        )

    async def signal_resolver_loop(self) -> None:
        """Hourly: fill realized_* columns for signal_history rows past their horizon."""
        ctx = self._ctx
        while not ctx.shutdown_event.is_set():
            try:
                await asyncio.sleep(3600)
                if ctx.shutdown_event.is_set():
                    break
                now_ms = int(time.time() * 1000)
                resolved = await asyncio.to_thread(ctx.candle_db.resolve_signal_history, now_ms)
                if resolved:
                    logger.info("signal_resolver: resolved %d rows", resolved)
            except Exception:
                logger.exception("signal_resolver loop error")

    async def subscribe_candle_handler(self) -> None:
        """Subscribe to new_candle events and run the strategy on each."""
        ctx = self._ctx

        async def _on_new_candle(data: Any) -> None:
            if not isinstance(data, Candle) or not data.is_confirmed:
                return
            await self.process_candle(data)

        ctx.event_bus.subscribe(EVENT_NEW_CANDLE, _on_new_candle)
        await ctx.shutdown_event.wait()

    async def process_candle(self, candle: Candle) -> None:
        """Run per-candle logic (stop-loss + take-profit checks for IG TopK)."""
        ctx = self._ctx
        if not isinstance(candle, Candle):
            return

        symbol = candle.symbol
        candles = ctx.store.get_candles(symbol)
        current_position = ctx.state.positions.get(symbol)

        if ctx.topk_strategy is not None:
            if len(candles) < 2:
                return
            await self.process_candle_ig_topk(symbol, candle.close, current_position)

    async def process_candle_ig_topk(
        self,
        symbol: str,
        current_price: float,
        current_position: Any,
    ) -> None:
        """Per-candle logic for pure TopK IG mode.

        Exits: stop-loss breach on every candle; rerank-driven exits handled
        by ``topk_rerank_loop``.
        Entries: fired on the first candle after the symbol enters the top-K
        selection (i.e. after ``_topk_scanned`` is True and symbol is selected).
        """
        if current_position is not None:
            await self._handle_open_position(symbol, current_price, current_position)
            return
        await self._maybe_enter(symbol, current_price)

    async def _handle_open_position(
        self,
        symbol: str,
        current_price: float,
        current_position: Position,
    ) -> None:
        """Exit logic for an open position: stop-loss check (priority 1) then
        take-profit evaluation (priorities 2–6)."""
        ctx = self._ctx
        # Stop-loss check (priority 1): use vol-adjusted stop from the last scan signal
        topk_signal = next((s for s in ctx.topk_signals if s.symbol == symbol), None)
        stop_pct = topk_signal.stop_pct if topk_signal is not None else ctx.config.topk_min_stop_pct
        # Convert Twelve Data price to IG level for comparison with entry_price.
        # IG quotes JPY pairs at 100× the FX rate (e.g. USD/JPY 157.05 → level 15705).
        ig_current = current_price * ig_quote_scale(symbol)
        loss_pct = (current_position.entry_price - ig_current) / current_position.entry_price
        if loss_pct >= stop_pct:
            logger.info(
                "TopK stop-loss: %s loss=%.2f%% >= stop=%.2f%%",
                symbol,
                loss_pct * 100,
                stop_pct * 100,
            )
            closed = await ctx.closer.close_ig_position(symbol, current_position)
            if closed is None:
                # Deferred — IG funding/maintenance window. Skip reconcile +
                # alert; next candle will retry once the window ends.
                pass
            elif closed is False:
                await ctx.closer.handle_close_failure(
                    symbol,
                    f"Stop-loss close failed for {symbol} — position still open on IG, "
                    f"will retry on next candle",
                )
            else:
                # Success — ``closed`` is the realised IG-level fill price.
                if ctx.tp_manager is not None:
                    ctx.tp_manager.deregister_position(symbol)
            return

        # Take-profit evaluation (priorities 2–6, via evaluate_price)
        if ctx.tp_manager is not None:
            now_ms = int(time.time() * 1000)
            tp_decision = ctx.tp_manager.evaluate_price(symbol, ig_current, now_ms)
            if tp_decision.should_exit:
                await ctx.closer.request_close(
                    symbol, tp_decision.reason.value, tp_decision.reasoning
                )

    async def _maybe_enter(self, symbol: str, current_price: float) -> None:
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
            balance = await ctx.ig_client.fetch_balance()
            equity_gbp = balance["equity"]
            margin_used = balance["margin"]
        except Exception:
            logger.warning("Could not fetch IG balance", exc_info=True)
            return None, None

        logger.info("TopK balance: %s equity=%.2f margin=%.2f", symbol, equity_gbp, margin_used)
        ctx.risk_manager.update_equity(equity_gbp)
        ctx.state.cash = balance["balance"]
        ctx.state.open_pnl = balance["open_pnl"]
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
        # Pre-trade tier-aware margin estimate.  ``estimate_margin_gbp``
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
        try:
            # Pre-trade market-status gate.
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
