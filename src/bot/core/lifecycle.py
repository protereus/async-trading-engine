"""Bot lifecycle — broker init, feed construction, start / shutdown sequencing.

Extracted from ``main.py`` as the final step (§1 step 6) of the
 plan.  Owns:

* ``init_ig`` — wires IG-specific subsystems (IGClient, IGFeed, TopKStrategy,
  TakeProfitManager, trailing-stop lookup wiring).
* ``build_feed_task`` — constructs the right ``Feed`` for the configured
  ``candle_exchange`` and schedules it as an asyncio task.
* ``start`` — the public lifecycle entry point: connect → restore state →
  reconcile → wire events → spawn feeds + strategy + heartbeat → wait for
  shutdown.
* ``shutdown`` — bounded graceful teardown: signal loops → cancel tasks →
  persist state → close DB → close broker / feed connections → send Telegram
  shutdown alert.

Shared state flows through ``BotContext`` like every other collaborator
(``RerankRunner`` / ``IGCloseManager`` / ``HealthMonitor`` / ``EventWiring``),
but Lifecycle additionally keeps the ``bot`` handle: it is the composition
root's delegate and populates the context's late-bound broker/strategy/feed
fields during ``init_ig()`` / ``start()``.  TradingBot keeps thin public
``start`` / ``shutdown`` delegates so external callers (``main()`` entry
point, tests) keep working unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from typing import TYPE_CHECKING, Any

from bot.core.models import TopkState

if TYPE_CHECKING:
    from bot.main import TradingBot

logger = logging.getLogger(__name__)

# Bounded wait when cancelling background tasks during shutdown.  Mirrors the
# constant the old in-line ``shutdown`` used; lives here now so changes are in
# one place.
SHUTDOWN_TIMEOUT = 10.0  # seconds to wait for tasks to cancel


class Lifecycle:
    """Broker init + start / shutdown sequencing collaborator for TradingBot."""

    def __init__(self, bot: TradingBot) -> None:
        self._ctx = bot.ctx

    def init_ig(self) -> None:
        """Wire IG-specific subsystems."""
        from bot.execution.ig_client import IGClient
        from bot.risk.spread_monitor import SpreadMonitor

        ctx = self._ctx
        ctx.ig_client = IGClient(ctx.config)

        # Construct once here (composition root) so the pre-trade
        # spread gate runs regardless of which candle feed ends up live.
        ctx.spread_monitor = SpreadMonitor()

        if ctx.config.candle_exchange == "ig":
            from bot.data.ig_feed import IGFeed

            ctx.ig_feed = IGFeed(
                ctx.ig_client,
                ctx.store,
                ctx.event_bus,
                ctx.config,
                candle_db=ctx.candle_db,
                spread_monitor=ctx.spread_monitor,
            )

        if ctx.config.topk_enabled:
            from bot.strategy.topk_strategy import TopKConfig, TopKStrategy

            # Prepend Kronos source directory so `from model import` works inside
            # TopKStrategy._load_predictor(). Done here at startup, not inside the
            # strategy, so sys.path manipulation stays in one place.
            kronos_dir = ctx.config.kronos_dir
            if kronos_dir and kronos_dir not in sys.path:
                sys.path.insert(0, kronos_dir)

            ctx.topk_strategy = TopKStrategy(TopKConfig.from_bot_config(ctx.config))
            logger.info(
                "TopK strategy enabled: k=%d watchlist=%s",
                ctx.config.topk_k,
                ctx.config.topk_watchlist or ctx.candle_symbols,
            )

            from bot.strategy.take_profit import TakeProfitConfig, TakeProfitManager

            ctx.tp_manager = TakeProfitManager(
                TakeProfitConfig.from_bot_config(ctx.config),
                pred_len=ctx.config.topk_pred_len,
            )
            logger.info("TakeProfitManager initialised")

            # Let the risk-manager's total-risk gate shrink live risk-on as
            # the trail ratchets up.  TP keys by candle symbol; risk_manager
            # keys by EPIC — translate via ctx.candle_for.
            tp_manager = ctx.tp_manager

            def _trail_lookup(epic: str) -> float | None:
                trail: float | None = tp_manager.get_trailing_stop(ctx.candle_for(epic))
                return trail

            ctx.risk_manager.set_trailing_stop_lookup(_trail_lookup)

    def build_feed_task(self) -> asyncio.Task[None]:
        """Instantiate the appropriate data feed and schedule it as an asyncio Task."""
        ctx = self._ctx
        if ctx.config.candle_exchange == "ig":
            return asyncio.create_task(ctx.ig_feed.run(), name="ig_feed")
        if ctx.config.candle_exchange == "twelvedata":
            from bot.data.twelve_data_feed import TwelveDataFeed

            ctx.twelve_data_feed = TwelveDataFeed(
                ctx.store,
                ctx.event_bus,
                ctx.config,
                candle_db=ctx.candle_db,
            )
            return asyncio.create_task(ctx.twelve_data_feed.run(), name="twelve_data_feed")
        if ctx.config.candle_exchange == "eodhd":
            from bot.data.eodhd_feed import EODHDFeed

            ctx.eodhd_feed = EODHDFeed(
                ctx.store,
                ctx.event_bus,
                ctx.config,
                candle_db=ctx.candle_db,
                spread_monitor=ctx.spread_monitor,
            )
            return asyncio.create_task(ctx.eodhd_feed.run(), name="eodhd_feed")
        raise ValueError(
            f"Unsupported candle_exchange={ctx.config.candle_exchange!r}; "
            "expected one of 'ig', 'twelvedata', 'eodhd' "
            "(ccxt/binance feeds archived 2026-06-24)"
        )

    async def start(self) -> None:
        """Connect, load state, reconcile, and run until shutdown."""
        ctx = self._ctx
        # 1. Connect to broker
        await ctx.ig_client.connect()
        logger.info(
            "IG connected: account=%s env=%s",
            ctx.ig_client.account_id,
            ctx.config.bot_env,
        )

        # 2. Load persisted state
        self._restore_persisted_state()

        # 3. Reconcile deal IDs for IG: query live positions and rebuild _ig_deal_ids.
        #    Any restored position with no matching live IG deal is purged — it was
        #    closed externally (stop-loss, manual close) while the bot was offline.
        await ctx.closer.reconcile_positions_with_ig(verbose=True)

        # 3b. Fetch initial equity so drawdown tier is correct from the start
        await self._fetch_startup_balance()

        # 4. Wire event subscriptions
        ctx.events.wire()

        # 5. Build and start data feed
        feed_task = self.build_feed_task()

        strategy_task = asyncio.create_task(ctx.runner.subscribe_candle_handler(), name="strategy")
        health_task = asyncio.create_task(ctx.health.health_check(), name="health_check")

        ctx.tasks = [feed_task, strategy_task, health_task]
        if ctx.topk_strategy is not None:
            _restored_corr = await asyncio.to_thread(ctx.candle_db.read_latest_correlations)
            if _restored_corr:
                ctx.topk_strategy.restore_correlation(_restored_corr)
            topk_task = asyncio.create_task(ctx.runner.topk_rerank_loop(), name="topk_rerank")
            ctx.tasks.append(topk_task)
            resolver_task = asyncio.create_task(
                ctx.runner.signal_resolver_loop(), name="signal_resolver"
            )
            ctx.tasks.append(resolver_task)
            drift_task = asyncio.create_task(self._scale_drift_loop(), name="scale_drift_guard")
            ctx.tasks.append(drift_task)

        # Optional config-gated overlays — each guards on its own flag and
        # appends its task to ctx.tasks independently (order is log-readability
        # only, not correctness).
        self._maybe_start_ig_candle_feed()
        self._maybe_start_fred()
        self._maybe_start_sentiment()

        # 6. Startup alert
        broker_label = f"IG ({ctx.config.bot_env.upper()})"
        pairs_str = ", ".join(ctx.candle_symbols)
        topk_status = f"ON (k={ctx.config.topk_k})" if ctx.topk_strategy is not None else "OFF"
        sentiment_status = (
            f"ON (gate={'ON' if ctx.config.sentiment_gate_enabled else 'OFF'})"
            if ctx.config.sentiment_enabled
            else "OFF"
        )
        await ctx.alerter.send_startup(
            f"Broker: {broker_label}\n"
            f"Pairs: {pairs_str}\n"
            f"Strategy: Kronos TopK\n"
            f"TopK: {topk_status}\n"
            f"Sentiment: {sentiment_status}\n"
            f"Env: {ctx.config.bot_env.upper()}"
        )

        logger.info("Trading bot started. Waiting for shutdown signal...")
        await ctx.shutdown_event.wait()
        logger.info("Shutdown event received -- stopping tasks")

    def _restore_persisted_state(self) -> None:
        """Load persisted bot state and rehydrate risk / take-profit / topk state.

        Crash-recovery (behaviour-critical): a manual restart resets a
        consecutive-loss pause (operator acknowledgement), and topk state is only
        rehydrated if it is ≤4h old — the AssetSignal rebuild must stay
        field-for-field identical to the original save shape."""
        ctx = self._ctx
        recovered = ctx.state_manager.load()
        if recovered is not None:
            ctx.state = recovered
            # A manual restart is an implicit operator acknowledgement.  Reset
            # consecutive_losses if it would otherwise block all new entries.
            if recovered.risk.consecutive_losses >= ctx.risk_config.consecutive_loss_pause:
                logger.warning(
                    "Startup: resetting consecutive_losses=%d (was at pause threshold=%d) "
                    "— operator restart treated as acknowledgement",
                    recovered.risk.consecutive_losses,
                    ctx.risk_config.consecutive_loss_pause,
                )
                recovered.risk.consecutive_losses = 0
            ctx.risk_manager.load_state(ctx.state.risk)
            if ctx.tp_manager is not None and ctx.state.take_profit_state:
                ctx.tp_manager.restore(ctx.state.take_profit_state)
            topk_saved = ctx.state.topk_state
            if topk_saved is not None:
                age_h = (time.time() * 1000 - float(topk_saved.get("scanned_at", 0))) / 3_600_000
                if age_h <= 4.0:
                    from bot.strategy.topk_strategy import AssetSignal  # noqa: PLC0415

                    ctx.topk_selected = list(topk_saved.get("selected", []))
                    ctx.topk_signals = [
                        AssetSignal.from_persist(s) for s in topk_saved.get("signals", [])
                    ]
                    ctx.topk_scanned = True
                    logger.info(
                        "Restored topk state from %.1fh ago: selected=%s",
                        age_h,
                        ctx.topk_selected,
                    )
                else:
                    logger.info("Skipping stale topk state (%.1fh old)", age_h)

        ctx.state.bot_started_at = ctx.started_at

    async def _fetch_startup_balance(self) -> None:
        """Fetch initial IG equity so the drawdown tier is correct from the start.
        Best-effort: on failure the bot still starts (drawdown reads 100% until
        the first candle)."""
        ctx = self._ctx
        try:
            balance = await ctx.ig_client.fetch_balance()
            ctx.risk_manager.update_equity(balance["equity"])
            ctx.state.cash = balance["balance"]
            ctx.state.open_pnl = balance["open_pnl"]
            logger.info(
                "IG startup balance: equity=%.2f cash=%.2f open_pnl=%+.2f margin=%.2f",
                balance["equity"],
                balance["balance"],
                balance["open_pnl"],
                balance["margin"],
            )
        except Exception:
            logger.warning(
                "Could not fetch startup IG balance — drawdown may read 100%% until first candle"
            )

    def _maybe_start_ig_candle_feed(self) -> None:
        """IG-native LS candle feed for the symbols in IG_NATIVE_CANDLE_SYMBOLS
        (the two metals since 2026-06-19;
        "Metals IG-native cutover").  Runs whenever there are IG-native symbols
        — INCLUDING under candle_exchange='eodhd'.  It used to be skipped under
        EODHD because its delete-first REST backfill burned the IG
        historical-data allowance; that backfill is now gap-aware
        (IGCandleLSFeed._backfill_if_needed), so a buffered metal costs only a
        1-bar basis probe per restart.  Also runs under the twelvedata rollback
        path.  The feed self-no-ops if the set is empty."""
        ctx = self._ctx
        from bot.data.ig_candle_aggregator import IG_NATIVE_CANDLE_SYMBOLS

        if ctx.config.candle_exchange != "eodhd" or IG_NATIVE_CANDLE_SYMBOLS:
            from bot.data.ig_candle_feed import IGCandleLSFeed

            ctx.ig_candle_feed = IGCandleLSFeed(
                ctx.ig_client,
                ctx.store,
                ctx.event_bus,
                ctx.config,
                candle_db=ctx.candle_db,
                spread_monitor=ctx.spread_monitor,
            )
            ig_candle_task = asyncio.create_task(ctx.ig_candle_feed.run(), name="ig_candle_feed")
            ctx.tasks.append(ig_candle_task)
            if IG_NATIVE_CANDLE_SYMBOLS:
                logger.info(
                    "IGCandleLSFeed started: %d symbols (%s)",
                    len(IG_NATIVE_CANDLE_SYMBOLS),
                    ", ".join(sorted(IG_NATIVE_CANDLE_SYMBOLS)),
                )
            else:
                logger.info(
                    "IGCandleLSFeed wired but dormant (no symbols in "
                    "IG_NATIVE_CANDLE_SYMBOLS — D3 cutover activates)"
                )

    def _maybe_start_fred(self) -> None:
        """Optional FRED macro overlay — daily pulls into macro_features.
        Pure data collection in this phase; strategy integration is deferred."""
        ctx = self._ctx
        if ctx.config.fred_api_key and ctx.candle_db is not None:
            from bot.macro.fred import FREDClient

            ctx.fred_client = FREDClient(ctx.config.fred_api_key, ctx.candle_db)
            fred_task = asyncio.create_task(ctx.fred_client.run(), name="fred_macro")
            ctx.tasks.append(fred_task)
            logger.info("FREDClient started: macro overlay enabled")

    def _maybe_start_sentiment(self) -> None:
        """Optional sentiment overlay."""
        ctx = self._ctx
        if ctx.config.sentiment_enabled:
            from bot.sentiment.config import SentimentConfig
            from bot.sentiment.engine import SentimentEngine

            sentiment_cfg = SentimentConfig.from_bot_config(ctx.config)
            ctx.sentiment_engine = SentimentEngine(sentiment_cfg)
            ctx.sentiment_engine.set_rerank_busy_callback(lambda: ctx.rerank_in_progress)
            sentiment_task = asyncio.create_task(
                ctx.sentiment_engine.run(ctx.candle_symbols), name="sentiment"
            )
            ctx.tasks.append(sentiment_task)
            logger.info(
                "SentimentEngine started: scan_interval=%dm gate=%s",
                ctx.config.sentiment_scan_interval_minutes,
                ctx.config.sentiment_gate_enabled,
            )

    async def _scale_drift_loop(self) -> None:
        """Daily: check IG-quote-vs-candle scale drift for ETF-proxied symbols.

        Catches silent mis-scaling between a symbol's candle feed and its IG
        quote, which otherwise produces mis-stated P&L and ghost take-profits.
        Detection + alert only — no trading-state mutation.  Runs once
        shortly after startup, then every 24 h.
        """
        # Small initial delay so the candle store has been warmed by the
        # feeds before the first check (avoids spurious "no candle" skips).
        try:
            await asyncio.wait_for(self._ctx.shutdown_event.wait(), timeout=300.0)
            return  # shutdown came first
        except TimeoutError:
            pass
        while not self._ctx.shutdown_event.is_set():
            try:
                await self._run_scale_drift_check()
            except Exception:
                logger.exception("scale_drift check error")
            try:
                await asyncio.wait_for(self._ctx.shutdown_event.wait(), timeout=86_400.0)
                break
            except TimeoutError:
                pass

    async def _run_scale_drift_check(self) -> None:
        """Compare configured scale to live IG-vs-candle ratio for every
        explicitly-scaled symbol (the ``IG_SCALED_SYMBOLS`` set — forex
        pairs use stable defaults and can't drift)."""
        from bot.execution.ig_convert import safe_float
        from bot.execution.ig_quote_scale import IG_SCALED_SYMBOLS
        from bot.risk.scale_guard import DriftSeverity, compute_drift

        ctx = self._ctx
        checked = 0
        for symbol in IG_SCALED_SYMBOLS:
            epic = ctx.candle_epic_map.get(symbol)
            if epic is None:
                continue
            latest = ctx.store.get_latest_candle(symbol)
            if latest is None:
                continue
            try:
                details = await ctx.ig_client.fetch_market_details(epic)
                snap = details.get("snapshot", {})
                bid = safe_float(snap.get("bid"))
                offer = safe_float(snap.get("offer"))
            except Exception:
                logger.exception("scale_drift: market-details fetch failed for %s", symbol)
                continue
            if bid <= 0 or offer <= 0:
                continue
            ig_mid = (bid + offer) / 2.0
            result = compute_drift(symbol, latest.close, ig_mid)
            if result is None:
                continue
            checked += 1
            if result.severity is DriftSeverity.CRITICAL:
                logger.critical(
                    "SCALE DRIFT %s: %.1f%% — candle=%.4f × cfg_scale=%.2f ≠ IG_mid=%.2f "
                    "(real_scale=%.2f). P&L on any %s position is mis-stated by ~%.0f%%. "
                    "See docs/ig_native_candle_feed.md.",
                    symbol,
                    result.drift * 100,
                    result.candle_price,
                    result.expected_scale,
                    result.ig_mid,
                    result.real_scale,
                    symbol,
                    result.implied_pnl_error * 100,
                )
                try:
                    await ctx.alerter.send_error(
                        f"SCALE DRIFT {symbol}: {result.drift:+.1%} "
                        f"(real_scale={result.real_scale:.1f} vs cfg={result.expected_scale:.1f}). "
                        f"P&L mis-stated ~{result.implied_pnl_error:.0%}. Review before trusting "
                        f"{symbol} P&L; see docs/ig_native_candle_feed.md."
                    )
                except Exception:
                    logger.exception("scale_drift alert failed for %s", symbol)
            elif result.severity is DriftSeverity.WARN:
                logger.warning(
                    "scale drift %s: %.1f%% (real_scale=%.2f vs cfg=%.2f) — monitoring",
                    symbol,
                    result.drift * 100,
                    result.real_scale,
                    result.expected_scale,
                )
        logger.info("scale_drift check complete: %d symbols evaluated", checked)

    async def shutdown(self) -> None:
        """Graceful shutdown with bounded per-step timeouts.

        Sequencing:
          1. Signal every loop to stop (shutdown_event)
          2. Cancel background tasks and wait for them to finish their
             ``finally`` blocks — bounded ``SHUTDOWN_TIMEOUT``
          3. Snapshot live state to disk (after tasks finish so no concurrent
             writers race against the snapshot)
          4. Close the SQLite connection
          5. Close broker + feed connections — each bounded 5 s
          6. Send the Telegram shutdown alert — bounded 5 s

        Total worst-case wall-time is bounded; the unit file's
        ``TimeoutStopSec`` should be at least 60 s.
        """
        ctx = self._ctx
        ctx.shutdown_event.set()

        # Cancel tasks BEFORE persisting state — background loops (rerank,
        # sentiment, FRED) should finish their in-flight work or hit
        # their CancelledError handlers before we snapshot bot_state.json.
        for task in ctx.tasks:
            task.cancel()
        if ctx.tasks:
            done, pending = await asyncio.wait(ctx.tasks, timeout=SHUTDOWN_TIMEOUT)
            if pending:
                logger.warning("%d tasks did not finish within %ss", len(pending), SHUTDOWN_TIMEOUT)

        # Persist live state
        ctx.state.risk = ctx.risk_manager.get_state()
        if ctx.tp_manager is not None:
            ctx.state.take_profit_state = ctx.tp_manager.snapshot()
        if ctx.topk_scanned:
            ctx.state.topk_state = TopkState(
                selected=list(ctx.topk_selected),
                signals=[s.to_persist() for s in ctx.topk_signals],
                scanned_at=int(time.time() * 1000),
            )
        ctx.state.last_heartbeat = int(time.time() * 1000)
        logger.info("Saving state to disk...")
        if ctx.state_manager.save(ctx.state):
            logger.info("State saved successfully")
        else:
            logger.error("State save FAILED during shutdown — bot_state.json may be stale")

        ctx.candle_db.close()

        # Close broker + feed connections.  Each await is bounded so a single
        # stalled HTTP endpoint (Telegram, IG, Twelve Data) cannot push the
        # shutdown past systemd's TimeoutStopSec.
        async def _bounded_close(name: str, coro: Any, deadline_s: float = 5.0) -> None:
            try:
                await asyncio.wait_for(coro, timeout=deadline_s)
            except TimeoutError:
                logger.warning("Shutdown: %s close timed out after %.0fs", name, deadline_s)
            except Exception:
                logger.exception("Shutdown: %s close raised", name)

        if ctx.ig_client is not None:
            if ctx.ig_feed is not None:
                await _bounded_close("ig_feed", ctx.ig_feed.close())
            if ctx.twelve_data_feed is not None:
                await _bounded_close("twelve_data_feed", ctx.twelve_data_feed.close())
            if ctx.ig_candle_feed is not None:
                await _bounded_close("ig_candle_feed", ctx.ig_candle_feed.close())
            if ctx.fred_client is not None:
                await _bounded_close("fred_client", ctx.fred_client.close())
            await _bounded_close("ig_client", ctx.ig_client.close(), deadline_s=10.0)
            logger.info("IG connection closed")

        await _bounded_close("telegram", ctx.alerter.send_shutdown("Graceful shutdown"))
