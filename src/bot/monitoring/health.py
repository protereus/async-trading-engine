"""Heartbeat logging + external uptime ping.

Extracted from ``main.py``.  ``HealthMonitor`` runs once per minute
inside ``TradingBot.start()``:

* refresh IG balance (so cached equity / cash / open-pnl track within 60 s
  rather than the hourly rerank cadence)
* compose and log the HEARTBEAT line (candle counts, memory, drawdown
  tier, P&L summary, topk selection)
* snapshot ``bot_state.json`` atomically via ``StateManager.save``
* ping an external uptime watcher (``healthcheck_url``) if configured

Operates on the shared ``BotContext`` — the explicit collaborator seam
(same pattern as ``IGCloseManager`` / ``RerankRunner``).
"""

from __future__ import annotations

import asyncio
import logging
import resource
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import aiohttp

from bot.data.ig_candle_aggregator import IG_NATIVE_CANDLE_SYMBOLS
from bot.trading_hours import is_market_open

if TYPE_CHECKING:
    from bot.core.bot_context import BotContext

logger = logging.getLogger(__name__)

# A candle feed is considered stale if its freshest candle (across the symbols
# whose market is currently open) is older than this.  Hourly bars confirm at
# each :00 boundary, so a healthy feed's latest candle is always < 1h old; 3h
# tolerates a single missed boundary while still catching a silently-dropped
# WebSocket within ~3h (the 2026-06-23 EODHD stall ran ~20h undetected because
# the heartbeat's ``connected`` flag tracks the IG REST session, not the feeds).
_FEED_STALENESS_MS = 3 * 3_600 * 1_000


class HealthMonitor:
    """Heartbeat loop + external uptime-watcher ping."""

    def __init__(self, ctx: BotContext) -> None:
        self._ctx = ctx
        # Feed names currently flagged stale — used to alert once on the
        # healthy→stale transition (and again on recovery) rather than every
        # heartbeat.
        self._stale_feeds: set[str] = set()

    async def _check_feed_staleness(self, now_ms: int) -> None:
        """Alert if a candle feed has gone silent while its market is open.

        The bot runs two independent feeds that write to the same store: EODHD
        (12 FX + 14 US shares) and the IG-native Lightstreamer feed (XAU/XAG).
        A silent EODHD WS death (socket "connected" but no ticks flowing) froze
        every FX/share candle for ~20h on 2026-06-23 without tripping the
        heartbeat's ``connected`` flag — that flag tracks the IG REST session,
        not the data feeds.

        Each feed is checked via the *freshest* candle among its currently-open
        symbols: a liquid FX pair is always fresh when EODHD is healthy, so a
        just-opened (still-cold) US share doesn't false-positive at the session
        boundary, and a healthy metals feed can't mask a dead EODHD feed. A feed
        with no open symbols (e.g. weekend) is skipped — silence is expected.
        """
        ctx = self._ctx
        groups = {
            "EODHD (FX + US shares)": [
                s for s in ctx.candle_symbols if s not in IG_NATIVE_CANDLE_SYMBOLS
            ],
            "IG-native metals (XAU/XAG)": [
                s for s in ctx.candle_symbols if s in IG_NATIVE_CANDLE_SYMBOLS
            ],
        }
        for feed_name, syms in groups.items():
            open_syms = [s for s in syms if is_market_open(s)]
            if not open_syms:
                continue  # market closed for the whole group — silence is expected
            freshest = 0
            for s in open_syms:
                candle = ctx.store.get_latest_candle(s)
                if candle is not None and candle.timestamp > freshest:
                    freshest = candle.timestamp
            if freshest == 0:
                continue  # no candles yet (cold start) — not a stall
            age_ms = now_ms - freshest
            if age_ms > _FEED_STALENESS_MS:
                age_h = age_ms / 3_600_000
                logger.warning(
                    "FEED STALE: %s — freshest candle %.1fh old across %d open symbol(s)",
                    feed_name,
                    age_h,
                    len(open_syms),
                )
                if feed_name not in self._stale_feeds:
                    self._stale_feeds.add(feed_name)
                    try:
                        await ctx.alerter.send_error(
                            f"⚠️ {feed_name} candle feed stale — no new candles for "
                            f"{age_h:.1f}h while market open ({len(open_syms)} symbols). "
                            "Feed likely dropped silently; a bot restart re-establishes it "
                            "(EODHD self-backfills the gap)."
                        )
                    except Exception:
                        logger.exception("Feed-staleness alert failed for %s", feed_name)
            elif feed_name in self._stale_feeds:
                self._stale_feeds.discard(feed_name)
                # Log the recovery transition too — the stale state logs a
                # WARNING every heartbeat, so without this the journal shows a
                # staleness window that "never ends", leaving a postmortem
                # unable to see when the feed actually came back.
                logger.info(
                    "FEED RECOVERED: %s — fresh candle %.1fh old, back under the %.0fh threshold",
                    feed_name,
                    age_ms / 3_600_000,
                    _FEED_STALENESS_MS / 3_600_000,
                )
                try:
                    await ctx.alerter.send_error(
                        f"✅ {feed_name} candle feed recovered — fresh candles flowing again."
                    )
                except Exception:
                    logger.exception("Feed-recovery alert failed for %s", feed_name)

    async def ping_healthcheck(self) -> None:
        url = self._ctx.config.healthcheck_url
        if not url:
            return
        try:
            async with aiohttp.ClientSession() as session:
                await session.get(url, timeout=aiohttp.ClientTimeout(total=10))
        except Exception:
            logger.debug("Healthcheck ping failed", exc_info=True)

    async def _refresh_ig_state(self, now_ms: int) -> None:
        """Fetch balance and position prices from IG and update local state.

        Refresh IG balance once per heartbeat so cached equity/cash/open_pnl
        track IG within 60s rather than the hourly rerank cadence. Failures
        are non-fatal — we fall back to the last known values.

        Refresh each open position's live IG price (market bid, an IG
        level) so the dashboard shows a fresh per-position P&L at
        heartbeat cadence (~60s) instead of the last closed hourly
        candle. fetch_positions is keyed by IG epic → map to bot symbol.
        """
        ctx = self._ctx
        if ctx.ig_client is None:
            return

        try:
            await ctx.refresh_balance()
        except Exception:
            logger.debug("Heartbeat: IG balance refresh failed", exc_info=True)

        if ctx.state.positions:
            try:
                _live = {p.symbol: p for p in await ctx.ig_client.fetch_positions()}
                for _sym, _pos in list(ctx.state.positions.items()):
                    _lp = _live.get(ctx.epic_for(_sym))
                    if _lp is not None and _lp.current_price > 0:
                        ctx.state.positions[_sym] = replace(
                            _pos, current_price=_lp.current_price, updated_at=now_ms
                        )
            except Exception:
                logger.debug("Heartbeat: IG positions refresh failed", exc_info=True)

    def _snapshot_state(self, now_ms: int, risk: dict[str, Any]) -> None:
        """Snapshot live state to disk via StateManager.

        Used by the read-only webgui dashboard and for faster crash recovery.
        The atomic-rename pattern means concurrent readers either see the
        previous or the new file, never a torn one. Failures are logged
        inside StateManager.save; this call never raises.

        Snapshots the full RiskState too so risk_budgets, trade_results
        and trading_halted survive an ungraceful exit (kill -9, OOM,
        systemd SIGKILL after stop-timeout).
        """
        ctx = self._ctx
        ctx.state.last_heartbeat = now_ms
        ctx.state.equity = ctx.risk_manager.equity
        ctx.state.peak_equity = max(ctx.state.peak_equity, ctx.risk_manager.peak_equity)
        ctx.state.pnl_24h = float(risk["pnl_24h"])
        ctx.state.risk = ctx.risk_manager.get_state()
        ctx.state_manager.save(ctx.state)

    async def health_check(self) -> None:
        """Log heartbeat every 60 seconds."""
        ctx = self._ctx
        while not ctx.shutdown_event.is_set():
            try:
                await asyncio.wait_for(asyncio.shield(ctx.shutdown_event.wait()), timeout=60.0)
                break
            except TimeoutError:
                pass

            now_ms = int(time.time() * 1000)
            uptime_s = (now_ms - ctx.state.bot_started_at) // 1000
            candle_counts = {sym: ctx.store.get_candle_count(sym) for sym in ctx.candle_symbols}

            # Alert if a candle feed has silently stalled (connected-but-no-ticks)
            # while its market is open — the ``connected`` flag below only tracks
            # the IG REST session, not the EODHD/IG-native data feeds.
            await self._check_feed_staleness(now_ms)

            mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

            # Refresh IG balance and positions
            await self._refresh_ig_state(now_ms)

            risk = ctx.risk_manager.get_risk_summary()
            open_orders = len(ctx.ig_deal_ids)
            connected = bool(ctx.ig_client and ctx.ig_client.is_connected)

            topk_selected_str = ", ".join(ctx.topk_selected) if ctx.topk_selected else "none"

            logger.info(
                "HEARTBEAT | uptime=%ds | candles=%s | mem=%.1fMB | connected=%s | "
                "open_orders=%d | drawdown=%.1f%% | tier=%s | pnl_24h=%.2f | halted=%s | "
                "topk_selected=%s",
                uptime_s,
                candle_counts,
                mem_mb,
                connected,
                open_orders,
                risk["current_drawdown_pct"] * 100,
                risk["drawdown_tier"].upper(),
                risk["pnl_24h"],
                risk["trading_halted"],
                topk_selected_str,
            )

            self._snapshot_state(now_ms, risk)
            await self.ping_healthcheck()
