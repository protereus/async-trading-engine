"""Entry point for the trading bot.

Initialises the async event loop, wires up all modules, and manages
graceful shutdown on SIGINT / SIGTERM.

IG Group (``BROKER=ig``) is the only supported broker: IGClient + IGFeed /
EODHD / Twelve Data feeds + Kronos TopK strategy.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from typing import Any

from bot.config import BotConfig
from bot.core.bot_context import BotContext
from bot.core.event_bus import (
    EventBus,
)
from bot.core.event_wiring import EventWiring
from bot.core.lifecycle import Lifecycle
from bot.core.models import (
    BotState,
)
from bot.data.candle_db import CandleDB
from bot.data.store import DataStore
from bot.execution.ig_close import IGCloseManager
from bot.execution.ig_convert import (
    safe_float,
)
from bot.monitoring.health import HealthMonitor
from bot.monitoring.telegram_alerts import TelegramAlerter
from bot.risk.risk_config import RiskConfig
from bot.risk.risk_manager import RiskManager
from bot.state.rerank_status import RerankStatusWriter
from bot.state.state_manager import StateManager
from bot.strategy import _kronos_progress
from bot.strategy.rerank_runner import RerankRunner
from bot.trading_hours import in_equity_mark_blackout

logger = logging.getLogger(__name__)

PID_FILE = "/tmp/trading-bot.pid"

# IG enforces percentage-based minimum stop distances on volatile instruments.
# Verified via GET /markets/{epic} dealingRules.minNormalStopOrLimitDistance.
# All other EPICs in our universe use POINTS-based minimums (trivially small).
SHUTDOWN_TIMEOUT = 10.0  # seconds to wait for tasks to cancel


# ---------------------------------------------------------------------------
# PID lock
# ---------------------------------------------------------------------------


class PIDLockManager:
    """Prevents duplicate bot instances by writing a PID file."""

    def __init__(self, pidfile: str = PID_FILE) -> None:
        self.pidfile = pidfile
        self._locked = False

    def acquire(self) -> bool:
        """Return True if the lock was acquired, False if another instance is running."""
        if os.path.exists(self.pidfile):
            try:
                with open(self.pidfile) as f:
                    old_pid = int(f.read().strip())
                try:
                    os.kill(old_pid, 0)
                    logger.error(
                        "Another bot instance is already running (PID %d). "
                        "Kill it or remove %s to force start.",
                        old_pid,
                        self.pidfile,
                    )
                    return False
                except OSError:
                    logger.warning("Removing stale PID file (PID %d not running)", old_pid)
                    os.remove(self.pidfile)
            except Exception as exc:
                logger.warning("Error reading PID file: %s -- removing it", exc)
                import contextlib

                with contextlib.suppress(FileNotFoundError):
                    os.remove(self.pidfile)

        try:
            with open(self.pidfile, "w") as f:
                f.write(str(os.getpid()))
            self._locked = True
            logger.info("PID lock acquired: %s (PID %d)", self.pidfile, os.getpid())
            return True
        except OSError as exc:
            logger.error("Failed to write PID file %s: %s", self.pidfile, exc)
            return False

    def release(self) -> None:
        """Remove the PID file if we own it."""
        if not self._locked:
            return
        try:
            with open(self.pidfile) as f:
                file_pid = int(f.read().strip())
            if file_pid == os.getpid():
                os.remove(self.pidfile)
                logger.info("PID lock released: %s", self.pidfile)
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("Error releasing PID lock: %s", exc)
        self._locked = False


# ---------------------------------------------------------------------------
# TradingBot
# ---------------------------------------------------------------------------


class TradingBot:
    """Top-level orchestrator for all bot subsystems.

    IG is the sole supported broker (``config.broker`` must be ``"ig"``); Kronos
    TopK is the sole signal source.  The legacy ccxt/OKX execution stack was
    archived 2026-06-24.
    """

    def __init__(self, config: BotConfig) -> None:
        if config.broker != "ig":
            raise ValueError(f"Unsupported broker={config.broker!r}; only 'ig' is supported")

        self.ctx = self._build_context(config)
        self._init_collaborators()

    def _resolve_symbols(self, config: BotConfig) -> tuple[list[str], dict[str, str]]:
        # Symbol routing: candle symbols (what the feed streams, used as DB/strategy key)
        # and the map to IG EPICs (needed only at order-placement time).
        # When candle_exchange="ig" EPICs ARE the candle symbols (identity mapping).
        if config.candle_exchange in ("twelvedata", "eodhd"):
            # Both source the universe from a {bot_key: IG_EPIC} map. EODHD is the
            # post-migration single-vendor feed;
            # CANDLE_EXCHANGE flips between them with no other change.
            if config.candle_exchange == "eodhd":
                from bot.data.eodhd_symbols import SYMBOL_EPIC_MAP as _MAP
            else:
                from bot.data.twelve_data_feed import SYMBOL_EPIC_MAP as _MAP

            return list(_MAP.keys()), dict(_MAP)
        elif config.candle_exchange != "ig":
            return list(config.candle_epic_map.keys()), dict(config.candle_epic_map)
        else:
            return list(config.ig_epics), {epic: epic for epic in config.ig_epics}

    def _build_context(self, config: BotConfig) -> BotContext:
        event_bus = EventBus()
        candle_db = CandleDB()
        candle_db.init_db()
        risk_config = RiskConfig()
        # Freeze the drawdown breaker during the daily IG rollover/maintenance
        # window where account marks are unreliable (2026-06-05 incident).
        _blackout_fn = in_equity_mark_blackout if risk_config.drawdown_maintenance_guard else None

        candle_symbols, candle_epic_map = self._resolve_symbols(config)

        # All shared runtime state lives on the context — collaborators
        # receive it instead of the bot, so the seam contract is explicit
        # (see bot_context.py for the staged-init story).
        return BotContext(
            config=config,
            risk_config=risk_config,
            event_bus=event_bus,
            store=DataStore(buffer_size=config.candle_buffer_size),
            candle_db=candle_db,
            risk_manager=RiskManager(risk_config, event_bus, in_blackout_fn=_blackout_fn),
            state_manager=StateManager(),
            state=BotState(),
            alerter=TelegramAlerter(config.telegram_bot_token, config.telegram_chat_id),
            rerank_status=RerankStatusWriter(),
            candle_symbols=candle_symbols,
            candle_epic_map=candle_epic_map,
            # Cached inverse (IG EPIC -> candle symbol) built once so the
            # mapping has a single source; collaborators translate through
            # ctx.epic_for / ctx.candle_for.
            epic_to_candle={epic: sym for sym, epic in candle_epic_map.items()},
            # Serialises TopK entries: the risk gate reads shared position
            # state that isn't registered until after the order's network
            # round-trips, so concurrent hour-boundary candle handlers could
            # otherwise each pass the gate on a stale snapshot and overshoot
            # the caps. See RerankRunner.
            entry_lock=asyncio.Lock(),
            shutdown_event=asyncio.Event(),
            tasks=[],
            started_at=int(time.time() * 1000),
        )

    def _init_collaborators(self) -> None:
        # Collaborators (§1 steps 2-6 — see each class).  Constructed against
        # the context and wired onto it so they can reach each other through
        # the same explicit seam.
        self.ctx.health = HealthMonitor(self.ctx)
        self.ctx.closer = IGCloseManager(self.ctx)
        self.ctx.runner = RerankRunner(self.ctx)
        self.ctx.events = EventWiring(self.ctx)
        _kronos_progress.set_progress_callback(self._on_kronos_progress)

        # Lifecycle (broker init + start/shutdown sequencing) gets the bot
        # itself: it is the composition root's delegate and also owns
        # starting bot-level loops like _scale_drift_loop.
        self.ctx.lifecycle = Lifecycle(self)
        self.ctx.lifecycle.init_ig()

    async def start(self) -> None:
        """Public lifecycle entry — delegates to ``Lifecycle.start``."""
        await self.ctx.lifecycle.start()

    async def shutdown(self) -> None:
        """Public lifecycle exit — delegates to ``Lifecycle.shutdown``."""
        await self.ctx.lifecycle.shutdown()

    def _on_kronos_progress(self, batch_index: int, snapshot: dict[str, Any] | None) -> None:
        """Callback registered with ``_kronos_progress``.

        Fires on every throttled tqdm emission inside Kronos's
        ``auto_regressive_inference`` (verbose Pass-1 calls, with an inner-bar
        ``snapshot``) and once per silent Pass-2 variance call (``snapshot=None``
        via ``bump_batch``).  Forwarding both keeps the dashboard's overall N/total
        bar advancing across all ~42 ``predict_batch`` calls; ``current_batch=None``
        simply hides the inner 0/120 bar during the variance pass.
        """
        self.ctx.rerank_status.update(
            batches_done=max(0, batch_index - 1),
            current_batch=snapshot,
        )

    async def _scale_drift_loop(self) -> None:
        """Daily: check IG-quote-vs-candle scale drift for ETF-proxied symbols.

        D4 of .  Catches the silent
        mis-scaling that produced the 2026-05-28 USO/UNG P&L errors and
        ghost take-profits.  Detection + alert only — no trading-state
        mutation.  Runs once shortly after startup, then every 24 h.
        """
        # Small initial delay so the candle store has been warmed by the
        # feeds before the first check (avoids spurious "no candle" skips).
        try:
            await asyncio.wait_for(self.ctx.shutdown_event.wait(), timeout=300.0)
            return  # shutdown came first
        except TimeoutError:
            pass
        while not self.ctx.shutdown_event.is_set():
            try:
                await self._run_scale_drift_check()
            except Exception:
                logger.exception("scale_drift check error")
            try:
                await asyncio.wait_for(self.ctx.shutdown_event.wait(), timeout=86_400.0)
                break
            except TimeoutError:
                pass

    async def _run_scale_drift_check(self) -> None:
        """Compare configured scale to live IG-vs-candle ratio for every
        explicitly-scaled symbol (the ``IG_SCALED_SYMBOLS`` set — forex
        pairs use stable defaults and can't drift)."""
        from bot.execution.ig_quote_scale import IG_SCALED_SYMBOLS
        from bot.risk.scale_guard import DriftSeverity, compute_drift

        checked = 0
        for symbol in IG_SCALED_SYMBOLS:
            epic = self.ctx.candle_epic_map.get(symbol)
            if epic is None:
                continue
            latest = self.ctx.store.get_latest_candle(symbol)
            if latest is None:
                continue
            try:
                details = await self.ctx.ig_client.fetch_market_details(epic)
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
                    "(real_scale=%.2f). P&L on any %s position is mis-stated by ~%.0f%%.",
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
                    await self.ctx.alerter.send_error(
                        f"SCALE DRIFT {symbol}: {result.drift:+.1%} "
                        f"(real_scale={result.real_scale:.1f} vs cfg={result.expected_scale:.1f}). "
                        f"P&L mis-stated ~{result.implied_pnl_error:.0%}. Review before trusting "
                        f"{symbol} P&L."
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

    # ------------------------------------------------------------------
    # Strategy loop
    # ------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    config = BotConfig()
    config.validate_config()

    bot_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logging.getLogger("bot").setLevel(bot_level)

    pid_lock = PIDLockManager()
    if not pid_lock.acquire():
        sys.exit(1)

    bot = TradingBot(config)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _signal_handler(sig: int) -> None:
        logger.info("Received signal %d -- initiating shutdown", sig)
        bot.ctx.shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler, sig)

    try:
        loop.run_until_complete(bot.start())
    except Exception:
        logger.exception("Unhandled exception in main loop")
    finally:
        try:
            loop.run_until_complete(bot.shutdown())
        except Exception:
            logger.exception("Error during shutdown")
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        pid_lock.release()
        logger.info("Bot shutdown complete")
        # Force-exit to bypass non-daemon threads (torch, Kronos worker pools,
        # tokenizer background threads) that would otherwise keep the process
        # alive until systemd SIGKILL.  Everything Python-level is done at
        # this point — state file written, DB closed, IG session destroyed.
        # See 2026-05-19 shutdown audit: the bot consistently logged
        # "Bot shutdown complete" within 1.5 s but the process lingered for
        # the full systemd timeout because non-daemon threads kept the
        # interpreter alive.
        os._exit(0)


if __name__ == "__main__":
    main()
