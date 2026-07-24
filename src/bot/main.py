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
from pathlib import Path
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
from bot.monitoring.health import HealthMonitor
from bot.monitoring.telegram_alerts import TelegramAlerter
from bot.risk.risk_config import RiskConfig
from bot.risk.risk_manager import RiskManager
from bot.state.rerank_status import RerankStatusWriter
from bot.state.state_manager import StateManager
from bot.strategy import kronos_progress
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

    def __init__(self, pidfile: str | Path = PID_FILE) -> None:
        self.pidfile = Path(pidfile)
        self._locked = False

    def acquire(self) -> bool:
        """Return True if the lock was acquired, False if another instance is running."""
        if self.pidfile.exists():
            try:
                old_pid = int(self.pidfile.read_text().strip())
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
                    self.pidfile.unlink()
            except Exception as exc:
                logger.warning("Error reading PID file: %s -- removing it", exc)
                import contextlib

                with contextlib.suppress(FileNotFoundError):
                    self.pidfile.unlink()

        try:
            self.pidfile.write_text(str(os.getpid()))
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
            file_pid = int(self.pidfile.read_text().strip())
            if file_pid == os.getpid():
                self.pidfile.unlink()
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
        kronos_progress.set_progress_callback(self._on_kronos_progress)

        # Lifecycle (broker init + start/shutdown sequencing) is the
        # composition root's delegate; it owns _scale_drift_loop directly.
        self.ctx.lifecycle = Lifecycle(self)
        self.ctx.lifecycle.init_ig()

    async def start(self) -> None:
        """Public lifecycle entry — delegates to ``Lifecycle.start``."""
        await self.ctx.lifecycle.start()

    async def shutdown(self) -> None:
        """Public lifecycle exit — delegates to ``Lifecycle.shutdown``."""
        await self.ctx.lifecycle.shutdown()

    def _on_kronos_progress(self, batch_index: int, snapshot: dict[str, Any] | None) -> None:
        """Callback registered with ``kronos_progress``.

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
