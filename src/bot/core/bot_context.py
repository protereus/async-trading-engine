"""Shared runtime state for TradingBot's collaborators.

``BotContext`` is the explicit contract between ``TradingBot`` and the
collaborators that used to reach into ``bot._*`` private attributes
(``RerankRunner``, ``IGCloseManager``, ``HealthMonitor``, ``EventWiring``,
``Lifecycle``).  Everything a collaborator may read or mutate lives here as a
public, documented field; ``TradingBot`` itself keeps only what is genuinely
its own (the asyncio entrypoints and the Kronos progress callback).

Staged initialisation
---------------------
The bot cannot construct everything up front: the broker client, strategy,
feeds and overlay engines are built by ``Lifecycle.init_ig()`` / ``start()``
*after* the collaborators exist.  The context therefore has three field
groups:

* **Early-bound** — constructed with the bot, always present.
* **Late-bound slots** (``field(init=False)``) — the collaborators
  themselves, wired by ``TradingBot.__init__`` immediately after the context
  is created.  Reading one before wiring raises ``AttributeError`` loudly.
* **Broker/feed objects** (``Any = None``) — populated by ``Lifecycle``;
  ``None`` is a legitimate runtime value for the optional subsystems
  (take-profit, sentiment, FRED, the per-exchange feeds), so consumers keep
  their existing ``is not None`` guards.  They stay ``Any``-typed exactly as
  they were on ``TradingBot``; tightening them to real Optionals is future
  work that belongs with adding the missing guards.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bot.config import BotConfig
    from bot.core.event_bus import EventBus
    from bot.core.event_wiring import EventWiring
    from bot.core.lifecycle import Lifecycle
    from bot.core.models import BotState
    from bot.data.candle_db import CandleDB
    from bot.data.store import DataStore
    from bot.execution.ig_close import IGCloseManager
    from bot.monitoring.health import HealthMonitor
    from bot.monitoring.telegram_alerts import TelegramAlerter
    from bot.risk.risk_config import RiskConfig
    from bot.risk.risk_manager import RiskManager
    from bot.state.rerank_status import RerankStatusWriter
    from bot.state.state_manager import StateManager
    from bot.strategy.rerank_runner import RerankRunner
    from bot.strategy.topk_strategy import AssetSignal


@dataclass
class BotContext:
    """Explicit shared state passed to every TradingBot collaborator."""

    # --- configuration (early-bound, read-only in spirit) ---
    config: BotConfig
    risk_config: RiskConfig

    # --- core services (early-bound) ---
    event_bus: EventBus
    store: DataStore
    candle_db: CandleDB
    risk_manager: RiskManager
    state_manager: StateManager
    state: BotState
    alerter: TelegramAlerter
    rerank_status: RerankStatusWriter

    # --- symbol routing (early-bound; see epic_for/candle_for) ---
    candle_symbols: list[str]
    candle_epic_map: dict[str, str]
    epic_to_candle: dict[str, str]

    # --- run-loop state (early-bound) ---
    entry_lock: asyncio.Lock
    shutdown_event: asyncio.Event
    tasks: list[asyncio.Task[None]]
    started_at: int

    # --- collaborators (late-bound slots; wired by TradingBot.__init__) ---
    health: HealthMonitor = field(init=False)
    closer: IGCloseManager = field(init=False)
    runner: RerankRunner = field(init=False)
    events: EventWiring = field(init=False)
    lifecycle: Lifecycle = field(init=False)

    # --- broker / strategy / feed objects (populated by Lifecycle) ---
    ig_client: Any = None  # IGClient — always set by Lifecycle.init_ig()
    # Spread-anomaly detector — always set by Lifecycle.init_ig() so the
    # pre-trade gate runs regardless of which candle feed is live (EODHD,
    # IG-native metals, or the legacy all-IG path).
    spread_monitor: Any = None
    ig_feed: Any = None  # IGFeed — set when candle_exchange="ig"
    # IGCandleLSFeed — D1 IG-native candle aggregator (any IG path)
    ig_candle_feed: Any = None
    twelve_data_feed: Any = None  # TwelveDataFeed (candle_exchange="twelvedata")
    eodhd_feed: Any = None  # EODHDFeed (candle_exchange="eodhd")
    fred_client: Any = None  # FREDClient (when fred_api_key is set)
    topk_strategy: Any = None  # TopKStrategy — set by Lifecycle.start()
    sentiment_engine: Any = None  # SentimentEngine — set when sentiment_enabled=True
    tp_manager: Any = None  # TakeProfitManager — set when topk_enabled=True

    # --- live position / selection state (mutated at runtime) ---
    ig_deal_ids: dict[str, str] = field(default_factory=dict)  # candle_symbol → IG deal ID
    # IG dealIds we've already Telegram-alerted on as orphans (positions
    # present on IG but absent from state.positions). Reset on every restart —
    # a restart is the operator opting back into being notified about the
    # current set of orphans.
    alerted_orphan_deals: set[str] = field(default_factory=set)
    topk_selected: list[str] = field(default_factory=list)  # symbols approved for entry
    topk_signals: list[AssetSignal] = field(default_factory=list)  # last scan output
    topk_scanned: bool = False  # True after first successful rerank
    rerank_in_progress: bool = False  # True between rerank-loop entry and exit

    # ------------------------------------------------------------------
    # Symbol ↔ EPIC boundary
    # ------------------------------------------------------------------

    def epic_for(self, symbol: str) -> str:
        """IG EPIC for a candle symbol; identity when unmapped.

        The identity fallback keeps EPIC-keyed flows working when the
        universe is EPIC-native (``candle_exchange="ig"``) or when a caller
        already holds an EPIC.  This and :meth:`candle_for` are the only
        symbol↔EPIC seam — don't reach into the maps at call sites.
        """
        return self.candle_epic_map.get(symbol, symbol)

    def candle_for(self, epic: str) -> str:
        """Candle/bot symbol for an IG EPIC; identity when unmapped.

        Inverse of :meth:`epic_for` (built from the same map).
        """
        return self.epic_to_candle.get(epic, epic)

    # ------------------------------------------------------------------
    # Balance refresh
    # ------------------------------------------------------------------

    async def refresh_balance(self) -> dict[str, Any]:
        """Fetch IG balance and apply the equity/cash/open_pnl update triplet.

        Raises on fetch failure — callers keep their own try/except so each
        call site's existing failure-handling granularity (silent continue,
        return sentinel, debug vs warning log) is preserved rather than
        collapsed into one behaviour here.
        """
        balance: dict[str, Any] = await self.ig_client.fetch_balance()
        self.risk_manager.update_equity(balance["equity"])
        self.state.cash = balance["balance"]
        self.state.open_pnl = balance["open_pnl"]
        return balance

    # ------------------------------------------------------------------
    # Clock helpers
    # ------------------------------------------------------------------

    @staticmethod
    def mono_to_wall(mono_ts: float) -> float:
        """Convert a ``time.monotonic()`` value to a wall-clock seconds value.

        Used to expose the next-rerank deadline (carried as a monotonic value
        in the rerank loop) on ``rerank_status.json`` where the dashboard
        wants wall time.
        """
        return time.time() + (mono_ts - time.monotonic())
