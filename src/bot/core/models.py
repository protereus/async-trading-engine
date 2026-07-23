"""Shared frozen dataclasses and enums for the trading bot."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, TypedDict

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(StrEnum):
    PENDING = "pending"
    OPEN = "open"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class ErrorType(StrEnum):
    """Categorized error types for exchange operations."""

    # Transient errors (retryable)
    NETWORK_TIMEOUT = "network_timeout"
    RATE_LIMIT = "rate_limit"
    SERVICE_UNAVAILABLE = "service_unavailable"
    CONNECTION_ERROR = "connection_error"
    MARKET_CLOSED = "market_closed"

    # Permanent errors (non-retryable)
    AUTHENTICATION_FAILED = "authentication_failed"
    EXCHANGE_ERROR = "exchange_error"
    INVALID_ORDER = "invalid_order"
    UNKNOWN_ERROR = "unknown_error"

    @property
    def is_retryable(self) -> bool:
        return self in {
            ErrorType.NETWORK_TIMEOUT,
            ErrorType.RATE_LIMIT,
            ErrorType.SERVICE_UNAVAILABLE,
            ErrorType.CONNECTION_ERROR,
            ErrorType.MARKET_CLOSED,
        }


class RiskLevel(StrEnum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRITICAL = "critical"
    HALTED = "halted"


class DrawdownTier(StrEnum):
    NORMAL = "normal"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"

    @property
    def severity(self) -> int:
        """Ordinal severity (NORMAL=0 … RED=3) for directional comparisons.

        Used by the drawdown hysteresis to tell escalation (accept the raw
        tier immediately) from de-escalation (require a recovery band)."""
        return _DRAWDOWN_TIER_SEVERITY[self]


_DRAWDOWN_TIER_SEVERITY: dict[DrawdownTier, int] = {
    DrawdownTier.NORMAL: 0,
    DrawdownTier.YELLOW: 1,
    DrawdownTier.ORANGE: 2,
    DrawdownTier.RED: 3,
}


class MarginCircuitState(StrEnum):
    """Real-time margin-utilisation circuit-breaker state (IG_LIVE_RISK_REFERENCE.md §4.3).

    Ratios are ``equity / total_margin_required``.  Driven by LS ACCOUNT pushes
    plus on-tick recomputes for instruments with open positions.  Each state
    above NORMAL takes a more aggressive de-risking action; LIQUIDATION is the
    broker's own floor (0.50) — by the time we see it, the close-out has
    already started.  See IG_LIVE_RISK_REFERENCE.md §4.3.
    """

    NORMAL = "normal"
    HALT_ENTRIES = "halt_entries"  # ≤ 0.80 — refuse new orders, keep existing
    DEFENSIVE_CLOSE = "defensive_close"  # ≤ 0.65 — close worst performer
    EMERGENCY_FLATTEN = "emergency_flatten"  # ≤ 0.55 — flatten everything
    LIQUIDATION = "liquidation"  # ≤ 0.50 — broker is closing positions


# ---------------------------------------------------------------------------
# Exchange exceptions
# ---------------------------------------------------------------------------


class ExchangeError(Exception):
    """Raised when an exchange returns a non-retryable error.

    Lives in core so the execution layer (IGClient / IGHttp) and the strategy
    layer can raise and catch it without a cross-sibling import dependency.
    (Historically shared with a since-removed second broker path.)
    """

    def __init__(self, message: str, error_type: ErrorType) -> None:
        super().__init__(message)
        self.error_type = error_type


class MarketClosedError(ExchangeError):
    """Raised when the exchange rejects an order because the market is
    temporarily closed for edits/funding (e.g. IG ``MARKET_CLOSED_WITH_EDITS``
    during the 22:00 UTC daily reconciliation window).

    Callers should defer retry until the window ends — the position is still
    alive on the broker side, so a ghost-deal reconcile would mis-purge it.
    """


# ---------------------------------------------------------------------------
# IG-specific Literal type aliases (uppercase strings used by the IG REST API)
# ---------------------------------------------------------------------------

IGDirection = Literal["BUY", "SELL"]
IGOrderType = Literal["MARKET", "LIMIT"]

# ---------------------------------------------------------------------------
# Trading data models (frozen dataclasses for immutability)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candle:
    """A single OHLCV candle."""

    timestamp: int  # Unix ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    is_confirmed: bool  # False while the candle is still forming


@dataclass(frozen=True)
class IGOrderRequest:
    """Request to place an IG spread bet position.

    size is currency-per-point (e.g. £1/pt), not a quantity of asset.
    stop_distance and limit_distance are in points from the entry level.
    """

    epic: str
    direction: IGDirection
    size: float  # £ per point
    order_type: IGOrderType = "MARKET"
    stop_distance: float | None = None
    limit_distance: float | None = None
    guaranteed_stop: bool = False
    deal_reference: str = ""


@dataclass(frozen=True)
class OrderResult:
    """Result / state of a placed order."""

    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    status: OrderStatus
    requested_quantity: float
    filled_quantity: float
    average_price: float
    fee: float
    fee_currency: str
    timestamp: int
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Position:
    """An open trading position."""

    symbol: str
    side: OrderSide
    entry_price: float
    quantity: float
    current_price: float
    unrealised_pnl: float
    realised_pnl: float
    opened_at: int  # Unix ms
    updated_at: int  # Unix ms


# ---------------------------------------------------------------------------
# Risk models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskDecision:
    """Output of every risk check — returned by RiskManager.evaluate_ig_order()."""

    approved: bool
    original_quantity: float
    adjusted_quantity: float
    reason: str
    risk_level: RiskLevel


@dataclass
class RiskEvent:
    """A logged risk event (drawdown tier change, limit breach, halt, etc.)."""

    timestamp: int
    event_type: str  # e.g. "drawdown_yellow", "daily_limit_hit", "shutdown_triggered"
    details: dict[str, Any]
    resolved: bool = False
    resolved_at: int | None = None


@dataclass(frozen=True)
class PositionClosed:
    """Payload for ``EVENT_POSITION_CLOSED`` — realised P&L on a SELL fill."""

    symbol: str
    pnl: float


@dataclass(frozen=True)
class EquitySnapshot:
    """Point-in-time snapshot of portfolio performance."""

    timestamp: int
    equity: float
    peak_equity: float
    drawdown_pct: float
    daily_pnl: float
    open_position_count: int


@dataclass(frozen=True)
class AccountUpdate:
    """Normalised account snapshot from the IG Lightstreamer ACCOUNT channel.

    Used to drive the margin circuit breakers.  ``margin_required`` is the
    sum of margin held against open positions ("FUNDS" or "MARGIN" in IG's
    ACCOUNT fields, depending on subscription).  A value of 0 means no
    positions are open and the circuit-breaker ratio is undefined (treated
    as infinity / healthy).
    """

    timestamp: int
    equity: float
    margin_required: float
    available_to_deal: float
    unrealised_pnl: float


# De-risking action requested by a MarginBreakerEvent — the dispatch sites in
# event_wiring compare against these values, so the closed set lives in the
# type rather than a comment.
MarginAction = Literal["halt_entries", "close_worst", "flatten", "liquidation_alert"]


@dataclass(frozen=True)
class MarginBreakerEvent:
    """Emitted only on circuit-breaker *transitions* so action handlers don't
    loop on a steady high-utilisation state.  ``action`` tells main.py what
    to do; ``ratio`` and ``state`` are for logging / Telegram."""

    timestamp: int
    state: MarginCircuitState
    action: MarginAction
    ratio: float
    equity: float
    margin_required: float


# ---------------------------------------------------------------------------
# Persistent state
# ---------------------------------------------------------------------------


@dataclass
class RiskState:
    """Serialisable snapshot of risk manager internals — survives restarts."""

    peak_equity: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    consecutive_losses: int = 0
    trade_results: list[list[Any]] = field(default_factory=list)  # [[ts_ms, pnl], ...]
    trading_halted: bool = False
    halt_reason: str = ""
    risk_events: list[dict[str, Any]] = field(default_factory=list)  # last 100
    atr_values: dict[str, list[float]] = field(default_factory=dict)
    # Risk-on (stop-loss) budget per EPIC in GBP — set at fill time by main.py.
    # Used by evaluate_ig_order to enforce max_total_risk_pct.
    risk_budgets: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak_equity": self.peak_equity,
            "daily_pnl": self.daily_pnl,
            "weekly_pnl": self.weekly_pnl,
            "monthly_pnl": self.monthly_pnl,
            "consecutive_losses": self.consecutive_losses,
            "trade_results": self.trade_results,
            "trading_halted": self.trading_halted,
            "halt_reason": self.halt_reason,
            "risk_events": self.risk_events,
            "atr_values": self.atr_values,
            "risk_budgets": self.risk_budgets,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RiskState:
        return cls(
            peak_equity=d.get("peak_equity", 0.0),
            daily_pnl=d.get("daily_pnl", 0.0),
            weekly_pnl=d.get("weekly_pnl", 0.0),
            monthly_pnl=d.get("monthly_pnl", 0.0),
            consecutive_losses=d.get("consecutive_losses", 0),
            trade_results=d.get("trade_results", []),
            trading_halted=d.get("trading_halted", False),
            halt_reason=d.get("halt_reason", ""),
            risk_events=d.get("risk_events", []),
            atr_values=d.get("atr_values", {}),
            risk_budgets=d.get("risk_budgets", {}),
        )


class PersistedAssetSignal(TypedDict):
    """One TopK ``AssetSignal`` as persisted in ``BotState.topk_state``.

    Exactly the nine scalar fields — the per-draw diagnostic lists
    (``samples``, ``var_closes_at_horizons``) are rerank-scoped and dropped.
    ``AssetSignal.to_persist`` / ``from_persist`` are the only writer/reader,
    so the save and restore shapes cannot drift apart.
    """

    symbol: str
    mean_return: float
    std_return: float
    direction_confidence: float
    uncertainty: float
    stop_pct: float
    tradeable: bool
    predicted_close: float
    direction: Literal["LONG", "SHORT"]


class TopkState(TypedDict):
    """Snapshot of the last TopK scan, persisted across restarts.

    Restored by ``Lifecycle._restore_persisted_state`` only while fresh
    (≤ 4 h old, judged by ``scanned_at``) so a restart doesn't trade on a
    stale selection.
    """

    selected: list[str]
    signals: list[PersistedAssetSignal]
    scanned_at: int


@dataclass
class BotState:
    """Mutable bot state -- persisted to disk on shutdown."""

    positions: dict[str, Position] = field(default_factory=dict)
    open_orders: dict[str, OrderResult] = field(default_factory=dict)
    pending_signals: list[str] = field(default_factory=list)
    equity: float = 0.0
    peak_equity: float = 0.0
    cash: float = 0.0  # Closed-P&L cash balance (IG `balance` field). equity = cash + open_pnl.
    open_pnl: float = 0.0  # Unrealised P&L on open positions (IG `profitLoss` field).
    # Rolling 24-hour realised-P&L sum (NOT calendar-day P&L since 00:00 UTC).
    # Sourced from RiskManager._window_pnl(now, 24h) on every heartbeat.
    pnl_24h: float = 0.0
    last_candle_timestamps: dict[str, int] = field(default_factory=dict)
    bot_started_at: int = 0
    last_heartbeat: int = 0
    risk: RiskState = field(default_factory=RiskState)
    take_profit_state: dict[str, object] = field(default_factory=dict)
    topk_state: TopkState | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""

        def _pos(p: Position) -> dict[str, Any]:
            return {
                "symbol": p.symbol,
                "side": p.side.value,
                "entry_price": p.entry_price,
                "quantity": p.quantity,
                "current_price": p.current_price,
                "unrealised_pnl": p.unrealised_pnl,
                "realised_pnl": p.realised_pnl,
                "opened_at": p.opened_at,
                "updated_at": p.updated_at,
            }

        def _order(o: OrderResult) -> dict[str, Any]:
            return {
                "order_id": o.order_id,
                "client_order_id": o.client_order_id,
                "symbol": o.symbol,
                "side": o.side.value,
                "order_type": o.order_type.value,
                "status": o.status.value,
                "requested_quantity": o.requested_quantity,
                "filled_quantity": o.filled_quantity,
                "average_price": o.average_price,
                "fee": o.fee,
                "fee_currency": o.fee_currency,
                "timestamp": o.timestamp,
                "raw_response": o.raw_response,
            }

        return {
            "positions": {k: _pos(v) for k, v in self.positions.items()},
            "open_orders": {k: _order(v) for k, v in self.open_orders.items()},
            "pending_signals": list(self.pending_signals),
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "cash": self.cash,
            "open_pnl": self.open_pnl,
            "pnl_24h": self.pnl_24h,
            "last_candle_timestamps": self.last_candle_timestamps,
            "bot_started_at": self.bot_started_at,
            "last_heartbeat": self.last_heartbeat,
            "risk": self.risk.to_dict(),
            "take_profit_state": self.take_profit_state,
            "topk_state": self.topk_state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BotState:
        """Deserialise from a JSON-compatible dict."""

        def _pos(d: dict[str, Any]) -> Position:
            return Position(
                symbol=d["symbol"],
                side=OrderSide(d["side"]),
                entry_price=d["entry_price"],
                quantity=d["quantity"],
                current_price=d["current_price"],
                unrealised_pnl=d["unrealised_pnl"],
                realised_pnl=d["realised_pnl"],
                opened_at=d["opened_at"],
                updated_at=d["updated_at"],
            )

        def _order(d: dict[str, Any]) -> OrderResult:
            return OrderResult(
                order_id=d["order_id"],
                client_order_id=d["client_order_id"],
                symbol=d["symbol"],
                side=OrderSide(d["side"]),
                order_type=OrderType(d["order_type"]),
                status=OrderStatus(d["status"]),
                requested_quantity=d["requested_quantity"],
                filled_quantity=d["filled_quantity"],
                average_price=d["average_price"],
                fee=d["fee"],
                fee_currency=d["fee_currency"],
                timestamp=d["timestamp"],
                raw_response=d.get("raw_response", {}),
            )

        return cls(
            positions={k: _pos(v) for k, v in data.get("positions", {}).items()},
            open_orders={k: _order(v) for k, v in data.get("open_orders", {}).items()},
            pending_signals=data.get("pending_signals", []),
            equity=data.get("equity", 0.0),
            peak_equity=data.get("peak_equity", 0.0),
            cash=data.get("cash", 0.0),
            open_pnl=data.get("open_pnl", 0.0),
            # Legacy "daily_pnl" key supported for one restart cycle so the
            # rename doesn't lose the live value in bot_state.json.
            pnl_24h=data.get("pnl_24h", data.get("daily_pnl", 0.0)),
            last_candle_timestamps=data.get("last_candle_timestamps", {}),
            bot_started_at=data.get("bot_started_at", 0),
            last_heartbeat=data.get("last_heartbeat", 0),
            risk=RiskState.from_dict(data.get("risk", {})),
            take_profit_state=data.get("take_profit_state", {}),
            # ``or None`` folds the legacy empty-dict sentinel (pre-TopkState
            # state files) into the None default.
            topk_state=data.get("topk_state") or None,
        )
