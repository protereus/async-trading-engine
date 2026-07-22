"""IG Group REST API client.

Async wrapper around the IG REST API for spread betting accounts —
the sole broker client.

Authentication uses CST + X-SECURITY-TOKEN headers obtained from
POST /session.  Tokens are valid for 6 hours, auto-extended to 72 hours
on activity — they carry no process or connection affinity and can be
persisted across restarts.

Token caching: on connect(), cached tokens are loaded from
``.ig_session_cache.json`` in the project root.  A fresh POST /session
is only made when the cache is absent or stale (> TOKEN_CACHE_MAX_AGE_S).
This avoids the IG demo rate-limit on repeated logins during development.

On connect(), the client automatically switches to the SPREADBET account
if the session defaults to a CFD account.

Spread bet sizing: ``size`` is currency-per-point (e.g. £1/pt), not a
quantity of asset.  Use IGOrderRequest instead of OrderRequest for orders.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import aiohttp

from bot.core.models import (
    Candle,
    ErrorType,
    ExchangeError,
    IGOrderRequest,
    MarketClosedError,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from bot.execution.ig_http import IGHttp
from bot.execution.ig_parsers import mid_price, parse_ig_timestamp, parse_working_order
from bot.execution.ig_session import IGSession

if TYPE_CHECKING:
    from bot.config import BotConfig

logger = logging.getLogger(__name__)

# IG REST base URLs
_BASE_DEMO = "https://demo-api.ig.com/gateway/deal"
_BASE_LIVE = "https://api.ig.com/gateway/deal"

# Semantic retry for ``/confirms/{ref}`` returning 404 ``deal-not-found``.
# IG's confirms cache lags placement by a variable amount — empirically 1–3 s
# on demo, occasionally up to ~10 s.  Without this retry every laggy confirm
# created an orphan position (May 28 2026 GBP/NZD incident).  The delays are
# tried sequentially: attempt 1 at t=0, attempt 2 after +1.5 s, etc.  Four
# attempts ≈ 10.5 s total wall-time which still fits inside the per-candle
# entry path budget.  Beyond that the orphan-detection / manual-cleanup
# path is the fallback.
_CONFIRM_RETRY_DELAYS_S: tuple[float, ...] = (1.5, 3.0, 6.0)


# Maps bot timeframe strings to IG resolution strings
_IG_RESOLUTION: dict[str, str] = {
    "1m": "MINUTE",
    "2m": "MINUTE_2",
    "3m": "MINUTE_3",
    "5m": "MINUTE_5",
    "10m": "MINUTE_10",
    "15m": "MINUTE_15",
    "30m": "MINUTE_30",
    "1h": "HOUR",
    "2h": "HOUR_2",
    "3h": "HOUR_3",
    "4h": "HOUR_4",
    "1d": "DAY",
    "1w": "WEEK",
}

# Per-request timeout cap for every aiohttp call.  Without this an unhealthy
# IG endpoint can stall the whole event loop indefinitely (we observed a 4-minute
# hang on ``GET /accounts`` during cached-token verification on 2026-05-07,
# blocking the entire startup sequence with no log output).  30 s is well above
# IG's normal sub-second response time and still short enough that an operator
# notices a problem instead of waiting forever.
_REQUEST_TIMEOUT_S = 30.0


class IGClient:
    """Async IG REST client for spread betting accounts.

    Lifecycle::

        client = IGClient(config)
        await client.connect()
        # ... use client ...
        await client.close()
    """

    def __init__(self, config: BotConfig) -> None:
        self._config = config
        self._base = _BASE_DEMO if config.bot_env == "demo" else _BASE_LIVE
        self._session: aiohttp.ClientSession | None = None

        # Auth state — populated by connect() and refresh_session()
        self._cst: str = ""
        self._xst: str = ""
        self._account_id: str = ""
        self._ls_endpoint: str = ""
        # Serialises full re-auth so concurrent 401 retries don't stampede
        # POST /session (which is the most rate-limited endpoint on IG demo).
        self._auth_lock = asyncio.Lock()
        # Re-entrancy guard: True while a forced refresh is in flight so the
        # /accounts probe inside ``_switch_to_spreadbet`` cannot recurse into
        # another refresh on a transient 401 while we already hold the lock.
        self._refreshing: bool = False

        # Live quota from most recent /prices response
        self._datapoints_remaining: int | None = None
        self._datapoints_total: int | None = None

        # Collaborators — back-references to this IGClient.  HTTP transport
        # (rate limiting, retry, header construction) and session lifecycle
        # (POST /session, token cache, background refresh + keep-alive).
        self._http = IGHttp(self)
        self._sess = IGSession(self)

        # Background tasks
        self._refresh_task: asyncio.Task[None] | None = None
        self._keepalive_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Create aiohttp session, authenticate with IG, switch to SPREADBET account.

        Loads cached tokens first to avoid hammering POST /session during
        development.  Only falls back to a fresh login when the cache is
        absent, stale, or the cached tokens are rejected.
        """
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_S)
        )

        if not await self._sess.load_cached_tokens():
            await self._sess.create_session()
            await self._sess.switch_to_spreadbet()
            self._sess.save_tokens()
        else:
            # Verify cached tokens with a lightweight call; re-auth on rejection
            # or on a hung response (treat a timeout the same as a 401 — IG demo
            # occasionally stalls and re-auth is the safe recovery path).
            try:
                await self._http.get("/accounts", version="1", authenticated=True)
                logger.info("IGClient resumed from cached session  account=%s", self._account_id)
            except (ExchangeError, TimeoutError):
                logger.info("Cached tokens rejected or unresponsive, re-authenticating")
                await self._sess.create_session()
                await self._sess.switch_to_spreadbet()
                self._sess.save_tokens()

        self._refresh_task = asyncio.create_task(
            self._sess.refresh_loop(), name="ig-session-refresh"
        )
        self._keepalive_task = asyncio.create_task(
            self._sess.keepalive_loop(), name="ig-session-keepalive"
        )
        logger.info(
            "IGClient connected  account=%s  env=%s  ls=%s",
            self._account_id,
            self._config.bot_env,
            self._ls_endpoint,
        )

    async def close(self) -> None:
        """Cancel refresh + keep-alive tasks, logout, close HTTP session."""
        for task_attr in ("_refresh_task", "_keepalive_task"):
            task = getattr(self, task_attr)
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                setattr(self, task_attr, None)

        if self._session is not None:
            with contextlib.suppress(Exception):
                await self._http.delete("/session", version="1", authenticated=True)
            await self._session.close()
            self._session = None
            logger.info("IGClient disconnected")

    @property
    def is_connected(self) -> bool:
        return self._session is not None and bool(self._cst)

    @property
    def lightstreamer_endpoint(self) -> str:
        return self._ls_endpoint

    @property
    def account_id(self) -> str:
        return self._account_id

    @property
    def ls_password(self) -> str:
        """Lightstreamer connection password built from the current session tokens.

        The ``CST-…|XST-…`` format is IG-session knowledge, so the feed's
        Lightstreamer connectors read it here rather than reaching into the
        private ``_cst`` / ``_xst`` tokens.
        """
        return f"CST-{self._cst}|XST-{self._xst}"

    @property
    def datapoints_remaining(self) -> int | None:
        """Remaining historical price datapoints this week (from last /prices response)."""
        return self._datapoints_remaining

    # ------------------------------------------------------------------
    # Market data (REST)
    # ------------------------------------------------------------------

    async def fetch_ohlcv(
        self,
        epic: str,
        timeframe: str,
        limit: int = 100,
    ) -> list[Candle]:
        """Fetch historical OHLCV candles via REST.

        Uses the IG /prices endpoint.  After each call the live quota
        from the ``allowance`` field is stored in ``self.datapoints_remaining``.

        Args:
            epic:      IG EPIC code, e.g. ``"CS.D.AVXUSD.TODAY.IP"``
            timeframe: Bot timeframe string, e.g. ``"1m"``, ``"5m"``, ``"1h"``
            limit:     Number of candles to fetch (max ~1,000 per call)
        """
        resolution = _IG_RESOLUTION.get(timeframe)
        if resolution is None:
            raise ValueError(f"Unsupported timeframe for IG: {timeframe!r}")

        data = await self._http.get(
            f"/prices/{epic}/{resolution}/{limit}",
            version="2",
            authenticated=True,
        )

        # Update live quota from response
        allowance = data.get("allowance", {})
        if allowance:
            self._datapoints_remaining = int(allowance.get("remainingAllowance", 0))
            self._datapoints_total = int(allowance.get("totalAllowance", 0))
            expiry_s = int(allowance.get("allowanceExpiry", 0))
            if self._datapoints_remaining < 2_000:
                logger.error(
                    "IG price quota LOW: %d/%d remaining, resets in %dh",
                    self._datapoints_remaining,
                    self._datapoints_total,
                    expiry_s // 3600,
                )
            elif self._datapoints_remaining < 5_000:
                logger.warning(
                    "IG price quota warning: %d/%d remaining",
                    self._datapoints_remaining,
                    self._datapoints_total,
                )

        candles: list[Candle] = []
        for item in data.get("prices", []):
            ts = parse_ig_timestamp(item.get("snapshotTimeUTC") or item.get("snapshotTime", ""))
            mid_o = mid_price(item["openPrice"])
            mid_h = mid_price(item["highPrice"])
            mid_l = mid_price(item["lowPrice"])
            mid_c = mid_price(item["closePrice"])
            vol = float(item.get("lastTradedVolume") or 0)
            candles.append(
                Candle(
                    timestamp=ts,
                    open=mid_o,
                    high=mid_h,
                    low=mid_l,
                    close=mid_c,
                    volume=vol,
                    symbol=epic,
                    is_confirmed=True,
                )
            )
        candles.sort(key=lambda c: c.timestamp)
        return candles

    async def search_epic(self, term: str) -> list[dict[str, Any]]:
        """Search for markets by name. Returns list of market dicts with 'epic' field."""
        data = await self._http.get(f"/markets?searchTerm={term}", version="1", authenticated=True)
        result: list[dict[str, Any]] = data.get("markets", [])
        return result

    async def fetch_market_details(self, epic: str) -> dict[str, Any]:
        """Fetch full instrument and dealing rules for an EPIC."""
        return await self._http.get(f"/markets/{epic}", version="1", authenticated=True)

    async def require_tradeable(self, epic: str) -> float | None:
        """Block entry orders on a non-TRADEABLE market (IG_LIVE_RISK_REFERENCE.md §1.3).

        Calls ``GET /markets/{epic}`` and reads ``snapshot.marketStatus``.
        Raises ``MarketClosedError`` for any state other than ``TRADEABLE``
        (``CLOSED``, ``EDITS_ONLY``, ``MARKET_CLOSED_WITH_EDITS``,
        ``OFFLINE``, ``ON_AUCTION``, ``SUSPENDED`` etc.).

        This is the pre-trade gate from IG_LIVE_RISK_REFERENCE.md §1.3 — the
        demo environment hides all of these state transitions, so without it
        a scale-in or fresh entry into a closing-only market will sit pending
        until the next session, leaving the book unhedged.

        On a tradeable market, returns ``dealingRules.minDealSize.value`` (the
        smallest stake IG will accept, £/pt) so the caller can skip a
        risk-sized order that's below it rather than eat a
        ``MINIMUM_ORDER_SIZE_ERROR`` reject — notably the higher-priced US
        shares whose 1%-risk stake can fall under the 0.24 £/pt floor. Returns
        ``None`` if the field is absent.
        """
        try:
            data = await self.fetch_market_details(epic)
        except ExchangeError:
            # Surface the underlying error verbatim — callers already log
            # ExchangeError; wrapping it would lose the bucket/retry context.
            raise

        snapshot = data.get("snapshot") or {}
        status = str(snapshot.get("marketStatus", "")).upper()
        if status != "TRADEABLE":
            # Non-tradeable: classify so the risk layer can decide whether to
            # cancel scale-ins, reconcile, or just log-and-skip.
            raise MarketClosedError(
                f"IG market {epic} not tradeable: marketStatus={status or 'UNKNOWN'}",
                ErrorType.MARKET_CLOSED,
            )
        rules = data.get("dealingRules") or {}
        min_deal = rules.get("minDealSize") or {}
        try:
            value = min_deal.get("value")
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Account data
    # ------------------------------------------------------------------

    async def fetch_balance(self) -> dict[str, Any]:
        """Fetch account balance. Returns dict with equity, balance, open_pnl, margin (GBP)."""
        data = await self._http.get("/accounts", version="1", authenticated=True)
        for acc in data.get("accounts", []):
            if acc["accountId"] == self._account_id:
                bal = acc.get("balance", {})
                # IG REST API v1 balance object: balance, deposit, profitLoss, available.
                # No "equity" key — equity = balance + unrealised P&L (profitLoss).
                # "deposit" is margin committed to open trades.
                cash = float(bal.get("balance", 0))
                pnl = float(bal.get("profitLoss", 0))
                equity = float(bal.get("equity", cash + pnl))  # fallback for mocks
                return {
                    "equity": equity,
                    "available": float(bal.get("available", 0)),
                    "balance": cash,
                    "open_pnl": pnl,
                    "margin": float(bal.get("deposit", bal.get("margin", 0))),
                    "currency": acc.get("currency", "GBP"),
                }
        raise ExchangeError(
            f"Account {self._account_id} not found in /accounts response",
            ErrorType.EXCHANGE_ERROR,
        )

    async def fetch_positions(self, symbol: str | None = None) -> list[Position]:
        """Fetch all open spread bet positions."""
        data = await self._http.get("/positions", version="2", authenticated=True)
        positions = []
        for item in data.get("positions", []):
            pos = item.get("position", {})
            mkt = item.get("market", {})
            epic = mkt.get("epic", "")
            if symbol is not None and epic != symbol:
                continue
            direction = pos.get("direction", "BUY")
            side = OrderSide.BUY if direction == "BUY" else OrderSide.SELL
            positions.append(
                Position(
                    symbol=epic,
                    side=side,
                    entry_price=float(pos.get("level", 0)),
                    quantity=float(pos.get("size", 0)),  # £/pt in spread bets
                    current_price=float(mkt.get("bid", 0)),
                    unrealised_pnl=float(pos.get("upl", 0)),
                    realised_pnl=0.0,
                    opened_at=parse_ig_timestamp(pos.get("createdDateUTC", "")),
                    updated_at=int(time.time() * 1000),
                )
            )
        return positions

    async def fetch_positions_raw(self) -> list[dict[str, Any]]:
        """Raw IG ``/positions`` entries for reconciliation.

        ``fetch_positions`` parses each entry into a ``Position``, dropping the
        reconciliation-only fields (``dealId``, ``stopLevel``, ``direction``),
        so the close/reconcile path (``IGCloseManager``) needs the unparsed
        entries instead of reaching into the private ``_get``.
        """
        data = await self._http.get("/positions", version="2", authenticated=True)
        entries: list[dict[str, Any]] = data.get("positions", [])
        return entries

    async def fetch_open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        """Fetch working (pending) orders."""
        data = await self._http.get("/workingorders", version="2", authenticated=True)
        results = []
        for item in data.get("workingOrders", []):
            wo = item.get("workingOrderData", {})
            mkt = item.get("marketData", {})
            epic = mkt.get("epic", "")
            if symbol is not None and epic != symbol:
                continue
            results.append(parse_working_order(wo, mkt))
        return results

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    async def place_order(self, order: IGOrderRequest) -> OrderResult:
        """Place a spread bet position via POST /positions/otc.

        Returns an OrderResult with status=PENDING and order_id=dealReference.
        Call confirm_order(deal_reference) after to get the filled price.
        """
        body: dict[str, Any] = {
            "epic": order.epic,
            "expiry": "DFB",
            "direction": order.direction.upper(),
            "size": order.size,
            "orderType": order.order_type.upper(),
            "currencyCode": "GBP",
            "forceOpen": True,
            "guaranteedStop": order.guaranteed_stop,
        }
        if order.stop_distance is not None:
            body["stopDistance"] = order.stop_distance
        if order.limit_distance is not None:
            body["limitDistance"] = order.limit_distance
        if order.deal_reference:
            body["dealReference"] = order.deal_reference

        # idempotent=False: a lost ACK on a timeout/5xx must NOT be blind-retried
        # — IG may have already opened the position, and dealReference is not an
        # idempotency key. Fail closed; reconciliation surfaces any orphan.
        data = await self._http.post(
            "/positions/otc", body=body, version="2", authenticated=True, idempotent=False
        )
        deal_ref = data.get("dealReference", "")

        logger.info(
            "Order placed: %s %s size=%.2f £/pt  dealReference=%s",
            order.epic,
            order.direction,
            order.size,
            deal_ref,
        )
        return OrderResult(
            order_id=deal_ref,
            client_order_id=order.deal_reference or deal_ref,
            symbol=order.epic,
            side=OrderSide.BUY if order.direction.upper() == "BUY" else OrderSide.SELL,
            order_type=(
                OrderType.MARKET if order.order_type.upper() == "MARKET" else OrderType.LIMIT
            ),
            status=OrderStatus.PENDING,
            requested_quantity=order.size,
            filled_quantity=0.0,
            average_price=0.0,
            fee=0.0,
            fee_currency="GBP",
            timestamp=int(time.time() * 1000),
            raw_response=data,
        )

    async def confirm_order(self, deal_reference: str) -> OrderResult:
        """Fetch deal confirmation after placement — contains filled level and status.

        IG's ``/confirms/{ref}`` cache lags the actual deal by 1–10 s; an
        immediate query can 404 with ``error.confirms.deal-not-found`` even
        though the order has actually filled.  Pre-retry behaviour created a
        permanent orphan position on every laggy confirm (May 28 2026 GBP/NZD
        incident).  We now retry on that specific 404 with the delays in
        ``_CONFIRM_RETRY_DELAYS_S``.  Other 404s and all non-404 errors
        propagate immediately.
        """
        data: dict[str, Any] | None = None
        last_exc: ExchangeError | None = None
        max_attempts = len(_CONFIRM_RETRY_DELAYS_S) + 1  # first try + retries
        for attempt in range(max_attempts):
            if attempt > 0:
                await asyncio.sleep(_CONFIRM_RETRY_DELAYS_S[attempt - 1])
            try:
                data = await self._http.get(
                    f"/confirms/{deal_reference}", version="1", authenticated=True
                )
                break
            except ExchangeError as exc:
                if "deal-not-found" not in str(exc):
                    raise
                last_exc = exc
                remaining = max_attempts - attempt - 1
                if remaining > 0:
                    logger.warning(
                        "confirm_order deal-not-found for %s (attempt %d/%d) — retrying in %.1fs",
                        deal_reference,
                        attempt + 1,
                        max_attempts,
                        _CONFIRM_RETRY_DELAYS_S[attempt],
                    )
        if data is None:
            assert last_exc is not None
            logger.error(
                "confirm_order: %d attempts exhausted for %s — order may exist as "
                "orphan on IG; reconcile loop will detect and alert.",
                max_attempts,
                deal_reference,
            )
            raise last_exc

        status_map = {
            "ACCEPTED": OrderStatus.FILLED,
            "REJECTED": OrderStatus.REJECTED,
            "OPEN": OrderStatus.OPEN,
        }
        status_str = data.get("dealStatus", "REJECTED")
        status = status_map.get(status_str, OrderStatus.REJECTED)
        epic = data.get("epic", "")
        direction = data.get("direction", "BUY")

        if status == OrderStatus.REJECTED:
            reason = data.get("reason", "unknown")
            if "MARKET_CLOSED" in reason or "MARKET_OFFLINE" in reason:
                # Benign — the position/entry is deferred and retried once the
                # market reopens (see MarketClosedError callers); not a real error.
                logger.warning(
                    "Order rejected (market closed, deferred): dealReference=%s reason=%s",
                    deal_reference,
                    reason,
                )
                raise MarketClosedError(
                    f"IG market closed for {deal_reference}: {reason}",
                    ErrorType.MARKET_CLOSED,
                )
            logger.error("Order rejected: dealReference=%s reason=%s", deal_reference, reason)
            raise ExchangeError(
                f"IG rejected order {deal_reference}: {reason}",
                ErrorType.INVALID_ORDER,
            )

        return OrderResult(
            order_id=data.get("dealId", deal_reference),
            client_order_id=deal_reference,
            symbol=epic,
            side=OrderSide.BUY if direction == "BUY" else OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=status,
            requested_quantity=float(data.get("size", 0)),
            filled_quantity=float(data.get("size", 0)),
            average_price=float(data.get("level", 0)),
            fee=0.0,
            fee_currency="GBP",
            timestamp=int(time.time() * 1000),
            raw_response=data,
        )

    async def close_position(
        self, deal_id: str, epic: str, direction: str, size: float
    ) -> OrderResult:
        """Close an open spread bet position via DELETE /positions/otc.

        ``direction`` is the CLOSING direction (opposite of the open):
        opened BUY → close with SELL, and vice versa.
        """
        # IG REST v1: dealId is mutually exclusive with epic/expiry.
        # Close by dealId only — do not include epic or expiry.
        body = {
            "dealId": deal_id,
            "direction": direction.upper(),
            "size": size,
            "orderType": "MARKET",
        }
        data = await self._http.delete_with_body(
            "/positions/otc", body=body, version="1", authenticated=True
        )
        deal_ref = data.get("dealReference", "")
        if not deal_ref:
            raise RuntimeError(f"IG close_position returned no dealReference for dealId={deal_id}")
        logger.info("Position close requested: dealId=%s dealReference=%s", deal_id, deal_ref)
        return await self.confirm_order(deal_ref)

    async def fetch_closed_transaction(
        self,
        *,
        opened_at_ms: int,
        instrument_name_contains: str | None = None,
        lookback_extra_hours: int = 2,
    ) -> dict[str, Any] | None:
        """Find the ``GET /history/transactions`` record for a position by its
        open timestamp.  Returns the raw transaction dict — keys of interest:
        ``openLevel`` (str), ``closeLevel`` (str), ``profitAndLoss`` (str like
        ``"-£12.30"`` or ``"E-5.40"``), ``size``, ``instrumentName``, ``period``.

        ``opened_at_ms`` is the position open time we recorded locally; the
        matching transaction will have ``openDateUtc`` within ±2s of that.

        Returns None if no match within the lookback window.  ``lookback_extra_hours``
        widens the search window past "now" by that many hours to handle clock
        skew at the boundary.

        The endpoint is paginated; we request a generous ``pageSize`` (50) which
        is enough for the per-EPIC history this method looks at.
        """
        if opened_at_ms <= 0:
            return None
        from_ts = datetime.fromtimestamp(opened_at_ms / 1000, tz=UTC) - timedelta(seconds=5)
        to_ts = datetime.now(UTC) + timedelta(hours=lookback_extra_hours)
        from_str = from_ts.strftime("%Y-%m-%dT%H:%M:%S")
        to_str = to_ts.strftime("%Y-%m-%dT%H:%M:%S")
        path = f"/history/transactions?from={from_str}&to={to_str}&pageSize=50"
        try:
            data = await self._http.get(path, version="2", authenticated=True)
        except Exception:
            logger.exception("fetch_closed_transaction: /history/transactions request failed")
            return None
        for item in data.get("transactions", []):
            item_open_ms = parse_ig_timestamp(item.get("openDateUtc", ""))
            if item_open_ms == 0 or abs(item_open_ms - opened_at_ms) > 5000:
                continue
            if instrument_name_contains:
                name = item.get("instrumentName", "")
                if instrument_name_contains.lower() not in name.lower():
                    continue
            return dict(item)
        return None

    async def cancel_order(self, order_id: str, symbol: str) -> OrderResult:
        """Cancel a working (pending limit/stop) order."""
        data = await self._http.delete(
            f"/workingorders/otc/{order_id}",
            version="2",
            authenticated=True,
        )
        return OrderResult(
            order_id=order_id,
            client_order_id=order_id,
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.CANCELLED,
            requested_quantity=0.0,
            filled_quantity=0.0,
            average_price=0.0,
            fee=0.0,
            fee_currency="GBP",
            timestamp=int(time.time() * 1000),
            raw_response=data,
        )

    # ------------------------------------------------------------------
    # Delegates to collaborators
    # ------------------------------------------------------------------

    async def refresh_session(self) -> None:
        """Re-authenticate to get fresh CST/XST tokens before expiry.

        Public surface — called by external integrations (LS feed, tests)
        and by ``IGHttp`` on 401.  Delegates to ``IGSession``.
        """
        await self._sess.refresh_session()

    async def _post(
        self, path: str, body: dict[str, Any], version: str, authenticated: bool
    ) -> dict[str, Any]:
        """Back-compat alias for ``self._http.post`` — kept so existing tests
        that drive ``client._post`` directly still exercise the retry path."""
        return await self._http.post(path, body, version=version, authenticated=authenticated)

    async def _delete(self, path: str, version: str, authenticated: bool) -> dict[str, Any]:
        """Back-compat alias for ``self._http.delete``."""
        return await self._http.delete(path, version=version, authenticated=authenticated)
