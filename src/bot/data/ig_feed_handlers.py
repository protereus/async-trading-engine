"""Handlers for incoming Lightstreamer updates (CHART / TRADE / ACCOUNT).

Extracted from ``ig_feed.py``.  Owns:

* ``TickValidator`` — per-epic rolling-σ outlier filter for chart ticks
  (IG_LIVE_RISK_REFERENCE.md §2.3).
* ``IGFeedHandlers`` — dispatch logic for the three subscription types,
  reading shared state (``_store``, ``_candle_db``, ``_event_bus``,
  ``_spread_monitor``, ``_last_confirmed_ts``, ``_tick_validator``) via a
  back-reference to its parent ``IGFeed``.

These run on the asyncio main thread (the consumer in
``IGFeed._consume_updates`` is the only caller), so no extra locking is
required.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import time
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from bot.core.event_bus import EVENT_ACCOUNT_UPDATE, EVENT_NEW_CANDLE, EVENT_ORDER_FILLED
from bot.core.models import (
    AccountUpdate,
    Candle,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
)

if TYPE_CHECKING:
    from bot.data.ig_feed import IGFeed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tick outlier validator (IG_LIVE_RISK_REFERENCE.md §2.3).
#
# IG occasionally publishes corrupted ticks: zero/null prints, and (rarely)
# wildly off-mid values that survive the bid/offer sanity check.  Feeding
# those into the candle store skews indicators and can falsely trip the
# trailing stop on a real position.
#
# The validator keeps a rolling log-return deque per epic.  Once primed, any
# tick whose log-return-from-previous exceeds N standard deviations is
# rejected and the epic is suspended for a few ticks so a single bad print
# doesn't poison the window.  Suspension naturally recovers — the validator
# simply re-accepts after the cool-down.
# ---------------------------------------------------------------------------
_TICK_WINDOW = 100
_TICK_MIN_PRIME = 20  # accept-all until we have this many returns
_TICK_N_SIGMA = 6.0
_TICK_SUSPEND_AFTER_REJECT = 5  # burn this many ticks after a rejection


class TickValidator:
    """Per-epic rolling-σ outlier filter for Lightstreamer chart ticks.

    Returns False from ``accept()`` when the incoming mid-price would be an
    Nσ outlier (or when the epic is in cool-down).  Otherwise updates the
    rolling state and returns True.  Thread-safety: not held under a lock —
    the consumer in ``_handle_chart_update`` is the only caller, running on
    the asyncio main thread.
    """

    def __init__(
        self,
        *,
        window: int = _TICK_WINDOW,
        min_prime: int = _TICK_MIN_PRIME,
        n_sigma: float = _TICK_N_SIGMA,
        suspend_ticks: int = _TICK_SUSPEND_AFTER_REJECT,
    ) -> None:
        self._window = window
        self._min_prime = min_prime
        self._n_sigma = n_sigma
        self._suspend_ticks = suspend_ticks
        self._returns: dict[str, deque[float]] = {}
        self._last_price: dict[str, float] = {}
        self._tick_count: dict[str, int] = {}
        self._suspended_until: dict[str, int] = {}

    def accept(self, epic: str, mid_price: float) -> bool:
        if mid_price <= 0 or not math.isfinite(mid_price):
            return False

        count = self._tick_count.get(epic, 0) + 1
        self._tick_count[epic] = count

        sus_until = self._suspended_until.get(epic, 0)
        if count < sus_until:
            return False

        last = self._last_price.get(epic)
        if last is None or last <= 0:
            self._last_price[epic] = mid_price
            return True

        log_return = math.log(mid_price / last)

        returns = self._returns.setdefault(epic, deque(maxlen=self._window))
        if len(returns) >= self._min_prime:
            mean = statistics.fmean(returns)
            stdev = statistics.stdev(returns) if len(returns) > 1 else 0.0
            if stdev > 0 and abs(log_return - mean) > self._n_sigma * stdev:
                # Outlier — reject, suspend, do NOT update rolling state.
                self._suspended_until[epic] = count + self._suspend_ticks
                return False

        returns.append(log_return)
        self._last_price[epic] = mid_price
        return True

    def reset(self, epic: str | None = None) -> None:
        """Drop all rolling state for *epic* (or all epics if None).

        Use after a Lightstreamer reconnect / heartbeat-driven recovery —
        the new connection may resubscribe with a snapshot that breaks
        return continuity.
        """
        if epic is None:
            self._returns.clear()
            self._last_price.clear()
            self._tick_count.clear()
            self._suspended_until.clear()
            return
        self._returns.pop(epic, None)
        self._last_price.pop(epic, None)
        self._tick_count.pop(epic, None)
        self._suspended_until.pop(epic, None)


_LONDON = ZoneInfo("Europe/London")


def _safe_float(val: Any) -> float:
    """Convert a Lightstreamer field value to float, returning 0.0 on failure."""
    if val is None:
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _mid(bid: float, offer: float) -> float:
    """Return the mid-price between bid and offer."""
    if bid and offer:
        return (bid + offer) / 2.0
    return bid or offer


class IGFeedHandlers:
    """Dispatcher for CHART / TRADE / ACCOUNT updates.

    Holds a back-reference to its parent ``IGFeed`` so it can read/mutate
    the shared store / DB / event-bus / spread-monitor / dedup map.
    """

    def __init__(self, feed: IGFeed) -> None:
        self._feed = feed

    async def handle_chart_update(self, update: dict[str, Any]) -> None:
        """Parse a CHART item update and emit EVENT_NEW_CANDLE on confirmed close."""
        feed = self._feed
        item_name: str = update.get("item", "")
        fields: dict[str, Any] = update.get("fields", {})

        # Extract EPIC from item name: "CHART:{epic}:1MINUTE"
        parts = item_name.split(":")
        if len(parts) < 2:
            return
        epic = parts[1]

        # Parse OHLC mid-prices from bid/offer
        bid_o = _safe_float(fields.get("BID_OPEN"))
        bid_h = _safe_float(fields.get("BID_HIGH"))
        bid_l = _safe_float(fields.get("BID_LOW"))
        bid_c = _safe_float(fields.get("BID_CLOSE"))
        ofr_o = _safe_float(fields.get("OFR_OPEN"))
        ofr_h = _safe_float(fields.get("OFR_HIGH"))
        ofr_l = _safe_float(fields.get("OFR_LOW"))
        ofr_c = _safe_float(fields.get("OFR_CLOSE"))

        mid_o = _mid(bid_o, ofr_o)
        mid_h = _mid(bid_h, ofr_h)
        mid_l = _mid(bid_l, ofr_l)
        mid_c = _mid(bid_c, ofr_c)

        if mid_c == 0.0:
            return  # incomplete update — skip

        # Nσ outlier filter.  Suspends the epic for a few ticks on
        # rejection so a single bad print can't poison the rolling window.
        if not feed._tick_validator.accept(epic, mid_c):
            logger.warning(
                "Tick rejected for %s: mid=%.6f (outlier or epic in cool-down)",
                epic,
                mid_c,
            )
            return

        utms = fields.get("UTM")
        if utms is not None:
            # IG Lightstreamer UTM is epoch ms computed from London local time
            # (BST = UTC+1 in summer), not true UTC — confirmed IG Labs issue.
            # Apply the London UTC offset to convert to real UTC epoch.
            utc_offset = datetime.now(_LONDON).utcoffset()
            london_offset_ms = (
                int(utc_offset.total_seconds() * 1000) if utc_offset is not None else 0
            )
            ts = int(utms) + london_offset_ms
        else:
            ts = int(time.time() * 1000)

        volume = _safe_float(fields.get("LTV")) or 0.0
        cons_end = str(fields.get("CONS_END", "0")) == "1"

        candle = Candle(
            timestamp=ts,
            open=mid_o,
            high=mid_h,
            low=mid_l,
            close=mid_c,
            volume=volume,
            symbol=epic,
            is_confirmed=cons_end,
        )

        feed._store.add_candle(candle)

        if cons_end and ts > feed._last_confirmed_ts.get(epic, 0):
            feed._last_confirmed_ts[epic] = ts
            if feed._candle_db is not None:
                feed._candle_db.insert_candle(candle)
            # Sample the bid-ask spread (IG points) once per confirmed
            # candle.  Skipping intra-candle ticks keeps the rolling window
            # at ~30 days × 1-min cadence = 43k samples per epic.
            spread_pts = ofr_c - bid_c
            if spread_pts > 0:
                feed._spread_monitor.record(epic, spread_pts)
            await feed._event_bus.emit(EVENT_NEW_CANDLE, candle)
            logger.debug("New confirmed candle %s @ %d close=%.4f", epic, ts, mid_c)

    async def handle_trade_update(self, update: dict[str, Any]) -> None:
        """Parse a TRADE item update and emit EVENT_ORDER_FILLED for fills."""
        feed = self._feed
        fields: dict[str, Any] = update.get("fields", {})
        confirms_raw = fields.get("CONFIRMS")
        if not confirms_raw:
            return

        try:
            confirms: dict[str, Any] = json.loads(confirms_raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Could not parse CONFIRMS JSON: %r", confirms_raw)
            return

        deal_status = confirms.get("dealStatus", "")
        if deal_status != "ACCEPTED":
            logger.info(
                "TRADE CONFIRMS: dealReference=%s status=%s reason=%s",
                confirms.get("dealReference", ""),
                deal_status,
                confirms.get("reason", ""),
            )
            return

        epic = confirms.get("epic", "")
        direction = confirms.get("direction", "BUY")
        result = OrderResult(
            order_id=confirms.get("dealId", ""),
            client_order_id=confirms.get("dealReference", ""),
            symbol=epic,
            side=OrderSide.BUY if direction == "BUY" else OrderSide.SELL,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            requested_quantity=float(confirms.get("size", 0)),
            filled_quantity=float(confirms.get("size", 0)),
            average_price=float(confirms.get("level", 0)),
            fee=0.0,
            fee_currency="GBP",
            timestamp=int(time.time() * 1000),
            raw_response=confirms,
        )
        logger.info(
            "Fill confirmed via TRADE stream: %s %s %.2f £/pt @ %.2f",
            epic,
            direction,
            result.filled_quantity,
            result.average_price,
        )
        await feed._event_bus.emit(EVENT_ORDER_FILLED, result)

    async def handle_account_update(self, update: dict[str, Any]) -> None:
        """Parse the LS ACCOUNT push and emit ``EVENT_ACCOUNT_UPDATE``.

        Drives the real-time margin circuit breakers — RiskManager subscribes
        to the event and recomputes ``equity / margin_required`` on every push.
        LS sends partial frames so any field can be absent; we forward whatever
        arrived and let RiskManager decide whether the snapshot is actionable
        (it ignores 0/missing margin as "no open positions").
        """
        feed = self._feed
        fields: dict[str, Any] = update.get("fields", {})
        equity = _safe_float(fields.get("EQUITY"))
        margin = _safe_float(fields.get("MARGIN"))
        available = _safe_float(fields.get("AVAILABLE_TO_DEAL"))
        pnl = _safe_float(fields.get("PNL"))

        # Skip frames that carry none of the fields we care about — IG sends
        # heartbeat frames with empty bodies that would otherwise spam the bus.
        if not any(fields.get(k) is not None for k in ("EQUITY", "MARGIN", "PNL")):
            return

        logger.debug(
            "Account update: equity=%s margin=%s available=%s pnl=%s",
            equity,
            margin,
            available,
            pnl,
        )
        payload = AccountUpdate(
            timestamp=int(time.time() * 1000),
            equity=equity,
            margin_required=margin,
            available_to_deal=available,
            unrealised_pnl=pnl,
        )
        await feed._event_bus.emit(EVENT_ACCOUNT_UPDATE, payload)
