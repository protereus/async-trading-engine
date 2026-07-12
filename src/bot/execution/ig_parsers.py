"""Pure response-shape parsers for IG REST responses.

Extracted from ``ig_client.py``.  No state, no I/O, no aiohttp — these are
free functions that take dicts off the wire and return ``OrderResult`` /
``int`` / ``float`` shapes the rest of the bot understands.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from bot.core.models import OrderResult, OrderSide, OrderStatus, OrderType

logger = logging.getLogger(__name__)


def parse_working_order(wo: dict[str, Any], mkt: dict[str, Any]) -> OrderResult:
    direction = wo.get("direction", "BUY")
    return OrderResult(
        order_id=wo.get("dealId", ""),
        client_order_id=wo.get("dealReference", ""),
        symbol=mkt.get("epic", ""),
        side=OrderSide.BUY if direction == "BUY" else OrderSide.SELL,
        order_type=OrderType.LIMIT,
        status=OrderStatus.OPEN,
        requested_quantity=float(wo.get("orderSize", 0)),
        filled_quantity=0.0,
        average_price=float(wo.get("orderLevel", 0)),
        fee=0.0,
        fee_currency="GBP",
        timestamp=parse_ig_timestamp(wo.get("createdDateUTC", "")),
        raw_response={**wo, **mkt},
    )


def mid_price(price_dict: dict[str, Any]) -> float:
    """Return mid-price from an IG bid/ask price dict."""
    bid = price_dict.get("bid")
    ask = price_dict.get("ask")
    if bid is not None and ask is not None:
        return (float(bid) + float(ask)) / 2.0
    if bid is not None:
        return float(bid)
    if ask is not None:
        return float(ask)
    return 0.0


def parse_ig_timestamp(s: str) -> int:
    """Parse an IG UTC datetime string to Unix milliseconds.

    IG uses several formats depending on the endpoint:
      - ``"2024/10/01 05:28:00:000"``  /prices v2 with milliseconds
      - ``"2026/04/10 19:00:00"``       /prices v2 without milliseconds
      - ``"2024-10-01T05:28:00"``       /positions, /workingorders
      - ``"2024-10-01T05:28:00.000"``   /positions with milliseconds

    Returns 0 if the string is empty or unparseable.
    """
    if not s:
        return 0
    for fmt in (
        "%Y/%m/%d %H:%M:%S:%f",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    ):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=UTC)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    logger.warning("Could not parse IG timestamp %r", s)
    return 0
