"""IG-specific conversion + parsing helpers used by the main orchestrator.

Extracted from ``main.py``.  Pure functions with no class state:

* ``log_overnight_funding_estimate`` — overnight funding cost preview at order
  placement (with Wed/Fri ×3 multipliers).
* ``safe_float`` — best-effort float parse for IG ``str | float | None`` fields.
* ``parse_ig_pnl`` — parses IG's ``profitAndLoss`` strings
  (``"E-12.30"``, ``"-£12.30"``, ``"£+5.40"``, etc.) into a signed float.
* ``apply_sentiment_gate`` — direction-aware sentiment filter for TopK
  selections.

Also re-exports the per-EPIC stop-floor map ``IG_MIN_STOP_PCT`` so the order
placement path doesn't have to import the symbol from ``main``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Per-EPIC minimum stop-percentage floor.  Currently empty after the crypto
# EPICs were removed; the order placement gate in ``main.py`` consults this
# via ``IG_MIN_STOP_PCT.get(epic, 0.0)`` so the empty dict is a deliberate
# no-op default rather than a placeholder.
IG_MIN_STOP_PCT: dict[str, float] = {}


def log_overnight_funding_estimate(symbol: str, size_per_pt: float, ig_level: float) -> None:
    """Log the GBP cost of the *next* overnight roll for a new LONG
    position.  Includes Wed (FX) and Fri (equities/commodities) ×3 multipliers
    so the operator sees the realistic carry, not a generic hour-of-day
    warning.  Cheap: only fires at order-placement time."""
    from datetime import UTC, datetime

    from bot.risk.funding import (
        estimate_overnight_cost_gbp,
        is_equity_triple_day,
        is_fx_triple_day,
    )

    if size_per_pt <= 0 or ig_level <= 0:
        return
    now_utc = datetime.now(UTC)
    cost_gbp = estimate_overnight_cost_gbp(
        symbol=symbol,
        size_per_pt=size_per_pt,
        ig_level=ig_level,
        side="BUY",
        now_utc=now_utc,
    )
    triple = ""
    if is_fx_triple_day(now_utc):
        triple = " (Wed FX ×3)"
    elif is_equity_triple_day(now_utc):
        triple = " (Fri equity ×3)"
    logger.info(
        "Overnight funding estimate: %s LONG £%.2f/pt → next-roll cost £%.2f%s",
        symbol,
        size_per_pt,
        cost_gbp,
        triple,
    )


def safe_float(v: Any) -> float:
    """Best-effort float parse for fields returned by IG as ``str | float | None``."""
    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def parse_ig_pnl(s: str) -> float:
    """Parse IG ``profitAndLoss`` strings like ``"E-12.30"``, ``"-£12.30"``,
    ``"£+5.40"``, or ``"-12.30"`` into a signed float.  Returns 0.0 if
    unparseable.  IG prefixes the currency symbol (£/E/$) in front of the
    sign or after it depending on the endpoint version.
    """
    if not s:
        return 0.0
    cleaned = s.replace("£", "").replace("E", "").replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def apply_sentiment_gate(
    selected: list[str],
    signal_returns: dict[str, float],
    sentiment_scores: dict[str, float],
    long_threshold: float,
    short_threshold: float,
) -> tuple[list[str], list[str]]:
    """Direction-aware sentiment filter for TopK selections.

    LONG signals (mean_return >= 0) require sentiment >= long_threshold.
    SHORT signals (mean_return < 0) require sentiment <= short_threshold.
    Symbols missing sentiment or signal data pass through unchanged.
    Returns (passed, blocked).
    """
    passed: list[str] = []
    blocked: list[str] = []
    for sym in selected:
        sent = sentiment_scores.get(sym)
        mean_ret = signal_returns.get(sym)
        if sent is None or mean_ret is None:
            passed.append(sym)
            continue
        if (mean_ret >= 0 and sent >= long_threshold) or (mean_ret < 0 and sent <= short_threshold):
            passed.append(sym)
        else:
            blocked.append(sym)
    return passed, blocked
