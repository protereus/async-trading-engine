"""IG-specific conversion + parsing helpers used by the main orchestrator.

Extracted from ``main.py``.  Pure functions with no class state:

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
