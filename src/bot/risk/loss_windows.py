"""Rolling loss-window + consecutive-loss tracker (extracted from
risk_manager.py).

Owns the per-trade results deque used by the daily / weekly / monthly loss
limits, the consecutive-loss pause counter, and the cumulative PnL
accumulators that persist via ``RiskState``.  All API methods are
synchronous — the previous "in-loop emit" behaviour stays in
``RiskManager`` so this module has no event-bus dependency.
"""

from __future__ import annotations

from collections import deque

from bot.core.time_constants import MONTH_MS


class LossWindowTracker:
    """Tracks realised PnL across rolling time windows + a consecutive-loss
    counter used by the order-evaluation gates."""

    def __init__(self) -> None:
        self._trade_results: deque[tuple[int, float]] = deque()  # (ts_ms, pnl)
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._monthly_pnl: float = 0.0
        self._consecutive_losses: int = 0

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def record_close(self, now_ms: int, pnl: float) -> None:
        """Append a realised PnL to the rolling deque and roll the counters."""
        self._trade_results.append((now_ms, pnl))
        self._prune(now_ms)

        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        self._daily_pnl += pnl
        self._weekly_pnl += pnl
        self._monthly_pnl += pnl

    def _prune(self, now_ms: int) -> None:
        cutoff = now_ms - MONTH_MS
        while self._trade_results and self._trade_results[0][0] < cutoff:
            self._trade_results.popleft()

    def prune(self, now_ms: int) -> None:
        """Public form of :meth:`_prune` for snapshot callers."""
        self._prune(now_ms)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def window_pnl(self, now_ms: int, window_ms: int) -> float:
        cutoff = now_ms - window_ms
        return sum(pnl for ts, pnl in self._trade_results if ts >= cutoff)

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    @consecutive_losses.setter
    def consecutive_losses(self, value: int) -> None:
        self._consecutive_losses = max(0, int(value))

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def weekly_pnl(self) -> float:
        return self._weekly_pnl

    @property
    def monthly_pnl(self) -> float:
        return self._monthly_pnl

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, object]:
        return {
            "daily_pnl": self._daily_pnl,
            "weekly_pnl": self._weekly_pnl,
            "monthly_pnl": self._monthly_pnl,
            "consecutive_losses": self._consecutive_losses,
            "trade_results": [[ts, pnl] for ts, pnl in self._trade_results],
        }

    def load_snapshot(
        self,
        *,
        daily_pnl: float,
        weekly_pnl: float,
        monthly_pnl: float,
        consecutive_losses: int,
        trade_results: list[list[float]],
    ) -> None:
        self._daily_pnl = daily_pnl
        self._weekly_pnl = weekly_pnl
        self._monthly_pnl = monthly_pnl
        self._consecutive_losses = consecutive_losses
        self._trade_results = deque((int(row[0]), float(row[1])) for row in trade_results)
