"""Equity / peak-equity / drawdown-tier tracker (extracted from
risk_manager.py).

Tracks the rolling peak-equity high-water mark, exposes the current
drawdown percentage, classifies it into the four ``DrawdownTier`` buckets
from ``RiskConfig``, and reports tier transitions back to the caller via a
callback so the host can fan-out to logging / event-bus / RED-halt logic
without this module growing a bus dependency.
"""

from __future__ import annotations

from collections.abc import Callable

from bot.core.models import DrawdownTier
from bot.risk.risk_config import RiskConfig

# Callback signature: (old_tier, new_tier, drawdown_pct, equity) -> None
TierChangeCallback = Callable[[DrawdownTier, DrawdownTier, float, float], None]


class DrawdownTracker:
    """Equity + peak + tier classification.  Caller wires a tier-change
    callback to handle logging / event emission / trading-halt flips."""

    def __init__(
        self,
        config: RiskConfig,
        on_tier_change: TierChangeCallback | None = None,
    ) -> None:
        self._config = config
        self._on_tier_change = on_tier_change
        self._equity: float = 0.0
        self._peak_equity: float = 0.0
        self._tier: DrawdownTier = DrawdownTier.NORMAL

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def update_equity(self, equity: float) -> None:
        """Update current equity, advance the peak if hit, and reclassify the
        drawdown tier.  Fires ``on_tier_change`` if the bucket moves."""
        if equity < 0:
            equity = 0.0
        self._equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity

        drawdown = self.drawdown_pct
        old_tier = self._tier
        new_tier = self._tier_with_hysteresis(drawdown, old_tier)
        if new_tier != old_tier:
            self._tier = new_tier
            if self._on_tier_change is not None:
                self._on_tier_change(old_tier, new_tier, drawdown, equity)

    def set_peak_equity(self, value: float) -> None:
        """Restore the peak from persisted state without touching current equity."""
        self._peak_equity = max(0.0, value)

    def recompute_tier(self) -> None:
        """Recompute the current tier from current state.  Use after a state
        restore to bring the tier in line with the restored peak/equity."""
        if self._peak_equity > 0 and self._equity > 0:
            self._tier = self.tier_for_drawdown(self.drawdown_pct)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    @property
    def tier(self) -> DrawdownTier:
        return self._tier

    @property
    def drawdown_pct(self) -> float:
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - self._equity) / self._peak_equity

    def _tier_with_hysteresis(
        self, drawdown_pct: float, current_tier: DrawdownTier
    ) -> DrawdownTier:
        """Directional hysteresis around the raw tier thresholds.

        Escalation (raw tier as bad or worse than the current one) is accepted
        immediately — we never want to under-react to a worsening drawdown.
        De-escalation is sticky: the drawdown must recover a full
        ``drawdown_tier_rearm_band`` below the lower tier's line before the tier
        steps down, so equity parked on a threshold can't oscillate the tier
        (and spam tier-change alerts)."""
        raw = self.tier_for_drawdown(drawdown_pct)
        if raw.severity >= current_tier.severity:
            return raw
        # Downgrade candidate — re-classify with the band added back so we only
        # step down once we're band-below the line we'd be crossing.
        return self.tier_for_drawdown(drawdown_pct + self._config.drawdown_tier_rearm_band)

    def tier_for_drawdown(self, drawdown_pct: float) -> DrawdownTier:
        if drawdown_pct >= self._config.drawdown_red_pct:
            return DrawdownTier.RED
        if drawdown_pct >= self._config.drawdown_orange_pct:
            return DrawdownTier.ORANGE
        if drawdown_pct >= self._config.drawdown_yellow_pct:
            return DrawdownTier.YELLOW
        return DrawdownTier.NORMAL
