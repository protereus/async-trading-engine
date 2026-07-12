"""FSCS soft-ceiling guard (IG_LIVE_RISK_REFERENCE.md §7.2).

FSCS covers losses up to £120K per person per institution in the event of
broker / custodian insolvency (IG_LIVE_RISK_REFERENCE.md §7.2).  Surplus
equity beyond that line is uninsured at this broker.  We don't refuse to
trade above it — instead we:

1. Log a one-shot risk event when equity crosses the warn threshold
   (default £100K) on the way up.  Hysteresis avoids log-spam: once
   warned, the warning re-arms only after equity drops a configurable
   amount below the threshold.

2. Cap the equity used for *position sizing* at ``cap_gbp`` (default
   £120K).  Loss-limit, margin, and risk-budget checks continue to use
   real equity so the protective gates don't get neutered — what we
   block is reinvestment of profits beyond the FSCS line.

Stateless w.r.t. persistence: the warned flag is hysteresis-only and
re-derives from current equity on restart.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from bot.risk.risk_config import RiskConfig

logger = logging.getLogger(__name__)

# Callback signature: (event_type, details) -> None.
# Matches RiskManager._log_risk_event so the host can fan-out to the
# event ledger without this module growing a bus dependency.
RiskEventCallback = Callable[[str, dict[str, Any]], None]

# Hysteresis band: once warned, equity must drop this far below the warn
# threshold before the warning re-arms.  Keeps the log from oscillating
# around the threshold when equity ticks up and down across the line.
_REARM_BAND_GBP: float = 1_000.0


class FSCSCeiling:
    """One-shot warn + sizing-equity clamp around the FSCS £120K line."""

    def __init__(
        self,
        config: RiskConfig,
        risk_event_callback: RiskEventCallback | None = None,
    ) -> None:
        self._config = config
        self._on_risk_event = risk_event_callback
        self._warned: bool = False

    def update_equity(self, equity: float) -> None:
        """Drive the warn-once log on upward crossings of ``fscs_warn_gbp``.

        Called from ``RiskManager.update_equity`` before drawdown tracking.
        """
        warn_gbp = self._config.fscs_warn_gbp
        if not self._warned and equity >= warn_gbp:
            self._warned = True
            logger.warning(
                "FSCS soft ceiling: equity %.2f GBP ≥ warn threshold %.2f GBP. "
                "FSCS covers losses only up to %.2f GBP at this broker; "
                "consider splitting capital across independent brokerages.",
                equity,
                warn_gbp,
                self._config.fscs_cap_gbp,
            )
            if self._on_risk_event is not None:
                self._on_risk_event(
                    "fscs_warn",
                    {
                        "equity_gbp": equity,
                        "warn_gbp": warn_gbp,
                        "cap_gbp": self._config.fscs_cap_gbp,
                    },
                )
        elif self._warned and equity < warn_gbp - _REARM_BAND_GBP:
            self._warned = False

    def cap_for_sizing(self, equity: float) -> float:
        """Return the equity to use for position sizing — real equity
        clamped at ``fscs_cap_gbp``.  Returns ``equity`` unchanged when
        below the cap, so this is a no-op for AUM below the FSCS line.
        """
        cap = self._config.fscs_cap_gbp
        if equity <= cap:
            return equity
        return cap

    @property
    def warned(self) -> bool:
        """Current hysteresis state — true after we've crossed the warn
        threshold and have not yet dropped through the re-arm band."""
        return self._warned
