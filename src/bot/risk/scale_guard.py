"""Quote-scale drift guard (D4 of ).

Detects when the hard-coded ``ig_quote_scale(symbol)`` has drifted away
from the *actual* ratio between IG's quoted level and the candle-source
price (EODHD, or Twelve Data on the warm-standby path).

Why this exists
---------------
``_IG_PIP_VALUE`` is a static table calibrated on a single day.  For
symbols where the candle source is an ETF proxy for a different IG
instrument (notably USO↔WTI-Crude, UNG↔Nat-Gas), the ratio drifts as the
ETF and the underlying futures diverge through roll costs and tracking
error.  On 2026-05-28 USO had drifted −32 % and UNG +38 %, which silently
inflated P&L by the same amount and fired ghost take-profits.

This module turns that failure mode from "silent" into "loud": a cheap
periodic comparison of expected-vs-real scale, logged and alerted when it
crosses a threshold.  Detection only — it does NOT mutate trading state.
Enforcement (auto-excluding a badly-drifted symbol from selection) is a
deliberate follow-up; for now the operator decides what to do with the
alert.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from bot.execution.ig_quote_scale import ig_quote_scale

# Thresholds on |real_scale − expected_scale| / expected_scale.
_WARN_DRIFT = 0.10
_CRITICAL_DRIFT = 0.25


class DriftSeverity(StrEnum):
    OK = "ok"
    WARN = "warn"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ScaleDriftResult:
    """Outcome of one symbol's scale check."""

    symbol: str
    candle_price: float
    ig_mid: float
    expected_scale: float  # from ig_quote_scale(symbol)
    real_scale: float  # ig_mid / candle_price
    drift: float  # signed fraction: (real − expected) / expected
    severity: DriftSeverity

    @property
    def implied_pnl_error(self) -> float:
        """Magnitude of the P&L mis-statement this drift would produce.

        A position priced through a scale that's off by ``drift`` mis-reports
        its mark-to-market by the same fraction, so |drift| is a direct proxy
        for "how wrong is the displayed P&L".
        """
        return abs(self.drift)


def classify_drift(
    drift: float,
    *,
    warn: float = _WARN_DRIFT,
    critical: float = _CRITICAL_DRIFT,
) -> DriftSeverity:
    """Bucket a signed drift fraction into ok / warn / critical by magnitude."""
    mag = abs(drift)
    if mag >= critical:
        return DriftSeverity.CRITICAL
    if mag >= warn:
        return DriftSeverity.WARN
    return DriftSeverity.OK


def compute_drift(
    symbol: str,
    candle_price: float,
    ig_mid: float,
    *,
    warn: float = _WARN_DRIFT,
    critical: float = _CRITICAL_DRIFT,
) -> ScaleDriftResult | None:
    """Compare the configured scale to the live IG-vs-candle ratio.

    Returns ``None`` when the inputs can't yield a meaningful ratio
    (non-positive price or mid) — the caller treats that as "no data,
    skip" rather than a drift event, since a missing quote is a feed
    problem, not a scale problem.
    """
    if candle_price <= 0 or ig_mid <= 0:
        return None
    expected_scale = ig_quote_scale(symbol)
    real_scale = ig_mid / candle_price
    drift = (real_scale - expected_scale) / expected_scale
    return ScaleDriftResult(
        symbol=symbol,
        candle_price=candle_price,
        ig_mid=ig_mid,
        expected_scale=expected_scale,
        real_scale=real_scale,
        drift=drift,
        severity=classify_drift(drift, warn=warn, critical=critical),
    )
