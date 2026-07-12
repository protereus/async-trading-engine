"""Position-sizing math (extracted from risk_manager.py).

Pure functions over the per-trade inputs.  No state, no event-bus
integration, no logging — caller decides what to do with the result.
``RiskManager.compute_ig_size`` delegates here.
"""

from __future__ import annotations


def compute_ig_size(
    equity_gbp: float,
    risk_pct: float,
    entry_price: float,
    stop_pct: float,
    pip_value: float = 1.0,
    slippage_buffer_pts: float = 0.0,
) -> float:
    """Compute the IG spread bet stake (£/point) for a given risk budget.

    Args:
        equity_gbp:  Total account equity in GBP.
        risk_pct:    Fraction of equity to risk on this trade (e.g. 0.01 = 1%).
        entry_price: Entry price in native units (e.g. 0.9175 for EUR/CHF,
                     4700 for XAU/USD).
        stop_pct:    Stop distance as a fraction of entry (e.g. 0.005 = 0.5%).
        pip_value:   Size of one IG point in native price units.
                     Use 0.0001 for non-JPY forex, 0.01 for JPY pairs, 1.0
                     for gold/ETFs.
        slippage_buffer_pts: Extra IG points to add to the stop distance when
                     sizing — accounts for the real fill being worse than the
                     trigger level (IG_LIVE_RISK_REFERENCE.md §1.1).  Callers compute this via
                     ``bot.risk.ig_margin.estimate_slippage_pts``.  Default 0
                     preserves backward-compatible behaviour.

    Returns:
        Stake in £/point, rounded to 2 dp.  Returns 0.0 if inputs are invalid.
    """
    if equity_gbp <= 0 or risk_pct <= 0 or entry_price <= 0 or stop_pct <= 0 or pip_value <= 0:
        return 0.0
    risk_gbp = equity_gbp * risk_pct
    stop_distance_pts = entry_price * stop_pct / pip_value
    effective_stop_pts = stop_distance_pts + max(0.0, slippage_buffer_pts)
    return round(risk_gbp / effective_stop_pts, 2)
