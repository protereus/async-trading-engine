"""Risk management configuration.

Conservative-profile defaults are safe to run without external config.
Override via environment variables or a .env file if needed.
"""

from pydantic import BaseModel, field_validator


class RiskConfig(BaseModel):
    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    risk_per_trade_pct: float = 0.01  # 1% of equity risked per trade
    max_position_pct: float = 0.05  # 5% of portfolio in any single position
    atr_period: int = 14
    atr_multiplier: float = 2.0  # stop distance = ATR x this

    # ------------------------------------------------------------------
    # Exposure limits
    # ------------------------------------------------------------------
    # Soft sanity ceiling — set above the practical risk-on cap so the
    # real gate is ``max_total_risk_pct``.  With risk_per_trade_pct=0.01
    # and risk cap 0.05 the gate self-limits to ~5 concurrent full-size
    # positions; the count ceiling just stops a runaway from opening
    # dozens if the risk math goes sideways.
    max_open_positions: int = 8
    # Sum of risk-on (stop-loss) budgets across all open IG positions, as a
    # fraction of equity.  With risk_per_trade_pct=0.01, a cap of 0.05 allows
    # up to ~5 concurrent full-size positions — and headroom expands as
    # trailing stops ratchet (live risk shrinks).
    max_total_risk_pct: float = 0.05
    # Per-sector risk-on cap: max fraction of equity that can be at risk
    # (sum of stop-loss budgets) inside any single sector.  Half of
    # ``max_total_risk_pct`` by default — at most ~50 % of risk-on can
    # land in one sector, forcing diversification across the seven
    # buckets in ``bot.risk.sectors.SECTOR_MAP`` (fx_usd /
    # fx_eur_cross / fx_gbp_cross / fx_jpy_cross / metals /
    # equity_index / energy).  Complements the pairwise correlation
    # filter in ``correlation.py``: that catches pairs, this catches
    # clusters.
    max_sector_risk_pct: float = 0.025
    correlation_threshold: float = 0.70  # treat as same trade above this

    # ------------------------------------------------------------------
    # Drawdown tiers
    # ------------------------------------------------------------------
    drawdown_yellow_pct: float = 0.05  # 5%  -- reduce sizes 50%
    drawdown_orange_pct: float = 0.10  # 10% -- reduce sizes 75%
    drawdown_red_pct: float = 0.15  # 15% -- halt NEW ENTRIES (debounced; was full shutdown)
    # Drawdown breaker hardening (2026-06-05: a single transient gold mark during
    # the 22:00-23:00 metals maintenance window tripped RED and shut the bot down).
    drawdown_red_confirm_count: int = 3  # consecutive RED equity reads before halting
    drawdown_maintenance_guard: bool = True  # freeze the breaker during the IG rollover window
    # Tier hysteresis (2026-06-08: equity parked on a tier line oscillated
    # NORMAL↔YELLOW on tiny marks, spamming Telegram).  Escalation uses the raw
    # thresholds above; a tier only *downgrades* once drawdown recovers this far
    # below the lower tier's line (e.g. YELLOW→NORMAL needs dd < yellow−band).
    drawdown_tier_rearm_band: float = 0.01  # 1 percentage point

    # ------------------------------------------------------------------
    # Loss limits
    # ------------------------------------------------------------------
    daily_loss_limit_pct: float = 0.03  # 3% of equity
    weekly_loss_limit_pct: float = 0.05  # 5% of equity
    monthly_loss_limit_pct: float = 0.10  # 10% of equity
    consecutive_loss_pause: int = 4  # pause after N consecutive losses

    # ------------------------------------------------------------------
    # Volatility circuit breaker
    # ------------------------------------------------------------------
    volatility_atr_lookback: int = 20  # 20-period ATR average
    volatility_multiplier: float = 2.0  # pause if ATR > 2x average

    # ------------------------------------------------------------------
    # Order validation
    # ------------------------------------------------------------------
    max_single_order_pct: float = 0.05  # reject orders > 5% of balance
    max_price_deviation_pct: float = 0.03  # reject if price > 3% from market
    max_orders_per_hour: int = 60  # rate limit on order placement

    # ------------------------------------------------------------------
    # IG spread betting
    # ------------------------------------------------------------------
    max_margin_pct: float = 0.50  # hard cap: refuse new orders if margin > 50% of equity
    ig_overnight_warning_hour_utc: int = 18  # warn about DFB funding if opening after this hour

    # ------------------------------------------------------------------
    # Margin-utilisation circuit breakers (IG_LIVE_RISK_REFERENCE.md §4.3).
    # IG auto-liquidates retail positions when equity / total_margin_required
    # falls below 0.50 — no warning, no grace.  The three breakers below sit
    # well above that floor and trip on every LS ACCOUNT push or tick that
    # affects an open position, so we de-risk in stages instead of waking up
    # to a forced close.  The 0.50 floor itself is asserted-and-paged.
    # ------------------------------------------------------------------
    margin_halt_ratio: float = 0.80  # equity/margin ≤ this → refuse new entries
    margin_defensive_ratio: float = 0.65  # equity/margin ≤ this → close worst performer
    margin_emergency_ratio: float = 0.55  # equity/margin ≤ this → flatten everything
    margin_liquidation_floor: float = 0.50  # broker's number; assert-and-page if breached

    # ------------------------------------------------------------------
    # FSCS soft ceiling (IG_LIVE_RISK_REFERENCE.md §7.2).
    # FSCS covers £120K per person per institution; surplus beyond that is
    # uninsured at this broker.  ``fscs_warn_gbp`` triggers a single risk
    # event on the way up (with hysteresis so we don't spam); ``fscs_cap_gbp``
    # caps the equity used for position sizing so incremental profits past
    # the FSCS line don't increase per-trade £ risk.  Loss-limit and
    # margin checks continue to use real equity.
    # ------------------------------------------------------------------
    fscs_warn_gbp: float = 100_000.0
    fscs_cap_gbp: float = 120_000.0

    @field_validator("risk_per_trade_pct", "max_position_pct")
    @classmethod
    def _positive_pct(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError(f"Percentage must be in (0, 1], got {v}")
        return v

    @field_validator("drawdown_yellow_pct", "drawdown_orange_pct", "drawdown_red_pct")
    @classmethod
    def _drawdown_ascending(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError(f"Drawdown tier must be in (0, 1), got {v}")
        return v

    @field_validator("drawdown_tier_rearm_band")
    @classmethod
    def _rearm_band_range(cls, v: float) -> float:
        if not 0 <= v < 1:
            raise ValueError(f"Drawdown re-arm band must be in [0, 1), got {v}")
        return v
