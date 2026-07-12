"""Tests for RiskManager — position sizing, order evaluation, drawdown tracking,
loss windows, volatility circuit breaker, and state persistence.

All tests use fixed, deterministic values and a controlled clock function.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.core.event_bus import EventBus
from bot.core.models import (
    DrawdownTier,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    RiskLevel,
)
from bot.risk.risk_config import RiskConfig
from bot.risk.risk_manager import RiskManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW_MS = 1_700_000_000_000  # fixed "now" for deterministic tests


def _make_clock(ts_ms: int = _NOW_MS):
    """Return a clock function that always returns *ts_ms*."""
    return lambda: ts_ms


def _make_rm(config: RiskConfig | None = None, clock_ms: int = _NOW_MS) -> RiskManager:
    bus = MagicMock(spec=EventBus)
    bus.emit = AsyncMock()
    cfg = config or RiskConfig()
    return RiskManager(cfg, bus, clock_fn=_make_clock(clock_ms))


def _make_fill(
    symbol: str = "BTC/USDT",
    side: OrderSide = OrderSide.BUY,
    qty: float = 0.01,
    price: float = 50_000.0,
) -> OrderResult:
    return OrderResult(
        order_id="oid_001",
        client_order_id="test_order_001",
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        status=OrderStatus.FILLED,
        requested_quantity=qty,
        filled_quantity=qty,
        average_price=price,
        fee=0.0,
        fee_currency="USDT",
        timestamp=_NOW_MS,
    )


# ===========================================================================
# Position sizing
# ===========================================================================


class TestDrawdownTierStakeScaling:
    """Drawdown-tier stake reduction on the live entry path.

    The ATR-based *sizing math* itself is pinned by
    ``tests/test_sizing.py::TestComputeIgSize`` — these tests pin the
    tier multipliers ``evaluate_ig_order`` applies on top of the stake.
    """

    def test_stake_halved_by_yellow_drawdown(self):
        rm = _make_rm()
        rm.update_equity(10_000.0)  # establish peak
        rm.update_equity(9_400.0)  # 6% drawdown → YELLOW
        assert rm.current_drawdown_tier == DrawdownTier.YELLOW

        order = _make_ig_order(size=1.0)
        decision = rm.evaluate_ig_order(order, margin_used=0.0, equity_gbp=9_400.0)
        assert decision.approved
        assert decision.adjusted_quantity == pytest.approx(0.5)
        assert decision.risk_level == RiskLevel.ELEVATED

    def test_stake_quartered_by_orange_drawdown(self):
        rm = _make_rm()
        rm.update_equity(10_000.0)
        rm.update_equity(8_900.0)  # 11% drawdown → ORANGE
        assert rm.current_drawdown_tier == DrawdownTier.ORANGE

        order = _make_ig_order(size=1.0)
        decision = rm.evaluate_ig_order(order, margin_used=0.0, equity_gbp=8_900.0)
        assert decision.approved
        assert decision.adjusted_quantity == pytest.approx(0.25)


# ===========================================================================
# compute_ig_size with slippage buffer
# IG_LIVE_RISK_REFERENCE.md §1.1: stop level is a trigger, not a fill.
# ===========================================================================


class TestComputeIgSizeSlippage:
    def test_no_slippage_unchanged_behaviour(self) -> None:
        """slippage_buffer_pts=0 must match the buffer-free formula exactly."""
        size = RiskManager.compute_ig_size(
            equity_gbp=10_000.0,
            risk_pct=0.01,
            entry_price=11000.0,
            stop_pct=0.005,
            pip_value=0.0001,
        )
        # stop_distance_pts = 11000 × 0.005 / 0.0001 = 550 000
        # size = 100 / 550 000 = 0.000182 — rounds to 0.0
        assert size == pytest.approx(round(100 / (11000 * 0.005 / 0.0001), 2))

    def test_slippage_buffer_shrinks_size(self) -> None:
        """Adding slippage points to the stop distance must reduce stake (we
        keep the same £-risk but assume a worse fill)."""
        baseline = RiskManager.compute_ig_size(
            equity_gbp=10_000.0,
            risk_pct=0.01,
            entry_price=4500.0,
            stop_pct=0.01,  # 1 % stop
            pip_value=1.0,  # gold
        )
        with_slip = RiskManager.compute_ig_size(
            equity_gbp=10_000.0,
            risk_pct=0.01,
            entry_price=4500.0,
            stop_pct=0.01,
            pip_value=1.0,
            slippage_buffer_pts=5.0,  # 5-pt slippage on a 45-pt stop
        )
        assert with_slip < baseline
        # Concretely: stop=45, with slip→50, size = 100/50 = 2.00 vs 100/45 ≈ 2.22
        assert baseline == pytest.approx(100.0 / 45.0, abs=0.01)
        assert with_slip == pytest.approx(100.0 / 50.0, abs=0.01)

    def test_negative_slippage_treated_as_zero(self) -> None:
        """Negative buffer can't reduce the stop distance — clamped to 0."""
        with_neg = RiskManager.compute_ig_size(
            equity_gbp=10_000.0,
            risk_pct=0.01,
            entry_price=4500.0,
            stop_pct=0.01,
            pip_value=1.0,
            slippage_buffer_pts=-10.0,
        )
        baseline = RiskManager.compute_ig_size(
            equity_gbp=10_000.0,
            risk_pct=0.01,
            entry_price=4500.0,
            stop_pct=0.01,
            pip_value=1.0,
        )
        assert with_neg == pytest.approx(baseline)


# ===========================================================================
# evaluate_ig_order account-level gates — each check independently
# ===========================================================================


class TestEntryGates:
    """The account-level gates every live entry passes through
    (``_common_entry_gates`` + the shared rate-limit / volatility checks),
    exercised via ``evaluate_ig_order`` — the only entry path since the
    OKX/ccxt removal (2026-06-24)."""

    def test_rejected_when_trading_halted(self):
        rm = _make_rm()
        rm._trading_halted = True
        rm._halt_reason = "test halt"
        decision = rm.evaluate_ig_order(_make_ig_order(), margin_used=0.0, equity_gbp=10_000.0)
        assert not decision.approved
        assert decision.risk_level == RiskLevel.HALTED
        assert "halted" in decision.reason.lower()

    def test_rejected_on_daily_loss_limit(self):
        rm = _make_rm()
        # Inject a losing trade inside the daily window
        rm._loss_windows._trade_results.append((_NOW_MS - 1000, -300.1))  # -3.001% of 10000
        decision = rm.evaluate_ig_order(_make_ig_order(), margin_used=0.0, equity_gbp=10_000.0)
        assert not decision.approved
        assert "daily" in decision.reason.lower()

    def test_rejected_on_weekly_loss_limit(self):
        _DAY_MS = 24 * 3600 * 1000
        rm = _make_rm()
        # Loss within last 7 days but beyond 24h — daily limit NOT triggered
        two_days_ago = _NOW_MS - 2 * _DAY_MS
        rm._loss_windows._trade_results.append((two_days_ago, -500.1))  # -5.001% weekly
        decision = rm.evaluate_ig_order(_make_ig_order(), margin_used=0.0, equity_gbp=10_000.0)
        assert not decision.approved
        assert "weekly" in decision.reason.lower()

    def test_rejected_on_monthly_loss_limit(self):
        _DAY_MS = 24 * 3600 * 1000
        rm = _make_rm()
        # Loss 10 days ago — beyond weekly (7d) but within monthly (30d)
        ten_days_ago = _NOW_MS - 10 * _DAY_MS
        rm._loss_windows._trade_results.append((ten_days_ago, -1000.1))  # -10.001% monthly
        decision = rm.evaluate_ig_order(_make_ig_order(), margin_used=0.0, equity_gbp=10_000.0)
        assert not decision.approved
        assert "monthly" in decision.reason.lower()

    def test_rejected_after_consecutive_losses(self):
        cfg = RiskConfig(consecutive_loss_pause=4)
        rm = _make_rm(config=cfg)
        rm._loss_windows.consecutive_losses = 4
        decision = rm.evaluate_ig_order(_make_ig_order(), margin_used=0.0, equity_gbp=10_000.0)
        assert not decision.approved
        assert "consecutive" in decision.reason.lower()

    def test_rejected_when_max_positions_reached(self):
        cfg = RiskConfig(max_open_positions=2)
        rm = _make_rm(config=cfg)
        rm._open_positions["CS.D.EURUSD.TODAY.IP"] = MagicMock(spec=Position)
        rm._open_positions["CS.D.GBPUSD.TODAY.IP"] = MagicMock(spec=Position)
        decision = rm.evaluate_ig_order(
            _make_ig_order(epic="CS.D.USDJPY.TODAY.IP"), margin_used=0.0, equity_gbp=10_000.0
        )
        assert not decision.approved
        assert "positions" in decision.reason.lower()

    def test_rejected_when_order_rate_limit_hit(self):
        cfg = RiskConfig(max_orders_per_hour=5)
        rm = _make_rm(config=cfg)
        # Fill the rate limit bucket with 5 recent orders
        for _ in range(5):
            rm._orders_this_hour.append(_NOW_MS - 100)
        decision = rm.evaluate_ig_order(_make_ig_order(), margin_used=0.0, equity_gbp=10_000.0)
        assert not decision.approved
        assert "rate" in decision.reason.lower()

    def test_approved_with_full_stake_when_all_checks_pass(self):
        rm = _make_rm()
        rm.update_equity(10_000.0)
        decision = rm.evaluate_ig_order(
            _make_ig_order(size=1.0), margin_used=0.0, equity_gbp=10_000.0
        )
        assert decision.approved
        assert decision.original_quantity == 1.0
        assert decision.adjusted_quantity == 1.0
        assert decision.risk_level == RiskLevel.NORMAL


# ===========================================================================
# Drawdown tracking
# ===========================================================================


class TestDrawdownTracking:
    def test_tier_transitions_at_correct_thresholds(self):
        rm = _make_rm()
        rm.update_equity(10_000.0)

        rm.update_equity(9_600.0)  # 4% — still NORMAL
        assert rm.current_drawdown_tier == DrawdownTier.NORMAL

        rm.update_equity(9_400.0)  # 6% — YELLOW
        assert rm.current_drawdown_tier == DrawdownTier.YELLOW

        rm.update_equity(8_900.0)  # 11% — ORANGE
        assert rm.current_drawdown_tier == DrawdownTier.ORANGE

        rm.update_equity(8_400.0)  # 16% — RED
        assert rm.current_drawdown_tier == DrawdownTier.RED

    def test_events_logged_only_on_transitions(self):
        rm = _make_rm()
        rm.update_equity(10_000.0)
        rm.update_equity(9_400.0)  # transition to YELLOW
        assert len(rm._risk_events) == 1

        rm.update_equity(9_350.0)  # still YELLOW — no new event
        assert len(rm._risk_events) == 1

        rm.update_equity(8_900.0)  # transition to ORANGE
        assert len(rm._risk_events) == 2

    def test_hysteresis_no_oscillation_on_the_yellow_line(self):
        """Equity parked at ~5% drawdown must not flip NORMAL↔YELLOW: one
        escalation event, then nothing while it wiggles across the line."""
        rm = _make_rm()
        rm.update_equity(10_000.0)
        rm.update_equity(9_490.0)  # 5.1% — escalate to YELLOW (1 event)
        assert rm.current_drawdown_tier == DrawdownTier.YELLOW
        assert len(rm._risk_events) == 1

        # Tiny marks straddling the 5% line — would have spammed without the band.
        for eq in (9_510.0, 9_490.0, 9_505.0, 9_495.0, 9_520.0):
            rm.update_equity(eq)  # all in [4.8%, 5.2%] dd → still YELLOW
            assert rm.current_drawdown_tier == DrawdownTier.YELLOW
        assert len(rm._risk_events) == 1  # no further events

    def test_hysteresis_downgrade_requires_band_recovery(self):
        """YELLOW only steps back to NORMAL once drawdown recovers a full band
        below the 5% line (i.e. below 4% with the 1pp default)."""
        rm = _make_rm()
        rm.update_equity(10_000.0)
        rm.update_equity(9_400.0)  # 6% — YELLOW
        assert rm.current_drawdown_tier == DrawdownTier.YELLOW

        rm.update_equity(9_550.0)  # 4.5% — inside the band, stays YELLOW
        assert rm.current_drawdown_tier == DrawdownTier.YELLOW

        rm.update_equity(9_610.0)  # 3.9% — below 4%, downgrades to NORMAL
        assert rm.current_drawdown_tier == DrawdownTier.NORMAL

    def test_hysteresis_escalation_is_immediate(self):
        """A worsening drawdown is never delayed by the band."""
        rm = _make_rm()
        rm.update_equity(10_000.0)
        rm.update_equity(9_490.0)  # 5.1% — YELLOW at the raw line, no delay
        assert rm.current_drawdown_tier == DrawdownTier.YELLOW
        rm.update_equity(8_990.0)  # 10.1% — ORANGE immediately
        assert rm.current_drawdown_tier == DrawdownTier.ORANGE

    def test_peak_equity_never_moves_down(self):
        rm = _make_rm()
        rm.update_equity(10_000.0)
        assert rm._drawdown.peak_equity == 10_000.0

        rm.update_equity(9_000.0)
        assert rm._drawdown.peak_equity == 10_000.0  # peak unchanged

        rm.update_equity(11_000.0)
        assert rm._drawdown.peak_equity == 11_000.0  # peak updated

        rm.update_equity(9_000.0)
        assert rm._drawdown.peak_equity == 11_000.0  # peak unchanged again

    def test_red_single_read_does_not_halt(self):
        # Debounce: one transient RED read must NOT halt (the 2026-06-05 bug).
        rm = _make_rm()  # default drawdown_red_confirm_count=3
        rm.update_equity(10_000.0)
        rm.update_equity(8_400.0)  # 16% RED — read 1 of 3
        assert rm._consecutive_red == 1
        assert rm._trading_halted is False

    def test_red_halts_after_confirm_count(self):
        rm = _make_rm()  # confirm_count=3
        rm.update_equity(10_000.0)
        rm.update_equity(8_400.0)
        rm.update_equity(8_400.0)
        assert rm._trading_halted is False  # still only 2 reads
        rm.update_equity(8_400.0)  # 3rd consecutive RED → halt
        assert rm._trading_halted is True
        assert rm.is_trading_halted is True

    def test_red_debounce_resets_on_recovery(self):
        rm = _make_rm()
        rm.update_equity(10_000.0)
        rm.update_equity(8_400.0)  # RED read 1
        rm.update_equity(8_400.0)  # RED read 2
        rm.update_equity(9_400.0)  # recovered to YELLOW → counter resets
        assert rm._consecutive_red == 0
        assert rm._trading_halted is False

    def test_red_halt_auto_clears_on_recovery(self):
        rm = _make_rm(RiskConfig(drawdown_red_confirm_count=1))
        rm.update_equity(10_000.0)
        rm.update_equity(8_400.0)  # RED (confirm=1) → halted
        assert rm._trading_halted is True
        rm.update_equity(9_400.0)  # recover to YELLOW → auto-clear
        assert rm._trading_halted is False
        assert rm._halt_reason == ""

    def test_maintenance_guard_freezes_breaker(self):
        # During the blackout window a RED-looking equity must neither halt nor
        # pollute the peak — the update is skipped entirely.
        rm = _make_rm(RiskConfig(drawdown_red_confirm_count=1))
        rm._in_blackout_fn = lambda: True
        rm.update_equity(10_000.0)  # frozen — not tracked
        rm.update_equity(8_400.0)  # would be RED but guarded
        assert rm._trading_halted is False
        assert rm._drawdown.peak_equity == 0.0  # nothing tracked during blackout

    def test_trading_halt_survives_restart(self):
        rm = _make_rm(RiskConfig(drawdown_red_confirm_count=1))
        rm.update_equity(10_000.0)
        rm.update_equity(8_400.0)  # RED (confirm=1) — halted

        state = rm.get_state()
        assert state.trading_halted is True

        rm2 = _make_rm(RiskConfig(drawdown_red_confirm_count=1))
        rm2.update_equity(8_400.0)  # set _equity so drawdown is computable
        rm2.load_state(state)
        assert rm2.is_trading_halted is True

    def test_drawdown_pct_calculation(self):
        rm = _make_rm()
        rm.update_equity(10_000.0)
        rm.update_equity(9_000.0)
        assert rm.current_drawdown_pct == pytest.approx(0.10)

    def test_red_halt_does_not_emit_shutdown(self):
        # RED entry-halt must NOT shut the bot down (2026-06-05 change) — it
        # raises a Telegram risk alert and halts new entries instead.
        import asyncio

        from bot.core.event_bus import EVENT_RISK_ALERT, EVENT_SHUTDOWN

        rm = _make_rm(RiskConfig(drawdown_red_confirm_count=1))
        bus = EventBus()
        shutdowns: list[Any] = []
        alerts: list[Any] = []

        async def _cap_shutdown(p: Any) -> None:
            shutdowns.append(p)

        async def _cap_alert(p: Any) -> None:
            alerts.append(p)

        bus.subscribe(EVENT_SHUTDOWN, _cap_shutdown)
        bus.subscribe(EVENT_RISK_ALERT, _cap_alert)
        rm._event_bus = bus

        async def _run() -> None:
            rm.update_equity(10_000.0)
            rm.update_equity(8_400.0)  # RED (confirm=1) → halt, no shutdown
            await asyncio.sleep(0.01)

        asyncio.run(_run())

        assert shutdowns == []
        assert rm._trading_halted is True
        assert any(getattr(a, "event_type", "") == "drawdown_red_halt" for a in alerts)


# ===========================================================================
# Loss window tracking
# ===========================================================================


class TestLossWindowTracking:
    def test_pnl_24h_accumulates_correctly(self):
        rm = _make_rm()
        rm._drawdown._equity = 10_000.0
        rm.on_position_closed("BTC/USDT", -100.0)
        rm.on_position_closed("ETH/USDT", -50.0)
        _DAY_MS = 24 * 3600 * 1000
        total = rm._window_pnl(_NOW_MS, _DAY_MS)
        assert total == pytest.approx(-150.0)

    def test_pnl_24h_excludes_trades_older_than_24h(self):
        _DAY_MS = 24 * 3600 * 1000
        rm = _make_rm()
        # Trade from 25 hours ago — outside daily window
        rm._loss_windows._trade_results.append((_NOW_MS - 25 * 3600 * 1000, -999.0))
        # Trade from 1 hour ago — inside window
        rm._loss_windows._trade_results.append((_NOW_MS - 1 * 3600 * 1000, -50.0))
        total = rm._window_pnl(_NOW_MS, _DAY_MS)
        assert total == pytest.approx(-50.0)

    def test_weekly_window_correct(self):
        _DAY_MS = 24 * 3600 * 1000
        _WEEK_MS = 7 * _DAY_MS
        rm = _make_rm()
        rm._loss_windows._trade_results.append((_NOW_MS - 2 * _DAY_MS, -100.0))  # within week
        rm._loss_windows._trade_results.append((_NOW_MS - 8 * _DAY_MS, -999.0))  # outside week
        total = rm._window_pnl(_NOW_MS, _WEEK_MS)
        assert total == pytest.approx(-100.0)

    def test_monthly_window_correct(self):
        _DAY_MS = 24 * 3600 * 1000
        _MONTH_MS = 30 * _DAY_MS
        rm = _make_rm()
        rm._loss_windows._trade_results.append((_NOW_MS - 15 * _DAY_MS, -200.0))  # within month
        rm._loss_windows._trade_results.append((_NOW_MS - 31 * _DAY_MS, -999.0))  # outside month
        total = rm._window_pnl(_NOW_MS, _MONTH_MS)
        assert total == pytest.approx(-200.0)

    def test_consecutive_losses_reset_on_win(self):
        rm = _make_rm()
        rm._drawdown._equity = 10_000.0
        rm.on_position_closed("BTC/USDT", -100.0)
        rm.on_position_closed("BTC/USDT", -100.0)
        rm.on_position_closed("BTC/USDT", -100.0)
        assert rm._loss_windows.consecutive_losses == 3

        rm.on_position_closed("ETH/USDT", 200.0)  # win
        assert rm._loss_windows.consecutive_losses == 0

    def test_consecutive_losses_increment_per_loss(self):
        rm = _make_rm()
        rm._drawdown._equity = 10_000.0
        for _ in range(5):
            rm.on_position_closed("BTC/USDT", -10.0)
        assert rm._loss_windows.consecutive_losses == 5


# ===========================================================================
# Volatility circuit breaker
# ===========================================================================


class TestVolatilityCircuitBreaker:
    """The breaker keys on the order's IG EPIC inside ``evaluate_ig_order``."""

    _EPIC = "CS.D.EURUSD.MINI.IP"

    def test_triggers_when_atr_exceeds_2x_average(self):
        rm = _make_rm()
        for _ in range(20):
            rm.update_atr(self._EPIC, 100.0)
        rm._current_atr[self._EPIC] = 201.0  # just above 2× average

        order = _make_ig_order(epic=self._EPIC)
        decision = rm.evaluate_ig_order(order, margin_used=0.0, equity_gbp=10_000.0)
        assert not decision.approved
        assert "volatility" in decision.reason.lower() or "circuit" in decision.reason.lower()

    def test_does_not_trigger_during_normal_volatility(self):
        rm = _make_rm()
        for _ in range(20):
            rm.update_atr(self._EPIC, 100.0)
        rm._current_atr[self._EPIC] = 180.0  # below 2× average

        order = _make_ig_order(epic=self._EPIC)
        decision = rm.evaluate_ig_order(order, margin_used=0.0, equity_gbp=10_000.0)
        assert decision.approved

    def test_no_trigger_with_insufficient_atr_data(self):
        """With only one ATR value, circuit breaker should not fire."""
        rm = _make_rm()
        rm.update_atr(self._EPIC, 100.0)  # single data point
        rm._current_atr[self._EPIC] = 500.0

        order = _make_ig_order(epic=self._EPIC)
        decision = rm.evaluate_ig_order(order, margin_used=0.0, equity_gbp=10_000.0)
        # Only 1 data point, so the breaker uses the buffer but len(buf) < 2 → skipped
        assert decision.approved

    def test_circuit_breaker_uses_rolling_lookback(self):
        """Only the last N ATR values feed the average."""
        cfg = RiskConfig(volatility_atr_lookback=5)
        rm = _make_rm(config=cfg)
        # Old high values get evicted by the rolling deque
        for _ in range(5):
            rm.update_atr(self._EPIC, 1000.0)  # establish high average
        for _ in range(5):
            rm.update_atr(self._EPIC, 10.0)  # replace with low values → avg=10
        rm._current_atr[self._EPIC] = 21.0  # 2.1× avg of 10

        order = _make_ig_order(epic=self._EPIC)
        decision = rm.evaluate_ig_order(order, margin_used=0.0, equity_gbp=10_000.0)
        assert not decision.approved


# ===========================================================================
# IG total-risk gate
# ===========================================================================


def _make_ig_order(
    epic: str = "CS.D.EURUSD.MINI.IP",
    size: float = 1.0,
    stop_distance: float = 50.0,
    direction: str = "BUY",
):
    from bot.core.models import IGOrderRequest

    return IGOrderRequest(
        epic=epic,
        direction=direction,
        size=size,
        stop_distance=stop_distance,
    )


def _make_ig_pos(
    epic: str = "CS.D.EURUSD.MINI.IP",
    entry_level: float = 11_000.0,
    size: float = 1.0,
):
    """An open IG-style Position: entry_price is an IG level (e.g. 11000 for 1.1000)."""
    return Position(
        symbol=epic,
        side=OrderSide.BUY,
        entry_price=entry_level,
        quantity=size,
        current_price=entry_level,
        unrealised_pnl=0.0,
        realised_pnl=0.0,
        opened_at=_NOW_MS,
        updated_at=_NOW_MS,
    )


class TestIGTotalRiskGate:
    """aggregate risk-on cap across all open IG positions."""

    def test_blocks_when_sum_exceeds_cap(self):
        # Equity £10,000, cap 5% (£500).  Two open positions each booked at
        # £200 risk = £400 used. A new £200 order pushes total to £600 > £500.
        cfg = RiskConfig(
            max_total_risk_pct=0.05,
            max_open_positions=10,
            # Disable the per-sector cap for total-risk-gate tests so the
            # synthetic EPIC_A/B/C buckets (which all land in "other") don't
            # trip the concentration check.  Covered by TestSectorRiskCap below.
            max_sector_risk_pct=1.0,
        )
        rm = _make_rm(config=cfg)
        rm._open_positions["EPIC_A"] = _make_ig_pos(epic="EPIC_A")
        rm._open_positions["EPIC_B"] = _make_ig_pos(epic="EPIC_B")
        rm.set_risk_budget("EPIC_A", 200.0)
        rm.set_risk_budget("EPIC_B", 200.0)

        order = _make_ig_order(epic="EPIC_C", size=2.0, stop_distance=100.0)  # £200
        decision = rm.evaluate_ig_order(order, margin_used=0.0, equity_gbp=10_000.0)
        assert not decision.approved
        assert "total risk-on" in decision.reason.lower()

    def test_allows_when_sum_under_cap(self):
        cfg = RiskConfig(
            max_total_risk_pct=0.05,
            max_open_positions=10,
            # Disable the per-sector cap for total-risk-gate tests so the
            # synthetic EPIC_A/B/C buckets (which all land in "other") don't
            # trip the concentration check.  Covered by TestSectorRiskCap below.
            max_sector_risk_pct=1.0,
        )
        rm = _make_rm(config=cfg)
        rm._open_positions["EPIC_A"] = _make_ig_pos(epic="EPIC_A")
        rm.set_risk_budget("EPIC_A", 200.0)

        # New £100 risk → total £300 / £10,000 = 3% < 5%.
        order = _make_ig_order(epic="EPIC_C", size=1.0, stop_distance=100.0)
        decision = rm.evaluate_ig_order(order, margin_used=0.0, equity_gbp=10_000.0)
        assert decision.approved

    def test_trail_armed_shrinks_live_risk_and_allows_entry(self):
        """If TP has armed a trailing stop that locks in profit, the live
        risk-on for that position should drop to zero — freeing headroom
        for a new entry that would otherwise blow the cap."""
        cfg = RiskConfig(
            max_total_risk_pct=0.05,
            max_open_positions=10,
            # Disable the per-sector cap for total-risk-gate tests so the
            # synthetic EPIC_A/B/C buckets (which all land in "other") don't
            # trip the concentration check.  Covered by TestSectorRiskCap below.
            max_sector_risk_pct=1.0,
        )
        rm = _make_rm(config=cfg)
        # Open £400 of risk at entry, with cap = £500.  A new £200 order
        # would normally be blocked (total £600 > £500).
        pos = _make_ig_pos(epic="EPIC_A", entry_level=11_000.0, size=1.0)
        rm._open_positions["EPIC_A"] = pos
        rm.set_risk_budget("EPIC_A", 400.0)
        # Original stop distance = budget/size = 400 pts → stop @ 10,600.
        # Trail at 11,050 (above entry) → fully locked-in profit, live risk 0.
        rm.set_trailing_stop_lookup(lambda epic: 11_050.0 if epic == "EPIC_A" else None)

        order = _make_ig_order(epic="EPIC_B", size=2.0, stop_distance=100.0)  # £200
        decision = rm.evaluate_ig_order(order, margin_used=0.0, equity_gbp=10_000.0)
        assert decision.approved, decision.reason

    def test_trail_below_entry_partially_shrinks_live_risk(self):
        """Trail at half-way between original stop and entry should leave
        ~half the original budget on the table."""
        cfg = RiskConfig(
            max_total_risk_pct=0.05,
            max_open_positions=10,
            # Disable the per-sector cap for total-risk-gate tests so the
            # synthetic EPIC_A/B/C buckets (which all land in "other") don't
            # trip the concentration check.  Covered by TestSectorRiskCap below.
            max_sector_risk_pct=1.0,
        )
        rm = _make_rm(config=cfg)
        pos = _make_ig_pos(epic="EPIC_A", entry_level=11_000.0, size=1.0)
        rm._open_positions["EPIC_A"] = pos
        rm.set_risk_budget("EPIC_A", 400.0)  # original stop @ 10,600
        # Trail at 10,800 → still 200 pts × £1/pt = £200 risk remaining.
        rm.set_trailing_stop_lookup(lambda epic: 10_800.0 if epic == "EPIC_A" else None)

        # New £200 order → total live £200 + £200 = £400 / £10,000 = 4% < 5%.
        order = _make_ig_order(epic="EPIC_B", size=2.0, stop_distance=100.0)
        decision = rm.evaluate_ig_order(order, margin_used=0.0, equity_gbp=10_000.0)
        assert decision.approved
        # But a £400 order would push it to £600 > £500.
        order_big = _make_ig_order(epic="EPIC_C", size=4.0, stop_distance=100.0)
        decision_big = rm.evaluate_ig_order(order_big, margin_used=0.0, equity_gbp=10_000.0)
        assert not decision_big.approved

    def test_lookup_returning_none_uses_full_budget(self):
        cfg = RiskConfig(
            max_total_risk_pct=0.05,
            max_open_positions=10,
            # Disable the per-sector cap for total-risk-gate tests so the
            # synthetic EPIC_A/B/C buckets (which all land in "other") don't
            # trip the concentration check.  Covered by TestSectorRiskCap below.
            max_sector_risk_pct=1.0,
        )
        rm = _make_rm(config=cfg)
        rm._open_positions["EPIC_A"] = _make_ig_pos(epic="EPIC_A")
        rm.set_risk_budget("EPIC_A", 400.0)
        rm.set_trailing_stop_lookup(lambda epic: None)

        order = _make_ig_order(epic="EPIC_B", size=2.0, stop_distance=100.0)  # £200
        decision = rm.evaluate_ig_order(order, margin_used=0.0, equity_gbp=10_000.0)
        assert not decision.approved

    def test_raising_lookup_falls_back_to_full_budget(self):
        cfg = RiskConfig(
            max_total_risk_pct=0.05,
            max_open_positions=10,
            # Disable the per-sector cap for total-risk-gate tests so the
            # synthetic EPIC_A/B/C buckets (which all land in "other") don't
            # trip the concentration check.  Covered by TestSectorRiskCap below.
            max_sector_risk_pct=1.0,
        )
        rm = _make_rm(config=cfg)
        rm._open_positions["EPIC_A"] = _make_ig_pos(epic="EPIC_A")
        rm.set_risk_budget("EPIC_A", 400.0)

        def _boom(epic: str) -> float | None:
            raise RuntimeError("lookup failed")

        rm.set_trailing_stop_lookup(_boom)

        order = _make_ig_order(epic="EPIC_B", size=2.0, stop_distance=100.0)
        decision = rm.evaluate_ig_order(order, margin_used=0.0, equity_gbp=10_000.0)
        assert not decision.approved  # conservative: assume no trail

    def test_on_sell_fill_clears_risk_budget(self):
        rm = _make_rm()
        rm.set_risk_budget("EPIC_A", 150.0)
        rm._open_positions["EPIC_A"] = _make_ig_pos(epic="EPIC_A")
        rm.on_fill(_make_fill(symbol="EPIC_A", side=OrderSide.SELL, qty=1.0, price=11_050.0))
        assert "EPIC_A" not in rm._budget_ledger._budgets

    def test_clear_risk_budget(self):
        rm = _make_rm()
        rm.set_risk_budget("EPIC_A", 100.0)
        rm.clear_risk_budget("EPIC_A")
        assert "EPIC_A" not in rm._budget_ledger._budgets

    def test_risk_budgets_roundtrip_via_state(self):
        rm = _make_rm()
        rm.set_risk_budget("EPIC_A", 123.45)
        rm.set_risk_budget("EPIC_B", 67.89)
        state = rm.get_state()
        assert state.risk_budgets == {"EPIC_A": 123.45, "EPIC_B": 67.89}

        rm2 = _make_rm()
        rm2.load_state(state)
        assert rm2._budget_ledger._budgets == {"EPIC_A": 123.45, "EPIC_B": 67.89}


# ===========================================================================
# State persistence
# ===========================================================================


class TestStatePersistence:
    def test_get_and_load_state_roundtrip(self):
        rm = _make_rm()
        rm.update_equity(10_000.0)
        rm._loss_windows.consecutive_losses = 2
        rm._loss_windows._trade_results.append((_NOW_MS - 1000, -50.0))
        rm._trading_halted = True
        rm._halt_reason = "test"
        rm.update_atr("BTC/USDT", 150.0)

        state = rm.get_state()

        rm2 = _make_rm()
        rm2.update_equity(10_000.0)
        rm2.load_state(state)

        assert rm2._drawdown.peak_equity == pytest.approx(10_000.0)
        assert rm2._loss_windows.consecutive_losses == 2
        assert rm2._trading_halted is True
        assert rm2._halt_reason == "test"
        assert len(rm2._loss_windows._trade_results) == 1
        assert "BTC/USDT" in rm2._atr_values

    def test_drawdown_tier_correct_after_restore(self):
        rm = _make_rm()
        rm.update_equity(10_000.0)
        rm.update_equity(9_400.0)  # YELLOW
        assert rm.current_drawdown_tier == DrawdownTier.YELLOW

        state = rm.get_state()

        rm2 = _make_rm()
        rm2._drawdown._equity = 9_400.0  # must set equity before loading
        rm2.load_state(state)
        assert rm2.current_drawdown_tier == DrawdownTier.YELLOW

    def test_loss_limits_enforced_after_restart(self):
        rm = _make_rm()
        rm.update_equity(10_000.0)
        rm._loss_windows._trade_results.append((_NOW_MS - 1000, -300.1))  # daily limit breach

        state = rm.get_state()

        rm2 = _make_rm()
        rm2.update_equity(10_000.0)
        rm2.load_state(state)

        decision = rm2.evaluate_ig_order(_make_ig_order(), margin_used=0.0, equity_gbp=10_000.0)
        assert not decision.approved
        assert "daily" in decision.reason.lower()

    def test_trading_halt_persists_after_restart(self):
        rm = _make_rm(RiskConfig(drawdown_red_confirm_count=1))
        rm.update_equity(10_000.0)
        rm.update_equity(8_400.0)  # RED (confirm=1) — sets halt
        state = rm.get_state()

        rm2 = _make_rm(RiskConfig(drawdown_red_confirm_count=1))
        rm2._drawdown._equity = 8_400.0
        rm2.load_state(state)

        assert rm2.is_trading_halted
        decision = rm2.evaluate_ig_order(_make_ig_order(), margin_used=0.0, equity_gbp=8_400.0)
        assert not decision.approved
        assert decision.risk_level == RiskLevel.HALTED


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_first_trade_ever_no_history(self):
        """Brand new risk manager should approve a sensible order."""
        rm = _make_rm()
        rm.update_equity(10_000.0)
        decision = rm.evaluate_ig_order(_make_ig_order(), margin_used=0.0, equity_gbp=10_000.0)
        assert decision.approved

    def test_negative_pnl_larger_than_equity(self):
        """Catastrophic loss should not cause errors; halt should trigger."""
        rm = _make_rm(RiskConfig(drawdown_red_confirm_count=1))
        rm.update_equity(10_000.0)
        rm.update_equity(1.0)  # essentially wiped out → RED
        assert rm.is_trading_halted

    def test_multiple_simultaneous_position_closes(self):
        rm = _make_rm()
        rm._drawdown._equity = 10_000.0
        # Add multiple positions then close them
        for sym in ("BTC/USDT", "ETH/USDT", "AVAX/USDT"):
            fill = _make_fill(symbol=sym)
            rm.on_fill(fill)

        assert len(rm._open_positions) == 3

        for sym in ("BTC/USDT", "ETH/USDT", "AVAX/USDT"):
            rm.on_position_closed(sym, -10.0)

        assert len(rm._open_positions) == 0
        assert rm._loss_windows.consecutive_losses == 3

    def test_get_risk_summary_structure(self):
        rm = _make_rm()
        rm.update_equity(10_000.0)
        summary = rm.get_risk_summary()
        assert "current_drawdown_pct" in summary
        assert "drawdown_tier" in summary
        assert "pnl_24h" in summary
        assert "consecutive_losses" in summary
        assert "open_positions" in summary
        assert "trading_halted" in summary
        assert "total_exposure_pct" in summary


# ===========================================================================
# Margin circuit breakers — update_margin_state + state classification
# IG_LIVE_RISK_REFERENCE.md §4.3
# ===========================================================================


@pytest.mark.preflight
class TestMarginCircuitBreakers:
    @staticmethod
    def _account(equity: float, margin: float) -> Any:
        from bot.core.models import AccountUpdate

        return AccountUpdate(
            timestamp=_NOW_MS,
            equity=equity,
            margin_required=margin,
            available_to_deal=max(0.0, equity - margin),
            unrealised_pnl=0.0,
        )

    def test_no_positions_ratio_infinite_state_normal(self) -> None:
        from bot.core.models import MarginCircuitState

        rm = _make_rm()
        state = rm.update_margin_state(self._account(equity=10_000, margin=0))
        assert state == MarginCircuitState.NORMAL
        assert rm.margin_ratio == float("inf")

    def test_classifies_each_threshold(self) -> None:
        """Ratios at each threshold land in the correct state."""
        from bot.core.models import MarginCircuitState

        cases = [
            (10_000.0, 5_000.0, MarginCircuitState.NORMAL),  # 2.0 ratio
            (10_000.0, 12_500.0, MarginCircuitState.HALT_ENTRIES),  # 0.80
            (10_000.0, 15_385.0, MarginCircuitState.DEFENSIVE_CLOSE),  # ~0.65
            (10_000.0, 18_182.0, MarginCircuitState.EMERGENCY_FLATTEN),  # ~0.55
            (10_000.0, 25_000.0, MarginCircuitState.LIQUIDATION),  # 0.40
        ]
        for equity, margin, expected in cases:
            rm = _make_rm()
            actual = rm.update_margin_state(self._account(equity, margin))
            assert actual == expected, (
                f"equity={equity} margin={margin} ratio={equity / margin:.2f}"
            )

    def test_emits_event_only_on_transition(self) -> None:
        """Two consecutive updates at the same state should fire one event."""
        rm = _make_rm()
        emits: list[tuple[str, Any]] = []

        async def _capture(payload: Any) -> None:
            emits.append(("captured", payload))

        # Replace the bus mock with a real one + capture
        bus = EventBus()
        from bot.core.event_bus import EVENT_MARGIN_BREAKER

        bus.subscribe(EVENT_MARGIN_BREAKER, _capture)
        rm._event_bus = bus

        import asyncio

        async def _run() -> None:
            rm.update_margin_state(self._account(10_000, 12_500))  # HALT_ENTRIES
            rm.update_margin_state(self._account(10_000, 12_500))  # same state
            rm.update_margin_state(self._account(10_000, 15_385))  # DEFENSIVE_CLOSE
            await asyncio.sleep(0.01)

        asyncio.run(_run())

        # Two transitions emitted: NORMAL→HALT and HALT→DEFENSIVE
        actions = [c[1].action for c in emits]
        assert actions == ["halt_entries", "close_worst"]

    def test_pre_trade_gate_rejects_when_breaker_active(self) -> None:
        from bot.core.models import IGOrderRequest, RiskLevel

        rm = _make_rm()
        rm.update_margin_state(self._account(10_000, 12_500))  # → HALT_ENTRIES

        order = IGOrderRequest(epic="CS.D.EURUSD.MINI.IP", direction="BUY", size=1.0)
        decision = rm.evaluate_ig_order(order, margin_used=100.0, equity_gbp=10_000.0)
        assert decision.approved is False
        assert decision.risk_level == RiskLevel.CRITICAL
        assert "Margin circuit breaker" in decision.reason

    def test_recovery_back_to_normal_re_enables_entries(self) -> None:
        """If margin ratio recovers above the halt threshold, new entries are
        accepted again — the breaker isn't sticky."""
        from bot.core.models import IGOrderRequest

        rm = _make_rm()
        rm.update_margin_state(self._account(10_000, 12_500))  # HALT_ENTRIES
        rm.update_margin_state(self._account(10_000, 5_000))  # back to NORMAL (ratio 2.0)

        order = IGOrderRequest(
            epic="CS.D.EURUSD.MINI.IP", direction="BUY", size=1.0, stop_distance=5.0
        )
        decision = rm.evaluate_ig_order(order, margin_used=100.0, equity_gbp=10_000.0)
        assert decision.approved is True

    def test_liquidation_emits_shutdown(self) -> None:
        """Below the broker's 0.50 floor we shut down — the broker is already
        closing positions and we have no defensive options left."""
        from bot.core.event_bus import EVENT_SHUTDOWN

        rm = _make_rm()
        bus = EventBus()
        shutdown_payloads: list[Any] = []

        async def _capture(payload: Any) -> None:
            shutdown_payloads.append(payload)

        bus.subscribe(EVENT_SHUTDOWN, _capture)
        rm._event_bus = bus

        import asyncio

        async def _run() -> None:
            rm.update_margin_state(self._account(10_000, 25_000))  # ratio 0.40
            await asyncio.sleep(0.01)

        asyncio.run(_run())

        assert shutdown_payloads
        assert "margin_liquidation_floor" in str(shutdown_payloads[0])


# ===========================================================================
# Tier-aware pre-trade margin estimate (evaluate_ig_order step 7c)
# ===========================================================================


class TestPreTradeMarginGate:
    @staticmethod
    def _order() -> Any:
        from bot.core.models import IGOrderRequest

        return IGOrderRequest(
            epic="CS.D.EURUSD.MINI.IP",
            direction="BUY",
            size=1.0,
            stop_distance=5.0,
        )

    def test_zero_estimate_skips_the_gate(self) -> None:
        """Backward-compatible: callers that don't pass an estimate must not be
        affected by this check."""
        rm = _make_rm()
        decision = rm.evaluate_ig_order(self._order(), margin_used=0.0, equity_gbp=10_000.0)
        assert decision.approved is True

    def test_safe_estimate_passes(self) -> None:
        """Adding £500 of margin to an empty £10k account → post ratio 20.0,
        well above the 0.80 halt threshold."""
        rm = _make_rm()
        decision = rm.evaluate_ig_order(
            self._order(),
            margin_used=0.0,
            equity_gbp=10_000.0,
            estimated_margin_gbp=500.0,
        )
        assert decision.approved is True

    def test_estimate_that_would_breach_halt_ratio_rejects(self) -> None:
        """£3k margin used (under the 50 % cap) + £10k proposed → projected
        ratio 10000/13000 ≈ 0.77, below the 0.80 halt threshold — must reject
        on step 7c specifically (not on the legacy 50 % cap at step 7)."""
        from bot.core.models import RiskLevel

        rm = _make_rm()
        decision = rm.evaluate_ig_order(
            self._order(),
            margin_used=3_000.0,
            equity_gbp=10_000.0,
            estimated_margin_gbp=10_000.0,
        )
        assert decision.approved is False
        assert decision.risk_level == RiskLevel.CRITICAL
        assert "Pre-trade margin check" in decision.reason

    def test_just_under_halt_ratio_passes(self) -> None:
        """£2k margin used + £500 proposed → projected ratio 10000/2500 = 4.0,
        comfortably above 0.80 — must approve."""
        rm = _make_rm()
        decision = rm.evaluate_ig_order(
            self._order(),
            margin_used=2_000.0,
            equity_gbp=10_000.0,
            estimated_margin_gbp=500.0,
        )
        assert decision.approved is True

    def test_zero_equity_skips_the_gate(self) -> None:
        """If we don't have a positive equity reading yet, don't gate on a
        bogus ratio — other checks earlier in the pipeline handle that case."""
        rm = _make_rm()
        decision = rm.evaluate_ig_order(
            self._order(),
            margin_used=100.0,
            equity_gbp=0.0,
            estimated_margin_gbp=500.0,
        )
        # zero equity won't trip the gate; other checks may still reject
        # (loss limits etc. are inert in this minimal setup so we just verify
        # the margin gate didn't fire)
        assert "Pre-trade margin check" not in (decision.reason or "")


# ===========================================================================
# Per-sector concentration cap (bot/risk/sectors.py + RiskConfig.max_sector_risk_pct)
# Complements the total-risk gate: catches concentration that pairwise
# correlation_threshold doesn't (e.g. multiple yen-cross longs each below
# 0.65 pairwise but all moving with the yen).
# ===========================================================================


class TestSectorRiskCap:
    # Real EPIC strings so sector_for() resolves to real buckets.
    _EUR_USD = "CS.D.EURUSD.TODAY.IP"  # fx_usd
    _GBP_USD = "CS.D.GBPUSD.TODAY.IP"  # fx_usd
    _USD_JPY = "CS.D.USDJPY.TODAY.IP"  # fx_usd
    _XAU = "CS.D.USCGC.TODAY.IP"  # metals
    _XAG = "CS.D.USCSI.TODAY.IP"  # metals
    _GOLD = "CS.D.USCGC.TODAY.IP"  # metals

    def _cfg(self) -> RiskConfig:
        # Total-risk cap loose so the sector cap is the binding one in tests.
        return RiskConfig(
            max_total_risk_pct=1.0,
            max_open_positions=10,
            max_sector_risk_pct=0.025,  # cap = £250 on £10k
        )

    def test_first_position_in_sector_accepted(self) -> None:
        rm = _make_rm(config=self._cfg())
        order = _make_ig_order(epic=self._EUR_USD, size=2.0, stop_distance=100.0)  # £200
        decision = rm.evaluate_ig_order(order, margin_used=0.0, equity_gbp=10_000.0)
        assert decision.approved, decision.reason

    def test_second_position_within_sector_cap_accepted(self) -> None:
        """£100 booked + new £100 = £200 ≤ £250 sector cap."""
        rm = _make_rm(config=self._cfg())
        rm._open_positions[self._EUR_USD] = _make_ig_pos(epic=self._EUR_USD)
        rm.set_risk_budget(self._EUR_USD, 100.0)

        order = _make_ig_order(epic=self._GBP_USD, size=1.0, stop_distance=100.0)
        decision = rm.evaluate_ig_order(order, margin_used=0.0, equity_gbp=10_000.0)
        assert decision.approved, decision.reason

    def test_concentration_within_one_sector_rejected(self) -> None:
        """Three USD-pair longs each at £100 risk = £300 in fx_usd > £250 cap.
        This is the cluster the pairwise correlation cap misses."""
        rm = _make_rm(config=self._cfg())
        rm._open_positions[self._EUR_USD] = _make_ig_pos(epic=self._EUR_USD)
        rm._open_positions[self._GBP_USD] = _make_ig_pos(epic=self._GBP_USD)
        rm.set_risk_budget(self._EUR_USD, 100.0)
        rm.set_risk_budget(self._GBP_USD, 100.0)

        order = _make_ig_order(epic=self._USD_JPY, size=1.0, stop_distance=100.0)
        decision = rm.evaluate_ig_order(order, margin_used=0.0, equity_gbp=10_000.0)
        assert not decision.approved
        assert "sector 'fx_usd'" in decision.reason.lower()

    def test_other_sector_still_accepted_when_one_is_full(self) -> None:
        """fx_usd at the cap must not block an entry in metals."""
        rm = _make_rm(config=self._cfg())
        rm._open_positions[self._EUR_USD] = _make_ig_pos(epic=self._EUR_USD)
        rm._open_positions[self._GBP_USD] = _make_ig_pos(epic=self._GBP_USD)
        # Cram £250 into fx_usd — at the sector cap exactly.
        rm.set_risk_budget(self._EUR_USD, 125.0)
        rm.set_risk_budget(self._GBP_USD, 125.0)

        order = _make_ig_order(epic=self._XAU, size=2.0, stop_distance=100.0)  # £200 → metals
        decision = rm.evaluate_ig_order(order, margin_used=0.0, equity_gbp=10_000.0)
        assert decision.approved, decision.reason

    def test_proposed_order_fills_its_own_sector(self) -> None:
        """XAU already booked at £100; XAG proposed at £200 → metals £300 > £250 cap."""
        rm = _make_rm(config=self._cfg())
        rm._open_positions[self._XAU] = _make_ig_pos(epic=self._XAU)
        rm.set_risk_budget(self._XAU, 100.0)

        order = _make_ig_order(epic=self._XAG, size=2.0, stop_distance=100.0)
        decision = rm.evaluate_ig_order(order, margin_used=0.0, equity_gbp=10_000.0)
        assert not decision.approved
        assert "sector 'metals'" in decision.reason.lower()

    def test_trail_armed_shrinks_sector_risk_and_allows_entry(self) -> None:
        """A trailing stop that locks in profit on an existing fx_usd position
        must free headroom for a new fx_usd entry — the cap operates on *live*
        risk, not entry-time budget."""
        rm = _make_rm(config=self._cfg())
        pos = _make_ig_pos(epic=self._EUR_USD, entry_level=11_000.0, size=1.0)
        rm._open_positions[self._EUR_USD] = pos
        rm.set_risk_budget(self._EUR_USD, 200.0)  # original stop @ 10,800
        # Trail above entry → fully locked-in, live risk = 0.
        rm.set_trailing_stop_lookup(lambda epic: 11_050.0 if epic == self._EUR_USD else None)

        # New £200 fx_usd order: sector live risk = 0 + 200 = 200 < 250 cap.
        order = _make_ig_order(epic=self._GBP_USD, size=2.0, stop_distance=100.0)
        decision = rm.evaluate_ig_order(order, margin_used=0.0, equity_gbp=10_000.0)
        assert decision.approved, decision.reason
