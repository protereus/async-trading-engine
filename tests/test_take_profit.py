"""Tests for TakeProfitManager — five independent exit components.

All tests are synchronous and inject deterministic inputs.
No network calls, no Kronos imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from bot.strategy.take_profit import (
    ExitReason,
    PositionTPState,
    TakeProfitConfig,
    TakeProfitManager,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_MS = 1_700_000_000_000  # arbitrary epoch start for tests


@dataclass
class FakeSignal:
    symbol: str
    mean_return: float
    std_return: float
    direction_confidence: float
    uncertainty: float
    stop_pct: float
    tradeable: bool = True
    predicted_close: float = 100.0


@dataclass
class FakeSentiment:
    asset: str
    sentiment: float
    confidence: float


def _mgr(overrides: dict | None = None) -> TakeProfitManager:
    """Return a TakeProfitManager with default config and pred_len=120."""
    cfg = TakeProfitConfig(**(overrides or {}))
    return TakeProfitManager(cfg, pred_len=120)


def _register(
    mgr: TakeProfitManager,
    symbol: str = "EUR/USD",
    entry_price: float = 1.1000,
    mean_return: float = 0.05,
    stop_pct: float = 0.02,
    opened_at_ms: int = BASE_MS,
) -> FakeSignal:
    sig = FakeSignal(
        symbol=symbol,
        mean_return=mean_return,
        std_return=0.01,
        direction_confidence=0.80,
        uncertainty=1.0,
        stop_pct=stop_pct,
    )
    mgr.register_position(symbol, entry_price, sig, opened_at_ms)
    return sig


# ---------------------------------------------------------------------------
# Static take-profit
# ---------------------------------------------------------------------------


class TestStaticTP:
    def test_tp_triggers_at_correct_price(self) -> None:
        """mean_return=0.05, stop_pct=0.02 → tp_pct = max(0.02×1.5=0.03, 0.05×0.8=0.04) = 0.04"""
        mgr = _mgr()
        _register(mgr, mean_return=0.05, stop_pct=0.02, entry_price=100.0)
        # Just below target: no exit
        dec = mgr.evaluate_price("EUR/USD", 103.99, BASE_MS + 1000)
        assert not dec.should_exit
        # At target (100 * 1.04 = 104.0): exit
        dec = mgr.evaluate_price("EUR/USD", 104.0, BASE_MS + 1000)
        assert dec.should_exit
        assert dec.reason == ExitReason.STATIC_TP

    def test_tp_floor_from_rr_multiplier(self) -> None:
        """mean_return=0.005, stop_pct=0.02 → tp_pct = max(0.02×1.5=0.03, 0.005×0.8=0.004) = 0.03"""
        mgr = _mgr()
        _register(mgr, mean_return=0.005, stop_pct=0.02, entry_price=100.0)
        # RR floor wins: tp at 103.0
        dec = mgr.evaluate_price("EUR/USD", 102.99, BASE_MS + 1000)
        assert not dec.should_exit
        dec = mgr.evaluate_price("EUR/USD", 103.0, BASE_MS + 1000)
        assert dec.should_exit
        assert dec.reason == ExitReason.STATIC_TP

    def test_disabled_never_fires(self) -> None:
        mgr = _mgr({"static_enabled": False})
        _register(mgr, mean_return=0.05, stop_pct=0.02, entry_price=100.0)
        dec = mgr.evaluate_price("EUR/USD", 200.0, BASE_MS + 1000)
        assert not dec.should_exit or dec.reason != ExitReason.STATIC_TP

    def test_unknown_symbol_returns_hold(self) -> None:
        mgr = _mgr()
        dec = mgr.evaluate_price("UNKNOWN", 100.0, BASE_MS)
        assert not dec.should_exit
        assert dec.reason == ExitReason.HOLD


# ---------------------------------------------------------------------------
# Trailing stop — Stage 1 (breakeven)
# ---------------------------------------------------------------------------


class TestTrailingBreakeven:
    def test_stage1_arms_at_exactly_1x_stop(self) -> None:
        """With stop_pct=0.02, Stage 1 arms when profit >= 2%."""
        mgr = _mgr({"trail_activation_mult": 2.0, "trail_multiplier": 0.5})
        _register(mgr, entry_price=100.0, stop_pct=0.02)
        # 1.9% profit → not armed yet
        mgr.evaluate_price("EUR/USD", 101.9, BASE_MS + 1)
        state: PositionTPState = mgr._positions["EUR/USD"]
        assert not state.breakeven_armed
        # Exactly 2% profit → breakeven armed
        mgr.evaluate_price("EUR/USD", 102.0, BASE_MS + 2)
        assert state.breakeven_armed
        # Stop is entry * (1 + buffer) = 100 * 1.001 = 100.1
        assert state.current_trailing_stop == pytest.approx(100.1, rel=1e-5)

    def test_stage1_exit_fires_when_price_crosses_stop(self) -> None:
        mgr = _mgr({"trail_activation_mult": 2.0})
        _register(mgr, entry_price=100.0, stop_pct=0.02)
        # Arm Stage 1
        mgr.evaluate_price("EUR/USD", 102.0, BASE_MS + 1)
        # Price comes back below breakeven stop (100.1)
        dec = mgr.evaluate_price("EUR/USD", 100.05, BASE_MS + 2)
        assert dec.should_exit
        assert dec.reason == ExitReason.TRAILING_STOP_BREAKEVEN

    def test_stage1_no_exit_above_stop(self) -> None:
        mgr = _mgr({"trail_activation_mult": 2.0})
        _register(mgr, entry_price=100.0, stop_pct=0.02)
        mgr.evaluate_price("EUR/USD", 102.0, BASE_MS + 1)
        dec = mgr.evaluate_price("EUR/USD", 100.2, BASE_MS + 2)
        assert not dec.should_exit


# ---------------------------------------------------------------------------
# Trailing stop — Stage 2 (ratchet)
# ---------------------------------------------------------------------------


class TestTrailingRatchet:
    def test_stage2_arms_at_2x_stop(self) -> None:
        """stop_pct=0.02, trail_activation_mult=2 → arm at 4% profit."""
        mgr = _mgr()
        _register(mgr, entry_price=100.0, stop_pct=0.02)
        # 3.9% — no Stage 2
        mgr.evaluate_price("EUR/USD", 103.9, BASE_MS + 1)
        state = mgr._positions["EUR/USD"]
        assert not state.trail_armed
        # 4% profit → Stage 2 armed
        mgr.evaluate_price("EUR/USD", 104.0, BASE_MS + 2)
        assert state.trail_armed
        # trail_pct = 0.02 * 0.5 = 0.01; stop = 104 * (1 - 0.01) = 102.96
        assert state.current_trailing_stop == pytest.approx(104.0 * (1 - 0.01), rel=1e-5)

    def test_stage2_ratchets_up_as_peak_rises(self) -> None:
        mgr = _mgr()
        _register(mgr, entry_price=100.0, stop_pct=0.02)
        # Arm Stage 2 at 104
        mgr.evaluate_price("EUR/USD", 104.0, BASE_MS + 1)
        s1 = mgr._positions["EUR/USD"].current_trailing_stop
        # Price rises to 110 → stop ratchets up
        mgr.evaluate_price("EUR/USD", 110.0, BASE_MS + 2)
        s2 = mgr._positions["EUR/USD"].current_trailing_stop
        assert s2 is not None and s1 is not None and s2 > s1
        # Expected: 110 * (1 - 0.01) = 108.9
        assert s2 == pytest.approx(108.9, rel=1e-5)

    def test_stage2_never_moves_down_on_retracement(self) -> None:
        mgr = _mgr()
        _register(mgr, entry_price=100.0, stop_pct=0.02)
        # Peak at 110 → stop = 108.9
        mgr.evaluate_price("EUR/USD", 110.0, BASE_MS + 1)
        peak_stop = mgr._positions["EUR/USD"].current_trailing_stop
        # Price retraces to 105 (still above stop) — stop must not move down
        mgr.evaluate_price("EUR/USD", 105.0, BASE_MS + 2)
        assert mgr._positions["EUR/USD"].current_trailing_stop == peak_stop

    def test_stage2_exit_fires_when_price_crosses_stop(self) -> None:
        mgr = _mgr()
        _register(mgr, entry_price=100.0, stop_pct=0.02)
        mgr.evaluate_price("EUR/USD", 110.0, BASE_MS + 1)
        # stop = 108.9; price drops to 108.89 → exit
        dec = mgr.evaluate_price("EUR/USD", 108.89, BASE_MS + 2)
        assert dec.should_exit
        assert dec.reason == ExitReason.TRAILING_STOP_RATCHET

    def test_stage2_priority_over_stage1(self) -> None:
        """When both stages could fire simultaneously, Stage 2 wins."""
        mgr = _mgr()
        _register(mgr, entry_price=100.0, stop_pct=0.02)
        # Jump straight to Stage 2 territory (4%+)
        mgr.evaluate_price("EUR/USD", 105.0, BASE_MS + 1)
        state = mgr._positions["EUR/USD"]
        # trail_armed should be set, and breakeven_armed set implicitly
        assert state.trail_armed
        # Drop to trailing stop → reason is RATCHET not BREAKEVEN
        stop = state.current_trailing_stop
        assert stop is not None
        dec = mgr.evaluate_price("EUR/USD", stop - 0.001, BASE_MS + 2)
        assert dec.reason == ExitReason.TRAILING_STOP_RATCHET

    def test_peak_price_updates_on_higher_close(self) -> None:
        mgr = _mgr()
        _register(mgr, entry_price=100.0)
        mgr.evaluate_price("EUR/USD", 105.0, BASE_MS + 1)
        assert mgr._positions["EUR/USD"].peak_price == 105.0
        mgr.evaluate_price("EUR/USD", 103.0, BASE_MS + 2)
        assert mgr._positions["EUR/USD"].peak_price == 105.0  # didn't drop
        mgr.evaluate_price("EUR/USD", 108.0, BASE_MS + 3)
        assert mgr._positions["EUR/USD"].peak_price == 108.0

    def test_trailing_disabled_never_fires(self) -> None:
        mgr = _mgr({"trailing_enabled": False})
        _register(mgr, entry_price=100.0, stop_pct=0.02)
        mgr.evaluate_price("EUR/USD", 105.0, BASE_MS + 1)
        dec = mgr.evaluate_price("EUR/USD", 99.0, BASE_MS + 2)
        trailing_reasons = {ExitReason.TRAILING_STOP_RATCHET, ExitReason.TRAILING_STOP_BREAKEVEN}
        assert not dec.should_exit or dec.reason not in trailing_reasons


# ---------------------------------------------------------------------------
# Signal decay
# ---------------------------------------------------------------------------


class TestSignalDecay:
    def test_mean_return_flip_immediate_exit(self) -> None:
        mgr = _mgr()
        _register(mgr, mean_return=0.05)
        flipped_sig = FakeSignal("EUR/USD", -0.02, 0.01, 0.75, 0.5, 0.02)
        dec = mgr.evaluate_signal("EUR/USD", flipped_sig, in_topk=True)
        assert dec.should_exit
        assert dec.reason == ExitReason.SIGNAL_DECAY_FLIP

    def test_confidence_drop_once_no_exit(self) -> None:
        mgr = _mgr()
        _register(mgr)
        low_conf = FakeSignal("EUR/USD", 0.05, 0.01, 0.50, 1.0, 0.02)
        dec = mgr.evaluate_signal("EUR/USD", low_conf, in_topk=True)
        assert not dec.should_exit
        assert mgr._positions["EUR/USD"].signal_decay_strikes == 1

    def test_confidence_drop_twice_exits(self) -> None:
        mgr = _mgr({"signal_decay_max_strikes": 2})
        _register(mgr)
        low_conf = FakeSignal("EUR/USD", 0.05, 0.01, 0.50, 1.0, 0.02)
        mgr.evaluate_signal("EUR/USD", low_conf, in_topk=True)
        dec = mgr.evaluate_signal("EUR/USD", low_conf, in_topk=True)
        assert dec.should_exit
        assert dec.reason == ExitReason.SIGNAL_DECAY_STRIKES

    def test_confidence_recovery_resets_strikes(self) -> None:
        mgr = _mgr()
        _register(mgr)
        low_conf = FakeSignal("EUR/USD", 0.05, 0.01, 0.50, 1.0, 0.02)
        healthy = FakeSignal("EUR/USD", 0.05, 0.01, 0.80, 0.5, 0.02)
        # 1 strike
        mgr.evaluate_signal("EUR/USD", low_conf, in_topk=True)
        assert mgr._positions["EUR/USD"].signal_decay_strikes == 1
        # Recovery → reset
        mgr.evaluate_signal("EUR/USD", healthy, in_topk=True)
        assert mgr._positions["EUR/USD"].signal_decay_strikes == 0
        # Another drop: strike=1 again, not 2
        mgr.evaluate_signal("EUR/USD", low_conf, in_topk=True)
        assert mgr._positions["EUR/USD"].signal_decay_strikes == 1

    def test_not_in_topk_uses_separate_counter(self) -> None:
        # Healthy signal + not_in_topk increments the topk_miss counter, not the
        # signal-quality counter, and does not exit before the topk threshold.
        mgr = _mgr({"signal_decay_max_strikes": 2, "signal_decay_max_topk_misses": 3})
        _register(mgr)
        healthy_sig = FakeSignal("EUR/USD", 0.05, 0.01, 0.80, 0.5, 0.02)
        for _ in range(2):
            dec = mgr.evaluate_signal("EUR/USD", healthy_sig, in_topk=False)
            assert not dec.should_exit
        state = mgr._positions["EUR/USD"]
        assert state.topk_miss_strikes == 2
        assert state.signal_decay_strikes == 0

    def test_not_in_topk_exits_after_max_misses(self) -> None:
        mgr = _mgr({"signal_decay_max_topk_misses": 3})
        _register(mgr)
        healthy_sig = FakeSignal("EUR/USD", 0.05, 0.01, 0.80, 0.5, 0.02)
        mgr.evaluate_signal("EUR/USD", healthy_sig, in_topk=False)
        mgr.evaluate_signal("EUR/USD", healthy_sig, in_topk=False)
        dec = mgr.evaluate_signal("EUR/USD", healthy_sig, in_topk=False)
        assert dec.should_exit
        assert dec.reason == ExitReason.SIGNAL_DECAY_STRIKES
        assert "topk_misses" in dec.reasoning

    def test_in_topk_resets_topk_miss_strikes(self) -> None:
        mgr = _mgr({"signal_decay_max_topk_misses": 3})
        _register(mgr)
        healthy_sig = FakeSignal("EUR/USD", 0.05, 0.01, 0.80, 0.5, 0.02)
        mgr.evaluate_signal("EUR/USD", healthy_sig, in_topk=False)
        mgr.evaluate_signal("EUR/USD", healthy_sig, in_topk=False)
        assert mgr._positions["EUR/USD"].topk_miss_strikes == 2
        # Reselection resets
        mgr.evaluate_signal("EUR/USD", healthy_sig, in_topk=True)
        assert mgr._positions["EUR/USD"].topk_miss_strikes == 0

    def test_signal_decay_disabled_no_strikes(self) -> None:
        mgr = _mgr({"signal_decay_enabled": False})
        _register(mgr)
        low_conf = FakeSignal("EUR/USD", 0.05, 0.01, 0.50, 1.0, 0.02)
        dec = mgr.evaluate_signal("EUR/USD", low_conf, in_topk=False)
        assert not dec.should_exit

    def test_high_uncertainty_increments_strikes(self) -> None:
        mgr = _mgr({"signal_decay_max_strikes": 2, "signal_decay_max_uncertainty": 3.0})
        _register(mgr)
        high_unc = FakeSignal("EUR/USD", 0.05, 0.01, 0.80, 3.5, 0.02)  # uncertainty > 3.0
        mgr.evaluate_signal("EUR/USD", high_unc, in_topk=True)
        dec = mgr.evaluate_signal("EUR/USD", high_unc, in_topk=True)
        assert dec.should_exit
        assert dec.reason == ExitReason.SIGNAL_DECAY_STRIKES


# ---------------------------------------------------------------------------
# Time-based exit
# ---------------------------------------------------------------------------


class TestTimeExit:
    # Use entry_price=1.10 and evaluate at the same price (no profit) so no
    # other component (static TP or trailing stop) fires before the time limit.
    ENTRY = 1.1000

    def test_exit_after_pred_len_hours(self) -> None:
        """Default: max_age = 120h × 1.0 = 432_000_000ms."""
        mgr = _mgr()
        _register(mgr, entry_price=self.ENTRY, opened_at_ms=BASE_MS)
        max_age_ms = 120 * 3_600_000
        # 1ms before limit → no time exit
        dec = mgr.evaluate_price("EUR/USD", self.ENTRY, BASE_MS + max_age_ms - 1)
        assert not dec.should_exit or dec.reason != ExitReason.TIME_LIMIT
        # At limit → time exit
        dec = mgr.evaluate_price("EUR/USD", self.ENTRY, BASE_MS + max_age_ms)
        assert dec.should_exit
        assert dec.reason == ExitReason.TIME_LIMIT

    def test_no_exit_before_limit(self) -> None:
        mgr = _mgr()
        _register(mgr, entry_price=self.ENTRY, opened_at_ms=BASE_MS)
        dec = mgr.evaluate_price("EUR/USD", self.ENTRY, BASE_MS + 3_600_000)  # 1h
        assert not dec.should_exit or dec.reason != ExitReason.TIME_LIMIT

    def test_multiplier_configurable(self) -> None:
        """time_horizon_multiplier=0.5 → max_age = 60h."""
        mgr = _mgr({"time_horizon_multiplier": 0.5})
        _register(mgr, entry_price=self.ENTRY, opened_at_ms=BASE_MS)
        max_age_ms = 60 * 3_600_000
        dec = mgr.evaluate_price("EUR/USD", self.ENTRY, BASE_MS + max_age_ms)
        assert dec.should_exit
        assert dec.reason == ExitReason.TIME_LIMIT

    def test_time_disabled_never_fires(self) -> None:
        mgr = _mgr({"time_enabled": False})
        _register(mgr, entry_price=self.ENTRY, opened_at_ms=BASE_MS)
        dec = mgr.evaluate_price("EUR/USD", self.ENTRY, BASE_MS + 10 * 24 * 3_600_000)
        assert not dec.should_exit or dec.reason != ExitReason.TIME_LIMIT


# ---------------------------------------------------------------------------
# Sentiment reversal
# ---------------------------------------------------------------------------


class TestSentimentReversal:
    def test_exits_on_confident_bearish_signal(self) -> None:
        mgr = _mgr({"sentiment_reversal_enabled": True})
        _register(mgr)
        sent = FakeSentiment("EUR/USD", sentiment=-0.4, confidence=0.7)
        dec = mgr.evaluate_sentiment("EUR/USD", sent)
        assert dec.should_exit
        assert dec.reason == ExitReason.SENTIMENT_REVERSAL

    def test_no_exit_low_confidence(self) -> None:
        mgr = _mgr({"sentiment_reversal_enabled": True})
        _register(mgr)
        sent = FakeSentiment("EUR/USD", sentiment=-0.4, confidence=0.5)  # < 0.6
        dec = mgr.evaluate_sentiment("EUR/USD", sent)
        assert not dec.should_exit

    def test_no_exit_neutral_sentiment(self) -> None:
        mgr = _mgr({"sentiment_reversal_enabled": True})
        _register(mgr)
        sent = FakeSentiment("EUR/USD", sentiment=0.0, confidence=0.9)
        dec = mgr.evaluate_sentiment("EUR/USD", sent)
        assert not dec.should_exit

    def test_disabled_never_fires(self) -> None:
        mgr = _mgr({"sentiment_reversal_enabled": False})
        _register(mgr)
        sent = FakeSentiment("EUR/USD", sentiment=-1.0, confidence=1.0)
        dec = mgr.evaluate_sentiment("EUR/USD", sent)
        assert not dec.should_exit

    def test_exact_threshold_boundary(self) -> None:
        """sentiment == threshold (-0.3) should trigger exit."""
        mgr = _mgr({"sentiment_reversal_enabled": True, "sentiment_reversal_threshold": -0.3})
        _register(mgr)
        sent = FakeSentiment("EUR/USD", sentiment=-0.3, confidence=0.6)
        dec = mgr.evaluate_sentiment("EUR/USD", sent)
        assert dec.should_exit

    def test_above_threshold_no_exit(self) -> None:
        mgr = _mgr({"sentiment_reversal_enabled": True})
        _register(mgr)
        sent = FakeSentiment("EUR/USD", sentiment=-0.2, confidence=0.9)
        dec = mgr.evaluate_sentiment("EUR/USD", sent)
        assert not dec.should_exit


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    def test_trailing_ratchet_beats_static_tp(self) -> None:
        """When both trailing Stage 2 and static TP fire on the same candle,
        trailing wins (higher priority)."""
        # Static TP at 4%: entry=100, mean_return=0.05, stop_pct=0.02
        # tp_pct = max(0.03, 0.04) = 0.04 → tp_price = 104
        # Stage 2 armed at 4% (same point); trail_stop = 104 * (1-0.01) = 102.96
        mgr = _mgr()
        _register(mgr, entry_price=100.0, mean_return=0.05, stop_pct=0.02)
        # Price hits 104 — arms Stage 2 and reaches static TP
        dec = mgr.evaluate_price("EUR/USD", 104.0, BASE_MS + 1)
        # Could be STATIC_TP or RATCHET; but trailing stop isn't triggered yet
        # (price is AT 104, stop is 102.96 which price is above)
        # So static TP fires: no trailing stop breach
        assert dec.should_exit
        # This verifies evaluate_price checks trailing stop BEFORE static TP,
        # and that static TP fires when trailing stop hasn't been breached
        assert dec.reason in {ExitReason.STATIC_TP, ExitReason.TRAILING_STOP_RATCHET}

    def test_trailing_ratchet_beats_static_tp_on_breach(self) -> None:
        """After Stage 2 armed, if price falls below trailing stop and is also
        at static TP, trailing wins because it's checked first."""
        mgr = _mgr()
        _register(mgr, entry_price=100.0, mean_return=0.05, stop_pct=0.02)
        # Arm Stage 2 with a high peak
        mgr.evaluate_price("EUR/USD", 110.0, BASE_MS + 1)
        # stop = 110 * 0.99 = 108.9
        # Now price drops below trailing stop (108.89)
        dec = mgr.evaluate_price("EUR/USD", 108.89, BASE_MS + 2)
        assert dec.should_exit
        assert dec.reason == ExitReason.TRAILING_STOP_RATCHET


# ---------------------------------------------------------------------------
# State persistence (snapshot / restore)
# ---------------------------------------------------------------------------


class TestSnapshotRestore:
    def test_snapshot_restore_preserves_state(self) -> None:
        mgr = _mgr()
        _register(mgr, entry_price=1.2000, mean_return=0.04, stop_pct=0.015)
        # Simulate some price action to update state
        mgr.evaluate_price("EUR/USD", 1.23, BASE_MS + 1)  # arm Stage 2
        original_state = mgr._positions["EUR/USD"]

        snap = mgr.snapshot()
        mgr2 = _mgr()
        mgr2.restore(snap)

        restored = mgr2._positions["EUR/USD"]
        assert restored.symbol == original_state.symbol
        assert restored.entry_price == original_state.entry_price
        assert restored.entry_mean_return == original_state.entry_mean_return
        assert restored.entry_stop_pct == original_state.entry_stop_pct
        assert restored.peak_price == original_state.peak_price
        assert restored.opened_at_ms == original_state.opened_at_ms
        assert restored.signal_decay_strikes == original_state.signal_decay_strikes
        assert restored.topk_miss_strikes == original_state.topk_miss_strikes
        assert restored.breakeven_armed == original_state.breakeven_armed
        assert restored.trail_armed == original_state.trail_armed
        assert restored.current_trailing_stop == pytest.approx(
            original_state.current_trailing_stop or 0.0, rel=1e-9
        )

    def test_restore_handles_missing_symbols(self) -> None:
        """Positions closed during downtime are absent from snapshot; restore is graceful."""
        mgr = _mgr()
        _register(mgr, "EUR/USD")
        _register(mgr, "GBP/USD")
        snap = mgr.snapshot()

        # Simulate EUR/USD was closed while bot was down
        del snap["EUR/USD"]

        mgr2 = _mgr()
        mgr2.restore(snap)
        assert "EUR/USD" not in mgr2._positions
        assert "GBP/USD" in mgr2._positions

    def test_restore_handles_empty_snapshot(self) -> None:
        mgr = _mgr()
        _register(mgr)
        mgr.restore({})
        assert len(mgr._positions) == 0

    def test_snapshot_of_empty_manager(self) -> None:
        mgr = _mgr()
        assert mgr.snapshot() == {}

    def test_restore_preserves_trailing_stop_none(self) -> None:
        """current_trailing_stop=None must survive a round-trip."""
        mgr = _mgr()
        _register(mgr, entry_price=100.0)
        # Don't trigger any trailing stage
        snap = mgr.snapshot()
        mgr2 = _mgr()
        mgr2.restore(snap)
        assert mgr2._positions["EUR/USD"].current_trailing_stop is None

    def test_deregister_removes_position(self) -> None:
        mgr = _mgr()
        _register(mgr)
        mgr.deregister_position("EUR/USD")
        assert "EUR/USD" not in mgr._positions
        # Evaluating deregistered symbol returns HOLD
        dec = mgr.evaluate_price("EUR/USD", 100.0, BASE_MS)
        assert not dec.should_exit


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_position_evaluate_price_hold(self) -> None:
        mgr = _mgr()
        dec = mgr.evaluate_price("EUR/USD", 100.0, BASE_MS)
        assert dec.reason == ExitReason.HOLD

    def test_no_position_evaluate_signal_hold(self) -> None:
        mgr = _mgr()
        sig = FakeSignal("EUR/USD", 0.05, 0.01, 0.8, 0.5, 0.02)
        dec = mgr.evaluate_signal("EUR/USD", sig, in_topk=True)
        assert dec.reason == ExitReason.HOLD

    def test_no_position_evaluate_sentiment_hold(self) -> None:
        mgr = _mgr({"sentiment_reversal_enabled": True})
        sent = FakeSentiment("EUR/USD", -1.0, 1.0)
        dec = mgr.evaluate_sentiment("EUR/USD", sent)
        assert dec.reason == ExitReason.HOLD

    def test_multiple_positions_tracked_independently(self) -> None:
        mgr = _mgr()
        _register(mgr, "EUR/USD", entry_price=1.10, mean_return=0.05, stop_pct=0.02)
        _register(mgr, "GBP/USD", entry_price=1.30, mean_return=0.03, stop_pct=0.02)
        # EUR/USD hits TP; GBP/USD does not
        dec_eur = mgr.evaluate_price("EUR/USD", 1.1 * 1.04, BASE_MS + 1)
        dec_gbp = mgr.evaluate_price("GBP/USD", 1.30, BASE_MS + 1)
        assert dec_eur.should_exit
        assert not dec_gbp.should_exit

    def test_evaluate_signal_none_signal_increments_strikes(self) -> None:
        mgr = _mgr({"signal_decay_max_strikes": 2})
        _register(mgr)
        mgr.evaluate_signal("EUR/USD", None, in_topk=False)
        assert mgr._positions["EUR/USD"].signal_decay_strikes == 1
        dec = mgr.evaluate_signal("EUR/USD", None, in_topk=False)
        assert dec.should_exit
        assert dec.reason == ExitReason.SIGNAL_DECAY_STRIKES


# ---------------------------------------------------------------------------
# Preflight #3 — synthetic ex-div step-down audit.
# IG_LIVE_RISK_REFERENCE.md §1.2: a 2% ex-div price step-down on a held
# index is not a real adverse move.  Ex-dividend suppression is the
# planned fix.  Until it lands, the bot does NOT distinguish ex-div
# from a real move — these tests pin the current behaviour so the
# suppression change can flip the assertions and see the surface change.
# ---------------------------------------------------------------------------


@pytest.mark.preflight
class TestExDivSyntheticDrop:
    """Behaviour audit: 2% step-down on a held index.

    Fresh-position case is genuinely safe today (no trailing stop armed →
    HOLD).  Profitable-position case is the actual exposure: the 2% drop
    would knock out a position whose trailing stop has armed at breakeven.
    That second outcome is what ex-dividend suppression must prevent.
    """

    def test_fresh_position_2pct_drop_holds(self) -> None:
        """Just-opened position, no profit accumulated: a 2% drop sits well
        within the entry stop_pct=2.5% buffer and the trailing stop is not
        armed.  No exit fires."""
        mgr = _mgr()
        _register(mgr, "SPY", entry_price=500.0, mean_return=0.05, stop_pct=0.025)
        # 2% ex-div drop, immediately after open
        dec = mgr.evaluate_price("SPY", 500.0 * 0.98, BASE_MS + 1)
        assert not dec.should_exit, (
            "Fresh position should hold through a 2% drop — trailing stop not "
            "armed yet and static TP only fires upward. If this changes, the "
            "exit-component priority order has shifted."
        )
        assert dec.reason == ExitReason.HOLD

    def test_profitable_position_2pct_drop_currently_triggers_ratchet_exit(self) -> None:
        """A position that has rallied past Stage 2 activation has its
        trailing stop ratcheted up just below the peak.  A 2% ex-div drop
        from that peak breaches the trail and force-closes the position.
        This documents the gap that ex-dividend suppression closes:
        the bot cannot distinguish an ex-div print from a real drop, so
        the position is liquidated for what is in fact a cash distribution.
        When it lands, flip the assertion to ``not dec.should_exit``."""
        mgr = _mgr()
        # stop_pct=2.5%, trail_activation_mult=2.0 ⇒ Stage 2 arms at +5% profit.
        # trail_multiplier=0.5 ⇒ ratchet sits at peak × (1 − 1.25%) = peak × 0.9875.
        # A 2% drop from peak (peak × 0.98) therefore crosses below the trail.
        _register(mgr, "SPY", entry_price=500.0, mean_return=0.05, stop_pct=0.025)
        peak = 500.0 * 1.05  # 5% profit — Stage 2 armed
        mgr.evaluate_price("SPY", peak, BASE_MS + 1)
        post_div = peak * 0.98  # 2% ex-div drop from peak
        dec = mgr.evaluate_price("SPY", post_div, BASE_MS + 2)
        assert dec.should_exit, (
            "Audit pin: without ex-div suppression, a 2% drop on a ratchet-armed "
            "position closes it.  If this assertion now fails, suppression (or "
            "equivalent) has landed — update the gate and the audit doc."
        )
        assert dec.reason == ExitReason.TRAILING_STOP_RATCHET
