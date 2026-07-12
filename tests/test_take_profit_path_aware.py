"""Tests for path-aware take-profit logic.

Covers:
  - register_position stores path fields from KronosPathSignal
  - register_position with no path_signal leaves path fields as None
  - Static TP uses predicted_mfe_pct × kronos_mfe_capture_fraction when present
  - Static TP falls back to mean_return × kronos_target_fraction when absent
  - Time exit uses opened_at_ms + predicted_peak_bar × bar_interval_ms + grace when present
  - Time exit falls back to pred_len × time_horizon_multiplier when absent
  - peak_bar=0 does not cause immediate exit (still need to wait for grace period)
  - snapshot / restore roundtrips path fields correctly
"""

from __future__ import annotations

from dataclasses import dataclass

from bot.strategy.take_profit import (
    ExitReason,
    TakeProfitConfig,
    TakeProfitManager,
)

# ---------------------------------------------------------------------------
# Minimal signal stubs
# ---------------------------------------------------------------------------


@dataclass
class _Signal:
    mean_return: float = 0.010
    stop_pct: float = 0.005
    direction_confidence: float = 0.80
    uncertainty: float = 0.50


@dataclass
class _PathSignal:
    predicted_mfe_pct: float = 0.020
    predicted_mae_pct: float = 0.006
    predicted_peak_bar: int = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HOUR_MS = 3_600_000


def _manager(
    pred_len: int = 120,
    static: bool = True,
    trailing: bool = False,
    time_exit: bool = True,
    mfe_fraction: float = 0.85,
    grace_hours: float = 12.0,
) -> TakeProfitManager:
    cfg = TakeProfitConfig(
        static_enabled=static,
        trailing_enabled=trailing,
        time_enabled=time_exit,
        signal_decay_enabled=False,
        sentiment_reversal_enabled=False,
        kronos_mfe_capture_fraction=mfe_fraction,
        time_post_peak_grace_hours=grace_hours,
    )
    return TakeProfitManager(cfg, pred_len)


# ---------------------------------------------------------------------------
# register_position — path field storage
# ---------------------------------------------------------------------------


class TestRegisterPositionPathFields:
    def test_path_fields_stored_when_path_signal_present(self) -> None:
        mgr = _manager()
        sig = _Signal()
        path = _PathSignal(predicted_mfe_pct=0.025, predicted_mae_pct=0.008, predicted_peak_bar=15)
        mgr.register_position("EUR/USD", 1.0850, sig, opened_at_ms=0, path_signal=path)
        state = mgr._positions["EUR/USD"]
        assert abs(state.predicted_mfe_pct - 0.025) < 1e-9  # type: ignore[operator]
        assert abs(state.predicted_mae_pct - 0.008) < 1e-9  # type: ignore[operator]
        assert state.predicted_peak_bar == 15

    def test_path_fields_none_when_no_path_signal(self) -> None:
        mgr = _manager()
        mgr.register_position("EUR/USD", 1.0850, _Signal(), opened_at_ms=0)
        state = mgr._positions["EUR/USD"]
        assert state.predicted_mfe_pct is None
        assert state.predicted_mae_pct is None
        assert state.predicted_peak_bar is None

    def test_bar_interval_stored(self) -> None:
        mgr = _manager()
        mgr.register_position(
            "EUR/USD",
            1.0850,
            _Signal(),
            opened_at_ms=0,
            path_signal=_PathSignal(),
            bar_interval_ms=900_000,
        )
        state = mgr._positions["EUR/USD"]
        assert state.bar_interval_ms == 900_000

    def test_default_bar_interval_is_one_hour(self) -> None:
        mgr = _manager()
        mgr.register_position("EUR/USD", 1.0850, _Signal(), opened_at_ms=0)
        assert mgr._positions["EUR/USD"].bar_interval_ms == _HOUR_MS


# ---------------------------------------------------------------------------
# Static TP — path-aware vs fallback
# ---------------------------------------------------------------------------


class TestStaticTPPathAware:
    def test_tp_uses_mfe_pct_when_available(self) -> None:
        """
        Entry = 1.0850; mfe_pct=0.040; fraction=0.85 → tp_pct = 0.034
        stop_pct = 0.005; min_rr = 1.5 → rr_floor = 0.0075
        So tp_pct = max(0.0075, 0.034) = 0.034
        tp_price = 1.0850 * 1.034 = 1.12219
        """
        mgr = _manager(static=True, trailing=False, time_exit=False, mfe_fraction=0.85)
        sig = _Signal(mean_return=0.010, stop_pct=0.005)
        path = _PathSignal(predicted_mfe_pct=0.040, predicted_mae_pct=0.005, predicted_peak_bar=5)
        entry = 1.0850
        mgr.register_position("S", entry, sig, opened_at_ms=0, path_signal=path)

        tp_pct = 0.040 * 0.85  # 0.034
        tp_price = entry * (1.0 + tp_pct)

        # Just below → HOLD
        hold = mgr.evaluate_price("S", tp_price - 0.0001, now_ms=1)
        assert not hold.should_exit

        # At / above → STATIC_TP
        fire = mgr.evaluate_price("S", tp_price + 0.0001, now_ms=2)
        assert fire.should_exit
        assert fire.reason == ExitReason.STATIC_TP
        assert "predicted_mfe_pct" in fire.reasoning

    def test_tp_fallback_to_mean_return_when_no_path(self) -> None:
        """
        Entry = 1.0850; mean_return=0.020; target_fraction=0.80
        rr_floor = 0.005 * 1.5 = 0.0075
        tp_pct = max(0.0075, 0.020 * 0.80) = max(0.0075, 0.016) = 0.016
        tp_price = 1.0850 * 1.016 ≈ 1.10224
        """
        mgr = _manager(static=True, trailing=False, time_exit=False)
        sig = _Signal(mean_return=0.020, stop_pct=0.005)
        entry = 1.0850
        mgr.register_position("S", entry, sig, opened_at_ms=0)  # no path_signal

        tp_pct = max(0.005 * 1.5, 0.020 * 0.80)  # 0.016
        tp_price = entry * (1.0 + tp_pct)

        hold = mgr.evaluate_price("S", tp_price - 0.0001, now_ms=1)
        assert not hold.should_exit

        fire = mgr.evaluate_price("S", tp_price + 0.0001, now_ms=2)
        assert fire.should_exit
        assert fire.reason == ExitReason.STATIC_TP
        assert "mean_return" in fire.reasoning

    def test_rr_floor_respected_when_mfe_too_small(self) -> None:
        """mfe_pct × fraction < stop × min_rr → floor wins."""
        mgr = _manager(static=True, trailing=False, time_exit=False, mfe_fraction=0.85)
        sig = _Signal(mean_return=0.001, stop_pct=0.010)
        # mfe_pct=0.005, fraction=0.85 → 0.00425 < rr_floor=0.015 → floor
        path = _PathSignal(predicted_mfe_pct=0.005, predicted_mae_pct=0.003, predicted_peak_bar=2)
        entry = 1.0
        mgr.register_position("S", entry, sig, opened_at_ms=0, path_signal=path)

        rr_floor = 0.010 * 1.5  # 0.015
        tp_price = entry * (1.0 + rr_floor)

        fire = mgr.evaluate_price("S", tp_price + 0.0001, now_ms=1)
        assert fire.should_exit
        assert fire.reason == ExitReason.STATIC_TP


# ---------------------------------------------------------------------------
# Time exit — path-aware vs fallback
# ---------------------------------------------------------------------------


class TestTimeExitPathAware:
    def test_time_exit_uses_peak_bar_plus_grace(self) -> None:
        """
        opened_at_ms = 0, peak_bar = 10, bar_interval = 1h, grace = 12h
        deadline = 0 + 10 * 3600000 + 12 * 3600000 = 22 * 3600000
        """
        mgr = _manager(static=False, trailing=False, time_exit=True, grace_hours=12.0)
        sig = _Signal(stop_pct=0.0)  # stop_pct=0 disables static TP (also static=False)
        path = _PathSignal(predicted_peak_bar=10, predicted_mfe_pct=0.02, predicted_mae_pct=0.005)
        opened_at = 0
        mgr.register_position("S", 1.0, sig, opened_at_ms=opened_at, path_signal=path)

        deadline_ms = opened_at + 10 * _HOUR_MS + 12 * _HOUR_MS  # = 22 hours

        # One ms before deadline → HOLD
        hold = mgr.evaluate_price("S", 1.01, now_ms=deadline_ms - 1)
        assert not hold.should_exit

        # One ms after deadline → TIME_LIMIT
        fire = mgr.evaluate_price("S", 1.01, now_ms=deadline_ms + 1)
        assert fire.should_exit
        assert fire.reason == ExitReason.TIME_LIMIT
        assert "predicted peak bar" in fire.reasoning

    def test_time_exit_fallback_to_pred_len(self) -> None:
        """
        pred_len=120, multiplier=1.0 → max_age = 120h
        No path_signal → uses old formula.
        """
        mgr = _manager(static=False, trailing=False, time_exit=True, pred_len=120)
        sig = _Signal(stop_pct=0.0)
        opened_at = 0
        mgr.register_position("S", 1.0, sig, opened_at_ms=opened_at)

        max_age_ms = 120 * _HOUR_MS

        hold = mgr.evaluate_price("S", 1.01, now_ms=max_age_ms - 1)
        assert not hold.should_exit

        fire = mgr.evaluate_price("S", 1.01, now_ms=max_age_ms)
        assert fire.should_exit
        assert fire.reason == ExitReason.TIME_LIMIT
        assert "predicted peak bar" not in fire.reasoning

    def test_peak_bar_zero_does_not_fire_immediately(self) -> None:
        """peak_bar=0 means price peaks immediately; still need to wait grace hours."""
        mgr = _manager(static=False, trailing=False, time_exit=True, grace_hours=12.0)
        sig = _Signal(stop_pct=0.0)
        path = _PathSignal(predicted_peak_bar=0, predicted_mfe_pct=0.01, predicted_mae_pct=0.003)
        opened_at = 0
        mgr.register_position("S", 1.0, sig, opened_at_ms=opened_at, path_signal=path)

        # Deadline = 0 + 0 * interval + 12h = 12h
        deadline_ms = 12 * _HOUR_MS

        # Right after entry — must HOLD
        hold = mgr.evaluate_price("S", 1.01, now_ms=1)
        assert not hold.should_exit

        # Just past 12h grace → TIME_LIMIT
        fire = mgr.evaluate_price("S", 1.01, now_ms=deadline_ms + 1)
        assert fire.should_exit
        assert fire.reason == ExitReason.TIME_LIMIT

    def test_custom_bar_interval_propagates(self) -> None:
        """15-min bars (bar_interval_ms=900_000): peak at bar 20 + 12h grace."""
        bar_ms = 900_000  # 15 minutes
        grace_hours = 12.0
        mgr = _manager(static=False, trailing=False, time_exit=True, grace_hours=grace_hours)
        sig = _Signal(stop_pct=0.0)
        path = _PathSignal(predicted_peak_bar=20, predicted_mfe_pct=0.01, predicted_mae_pct=0.003)
        opened_at = 0
        mgr.register_position(
            "S",
            1.0,
            sig,
            opened_at_ms=opened_at,
            path_signal=path,
            bar_interval_ms=bar_ms,
        )

        deadline_ms = opened_at + 20 * bar_ms + int(grace_hours * _HOUR_MS)

        hold = mgr.evaluate_price("S", 1.01, now_ms=deadline_ms - 1)
        assert not hold.should_exit

        fire = mgr.evaluate_price("S", 1.01, now_ms=deadline_ms + 1)
        assert fire.should_exit
        assert fire.reason == ExitReason.TIME_LIMIT


# ---------------------------------------------------------------------------
# snapshot / restore — path fields roundtrip
# ---------------------------------------------------------------------------


class TestSnapshotRestorePathFields:
    def test_snapshot_includes_path_fields(self) -> None:
        mgr = _manager()
        sig = _Signal()
        path = _PathSignal(predicted_mfe_pct=0.030, predicted_mae_pct=0.009, predicted_peak_bar=7)
        mgr.register_position("EUR/USD", 1.0850, sig, opened_at_ms=1_000_000, path_signal=path)
        snap = mgr.snapshot()
        state = snap["EUR/USD"]
        assert abs(state["predicted_mfe_pct"] - 0.030) < 1e-9
        assert abs(state["predicted_mae_pct"] - 0.009) < 1e-9
        assert state["predicted_peak_bar"] == 7

    def test_restore_roundtrips_path_fields(self) -> None:
        mgr1 = _manager()
        sig = _Signal()
        path = _PathSignal(predicted_mfe_pct=0.030, predicted_mae_pct=0.009, predicted_peak_bar=7)
        mgr1.register_position("EUR/USD", 1.0850, sig, opened_at_ms=1_000_000, path_signal=path)
        snap = mgr1.snapshot()

        mgr2 = _manager()
        mgr2.restore(snap)
        state = mgr2._positions["EUR/USD"]
        assert abs(state.predicted_mfe_pct - 0.030) < 1e-9  # type: ignore[operator]
        assert abs(state.predicted_mae_pct - 0.009) < 1e-9  # type: ignore[operator]
        assert state.predicted_peak_bar == 7

    def test_restore_handles_null_path_fields(self) -> None:
        mgr1 = _manager()
        mgr1.register_position("GBP/USD", 1.2700, _Signal(), opened_at_ms=0)
        snap = mgr1.snapshot()

        mgr2 = _manager()
        mgr2.restore(snap)
        state = mgr2._positions["GBP/USD"]
        assert state.predicted_mfe_pct is None
        assert state.predicted_peak_bar is None

    def test_restored_position_still_exits_via_peak_bar(self) -> None:
        """After restore, time exit using predicted_peak_bar must still fire."""
        mgr1 = _manager(static=False, trailing=False, time_exit=True, grace_hours=0.0)
        sig = _Signal(stop_pct=0.0)
        path = _PathSignal(predicted_peak_bar=5, predicted_mfe_pct=0.01, predicted_mae_pct=0.003)
        opened_at = 0
        mgr1.register_position("USD/JPY", 150.0, sig, opened_at_ms=opened_at, path_signal=path)
        snap = mgr1.snapshot()

        mgr2 = _manager(static=False, trailing=False, time_exit=True, grace_hours=0.0)
        mgr2.restore(snap)

        deadline_ms = opened_at + 5 * _HOUR_MS  # grace=0 → exactly 5h
        fire = mgr2.evaluate_price("USD/JPY", 150.5, now_ms=deadline_ms + 1)
        assert fire.should_exit
        assert fire.reason == ExitReason.TIME_LIMIT
