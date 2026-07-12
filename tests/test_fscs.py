"""Tests for the FSCS soft ceiling (IG_LIVE_RISK_REFERENCE.md §7.2).

Covers FSCSCeiling in isolation and via RiskManager integration:
- warn fires exactly once on upward £100K crossing
- hysteresis re-arms after equity drops below warn − rearm band
- cap_for_sizing clamps at £120K, passes through below it
- RiskManager.equity_for_sizing wires to the FSCS clamp
- compute_ig_size with capped equity yields a smaller stake
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from bot.core.event_bus import EventBus
from bot.risk.fscs import FSCSCeiling
from bot.risk.risk_config import RiskConfig
from bot.risk.risk_manager import RiskManager


def _captured_events() -> tuple[FSCSCeiling, list[tuple[str, dict]]]:
    """Build an FSCSCeiling wired to a list-capturing risk-event callback."""
    events: list[tuple[str, dict]] = []
    fscs = FSCSCeiling(
        RiskConfig(),
        risk_event_callback=lambda evt, details: events.append((evt, details)),
    )
    return fscs, events


class TestFSCSCeiling:
    def test_no_warn_below_threshold(self) -> None:
        fscs, events = _captured_events()
        fscs.update_equity(99_999.0)
        assert events == []
        assert fscs.warned is False

    def test_warn_fires_once_on_crossing(self) -> None:
        fscs, events = _captured_events()
        fscs.update_equity(100_000.0)
        assert fscs.warned is True
        assert len(events) == 1
        evt_type, details = events[0]
        assert evt_type == "fscs_warn"
        assert details["equity_gbp"] == 100_000.0
        assert details["warn_gbp"] == 100_000.0
        assert details["cap_gbp"] == 120_000.0

    def test_no_warn_spam_on_repeat_updates(self) -> None:
        fscs, events = _captured_events()
        fscs.update_equity(100_500.0)
        fscs.update_equity(101_000.0)
        fscs.update_equity(105_000.0)
        fscs.update_equity(130_000.0)  # well past cap, still one event
        assert len(events) == 1

    def test_hysteresis_rearm_after_drop_below_band(self) -> None:
        """Once equity drops more than the rearm band below the warn line
        the next upward crossing re-fires the event."""
        fscs, events = _captured_events()
        fscs.update_equity(100_500.0)
        assert len(events) == 1
        # Brief dip just below warn — still inside rearm band, must not re-arm
        fscs.update_equity(99_500.0)
        assert fscs.warned is True
        fscs.update_equity(100_500.0)
        assert len(events) == 1

        # Deep dip below the rearm band — re-arms
        fscs.update_equity(95_000.0)
        assert fscs.warned is False
        fscs.update_equity(100_500.0)
        assert len(events) == 2

    def test_cap_for_sizing_below_cap_is_passthrough(self) -> None:
        fscs, _ = _captured_events()
        assert fscs.cap_for_sizing(50_000.0) == 50_000.0
        assert fscs.cap_for_sizing(120_000.0) == 120_000.0

    def test_cap_for_sizing_clamps_above_cap(self) -> None:
        fscs, _ = _captured_events()
        assert fscs.cap_for_sizing(120_000.01) == 120_000.0
        assert fscs.cap_for_sizing(500_000.0) == 120_000.0

    def test_custom_thresholds_honoured(self) -> None:
        cfg = RiskConfig(fscs_warn_gbp=50_000.0, fscs_cap_gbp=60_000.0)
        events: list[tuple[str, dict]] = []
        fscs = FSCSCeiling(cfg, risk_event_callback=lambda e, d: events.append((e, d)))
        fscs.update_equity(49_999.0)
        assert events == []
        fscs.update_equity(50_000.0)
        assert len(events) == 1
        assert fscs.cap_for_sizing(75_000.0) == 60_000.0


class TestRiskManagerFSCSIntegration:
    def _make_rm(self) -> RiskManager:
        bus = MagicMock(spec=EventBus)
        bus.emit = AsyncMock()
        return RiskManager(RiskConfig(), bus, clock_fn=lambda: 0)

    def test_update_equity_records_fscs_warn_event(self) -> None:
        rm = self._make_rm()
        rm.update_equity(50_000.0)
        rm.update_equity(110_000.0)
        # Inspect the persisted risk-event log
        events = rm.get_state().risk_events
        fscs_events = [e for e in events if e["event_type"] == "fscs_warn"]
        assert len(fscs_events) == 1
        assert fscs_events[0]["details"]["equity_gbp"] == 110_000.0

    def test_equity_for_sizing_passes_through_under_cap(self) -> None:
        rm = self._make_rm()
        assert rm.equity_for_sizing(50_000.0) == 50_000.0
        assert rm.equity_for_sizing(120_000.0) == 120_000.0

    def test_equity_for_sizing_clamps_over_cap(self) -> None:
        rm = self._make_rm()
        assert rm.equity_for_sizing(200_000.0) == 120_000.0

    def test_compute_ig_size_with_capped_equity_smaller_than_raw(self) -> None:
        """Sanity-check that callers using equity_for_sizing actually shrink
        their stake.  Comparing raw vs capped at the same risk_pct."""
        rm = self._make_rm()
        raw = RiskManager.compute_ig_size(
            equity_gbp=200_000.0,
            risk_pct=0.01,
            entry_price=4500.0,
            stop_pct=0.01,
            pip_value=1.0,
        )
        capped = RiskManager.compute_ig_size(
            equity_gbp=rm.equity_for_sizing(200_000.0),
            risk_pct=0.01,
            entry_price=4500.0,
            stop_pct=0.01,
            pip_value=1.0,
        )
        assert capped < raw
        # Cap is 120K, raw is 200K → ratio 0.6 (allow ±1 cent for round-trip
        # rounding inside compute_ig_size).
        assert abs(capped - raw * 0.6) < 0.02
