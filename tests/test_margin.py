"""Tests for bot.risk.margin — the margin-utilisation circuit breaker.

Exercises ``MarginCircuitBreaker`` directly (its behaviour through the
``RiskManager`` facade is covered in ``test_risk_manager.py``): threshold
classification, edge-triggered event emission, the typed
``MarginBreakerEvent`` payload, the LIQUIDATION shutdown escalation, and
the risk-event ledger callback.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.core.event_bus import EVENT_MARGIN_BREAKER, EVENT_SHUTDOWN, EventBus
from bot.core.models import AccountUpdate, MarginBreakerEvent, MarginCircuitState
from bot.risk.risk_config import RiskConfig

_NOW_MS = 1_700_000_000_000


def _make_breaker(config: RiskConfig | None = None):
    from bot.risk.margin import MarginCircuitBreaker

    bus = MagicMock(spec=EventBus)
    bus.emit = AsyncMock()
    events: list[tuple[str, dict[str, Any]]] = []
    breaker = MarginCircuitBreaker(
        config or RiskConfig(),
        bus,
        clock_fn=lambda: _NOW_MS,
        risk_event_callback=lambda et, payload: events.append((et, payload)),
    )
    return breaker, bus, events


def _update(equity: float, margin_required: float) -> AccountUpdate:
    return AccountUpdate(
        timestamp=_NOW_MS,
        equity=equity,
        margin_required=margin_required,
        available_to_deal=equity - margin_required,
        unrealised_pnl=0.0,
    )


class TestClassify:
    """Threshold mapping per IG_LIVE_RISK_REFERENCE.md §4.3 (defaults
    0.80 / 0.65 / 0.55 / 0.50, all inclusive lower bounds)."""

    @pytest.mark.parametrize(
        ("ratio", "expected"),
        [
            (float("inf"), MarginCircuitState.NORMAL),
            (1.50, MarginCircuitState.NORMAL),
            (0.81, MarginCircuitState.NORMAL),
            (0.80, MarginCircuitState.HALT_ENTRIES),  # boundary is inclusive
            (0.66, MarginCircuitState.HALT_ENTRIES),
            (0.65, MarginCircuitState.DEFENSIVE_CLOSE),
            (0.56, MarginCircuitState.DEFENSIVE_CLOSE),
            (0.55, MarginCircuitState.EMERGENCY_FLATTEN),
            (0.51, MarginCircuitState.EMERGENCY_FLATTEN),
            (0.50, MarginCircuitState.LIQUIDATION),
            (0.10, MarginCircuitState.LIQUIDATION),
        ],
    )
    def test_thresholds(self, ratio: float, expected: MarginCircuitState) -> None:
        breaker, _, _ = _make_breaker()
        assert breaker.classify(ratio) == expected

    def test_thresholds_come_from_config(self) -> None:
        cfg = RiskConfig(margin_halt_ratio=1.20)
        breaker, _, _ = _make_breaker(cfg)
        assert breaker.classify(1.10) == MarginCircuitState.HALT_ENTRIES
        assert breaker.classify(1.30) == MarginCircuitState.NORMAL


class TestUpdate:
    def test_no_positions_is_healthy_infinity(self) -> None:
        breaker, bus, _ = _make_breaker()
        state = breaker.update(_update(equity=10_000.0, margin_required=0.0))
        assert state == MarginCircuitState.NORMAL
        assert breaker.ratio == float("inf")
        bus.emit.assert_not_awaited()

    def test_ratio_and_margin_required_tracked(self) -> None:
        breaker, _, _ = _make_breaker()
        breaker.update(_update(equity=9_000.0, margin_required=10_000.0))
        assert breaker.ratio == pytest.approx(0.9)
        assert breaker.margin_required == pytest.approx(10_000.0)

    def test_negative_inputs_clamped(self) -> None:
        """A garbled LS push with negative numbers must not flip the sign of
        the ratio — both inputs clamp to 0, margin 0 reads as healthy."""
        breaker, _, _ = _make_breaker()
        state = breaker.update(_update(equity=-500.0, margin_required=-100.0))
        assert state == MarginCircuitState.NORMAL
        assert breaker.ratio == float("inf")

    async def test_transition_emits_typed_breaker_event(self) -> None:
        breaker, bus, _ = _make_breaker()
        state = breaker.update(_update(equity=7_000.0, margin_required=10_000.0))  # 0.70
        await asyncio.sleep(0)  # let the create_task emission run
        assert state == MarginCircuitState.HALT_ENTRIES

        bus.emit.assert_awaited_once()
        event_type, event = bus.emit.await_args.args
        assert event_type == EVENT_MARGIN_BREAKER
        assert isinstance(event, MarginBreakerEvent)
        assert event.state == MarginCircuitState.HALT_ENTRIES
        assert event.action == "halt_entries"
        assert event.ratio == pytest.approx(0.7)
        assert event.equity == pytest.approx(7_000.0)
        assert event.timestamp == _NOW_MS

    async def test_steady_state_does_not_re_emit(self) -> None:
        """Edge-triggered: staying inside HALT_ENTRIES must not spam the
        action handler on every ACCOUNT push."""
        breaker, bus, _ = _make_breaker()
        breaker.update(_update(equity=7_000.0, margin_required=10_000.0))  # 0.70
        breaker.update(_update(equity=6_900.0, margin_required=10_000.0))  # 0.69, same state
        breaker.update(_update(equity=7_100.0, margin_required=10_000.0))  # 0.71, same state
        await asyncio.sleep(0)
        assert bus.emit.await_count == 1

    @pytest.mark.parametrize(
        ("ratio_equity", "expected_action"),
        [
            (7_000.0, "halt_entries"),  # 0.70 → HALT_ENTRIES
            (6_000.0, "close_worst"),  # 0.60 → DEFENSIVE_CLOSE
            (5_200.0, "flatten"),  # 0.52 → EMERGENCY_FLATTEN
            (4_000.0, "liquidation_alert"),  # 0.40 → LIQUIDATION
        ],
    )
    async def test_each_state_dispatches_its_action(
        self, ratio_equity: float, expected_action: str
    ) -> None:
        breaker, bus, _ = _make_breaker()
        breaker.update(_update(equity=ratio_equity, margin_required=10_000.0))
        await asyncio.sleep(0)
        event = bus.emit.await_args_list[0].args[1]
        assert event.action == expected_action

    async def test_liquidation_also_emits_shutdown(self) -> None:
        breaker, bus, _ = _make_breaker()
        breaker.update(_update(equity=4_000.0, margin_required=10_000.0))  # 0.40
        await asyncio.sleep(0)
        emitted = {call.args[0] for call in bus.emit.await_args_list}
        assert emitted == {EVENT_MARGIN_BREAKER, EVENT_SHUTDOWN}

    async def test_recovery_transitions_without_breaker_event(self) -> None:
        """HALT_ENTRIES → NORMAL is a transition (logged via the ledger
        callback) but carries no de-risking action, so no breaker event."""
        breaker, bus, events = _make_breaker()
        breaker.update(_update(equity=7_000.0, margin_required=10_000.0))  # → HALT
        breaker.update(_update(equity=12_000.0, margin_required=10_000.0))  # → NORMAL
        await asyncio.sleep(0)
        assert breaker.state == MarginCircuitState.NORMAL
        assert bus.emit.await_count == 1  # only the HALT transition emitted
        assert [(e[1]["from"], e[1]["to"]) for e in events] == [
            ("normal", "halt_entries"),
            ("halt_entries", "normal"),
        ]

    def test_ledger_callback_payload(self) -> None:
        breaker, _, events = _make_breaker()
        breaker.update(_update(equity=6_000.0, margin_required=10_000.0))  # 0.60
        assert len(events) == 1
        event_type, payload = events[0]
        assert event_type == "margin_circuit_transition"
        assert payload["from"] == "normal"
        assert payload["to"] == "defensive_close"
        assert payload["ratio"] == pytest.approx(0.6)
