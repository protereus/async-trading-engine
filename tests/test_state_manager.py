"""Tests for the state manager."""

from __future__ import annotations

import os

import pytest

from bot.core.models import BotState, OrderSide, Position
from bot.state.state_manager import StateManager


def make_state() -> BotState:
    state = BotState(
        equity=12_000.0,
        peak_equity=12_500.0,
        pnl_24h=100.0,
        bot_started_at=1_700_000_000_000,
        last_heartbeat=1_700_000_060_000,
    )
    state.last_candle_timestamps["AVAX/USDT"] = 1_700_000_060_000
    return state


class TestStateManager:
    def test_save_and_load_roundtrip(self, tmp_path: pytest.TempPathFactory) -> None:
        state_file = str(tmp_path / "state.json")
        mgr = StateManager(state_file)
        original = make_state()
        mgr.save(original)
        recovered = mgr.load()
        assert recovered is not None
        assert recovered.equity == original.equity
        assert recovered.peak_equity == original.peak_equity
        assert recovered.pnl_24h == original.pnl_24h
        assert recovered.bot_started_at == original.bot_started_at
        assert recovered.last_heartbeat == original.last_heartbeat

    def test_load_missing_file_returns_none(self, tmp_path: pytest.TempPathFactory) -> None:
        state_file = str(tmp_path / "nonexistent.json")
        mgr = StateManager(state_file)
        assert mgr.load() is None

    def test_atomic_write_no_tmp_file_left(self, tmp_path: pytest.TempPathFactory) -> None:
        state_file = str(tmp_path / "state.json")
        mgr = StateManager(state_file)
        mgr.save(make_state())
        assert os.path.exists(state_file)
        # No stray temp file (unique-named .bot_state.*.tmp) should survive.
        assert list(tmp_path.glob("*.tmp")) == []  # type: ignore[attr-defined]

    def test_save_tolerates_foreign_tmp_leftover(self, tmp_path: pytest.TempPathFactory) -> None:
        # A leftover fixed-name tmp from the old scheme must not block saves:
        # the unique-name strategy never collides with it. (2026-06-04 regression.)
        state_file = str(tmp_path / "state.json")  # type: ignore[operator]
        stale = tmp_path / "state.json.tmp"  # type: ignore[attr-defined]
        stale.write_text("stale")
        mgr = StateManager(state_file)
        mgr.save(make_state())
        recovered = mgr.load()
        assert recovered is not None
        assert recovered.equity == 12_000.0

    def test_save_and_load_with_position(self, tmp_path: pytest.TempPathFactory) -> None:
        state_file = str(tmp_path / "state.json")
        mgr = StateManager(state_file)
        state = make_state()
        state.positions["AVAX/USDT"] = Position(
            symbol="AVAX/USDT",
            side=OrderSide.BUY,
            entry_price=41.0,
            quantity=10.0,
            current_price=43.0,
            unrealised_pnl=20.0,
            realised_pnl=5.0,
            opened_at=1_700_000_000_000,
            updated_at=1_700_000_001_000,
        )
        mgr.save(state)
        recovered = mgr.load()
        assert recovered is not None
        assert "AVAX/USDT" in recovered.positions
        assert recovered.positions["AVAX/USDT"].entry_price == 41.0

    def test_load_corrupted_returns_none(self, tmp_path: pytest.TempPathFactory) -> None:
        state_file = str(tmp_path / "state.json")
        with open(state_file, "w") as f:
            f.write("this is not valid json {{{{{{")
        mgr = StateManager(state_file)
        result = mgr.load()
        assert result is None
