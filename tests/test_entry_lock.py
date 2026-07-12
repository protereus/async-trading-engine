"""Entry serialisation: concurrent hour-boundary candle handlers must not each
pass the risk gate on a stale ``_open_positions`` snapshot and overshoot the
caps. ``process_candle_ig_topk`` holds ``ctx.entry_lock`` across the gate →
place → register critical section, and re-checks selection/position under it."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from bot.strategy.rerank_runner import RerankRunner


def _stub_ctx() -> SimpleNamespace:
    return SimpleNamespace(
        entry_lock=asyncio.Lock(),
        state=SimpleNamespace(positions={}),
        topk_scanned=True,
        topk_selected={"EUR/USD"},
    )


@pytest.mark.asyncio
async def test_entries_serialised_under_entry_lock() -> None:
    ctx = _stub_ctx()
    runner = RerankRunner(ctx)  # type: ignore[arg-type]

    live = {"now": 0, "max": 0}
    lock_held: list[bool] = []

    async def fake_entry(epic: str, price: float) -> None:
        lock_held.append(ctx.entry_lock.locked())
        live["now"] += 1
        live["max"] = max(live["max"], live["now"])
        await asyncio.sleep(0.01)  # hold across an await, as the real I/O does
        live["now"] -= 1

    runner._attempt_topk_entry = fake_entry  # type: ignore[method-assign]

    with patch("bot.strategy.rerank_runner.is_safe_for_entry", return_value=True):
        await asyncio.gather(
            runner.process_candle_ig_topk("EUR/USD", 1.1, None),
            runner.process_candle_ig_topk("EUR/USD", 1.1, None),
        )

    assert lock_held == [True, True]  # helper always runs under the lock
    assert live["max"] == 1  # never two entries in flight at once


@pytest.mark.asyncio
async def test_same_epic_second_entry_skipped_when_peer_already_filled() -> None:
    # The under-lock re-check guards against double-entering one epic when a
    # peer entry registered the position while we waited for the lock.
    ctx = _stub_ctx()
    runner = RerankRunner(ctx)  # type: ignore[arg-type]

    calls: list[str] = []

    async def fake_entry(epic: str, price: float) -> None:
        calls.append(epic)
        ctx.state.positions[epic] = object()  # the fill registers the position

    runner._attempt_topk_entry = fake_entry  # type: ignore[method-assign]

    with patch("bot.strategy.rerank_runner.is_safe_for_entry", return_value=True):
        await runner.process_candle_ig_topk("EUR/USD", 1.1, None)
        await runner.process_candle_ig_topk("EUR/USD", 1.1, None)

    assert calls == ["EUR/USD"]  # second call skipped by the under-lock re-check


@pytest.mark.asyncio
async def test_deselected_epic_skipped_under_lock() -> None:
    # If the epic dropped out of selection while waiting for the lock, skip.
    ctx = _stub_ctx()
    runner = RerankRunner(ctx)  # type: ignore[arg-type]
    called = False

    async def fake_entry(epic: str, price: float) -> None:
        nonlocal called
        called = True
        ctx.topk_selected.discard(epic)

    runner._attempt_topk_entry = fake_entry  # type: ignore[method-assign]

    with patch("bot.strategy.rerank_runner.is_safe_for_entry", return_value=True):
        # First runs; it deselects EUR/USD. Second must skip (not selected).
        await runner.process_candle_ig_topk("EUR/USD", 1.1, None)
        called = False
        await runner.process_candle_ig_topk("EUR/USD", 1.1, None)

    assert called is False
