"""Tests for the async event bus."""

from __future__ import annotations

import pytest

from bot.core.event_bus import EventBus


@pytest.mark.asyncio
async def test_subscribe_and_emit() -> None:
    bus = EventBus()
    received: list[object] = []

    async def handler(data: object) -> None:
        received.append(data)

    bus.subscribe("test_event", handler)
    await bus.emit("test_event", "hello")
    assert received == ["hello"]


@pytest.mark.asyncio
async def test_emit_no_subscribers() -> None:
    """Emitting an event with no subscribers should not raise."""
    bus = EventBus()
    await bus.emit("no_one_listening", {"key": "value"})


@pytest.mark.asyncio
async def test_multiple_subscribers() -> None:
    bus = EventBus()
    calls: list[str] = []

    async def h1(data: object) -> None:
        calls.append("h1")

    async def h2(data: object) -> None:
        calls.append("h2")

    bus.subscribe("evt", h1)
    bus.subscribe("evt", h2)
    await bus.emit("evt", None)
    assert sorted(calls) == ["h1", "h2"]


@pytest.mark.asyncio
async def test_handler_exception_does_not_crash_bus() -> None:
    """An exception in one handler should not prevent other handlers from running."""
    bus = EventBus()
    called: list[str] = []

    async def bad_handler(data: object) -> None:
        raise RuntimeError("I crashed")

    async def good_handler(data: object) -> None:
        called.append("good")

    bus.subscribe("evt", bad_handler)
    bus.subscribe("evt", good_handler)
    await bus.emit("evt", None)
    assert called == ["good"]


@pytest.mark.asyncio
async def test_unsubscribe() -> None:
    bus = EventBus()
    calls: list[str] = []

    async def handler(data: object) -> None:
        calls.append("called")

    bus.subscribe("evt", handler)
    bus.unsubscribe("evt", handler)
    await bus.emit("evt", None)
    assert calls == []


@pytest.mark.asyncio
async def test_unsubscribe_nonexistent_does_not_raise() -> None:
    bus = EventBus()

    async def handler(data: object) -> None:
        pass

    bus.unsubscribe("evt", handler)  # should not raise


@pytest.mark.asyncio
async def test_sync_handler_is_called() -> None:
    """Synchronous (non-async) handlers should also be callable."""
    bus = EventBus()
    results: list[object] = []

    def sync_handler(data: object) -> None:
        results.append(data)

    bus.subscribe("evt", sync_handler)
    await bus.emit("evt", 42)
    assert results == [42]
